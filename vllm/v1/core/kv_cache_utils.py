# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-Cache 유틸리티."""

import copy
import hashlib
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, NewType, TypeAlias, overload

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.hashing import sha256_cbor, xxhash_cbor
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import format_gib
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request
from vllm.v1.utils import tensor_data

# BlockHash는 프리픽스 캐싱에 사용되는 단일 KV-cache 블록의 해시를 나타낸다.
# `bytes`와 구분되는 별도 타입으로 다루면,
# raw 바이트 문자열 전달 시의 오용을 줄일 수 있다.
BlockHash = NewType("BlockHash", bytes)

# `BlockHashWithGroupId`는 `BlockHash`와 KV cache group ID를 결합한 타입이다.
# 메모리/성능 효율을 위해 raw bytes로 표현한다.
# 아래 헬퍼 함수는 이 키에서 `BlockHash`와 group id를 pack/unpack 한다.
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

# ExternalBlockHash는 재현 가능한(prefix-cache) 블록 해시에 사용된다.
# 기본 해시 표현이 sha256 bytes로 바뀐 뒤에도 하위 호환을 위해
# `bytes | int` 유니온 타입을 사용한다.
ExternalBlockHash: TypeAlias = bytes | int


def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int
) -> BlockHashWithGroupId:
    """`BlockHash`와 group id를 `BlockHashWithGroupId`로 패킹한다.

    group id는 4바이트 big-endian으로 인코딩해 block hash bytes 뒤에 붙인다.
    이 표현은 tuple 생성을 피하면서도, 필요 시 두 요소를 다시 복원할 수 있다.
    """
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))


def get_block_hash(key: BlockHashWithGroupId) -> BlockHash:
    """`BlockHashWithGroupId`에서 `BlockHash`를 추출한다."""
    return BlockHash(key[:-4])


def get_group_id(key: BlockHashWithGroupId) -> int:
    """`BlockHashWithGroupId`에서 group id를 추출한다."""
    return int.from_bytes(key[-4:], "big", signed=False)


def maybe_convert_block_hash(hash_bytes: BlockHash) -> ExternalBlockHash:
    if not envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES:
        return hash_bytes
    return int.from_bytes(hash_bytes, byteorder="big") & ((1 << 64) - 1)


logger = init_logger(__name__)

# 어떤 prefix 블록 시퀀스든 첫 블록에 사용할 해시 시드.
#
# 해시 충돌을 줄이기 위해 랜덤 값을 사용한다. 필요하면 PYTHONHASHSEED를 지정해
# 프로세스 간 시드를 공유할 수 있다. 이는 PYTHONHASHSEED 미설정 시 랜덤 시드를
# 사용하는 Python hash()의 동작과도 일치한다.
#
# `init_none_hash`가 이 전역 변수를 초기화한다.
NONE_HASH: BlockHash
_CBOR_HASH_FUNCTIONS = frozenset({sha256_cbor, xxhash_cbor})


def init_none_hash(hash_fn: Callable[[Any], bytes]):
    global NONE_HASH

    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is None and hash_fn in _CBOR_HASH_FUNCTIONS:
        logger.warning(
            "PYTHONHASHSEED is not set. This will lead to non-reproducible "
            "block-hashes when using CBOR-based hash functions such as "
            "sha256_cbor or xxhash_cbor. Consider setting PYTHONHASHSEED to a "
            "fixed value for reproducibility."
        )

    if hash_seed is None:
        NONE_HASH = BlockHash(os.urandom(32))
    else:
        NONE_HASH = BlockHash(hash_fn(hash_seed))


@dataclass
class KVCacheBlock:
    """KV-cache 블록 메타데이터."""

    # 블록 ID. 범위는 0 ~ num_gpu_blocks - 1.
    block_id: int
    # 참조 카운트.
    ref_cnt: int = 0
    # 블록의 해시 키(block hash + group id).
    # 블록이 가득 차서 캐시된 경우에만 존재한다.
    _block_hash: BlockHashWithGroupId | None = None

    # free 블록용 이중 연결 리스트를 구성하기 위한 포인터.
    # 아래 두 속성은 FreeKVCacheBlockQueue에서만 조작해야 한다.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    # 절대 캐시되면 안 되는 null 블록인지 여부.
    is_null: bool = False

    @property
    def block_hash(self) -> BlockHashWithGroupId | None:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, block_hash: BlockHashWithGroupId):
        assert self.block_hash is None, (
            "The block already has a hash. This should not happen."
        )
        self._block_hash = block_hash

    def reset_hash(self):
        """블록이 축출(evict)될 때 블록 해시를 초기화한다."""
        self._block_hash = None

    def __repr__(self) -> str:
        # KVCacheBlock 객체를 직접 출력하면 __repr__가 재귀 호출될 수 있으므로
        # 객체 대신 block_id를 사용한다.
        prev_block_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_block_id = self.next_free_block.block_id if self.next_free_block else None
        return (
            f"KVCacheBlock(block_id={self.block_id}, "
            f"ref_cnt={self.ref_cnt}, "
            f"_block_hash={self._block_hash!r}, "
            f"prev_free_block={prev_block_id}, "
            f"next_free_block={next_block_id})"
        )


class FreeKVCacheBlockQueue:
    """KVCacheBlock 리스트를 free 블록 이중 연결 리스트로 관리한다.

    큐 중간 블록을 O(1)에 제거하기 위해 Python 내장 deque 대신 별도 구현을 쓴다.
    C++로 구현된 deque와의 성능 차이를 줄이기 위해, 리스트 조작 중 추가 Python
    객체를 만들지 않고 각 블록의 prev_free_block/next_free_block만 조작한다.

    큐는 초기에는 block ID 순서를 따른다. 블록이 할당됐다가 반납되면,
    아래 축출 우선순위로 다시 뒤에 붙는다.
    1. 가장 오래전에 사용된 블록이 앞쪽(LRU)
    2. 마지막 접근 시점이 같다면(같은 시퀀스에서 할당), 해시 토큰 수가 더 많은
       블록 체인의 꼬리 쪽 블록이 앞쪽
    이 순서를 유지하기 위해 요청 단위로 free할 때 블록 순서를 뒤집는 처리를 하며,
    그 동작은 이 클래스 바깥에서 수행된다.

    인자:
        blocks: KVCacheBlock 객체 리스트.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)

        # 인접 블록 간 이중 연결 포인터를 초기화한다.
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # 분기 수를 줄이기 위해 이중 연결 리스트에 가짜 head/tail을 둔다.
        #
        # 구현상 가짜 head/tail은 절대 pop되지 않으므로,
        # 큐 내부의 실제 블록은 항상 prev/next를 가진다고 가정할 수 있다.
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            # fake_head/fake_tail을 각각 첫/마지막 실제 블록과 연결한다.
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            # 빈 리스트면 가짜 head와 tail만 서로 연결한다.
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

    def popleft(self) -> KVCacheBlock:
        """맨 앞 free 블록을 pop하고 num_free_blocks를 1 줄인다.

        반환:
            맨 앞 free 블록.
        """
        if (
            self.fake_free_list_head.next_free_block is self.fake_free_list_tail
            or self.fake_free_list_head.next_free_block is None
        ):
            assert self.num_free_blocks == 0, (
                f"num_free_blocks ({self.num_free_blocks}) is out of sync "
                "with the free list."
            )
            raise ValueError("No free blocks available")

        first_block: KVCacheBlock = self.fake_free_list_head.next_free_block

        if first_block.next_free_block is None:
            # 블록이 실제 free 리스트에서 왔다면 발생하면 안 된다.
            # 호출 측 로직 버그를 의미한다.
            raise RuntimeError(
                "Invalid block found in popleft() "
                "which doesn't have a valid next_free_block"
            )

        # fake_head를 first_block의 다음 블록(둘째 블록 또는 fake tail)과 연결한다.
        self.fake_free_list_head.next_free_block = first_block.next_free_block
        first_block.next_free_block.prev_free_block = self.fake_free_list_head

        # 연결 리스트에서 해당 블록을 분리한다.
        first_block.prev_free_block = first_block.next_free_block = None

        self.num_free_blocks -= 1
        return first_block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """앞에서 free 블록 n개를 pop하고 num_free_blocks를 n 줄인다.

        인자:
            n: pop할 블록 수.

        반환:
            free 블록 n개 리스트.
        """
        if n == 0:
            return []
        assert self.num_free_blocks >= n
        self.num_free_blocks -= n

        curr_block = self.fake_free_list_head.next_free_block
        # 리스트 head에서 n개를 pop한다.
        ret = []
        for _ in range(n):
            assert curr_block is not None
            ret.append(curr_block)
            last_block = curr_block
            curr_block = curr_block.next_free_block
            # pop된 모든 블록의 prev/next 포인터를 초기화한다.
            last_block.prev_free_block = None
            last_block.next_free_block = None

        if curr_block is not None:
            # 큐가 비어 있지 않다면 fake head를 새 첫 블록과 연결한다.
            self.fake_free_list_head.next_free_block = curr_block
            curr_block.prev_free_block = self.fake_free_list_head
        return ret

    def remove(self, block: KVCacheBlock) -> None:
        """free 리스트 중간의 블록 하나를 제거하고 num_free_blocks를 1 줄인다.

        인자:
            block: 제거할 블록.
        """
        if block.prev_free_block is None or block.next_free_block is None:
            # 블록이 실제 free 리스트 소속이라면 발생하면 안 된다.
            # 호출 측 로직 버그를 의미한다.
            raise RuntimeError(f"remove() called on an invalid block: {block}")

        # 이전 블록을 다음 블록과 연결한다.
        block.prev_free_block.next_free_block = block.next_free_block
        # 다음 블록을 이전 블록과 연결한다.
        block.next_free_block.prev_free_block = block.prev_free_block

        # 연결 리스트에서 해당 블록을 분리한다.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """블록을 free 리스트 뒤로 되돌리고 num_free_blocks를 1 늘린다.

        인자:
            block: 추가할 블록.
        """
        if self.fake_free_list_tail.prev_free_block is None:
            raise RuntimeError(
                "prev_free_block of fake_free_list_tail should always exist"
            )
        last_block: KVCacheBlock = self.fake_free_list_tail.prev_free_block

        # 마지막 블록 뒤에 새 블록을 연결한다.
        last_block.next_free_block = block
        block.prev_free_block = last_block

        # 새 블록 뒤에 fake tail을 연결한다.
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block

        self.num_free_blocks += 1

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """블록 리스트를 free 리스트 뒤로 되돌린다.

        인자:
            blocks: 추가할 블록들.
        """
        if len(blocks) == 0:
            return

        last_block = self.fake_free_list_tail.prev_free_block
        assert last_block is not None, (
            "prev_free_block of fake_free_list_tail should always exist"
        )
        # 연속 블록 간 내부 연결을 구성한다.
        for block in blocks:
            block.prev_free_block = last_block
            last_block.next_free_block = block
            last_block = block

        # <blocks>의 마지막 블록을 fake tail과 연결한다.
        last_block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = last_block

        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """free 리스트의 모든 free 블록을 가져온다. 주로 테스트에서 사용한다.

        반환:
            free 블록 리스트.
        """
        ret = []
        if self.fake_free_list_head.next_free_block is None:
            raise RuntimeError(
                "next_free_block of fake_free_list_head should always exist"
            )
        # 첫 번째 블록부터 시작한다.
        curr_block: KVCacheBlock = self.fake_free_list_head.next_free_block
        # next_free_block이 있으면 아직 fake tail에 도달하지 않은 상태다.
        while curr_block.next_free_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret


def need_extra_keys(request: Request) -> bool:
    """이 요청에 할당된 블록이 추가 해시 키를 필요로 하는지 확인한다.

    인자:
        request (Request): 요청 객체.

    반환:
        bool: 이 요청의 블록에 추가 해시 키가 필요한지 여부.
    """

    # 멀티모달 요청은 MM 해시를 포함해야 한다.
    # LoRA 요청은 LoRA 이름을 포함해야 한다.
    # cache_salt가 제공된 요청은 salt를 포함해야 한다.
    return (
        bool(request.mm_features)
        or (request.lora_request is not None)
        or (request.cache_salt is not None)
    )


def _gen_mm_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[list[Any], int]:
    """블록 해시 계산용 멀티모달 관련 추가 키를 생성한다.

    멀티모달 입력의 경우, 블록에 포함된 MM 입력과 시작 오프셋을 나타내는
    (mm_hash, start_offset) 정보가 추가 키가 된다.

    인자:
        request: 요청 객체.
        start_token_idx: 블록의 시작 토큰 인덱스.
        end_token_idx: 블록의 끝 토큰 인덱스.
        start_mm_idx: 블록 기준 멀티모달 시작 인덱스.

    반환:
        추가 키와 다음 멀티모달 인덱스를 담은 튜플.
    """
    extra_keys: list[Any] = []

    mm_features = request.mm_features
    if not mm_features:
        return extra_keys, start_mm_idx

    # mm_features는 mm_position.offset 기준 정렬되어 있다고 가정한다.
    # 시작 토큰 인덱스가 범위를 벗어나면 모든 MM 입력을 검사할 필요가 없다.
    # 이는 보통 prefill 후반이나 decode 단계에서 발생한다.
    last_pos = mm_features[-1].mm_position
    if last_pos.offset + last_pos.length < start_token_idx:
        return extra_keys, start_mm_idx

    # start_mm_idx == -1이면 마지막 MM 입력을 의미하도록 지원한다.
    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_features)
        start_mm_idx = len(mm_features) + start_mm_idx

    curr_mm_idx = start_mm_idx
    while mm_features and curr_mm_idx < len(mm_features):
        mm_feature = mm_features[curr_mm_idx]
        assert mm_feature.identifier is not None
        offset = mm_feature.mm_position.offset
        length = mm_feature.mm_position.length
        if end_token_idx > offset:
            if start_token_idx > offset + length:
                # 이 블록은 현재 MM 입력 구간을 이미 지난 상태다.
                curr_mm_idx += 1
                continue

            # 이 블록은 현재 MM 입력을 포함한다.
            extra_keys.append(mm_feature.identifier)

            if end_token_idx >= offset + length:
                # 현재 MM 입력의 끝을 포함하면, 다음 MM 입력도 포함할 수 있으므로
                # MM 인덱스를 다음으로 이동한다.
                curr_mm_idx += 1
            else:
                # 그렇지 않으면 이 블록의 MM 처리는 끝이다.
                break
        else:
            # 이 블록은 아직 현재 MM 입력 구간에 도달하지 않았다.
            break
    return extra_keys, curr_mm_idx


def _gen_lora_extra_hash_keys(request: Request) -> list[str]:
    """블록 해시 계산용 LoRA 관련 추가 키를 생성한다.

    인자:
        request: 요청 객체.

    반환:
        LoRA 요청이면 LoRA 이름을 반환하고, 아니면 빈 리스트를 반환한다.
    """
    if not request.lora_request:
        return []
    return [request.lora_request.lora_name]


def _gen_prompt_embeds_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int
) -> list[bytes]:
    """블록 해시 계산용 prompt embeds 관련 추가 키를 생성한다.

    인자:
        request: 요청 객체.
        start_token_idx: 블록의 시작 토큰 인덱스.
        end_token_idx: 블록의 끝 토큰 인덱스.

    반환:
        prompt embeds가 있으면 블록 범위 임베딩의 안정적인 해시를 반환하고,
        없으면 빈 리스트를 반환한다.
    """
    if request.prompt_embeds is None:
        return []
    block_range = (start_token_idx, end_token_idx)
    embeds_hash = request._prompt_embeds_per_block_hashes.get(block_range)
    if embeds_hash is None:
        block_prompt_embeds = request.prompt_embeds[start_token_idx:end_token_idx]
        # 블록 단위로 prompt embeds를 1회 해시해 request에 캐시한다.
        embeds_hash = hashlib.sha256(tensor_data(block_prompt_embeds)).digest()
        request._prompt_embeds_per_block_hashes[block_range] = embeds_hash
    return [embeds_hash]


def generate_block_hash_extra_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[tuple[Any, ...] | None, int]:
    """블록 해시용 추가 키를 생성한다.

    추가 키는 멀티모달 입력, 요청별 메타데이터(예: LoRA 이름),
    prompt 임베딩 해시 데이터에서 올 수 있다.

    인자:
        request: 요청 객체.
        start_token_idx: 블록의 시작 토큰 인덱스.
        end_token_idx: 블록의 끝 토큰 인덱스.
        start_mm_idx: 블록 기준 멀티모달 시작 인덱스.

    반환:
        추가 키와 다음 멀티모달 인덱스를 담은 튜플.
    """
    mm_extra_keys: list[Any]
    mm_extra_keys, new_start_mm_idx = _gen_mm_extra_hash_keys(
        request, start_token_idx, end_token_idx, start_mm_idx
    )
    lora_extra_keys: list[str] = _gen_lora_extra_hash_keys(request)
    cache_salt_keys: list[str] = (
        [request.cache_salt] if (start_token_idx == 0 and request.cache_salt) else []
    )
    prompt_embeds_keys = _gen_prompt_embeds_extra_hash_keys(
        request, start_token_idx, end_token_idx
    )

    extra_keys: list[Any] = (
        lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
    )

    if not extra_keys:
        return None, new_start_mm_idx

    return tuple(extra_keys), new_start_mm_idx


def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """현재 블록과 선행 블록(들)의 내용에 대응하는 해시를 계산한다.

    이 해시는 prefix caching에 사용된다. 동일한 블록 내용의 해시를
    반복 계산하지 않도록 LRU 캐시를 활용한다.
    인자:
        hash_function: 블록 해시 계산에 사용할 해시 함수.
        parent_block_hash: 부모 블록 해시. 첫 블록이면 None.
        curr_block_token_ids: 현재 블록의 토큰 ID 목록.
            현재 블록은 가득 찬 상태라고 가정한다.
        extra_keys: 블록용 추가 키.
    반환:
        블록 해시값과 블록 토큰 ID를 기반으로 한 최종 해시.
        전체 튜플이 블록 해시 키로 사용된다.
    """
    if not parent_block_hash:
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )


def get_request_block_hasher(
    block_size: int,
    caching_hash_fn: Callable[[Any], bytes],
) -> Callable[[Request], list[BlockHash]]:
    """요청에서 아직 계산되지 않은 블록 해시 목록을 계산하는 함수를 반환한다."""

    def request_block_hasher(request: Request) -> list[BlockHash]:
        start_token_idx = len(request.block_hashes) * block_size
        num_tokens = request.num_tokens

        if start_token_idx + block_size > num_tokens:
            # 새로 완성된(full) 블록이 없으면 즉시 종료한다.
            return []

        curr_mm_idx = 0
        if start_token_idx > 0:
            # curr_mm_idx = -1은 마지막 MM 입력을 의미한다.
            # 이 분기는 생성 토큰으로 블록이 완성된 경우에만 오므로
            # 마지막 MM 입력만 고려하면 된다.
            curr_mm_idx = -1

        prev_block_hash_value = (
            request.block_hashes[-1] if request.block_hashes else None
        )
        new_block_hashes: list[BlockHash] = []
        while True:
            end_token_idx = start_token_idx + block_size
            if end_token_idx > num_tokens:
                # 가득 찬 블록만 해싱한다.
                break

            # MM/LoRA 요청은 블록 해시 계산에 추가 키가 필요하다.
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start_token_idx, end_token_idx, curr_mm_idx
            )

            # 현재 블록의 해시를 계산한다.
            block_tokens = request.all_token_ids[start_token_idx:end_token_idx]
            block_hash = hash_block_tokens(
                caching_hash_fn, prev_block_hash_value, block_tokens, extra_keys
            )

            new_block_hashes.append(block_hash)
            start_token_idx += block_size
            prev_block_hash_value = block_hash

        return new_block_hashes

    return request_block_hasher


def _check_enough_kv_cache_memory(
    available_memory: int,
    get_needed_memory: Callable[[], int],
    max_model_len: int,
    estimate_max_model_len: Callable[[int], int],
):
    if available_memory <= 0:
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when initializing the engine. "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more details."
        )

    needed_memory = get_needed_memory()

    if needed_memory > available_memory:
        estimated_max_len = estimate_max_model_len(available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}. "
            )

        raise ValueError(
            f"To serve at least one request with the models's max seq len "
            f"({max_model_len}), ({format_gib(needed_memory)} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({format_gib(available_memory)} GiB). {estimated_msg}"
            f"Try increasing `gpu_memory_utilization` or decreasing `max_model_len` "
            f"when initializing the engine. "
            f"See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            f"for more details."
        )


def max_memory_usage_bytes(
    vllm_config: VllmConfig, kv_cache_specs: Iterable[KVCacheSpec]
) -> int:
    """주어진 KV cache spec들의 최대 메모리 사용량(바이트)을 구한다."""
    return sum(spec.max_memory_usage_bytes(vllm_config) for spec in kv_cache_specs)


def estimate_max_model_len(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
) -> int:
    """이진 탐색으로 가용 메모리에 맞는 최대 모델 길이를 추정한다.

    추정 과정에서 max_model_len을 임시로 바꾸지만, 반환 전에 원래 값으로
    복원하여 부작용이 없도록 한다.

    인자:
        vllm_config: 전역 VllmConfig.
        kv_cache_spec: 모델 각 어텐션 레이어의 KV cache spec.
        available_memory: KV cache에 사용할 수 있는 메모리(바이트).

    반환:
        가용 메모리에 맞는 최대 모델 길이 추정값.
    """
    # 추정 후 복원하기 위해 원래 max_model_len을 저장한다.
    original_max_model_len = vllm_config.model_config.max_model_len

    # 주어진 모델 길이가 메모리에 맞는지 확인하는 함수를 정의한다.
    def fits_in_memory(model_len: int) -> bool:
        # 계산을 위해 max_model_len을 임시로 변경한다.
        vllm_config.model_config.max_model_len = model_len
        # 해당 모델 길이에 필요한 메모리를 계산한다.
        memory_needed = max_memory_usage_bytes(vllm_config, kv_cache_spec.values())
        return memory_needed <= available_memory

    try:
        # 최대 모델 길이를 이진 탐색한다.
        left, right = 1, original_max_model_len

        # 최소 길이(1)도 맞지 않으면 0을 반환한다.
        if not fits_in_memory(left):
            return 0

        # 메모리에 맞는 최대 모델 길이를 이진 탐색한다.
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits_in_memory(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        # 부작용 방지를 위해 원래 max_model_len을 항상 복원한다.
        vllm_config.model_config.max_model_len = original_max_model_len


def check_enough_kv_cache_memory(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
):
    """`available_memory`가 KV cache에 충분한지 확인한다.

    기준은 모델의 max_model_len 요청 최소 1개를 수용할 수 있는지 여부다.

    인자:
        vllm_config: 전역 VllmConfig.
        kv_cache_spec: 모델 각 어텐션 레이어의 KV cache spec.
        available_memory: KV cache에 사용할 수 있는 메모리(바이트).

    예외:
        ValueError: KV cache용 메모리가 부족한 경우.
    """

    # kv_cache_spec이 비어 있으면(어텐션 프리) 메모리 체크가 필요 없다.
    if kv_cache_spec:
        _check_enough_kv_cache_memory(
            available_memory,
            lambda: max_memory_usage_bytes(vllm_config, kv_cache_spec.values()),
            vllm_config.model_config.max_model_len,
            lambda am: estimate_max_model_len(vllm_config, kv_cache_spec, am),
        )


def create_kv_cache_group_specs(
    kv_cache_spec: dict[str, KVCacheSpec], grouped_layer_names: list[list[str]]
) -> list[KVCacheGroupSpec]:
    """각 KV cache 그룹에 대한 KVCacheGroupSpec 객체를 생성한다.

    같은 그룹의 레이어는 동일한 KVCacheSpec을 공유해야 한다.

    인자:
        kv_cache_spec: 레이어 이름 -> 해당 KVCacheSpec 매핑.
        grouped_layer_names: KV cache 그룹 리스트.
            각 원소는 같은 그룹에 속해 동일 KVCacheSpec을 공유해야 하는
            레이어 이름 리스트다.
    반환:
        그룹별 KVCacheGroupSpec 리스트.
    """
    kv_cache_groups = []
    for layer_names_one_group in grouped_layer_names:
        layer_specs = [
            kv_cache_spec[layer_name] for layer_name in layer_names_one_group
        ]
        merged_layer_spec = layer_specs[0].merge(layer_specs)
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names_one_group, merged_layer_spec)
        )
    return kv_cache_groups


def is_kv_cache_spec_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """주어진 KVCacheSpec에서 모든 레이어 spec이 동일한지 확인한다.

    슬라이딩 윈도우 유무가 다른 FullAttentionSpec은 같은 타입으로 본다.

    인자:
        kv_cache_spec: 모델 각 어텐션 레이어의 KV cache spec.

    반환:
        모든 레이어 타입이 같으면 True, 아니면 False.
    """

    if not kv_cache_spec:
        # 인코더 전용 모델은 KV cache가 없으므로 uniform으로 간주한다.
        return True
    try:
        kv_cache_spec_values = list(kv_cache_spec.values())
        _ = kv_cache_spec_values[0].merge(kv_cache_spec_values)
    except AssertionError:
        return False
    return True


def get_max_concurrency_for_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> float:
    """주어진 KV cache 설정에서 가능한 최대 동시성을 계산한다."""
    num_layer_per_group = max(
        len(group.layer_names) for group in kv_cache_config.kv_cache_groups
    )
    max_memory_usage_per_request = num_layer_per_group * max_memory_usage_bytes(
        vllm_config, (group.kv_cache_spec for group in kv_cache_config.kv_cache_groups)
    )
    memory_per_block = (
        kv_cache_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
        * num_layer_per_group
    )
    num_block_per_request = cdiv(max_memory_usage_per_request, memory_per_block)
    max_concurrency = kv_cache_config.num_blocks / num_block_per_request
    return max_concurrency


def may_override_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> int:
    """`num_gpu_blocks_override`가 설정되어 있으면 블록 수를 덮어쓴다."""
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_gpu_blocks_override = vllm_config.cache_config.num_gpu_blocks_override
        logger.info(
            "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
            num_blocks,
            num_gpu_blocks_override,
        )
        num_blocks = num_gpu_blocks_override

    return num_blocks


def get_num_blocks(
    vllm_config: VllmConfig, num_layers: int, available_memory: int, page_size: int
) -> int:
    """KV cache 블록 수를 계산한다.

    인자:
        vllm_config: 전역 VllmConfig.
        num_layers: 레이어 수.
        available_memory: KV cache에 사용할 수 있는 메모리(바이트).
        page_size: KV cache 페이지 크기.
    """
    num_blocks = int(available_memory // page_size // num_layers)
    num_blocks = max(num_blocks, 0)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)
    return num_blocks


def get_uniform_page_size(kv_cache_specs: Iterable[KVCacheSpec]) -> int:
    """KV cache spec들의 공통 페이지 크기를 반환한다."""
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    assert len(page_sizes) == 1
    return page_sizes.pop()


def _get_kv_cache_groups_uniform_spec(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """모든 레이어의 KV cache spec이 동일한 모델용 그룹 설정을 생성한다.

    인자:
        kv_cache_specs: 모델 각 어텐션 레이어의 KV cache spec.

    반환:
        생성된 KVCacheGroupSpec들.
    """

    return create_kv_cache_group_specs(kv_cache_specs, [list(kv_cache_specs.keys())])


def _get_kv_cache_groups_uniform_type(
    spec: UniformTypeKVCacheSpecs,
) -> list[KVCacheGroupSpec]:
    """KV cache 타입은 하나지만 hidden size가 다른 모델용 그룹을 생성한다.

    이 경우 모든 레이어를 하나의 그룹으로 합친다.

    인자:
        spec: 모델의 UniformTypeKVCacheSpecs.

    반환:
        생성된 KVCacheGroupSpec들.
    """

    return [KVCacheGroupSpec(list(spec.kv_cache_specs.keys()), spec)]


def is_kv_cache_page_size_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """주어진 KVCacheSpec에서 모든 레이어의 페이지 크기가 같은지 확인한다.
    인자:
        kv_cache_spec: 모델 각 어텐션 레이어의 KVCacheSpec.

    반환:
        모든 레이어 페이지 크기가 같으면 True, 아니면 False.
    """

    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    return len(page_sizes) == 1


def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """주어진 KVCacheSpec의 페이지 크기를 통일한다.

    모든 레이어의 페이지 크기가 같으면 원본을 그대로 반환한다.
    다르면 페이지가 더 작은 레이어의 block size를 키워 페이지 크기를 맞춘다.
    통일이 불가능하면 NotImplementedError를 발생시킨다.

    인자:
        kv_cache_spec: 모델 각 어텐션 레이어의 KVCacheSpec.

    반환:
        page_size_bytes가 동일하도록 갱신된 KVCacheSpec.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # 모든 레이어 페이지 크기가 같으면 통일할 필요가 없다.
        return kv_cache_spec

    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size != 0:
                raise NotImplementedError(
                    "The page size of the layer is not divisible by the "
                    "maximum page size. Cannot unify by adjusting block_size."
                )
            ratio = max_page_size // layer_page_size
            new_block_size = layer_spec.block_size * ratio
            new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec


def is_kv_cache_type_attention_free(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    # attention-free 모델의 kv_cache_spec은 빈 dict다.
    return not kv_cache_spec


def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """하이브리드 모델(어텐션 타입 다수)의 KV cache 그룹을 생성한다.

    조건은 모든 레이어의 페이지 크기(레이어당 블록당 물리 메모리)가 동일해야 한다.

    하이브리드 모델 KV cache 관리 개요:
    레이어는 반복 패턴을 이루는 경우가 많다. 예를 들어 full 10층 + sliding
    window 20층 모델은 (1 * full, 2 * sw) 패턴을 10회 반복한 형태로 볼 수 있다.
    KVCacheManager는 패턴 내 각 레이어 타입별로 블록 테이블을 만들고, 이를 반복해
    전체 레이어 블록 테이블을 구성한다.

    예:
    1. full attention만 쓰는 모델:
       패턴은 (num_hidden_layers * full)이고 그룹은 1개이며 블록 테이블을 전체
       레이어가 공유한다. (`_get_kv_cache_config_uniform_type`에서 처리)
    2. full 10층 + sliding window 20층 모델:
       패턴은 (1 * full, 2 * sw)로 레이어 슬롯 3개이므로 kv_cache_group도 3개가 되며,
       각 그룹은 모델 내 10개 레이어를 대표한다.

    구현 단순화를 위한 가정:
    1. 블록당 물리 메모리: 모든 KV cache 그룹에서 동일해야 한다.
       크기가 다른 블록을 섞어 할당하면 메모리 단편화 이슈로 복잡해진다.
    2. 블록당 토큰 수(block_size): 현재는 모든 레이어에 `CacheConfig.block_size`
       를 그대로 사용한다. 그룹별로 다르게 확장할 수는 있지만, 같은 그룹 내부
       레이어들은 동일해야 한다.
    3. 레이어별 토큰당 물리 메모리: 모델 설정이 결정한다. 현재는 모든 레이어가
       동일한 경우만 지원한다.
    4. 그룹당 레이어 수: 현재는 동일하다고 가정한다.
    5. 그룹 내 어텐션 타입: 한 그룹의 레이어는 같은 어텐션 타입이어야 한다.
       단, `--disable-hybrid-kv-cache-manager`가 true면 full 그룹에 sliding window
       또는 LLaMA4 local attention 레이어가 섞일 수 있다.
       (`unify_hybrid_kv_cache_specs` 참고)
    6. 어텐션 타입 수: 설계 자체는 임의 개수 타입에 일반적이지만,
       `find_longest_cache_hit`은 현재 1개 타입 또는 full-attention 2종 + 다른 1종
       형태만 지원한다.

    현재는 블록당 토큰 수, 레이어별 토큰당 물리 메모리, 그룹당 레이어 수가 같다고
    가정하므로 모든 그룹에서 블록당 물리 메모리 동일성을 보장할 수 있다.

    인자:
        kv_cache_spec: 모델 각 어텐션 레이어의 KVCacheSpec.
    반환:
        생성된 KVCacheGroupSpec들.
    """
    # kv_cache_spec별로 레이어를 묶는다.
    # 예: full 2개 + sliding window 3개면
    # 결과 예: (full.0, full.1), (sw.0, sw.1, sw.2).
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec].append(layer_name)

    # 각 타입 그룹을 더 작은 그룹으로 나눠 그룹당 레이어 수를 맞춘다.
    # 필요하면 마지막 그룹에 padding 레이어를 추가한다.
    # 예: (full.0, full.1), (sw.0, sw.1, sw.2)
    # -> 2레이어씩 3그룹: (full.0, full.1), (sw.0, sw.2), (sw.1, padding).
    # FIXME(Chen): 이 코드 작성 시점(2025-06-02) 공개 하이브리드 모델은 대부분
    # 어텐션 타입 비율이 n:1 형태다(예: Gemma3 sw:full=5:1, LLaMA4 local:full=3:1).
    # 그래서 최소 레이어 수(비율의 1)를 그룹 크기로 쓸 수 있다.
    # 더 복잡한 패턴(예: full 20 + sw 30) 지원에는 개선이 필요하다.
    min_num_layers = min([len(layers) for layers in same_type_layers.values()])
    group_size = min_num_layers
    max_num_layers = max([len(layers) for layers in same_type_layers.values()])
    if max_num_layers < min_num_layers * 1.25:
        # 레이어 수가 최소값보다 크게 차이 나지 않으면, padding 과다를 피하려고
        # 최대 레이어 수를 그룹 크기로 사용한다.
        # 예: gpt-oss-20b + eagle(12 sw + 13 full)에서
        # (12 sw, 24 full) 대신 (13 sw, 13 full)로 맞춘다.
        # 1.25는 padding 과다를 피하기 위한 경험적 상수다.
        group_size = max_num_layers
    grouped_layers = []
    for layers in same_type_layers.values():
        num_padding_layers = group_size - len(layers) % group_size
        if num_padding_layers != group_size:
            logger.warning(
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",  # noqa
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)
        # PP 환경 예:
        # - stage 0 구성: full.0, sw.0, sw.1
        # - stage 1 구성: full.1, sw.2, sw.3
        # 올바른 3개 그룹은 (full.0, full.1), (sw.0, sw.2), (sw.1, sw.3)이다.
        # (full.0, full.1), (sw.0, sw.1), (sw.2, sw.3)로 나누면 stage 0의 그룹이
        # (full.0), (sw.0, sw.1), (empty)가 되어 padding이 늘고 메모리 낭비가 생긴다.
        # 이를 피하려고 layers[i::num_groups] 방식으로 그룹을 만든다.
        for i in range(num_groups):
            grouped_layers.append(layers[i::num_groups])
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """KV cache 그룹과 레이어 spec으로부터 KV cache 설정을 생성한다.

    인자:
        vllm_config: 전역 VllmConfig.
        kv_cache_groups: KV cache 그룹.
        available_memory: KV cache에 사용할 수 있는 메모리(바이트).
    반환:
        생성된 KVCacheConfig.
    """
    if len(kv_cache_groups) == 0:
        # attention-free 모델은 KV cache가 없다.
        # BlockPool은 null_block이 필요하므로 num_blocks=1을 반환한다.
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        )

    # 모델 러너가 KV cache 텐서를 어떻게 초기화할지 결정한다.
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # 특수 케이스: KV cache 타입은 같지만 hidden size가 레이어마다 다르다.
        # hidden size에 맞춰 레이어별 메모리를 다르게 할당한다.
        num_blocks = (
            available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
        )
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        kv_cache_tensors = [
            KVCacheTensor(
                size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                shared_by=[layer_name],
            )
            for layer_name in kv_cache_groups[0].layer_names
        ]
    else:
        # 일반 케이스:
        # group_size개의 메모리 풀을 두고, 각 풀은 각 그룹의 i번째 레이어가 공유한다.
        # 그룹별 블록 테이블이 다르므로 공유 텐서 내 서로 다른 영역을 사용한다.
        # 예: 3개 그룹 (full.0, full.1), (sw.0, sw.2), (sw.1, padding),
        # group_size=2이면
        # full.0/sw.0/sw.1이 텐서 하나(available_memory//2)를 공유,
        # full.1/sw.2가 다른 텐서 하나(available_memory//2)를 공유한다.
        group_size = max(len(group.layer_names) for group in kv_cache_groups)

        page_size = get_uniform_page_size(
            [group.kv_cache_spec for group in kv_cache_groups]
        )
        assert group_size > 0, "group_size must be greater than 0"
        num_blocks = get_num_blocks(
            vllm_config, group_size, available_memory, page_size
        )
        kv_cache_tensors = []
        for i in range(group_size):
            shared_by = []
            for j in range(len(kv_cache_groups)):
                if i < len(kv_cache_groups[j].layer_names):
                    shared_by.append(kv_cache_groups[j].layer_names[i])
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]):
    """하이브리드 모델의 KV cache spec을 가능한 한 단일 타입으로 통일한다.

    FullAttentionSpec과 SlidingWindowSpec이 함께 있으면 SlidingWindowSpec을
    FullAttentionSpec으로 변환한다.

    인자:
        kv_cache_spec: 모델 각 어텐션 레이어의 KV cache spec.
    """

    if is_kv_cache_spec_uniform(
        kv_cache_spec
    ) or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec):
        return

    logger.warning(
        "Hybrid KV cache manager is disabled for this hybrid model, "
        "This means we do not enable any optimizations for saving KV cache "
        "memory (e.g., dropping the KV cache outside the sliding window). "
        "The compute of layers like sliding window is still saved."
    )

    has_full_attention = any(
        isinstance(spec, FullAttentionSpec) for spec in kv_cache_spec.values()
    )
    has_sliding_window = any(
        isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values()
    )
    has_chunked_local_attention = any(
        isinstance(spec, ChunkedLocalAttentionSpec) for spec in kv_cache_spec.values()
    )
    if has_full_attention and (has_sliding_window or has_chunked_local_attention):
        for layer_name, spec in kv_cache_spec.items():
            if isinstance(spec, SlidingWindowSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    sliding_window=spec.sliding_window,
                    page_size_padded=spec.page_size_padded,
                )
            elif isinstance(spec, ChunkedLocalAttentionSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    attention_chunk_size=spec.attention_chunk_size,
                    page_size_padded=spec.page_size_padded,
                )

    if not (
        is_kv_cache_spec_uniform(kv_cache_spec)
        or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec)
    ):
        raise ValueError(
            "Hybrid KV cache manager is disabled but failed to "
            "convert the KV cache specs to one unified type."
        )


def get_kv_cache_groups(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
) -> list[KVCacheGroupSpec]:
    """모델 레이어를 동일한 KV cache spec을 갖는 그룹으로 분할한다.

    인자:
        vllm_config: 전역 VllmConfig.
        kv_cache_spec: 모델 각 어텐션 레이어의 KV cache spec.

    반환:
        생성된 KVCacheGroup들.
    """
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)

    if is_kv_cache_type_attention_free(kv_cache_spec):
        # KVCacheManager가 attention-free 모델을 처리할 수 있도록 빈 리스트를 반환한다.
        return []

    if is_kv_cache_spec_uniform(kv_cache_spec):
        # 대부분 모델처럼 모든 레이어 KV cache가 같으면
        # 각 레이어에 같은 양의 메모리를 할당한다.
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        # 모든 레이어가 같은 수의 토큰 슬롯이 필요하면(예: 전부 full attention,
        # 혹은 동일 윈도우 크기의 sliding window), 한 그룹으로 묶는다.
        return _get_kv_cache_groups_uniform_type(uniform_spec)

    # KVCacheManager는 한 가지 크기의 메모리만 할당할 수 있으므로
    # 레이어 페이지 크기를 통일해야 한다. 통일 불가 시 예외를 발생시킨다.
    kv_cache_spec = unify_kv_cache_spec_page_size(kv_cache_spec)
    # 모델에 여러 어텐션 타입이 있어도, 레이어당 블록당 물리 메모리가 같다면
    # 동일 레이어 수 기준으로 그룹화하여 총 페이지 크기를 맞춘다.
    return _get_kv_cache_groups_uniform_page_size(kv_cache_spec)


def generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],
) -> KVCacheConfig:
    """스케줄러용 KV cache 설정을 생성한다."""
    assert all(
        [cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs]
    )
    # 레이어 이름을 제외하면 모든 워커 kv_cache_config가 같으므로,
    # 임의의 하나로 스케줄러를 초기화한다.
    cfg = copy.deepcopy(kv_cache_configs[0])
    for group in cfg.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            # UniformTypeKVCacheSpecs 내부 레이어는 타입이 동일하므로
            # 임의의 하나를 스케줄러 초기값으로 사용한다.
            group.kv_cache_spec = next(
                iter(group.kv_cache_spec.kv_cache_specs.values())
            )
    return cfg


def _report_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> None:
    """해석된(resolved) KV cache 설정을 로그로 출력한다.

    인자:
        vllm_config: 전역 VllmConfig.
        kv_cache_config: 해석된 KV cache 설정.
    """
    min_block_size = min(
        [group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups]
    )

    # KV cache 크기와 최대 동시성을 로그로 남긴다.
    num_tokens = (
        kv_cache_config.num_blocks
        // len(kv_cache_config.kv_cache_groups)
        * min_block_size
    )
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    if pcp_size * dcp_size > 1:
        num_tokens *= pcp_size * dcp_size
        logger.info(
            "Multiplying the GPU KV cache size by the cp_world_size %d "
            "(pcp_world_size %d * dcp_world_size %d).",
            pcp_size * dcp_size,
            pcp_size,
            dcp_size,
        )
    num_tokens_str = f"{num_tokens:,}"
    logger.info_once("GPU KV cache size: %s tokens", num_tokens_str, scope="local")
    max_model_len_str = f"{vllm_config.model_config.max_model_len:,}"
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    logger.info_once(
        "Maximum concurrency for %s tokens per request: %.2fx",
        max_model_len_str,
        max_concurrency,
        scope="local",
    )


def _max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """KV cache 그룹 기준 최대 메모리 사용량(바이트)을 계산한다.

    하이브리드 모델의 padding까지 반영한다. 예를 들어 full 8층 + sliding window
    9층이면, 그룹 크기 통일을 위해 9 full + 9 sliding window로 패딩되는 상황을
    올바르게 고려한다.
    """
    if not kv_cache_groups:
        return 0

    # UniformTypeKVCacheSpecs 특수 케이스(단일 그룹, 레이어별 spec).
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        return sum(
            spec.max_memory_usage_bytes(vllm_config)
            for spec in per_layer_specs.values()
        )

    # 일반 케이스: group_size개의 풀을 두고, 각 풀을 그룹별 1개 레이어가 공유한다.
    # 메모리 = group_size * page_size * blocks_for_max_len
    group_size = max(len(group.layer_names) for group in kv_cache_groups)
    page_size = get_uniform_page_size(
        [group.kv_cache_spec for group in kv_cache_groups]
    )
    any_spec = kv_cache_groups[0].kv_cache_spec
    blocks_needed = cdiv(any_spec.max_memory_usage_bytes(vllm_config), page_size)

    return group_size * page_size * blocks_needed


def _estimate_max_model_len_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> int:
    """가용 메모리에 맞는 최대 모델 길이를 이진 탐색한다.

    토큰 1개도 맞지 않으면 0을 반환한다.
    """
    original_max = vllm_config.model_config.max_model_len

    def fits(model_len: int) -> bool:
        vllm_config.model_config.max_model_len = model_len
        return (
            _max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)
            <= available_memory
        )

    try:
        left, right = 1, original_max
        if not fits(left):
            return 0
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        vllm_config.model_config.max_model_len = original_max


def _auto_fit_max_model_len(
    vllm_config: VllmConfig,
    projected_groups_per_worker: list[list[KVCacheGroupSpec]],
    available_memory: list[int],
) -> None:
    """max_model_len이 -1일 때, 지원 가능한 최대 컨텍스트 길이를 추정한다.

    워커별 가용 GPU 메모리를 기준으로 전체 워커에 공통으로 맞는 최대 길이를
    이진 탐색으로 찾는다.

    인자:
        vllm_config: 전역 VllmConfig(함수 내에서 in-place 수정될 수 있음).
        projected_groups_per_worker: 워커별로 투영된 KV cache 그룹.
        available_memory: 워커별 KV cache 가용 메모리(바이트).
    """
    original_max = vllm_config.model_config.max_model_len

    if all(not groups for groups in projected_groups_per_worker):
        # 모든 워커 spec이 비어 있음(attention-free 모델).
        logger.info_once(
            "Auto-fit max_model_len: attention-free model, "
            "using derived max_model_len=%d",
            original_max,
            scope="local",
        )
        return

    # 전체 워커에서 공통으로 수용 가능한 max_model_len을 찾는다.
    auto_fit_max = original_max
    limiting_worker_mem = available_memory[0]
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        worker_max = _estimate_max_model_len_from_groups(vllm_config, groups, avail_mem)
        if worker_max < auto_fit_max:
            auto_fit_max = worker_max
            limiting_worker_mem = avail_mem

    if auto_fit_max <= 0:
        raise ValueError(
            "Cannot auto-fit max_model_len: not enough GPU memory available "
            "to serve even a single token. Try increasing `gpu_memory_utilization`."
        )

    if auto_fit_max >= original_max:
        # 모델의 전체 컨텍스트 길이가 메모리에 그대로 들어간다.
        logger.info_once(
            "Auto-fit max_model_len: full model context length %d fits in "
            "available GPU memory",
            original_max,
            scope="local",
        )
    else:
        # 메모리에 맞추기 위해 max_model_len 축소가 필요하다.
        vllm_config.model_config.max_model_len = auto_fit_max
        logger.info_once(
            "Auto-fit max_model_len: reduced from %d to %d to fit in "
            "available GPU memory (%s GiB available for KV cache)",
            original_max,
            auto_fit_max,
            format_gib(limiting_worker_mem),
            scope="local",
        )


def _project_kv_cache_groups_to_worker(
    global_kv_cache_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """전역 KV cache 그룹을 특정 워커가 담당하는 레이어로 투영한다.

    파이프라인 병렬에서는 워커마다 일부 레이어만 가진다. 이 함수는 전역 그룹에서
    해당 워커 레이어만 남기고, 필요 시 UniformTypeKVCacheSpecs도 워커 단위로 조정한다.

    인자:
        global_kv_cache_groups: 전체 모델 기준 전역 KV cache 그룹.
        worker_spec: 이 워커에 존재하는 각 레이어의 KV cache spec.

    반환:
        이 워커 레이어만 포함한 투영된 KV cache 그룹.
    """
    projected_groups: list[KVCacheGroupSpec] = []
    for group in global_kv_cache_groups:
        worker_layer_names = [
            layer_name for layer_name in group.layer_names if layer_name in worker_spec
        ]
        group_spec = group.kv_cache_spec
        if worker_layer_names and isinstance(group_spec, UniformTypeKVCacheSpecs):
            group_spec = UniformTypeKVCacheSpecs(
                block_size=group_spec.block_size,
                kv_cache_specs={
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in worker_layer_names
                },
            )
        projected_groups.append(KVCacheGroupSpec(worker_layer_names, group_spec))
    return projected_groups


def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """모델의 워커별 KV cache 설정(KVCacheConfig)을 생성한다.

    모든 워커가 공유 중앙 컨트롤러를 사용하므로, 할당 정책이 워커 전반에 적용되려면
    `kv_cache_config`가 워커 간 일관돼야 한다. 다만 워커마다 가용 메모리나
    레이어 구성이 다를 수 있으므로(특히 pipeline parallel), 현재 절차는 다음과 같다.
    1. 모든 워커의 KV cache spec을 병합해 모델 전체 KVCacheSpec을 만든다.
    2. 전체 모델의 레이어 비율을 기준으로 KV cache 그룹을 생성한다.
       (하이브리드 모델 spec 통일도 이 단계에서 처리)
    3. PP 샤딩을 반영하기 위해 워커별 투영 그룹으로 auto-fit max_model_len과
       메모리 검사를 수행한다.
    4. 해당 그룹 전략에 따라 각 워커의 KV cache config를 생성한다.
       (보통 PP stage 간 레이어 비율이 유사하므로 합리적)
    5. 모든 워커의 num_blocks를 최솟값으로 통일하고, 텐서 크기도 비례 축소해
       미사용 메모리 할당을 방지한다.

    인자:
        vllm_config: 전역 VllmConfig.
        kv_cache_specs: 워커별 dict[layer_name, KVCacheSpec] 리스트.
        available_memory: 워커별 KV cache 가용 메모리(바이트).

    반환:
        워커별 생성된 KVCacheConfig 리스트.
    """

    # 모든 워커의 KV cache spec을 병합한다.
    # PP stage가 다르면 레이어 이름이 다를 수 있고, 같은 PP stage의 TP rank들은
    # 동일한 KV cache spec을 가져야 한다.
    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_name] = layer_spec
            else:
                assert merged_kv_cache_specs[layer_name] == layer_spec, (
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet."
                )

    # 전역 KV cache 그룹을 얻는다. disable_hybrid_kv_cache_manager가 켜진 경우
    # 하이브리드 모델 spec 통일도 여기서 처리한다.
    # 이 호출 이후 merged_kv_cache_specs는 in-place로 변경될 수 있다.
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    # original_max_model_len이 -1이면, 가용 GPU 메모리에 맞는 최대 모델 길이를
    # 자동으로 결정한다.
    # PP 샤딩 반영을 위해 워커별 투영 그룹을 사용한다.
    projected_groups_per_worker = [
        _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec)
        for worker_spec in kv_cache_specs
    ]

    if vllm_config.model_config.original_max_model_len == -1:
        _auto_fit_max_model_len(
            vllm_config, projected_groups_per_worker, available_memory
        )

    # 워커별 가용 메모리가 충분한지 확인한다.
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        _check_enough_kv_cache_memory(
            avail_mem,
            partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(_estimate_max_model_len_from_groups, vllm_config, groups),
        )

    kv_cache_configs: list[KVCacheConfig] = []
    for projected_groups, kv_cache_spec_one_worker, available_memory_one_worker in zip(
        projected_groups_per_worker, kv_cache_specs, available_memory
    ):
        assert sum(len(group.layer_names) for group in projected_groups) == len(
            kv_cache_spec_one_worker
        ), "Some layers are not assigned to any group."
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(
                vllm_config, projected_groups, available_memory_one_worker
            )
        )

    # 모든 rank의 num_blocks를 최솟값으로 맞춘다.
    # 미사용 메모리 할당을 피하기 위해 텐서 크기도 비례 축소한다.
    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    for kv_cache_config in kv_cache_configs:
        num_blocks_old = kv_cache_config.num_blocks
        kv_cache_config.num_blocks = min_num_blocks

        # 텐서 크기를 비례 축소한다.
        for tensor in kv_cache_config.kv_cache_tensors:
            assert tensor.size % num_blocks_old == 0
            tensor.size = tensor.size // num_blocks_old * min_num_blocks

        if len(kv_cache_config.kv_cache_groups) > 0:
            _report_kv_cache_config(vllm_config, kv_cache_config)

    return kv_cache_configs


class BlockHashListWithBlockSize:
    """블록 해시 단위를 `hash_block_size`에서 `target_block_size`로 변환한다.

    KV cache 그룹마다 block size가 다를 때 사용한다.
    `hash_block_size`는 원본 `block_hashes` 계산 단위이고,
    `target_block_size`는 해당 그룹의 실제 block size다.

    현재는 정수 배 확대만 지원한다
    (`target_block_size`가 `hash_block_size`의 배수여야 함).
    성능을 위해 접근 시점에 지연(lazy) 변환하며, 연속된 해시를 이어붙여
    큰 block size 해시를 만든다.

    예 (`hash_block_size` = 16, `target_block_size` = 32):
    16 단위 해시 2개를 이어붙이면 32 단위 해시 1개가 된다.

    block_size 16의 블록 해시:
    | 토큰 범위   | 0-15 | 16-31 | 32-47 | 48-63 |
    |-------------|------|-------|-------|-------|
    | 해시        | A    | B     | C     | D     |

    block_size 32의 블록 해시:
    | 토큰 범위   | 0-31 | 32-63 |
    |-------------|------|-------|
    | 해시        | AB   | CD    |

    인자:
        block_hashes: 변환 대상 블록 해시(`hash_block_size` 기준 계산값).
        hash_block_size: `block_hashes`가 계산된 block size.
        target_block_size: 목표 block size(`hash_block_size`의 배수여야 함).
    """

    def __init__(
        self,
        block_hashes: list[BlockHash],
        hash_block_size: int,
        target_block_size: int,
    ):
        self.block_hashes = block_hashes
        assert target_block_size % hash_block_size == 0
        self.scale_factor = target_block_size // hash_block_size

    def __len__(self) -> int:
        return len(self.block_hashes) // self.scale_factor

    @overload
    def __getitem__(self, idx: int) -> BlockHash: ...

    @overload
    def __getitem__(self, idx: slice) -> list[BlockHash]: ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._get_value_at(idx)

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self._get_value_at(i) for i in range(start, stop, step)]

        raise TypeError(f"Invalid index type: {type(idx)!r}")

    def __iter__(self) -> Iterator[BlockHash]:
        for i in range(len(self)):
            yield self._get_value_at(i)

    def _get_value_at(self, idx: int) -> BlockHash:
        base = idx * self.scale_factor
        end = base + self.scale_factor
        merged_hash: bytes = self.block_hashes[base]
        for i in range(base + 1, end):
            merged_hash += self.block_hashes[i]
        return BlockHash(merged_hash)


BlockHashList = list[BlockHash] | BlockHashListWithBlockSize
