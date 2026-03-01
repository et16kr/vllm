# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashList,
    BlockHashWithGroupId,
    KVCacheBlock,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SinkFullAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request


class SingleTypeKVCacheManager(ABC):
    """
    kv 캐시 관리를 처리하는 관리자에 대한 추상 기본 클래스
    특정 유형의 주의 계층의 논리.
    """

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> None:
        """
        SingleTypeKVCacheManager를 초기화합니다.
        인수:
            kv_cache_spec: 이 관리자에 대한 kv_cache_spec.
            block_pool: 블록 풀.
            kv_cache_group_id: 이 관리자의 kv 캐시 그룹 ID.
        """
        self.block_size = kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size * pcp_world_size > 1:
            self.block_size *= dcp_world_size * pcp_world_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.enable_caching = enable_caching

        # 각 요청의 할당 블록을 추적하기 위한 req_id -> blocks 매핑.
        # 요청 완료 시 해당 블록을 해제할 때 사용한다.
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

        # {req_id: 해당 요청에 대해 캐시된 블록 수}
        # 이는 각 요청에 대해 캐시된 블록 수를 추적하는 데 사용됩니다.
        # 이는 실행 중인 요청을 추적하는 데만 사용되며 선점된 요청에 대한 
        # 데이터는 추적하지 않습니다.
        self.num_cached_block: dict[str, int] = {}

        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

    @classmethod
    def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):
        return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:
        """
        요청에 추가 할당해야 하는 블록 수를 계산합니다.

        인수:
            request_id: 요청 ID.
            num_tokens: 슬롯이 필요한 총 토큰 수
                (이미 할당된 토큰 포함).
            new_computed_blocks: prefix cache hit로 새로 확정된 블록들.
            total_computed_tokens: 로컬/외부 계산 토큰을 모두 포함한
                총 계산 토큰 수.
            num_tokens_main_model: 메인 모델 기준 토큰 수.
                speculative decoding이 없으면 `num_tokens`와 같고,
                있으면 `num_tokens - num_lookahead_tokens`입니다.

        반환:
            할당할 블록 수입니다.
        """

        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

        if request_id in self.num_cached_block:
            # 빠른 경로: 실행 중인 요청은 새 prefix cache hit가 없다.
            assert len(new_computed_blocks) == 0
            # speculative decoding에서는 나중에 거부될 draft 토큰용 블록이
            # 미리 잡혀 있을 수 있으므로, num_required_blocks가 더 작아질 수 있다.
            return max(num_required_blocks - num_req_blocks, 0)

        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
        # 주의 창에서 건너뛴 전체 블록 수.
        # 아무것도 건너뛰지 않은 경우 이는 0입니다.
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # 실제로 새 블록이 필요한 구간은 "건너뛰는 prefix 이후"의 suffix다.
        # 창 내부에 남아 있는 로컬 계산 블록 수와 건너뛴 블록 수 중 큰 값을
        # 기준으로 이미 커버된 범위를 계산한다.
        num_new_blocks = max(
            num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),
            0,
        )

        # `new_computed_blocks` 기준으로는 첫 `num_skipped_blocks`가 skip 대상이지만,
        # 그중 일부는 이미 `req_to_blocks`에 포함되어 있을 수 있다.
        # 따라서 `new_computed_blocks`에서 실제로 더 건너뛸 개수만 계산한다.
        num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)

        # 계산 블록이 퇴출 후보(free queue + ref_cnt == 0)였다면,
        # 이번 요청에서 touch되면서 free queue에서 빠진다.
        # 따라서 가용 용량 계산 시 evictable 블록 수도 함께 반영한다.
        num_evictable_blocks = self._get_num_evictable_blocks(
            new_computed_blocks[num_skipped_new_computed_blocks:]
        )
        return num_new_blocks + num_evictable_blocks

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        새로 계산된 블록을 요청에 반영합니다.

        처리 순서:
        1. 계산 블록을 touch하여 퇴출되지 않게 한다.
        2. (필요 시) sliding window로 skip되는 구간을 null block으로 채운다.
        3. 남은 계산 블록을 요청 블록 테이블에 추가한다.
        4. (필요 시) 외부 계산 토큰용 새 블록을 할당한다.

        인수:
            request_id: 요청 ID.
            new_computed_blocks: prefix cache hit로 새로 계산된 블록들.
            num_local_computed_tokens: 로컬 계산 토큰 수.
            num_external_computed_tokens: 외부 계산 토큰 수.
        """

        if request_id in self.num_cached_block:
            # 빠른 경로: 실행 중인 요청은 새 prefix cache hit가 없다.
            # 따라서 new_computed_blocks는 비어 있어야 한다.
            assert len(new_computed_blocks) == 0
            return

        # 새 요청.
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            # num_skipped_blocks가 더 크면 new_computed_blocks 전체가 skip될 수 있다.
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
            # 외부 계산 토큰도 일부 skip될 수 있다.
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )

        # 계산 블록을 touch해 퇴출되지 않도록 한다.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), (
                "Computed blocks should be empty when prefix caching is disabled"
            )

        # skip 구간은 null block으로 채운다.
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # 나머지 계산 블록을 추가한다.
        req_blocks.extend(new_computed_blocks)
        # 캐시 hit 블록(건너뛴 null 포함)은 이미 캐시에 있으므로,
        # cache_blocks()가 다시 캐시하려고 하지 않도록 표시한다.
        self.num_cached_block[request_id] = len(req_blocks)

        if num_external_computed_tokens > 0:
            # 외부 계산 토큰에 새 블록을 할당합니다.
            allocated_blocks = self.block_pool.get_new_blocks(
                cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
            )
            req_blocks.extend(allocated_blocks)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        """
        요청에 대해 새 블록을 할당해 최소 `num_tokens` 슬롯을 보장한다.

        인수:
            request_id: 요청 ID.
            num_tokens: 슬롯이 필요한 총 토큰 수
                (이미 할당된 토큰 포함).
            num_tokens_main_model: 메인 모델 기준 토큰 수.
                speculative decoding이 없으면 `num_tokens`와 같고,
                있으면 `num_tokens - num_lookahead_tokens`이다.
        반환:
            새로 할당된 블록.
        """
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            return new_blocks

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """
        요청에 대한 블록을 캐시합니다.

        인수:
            request: 요청.
            num_tokens: 캐시 대상으로 볼 총 토큰 수
                (이미 할당된 토큰 포함).
        """
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        if num_cached_blocks >= num_full_blocks:
            return

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
        )

        self.num_cached_block[request.request_id] = num_full_blocks

    def free(self, request_id: str) -> None:
        """
        요청에 대한 블록을 해제합니다.

        인수:
            request_id: 요청 ID.
        """
        # 할당 전에 요청이 중단/해제될 수 있으므로 기본값은 []로 둔다.
        req_blocks = self.req_to_blocks.pop(request_id, [])

        # tail 블록이 먼저 해제되도록 역순으로 free한다.
        ordered_blocks = reversed(req_blocks)

        self.block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request_id, None)

    @abstractmethod
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        현재 할당된 요청들 사이의 공통 접두사 블록 수를 반환한다.

        인수:
            running_request_id: 요청 ID.

        반환:
            모든 활성 요청에서 공통으로 공유되는 접두사 블록 수.
        """

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        `max_length` 이하 구간에서 최장 cache-hit 접두사를 찾는다.

        접두사는 `kv_cache_group_ids`의 모든 KV 캐시 그룹에서 공통이어야 한다.
        cache hit가 없으면 빈 목록을 반환한다.

        Eagle이 활성화된 경우 마지막 매칭 블록을 제거해, 마지막 블록을
        재계산하도록 강제한다. 이는 Eagle draft head에 필요한 hidden state를
        얻기 위함이다.

        attention 타입별 동작은 하위 클래스에서 구현한다.

        인수:
            block_hashes: 요청의 블록 해시 목록.
            max_length: 캐시 히트 prefix의 최대 길이.
            kv_cache_group_ids: KV cache group ID 목록.
            block_pool: 블록 풀.
            kv_cache_spec: KV cache 사양.
            use_eagle: Eagle 사용 여부.
            alignment_tokens: 반환되는 캐시 히트 길이(토큰 수)가
                반드시 나누어떨어져야 하는 정렬 단위. 기본값은 block_size.
            dcp_world_size: decode context parallel world size.
            pcp_world_size: prefill context parallel world size.

        반환:
            `kv_cache_group_ids`의 각 그룹에 대한 캐시 블록 목록.
            건너뛴 블록은 null_block으로 치환한다.
            반환 튜플 길이는 `len(kv_cache_group_ids)`이며,
            i번째 원소는 `kv_cache_group_ids`의 i번째 그룹에 대응한다.
            예를 들어 block_size=4, sliding_window=8, 그룹 수 1이면
            ([NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)]) 같은 형태가 된다.
        """

        raise NotImplementedError

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """
        attention 계산에 더 이상 필요하지 않은 블록을 제거하고,
        제거된 위치를 null_block으로 대체한다.

        이 함수는 attention 타입별로 다르게 구현되는
        `get_num_skipped_tokens` 결과에 의존한다.

        인수:
            request_id: 요청 ID.
            total_computed_tokens: 계산된 전체 토큰 수.
                로컬 계산 토큰 및 외부 계산 토큰을 포함하는 계산된 토큰의 총 수.
        """
        # attention 계산 중 건너뛰어야 하는 블록을 제거한다.
        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        if num_skipped_tokens <= 0:
            # 모든 토큰이 attention window 안에 있음을 의미한다.
            # 따라서 window 밖 블록을 해제할 필요가 없다.
            # 대표적으로 full attention은 요청 완료 전까지 skip이 없다.
            return
        blocks = self.req_to_blocks[request_id]
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # `num_skipped_tokens`에는 아직 블록이 할당되지 않은 토큰이 섞일 수 있다
        # (예: window가 외부 계산 토큰 영역까지 이동한 경우).
        # 따라서 현재 요청에 실제로 존재하는 블록 수로 상한을 건다.
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))
        removed_blocks: list[KVCacheBlock] = []
        # 블록 인덱스는 0부터 시작하므로, num_skipped_blocks개를 지우려면
        # [num_skipped_blocks - 1 .. 0] 범위를 순회한다.
        for i in range(num_skipped_blocks - 1, -1, -1):
            if blocks[i] == self._null_block:
                # 이미 null block을 만났다면 그보다 앞쪽도 이전 호출에서
                # null block으로 정리된 상태라고 본다.
                break
            removed_blocks.append(blocks[i])
            blocks[i] = self._null_block
        self.block_pool.free_blocks(removed_blocks)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        어텐션 계산을 위해 건너뛸 토큰 수를 가져옵니다.

        인수:
            num_computed_tokens: 계산된 토큰 수.

        반환:
            어텐션 계산을 위해 건너뛸 토큰 수.
        """
        # 기본 동작은 어떤 토큰도 건너뛰지 않는 것입니다.
        return 0

    def new_step_starts(self) -> None:
        # 기본적으로 아무것도 하지 않습니다.
        return None


class FullAttentionManager(SingleTypeKVCacheManager):
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(
            kv_cache_spec, FullAttentionSpec | ChunkedLocalAttentionSpec
        ), (
            "FullAttentionManager can only be used for full attention "
            "and chunked local attention groups"
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        if dcp_world_size * pcp_world_size > 1:
            block_size *= dcp_world_size * pcp_world_size
        max_num_blocks = max_length // block_size
        for block_hash in itertools.islice(block_hashes, max_num_blocks):
            # block_hashes는 체인 구조다. 현재 블록이 캐시에 없으면
            # 뒤 블록들도 확정 계산 상태라고 보장할 수 없다.
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_blocks[0]:
            # Eagle이 활성화된 경우 마지막으로 일치하는 블록을 삭제해야 합니다.
            for computed in computed_blocks:
                computed.pop()
        while (
            block_size != alignment_tokens  # 일반적인 경우 더 빠릅니다.
            and len(computed_blocks[0]) * block_size % alignment_tokens != 0
        ):
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        blocks = self.req_to_blocks[running_request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == len(self.req_to_blocks):
                num_common_blocks += 1
            else:
                break
        return num_common_blocks


class SlidingWindowManager(SingleTypeKVCacheManager):
    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, SlidingWindowSpec), (
            "SlidingWindowManager can only be used for sliding window groups"
        )
        assert dcp_world_size == 1, "DCP not support sliding window attn now."
        assert pcp_world_size == 1, "PCP not support sliding window attn now."

        # 접두사 캐시 히트에 필요한 연속 블록 수입니다.
        # -1 입력 토큰 자체도 창에 포함되어 있으므로 
        sliding_window_contiguous_blocks = cdiv(
            kv_cache_spec.sliding_window - 1, kv_cache_spec.block_size
        )
        if use_eagle:
            # Eagle이 활성화된 경우 마지막으로 일치하는 블록을 삭제해야 합니다. 
            # 슬라이딩 윈도우 레이어의 경우 접두사 캐시 히트에 필요한 
            # 연속 블록 수를 1만큼 늘리고 마지막으로 일치하는 블록
            # 을 삭제해 이를 달성한다.
            sliding_window_contiguous_blocks += 1

        # TODO: miss 시 i를 sliding_window_contiguous_blocks 단위로 건너뛰어
        # 시간 복잡도를 O(max_num_blocks)에서
        # O(max_num_blocks / sliding_window_contiguous_blocks +
        # sliding_window_contiguous_blocks)로 낮출 수 있다.
        max_num_blocks = max_length // kv_cache_spec.block_size
        computed_blocks = tuple(
            [block_pool.null_block] * max_num_blocks
            for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        num_contiguous_blocks = 0
        match_found = False
        # 오른쪽에서 왼쪽으로 검색하고 일치 항목이 발견되면 조기 중지합니다.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # 첫 일치 후보가 alignment_tokens 경계와 맞지 않으면
                # prefix match 검사 자체를 건너뛴다.
                if (
                    num_contiguous_blocks == 0
                    and block_size != alignment_tokens  # 일반적인 경우 더 빠릅니다.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                # 계산된 블록에 캐시된 블록을 추가합니다.
                for computed, cached in zip(computed_blocks, cached_block):
                    computed[i] = cached
                num_contiguous_blocks += 1
                if num_contiguous_blocks >= sliding_window_contiguous_blocks:
                    # 후행 블록을 다듬습니다.
                    # 예: [NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                    # Sliding_window_contiguous_blocks=2 예시.
                    for computed in computed_blocks:
                        del computed[i + num_contiguous_blocks :]
                    match_found = True
                    break
            else:
                num_contiguous_blocks = 0
        if not match_found:
            # 조건을 만족하는 연속 구간을 못 찾은 경우,
            # 현재까지 이어진 앞부분만 남긴다.
            for computed in computed_blocks:
                del computed[num_contiguous_blocks:]
            while (
                block_size != alignment_tokens  # 일반적인 경우 더 빠릅니다.
                and len(computed_blocks[0]) * block_size % alignment_tokens != 0
            ):
                for computed in computed_blocks:
                    computed.pop()
        if use_eagle and computed_blocks[0]:
            assert kv_cache_spec.block_size == alignment_tokens, (
                "aligned_length is not compatible with eagle now"
            )
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        어텐션 계산을 위해 건너뛸 토큰 수를 가져옵니다.

        슬라이딩 윈도우 attention의 경우,
        현재 윈도우보다 앞쪽에 있어 계산에서 제외할 토큰 수를 의미한다.

        예:
        예시: sliding_window=4, num_computed_tokens=7

        토큰: [ 0 1 2 3 4 5 6 7 ]
                  | ---- 계산됨 -----|
                                         ^ 계산할 다음 토큰
                               |------------| 다음 토큰을 위한 슬라이딩 창
                  |--건너뜀---|

        현재 윈도우에는 토큰 4~7이 포함된다.
        토큰 0~3은 윈도우 밖이므로 attention 계산에서 건너뛴다.
        따라서 get_num_skipped_tokens(7) == 4.

        인수:
            num_computed_tokens: 계산된 토큰 수.

        반환:
            어텐션 계산을 위해 건너뛸 토큰 수.
        """
        return max(0, num_computed_tokens - self.sliding_window + 1)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        NOTE(Chen): Sliding window 레이어의 prefix 영역은 null block이므로
        FullAttentionManager처럼 ref_cnt 기반 공통 접두사 계산이 유효하지 않다.
        정확성을 위해 항상 0을 반환한다.
        """
        return 0


class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):
    def __init__(self, kv_cache_spec: ChunkedLocalAttentionSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.attention_chunk_size = kv_cache_spec.attention_chunk_size

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        chunked local attention에서 `max_length` 이내의 최장 cache-hit 접두사를 찾는다.

        반환되는 접두사는 `kv_cache_group_ids`의 모든 KV 캐시 그룹에서 공통이어야 한다.
        cache hit가 없으면 빈 리스트를 반환한다.

        로컬 attention window 바깥의 완전 블록은 이미 계산된 것으로 간주해
        null block으로 채운다.

        예 1:
        - chunk_size=8, block_size=4, max_length=15
        - 다음 토큰은 인덱스 15(0-indexed)라고 할 때,
          8~14는 window 안(조회 필요), 0~7은 window 밖(이미 계산됨)이다.
        - 완전 블록인 block3(8~11)만 조회하며 hit라면
          `[null, null, block3]`, miss라면 `[null, null]`을 반환한다.

        예 2:
        - chunk_size=8, block_size=4, max_length=16
        - 0~15 전체가 window 밖이므로 이미 계산된 것으로 처리한다.
        - `[null, null, null, null]`을 반환한다.

        인수:
            block_hashes: 요청의 블록 해시 체인.
            max_length: 캐시 적중 접두사의 최대 길이.
            kv_cache_group_ids: KV 캐시 그룹 ID 목록.
            block_pool: 블록 풀.
            kv_cache_spec: KV 캐시 사양.
            use_eagle: Eagle 사용 여부.
            dcp_world_size: 디코드 컨텍스트 병렬 처리의 세계 크기.
            pcp_world_size: 사전 채우기 컨텍스트 병렬 처리의 세계 크기.
            alignment_tokens: 반환되는 cache-hit 길이(토큰)가 맞춰야 하는 정렬 단위.

        반환:
            캐시된 블록 목록.
        """
        assert isinstance(kv_cache_spec, ChunkedLocalAttentionSpec), (
            "ChunkedLocalAttentionManager can only be used for "
            "chunked local attention groups"
        )
        assert use_eagle is False, (
            "Hybrid KV cache is not supported for " + "eagle + chunked local attention."
        )
        assert dcp_world_size == 1, "DCP not support chunked local attn now."
        assert pcp_world_size == 1, "PCP not support chunked local attn now."
        assert kv_cache_spec.block_size == alignment_tokens, (
            "KV cache groups with different block sizes are not compatible with "
            "chunked local attention now"
        )
        max_num_blocks = max_length // kv_cache_spec.block_size
        if max_length > 0:
            local_attention_start_idx = (
                max_length
                // kv_cache_spec.attention_chunk_size
                * kv_cache_spec.attention_chunk_size
            )
        else:
            local_attention_start_idx = 0
        # window 바깥 블록은 null block(이미 계산됨)로 채우고,
        # window 안 블록은 cache lookup 결과를 순서대로 붙인다.
        # 결과 형태: [null] ... [null] [hit block 1] [hit block 2] ...
        local_attention_start_block_idx = (
            local_attention_start_idx // kv_cache_spec.block_size
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [block_pool.null_block] * local_attention_start_block_idx
            for _ in range(len(kv_cache_group_ids))
        )
        for i in range(local_attention_start_block_idx, max_num_blocks):
            block_hash = block_hashes[i]
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        return computed_blocks

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        어텐션 계산을 위해 건너뛸 토큰 수를 가져옵니다.

        청크된 로컬 주의의 경우 이는 현재 청크의 왼쪽에
        있는 토큰에 해당합니다.

        예 1:
        청크 크기 = 8, num_computed_tokens = 13
        토큰: [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | ----- 계산됨 ---------------|
                                                  ^^ 다음 계산할 토큰
                                   |----------------| <-- 
                                                          다음 토큰에 대한 주의 창
                 |--- 건너뜀 -----|
        출력: get_num_skipped_tokens(13) == 8

        예 2:
        청크 크기 = 8, num_computed_tokens = 8
        토큰: [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | --- 계산됨 ---|
                                     ^ 계산할 다음 토큰
                                   |--| <-- 다음 토큰에 대한 주의 창
                 | --- 건너뜀 ----|
        출력: get_num_skipped_tokens(8) == 8

        예 3:
        청크 크기 = 8, num_computed_tokens = 7
        토큰: [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 |---계산됨---|
                                 ^ 계산할 다음 토큰
                 |----| <-- 다음 토큰에 대한 주의 창
                 토큰을 건너뛰어야 합니다.
        출력: get_num_skipped_tokens(7) == 0

        인수:
            num_computed_tokens: 계산된 토큰 수.

        반환:
            어텐션 계산을 위해 건너뛸 토큰 수.
        """
        num_skipped_tokens = (
            num_computed_tokens // self.attention_chunk_size
        ) * self.attention_chunk_size
        return num_skipped_tokens

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        계단식 주의는 청크된 로컬 주의에서 지원되지 않습니다.
        """
        return 0


class MambaManager(SingleTypeKVCacheManager):
    def __init__(
        self, kv_cache_spec: MambaSpec, block_pool: BlockPool, **kwargs
    ) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.cached_blocks_this_step: set[BlockHashWithGroupId] = set()
        self.mamba_cache_mode = kv_cache_spec.mamba_cache_mode
        self.num_speculative_blocks: int = kv_cache_spec.num_speculative_blocks
        if self.mamba_cache_mode == "align":
            # 요청 ID -> 이전 단계에서 할당된 상태 블록 인덱스
            self.last_state_block_idx: dict[str, int] = {}
            # 할당된 요청 집합 블록
            self._allocated_block_reqs: set[str] = set()

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, MambaSpec), (
            "MambaManager can only be used for mamba groups"
        )
        assert dcp_world_size == 1, "DCP not support mamba now."
        assert pcp_world_size == 1, "PCP not support mamba now."
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )

        block_size = kv_cache_spec.block_size
        max_num_blocks = max_length // block_size
        # 오른쪽에서 왼쪽으로 검색하고 일치 항목이 발견되면 조기 중지합니다.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # Mamba prefix caching 사용 시, full attention과 Mamba 레이어 사이에서
                # `block_size` 정렬을 만족해야 prefix hit 길이 계산이 올바르다.
                if (
                    block_size != alignment_tokens  # 일반적인 경우 더 빠릅니다.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                for computed, cached in zip(computed_blocks, cached_block):
                    # hit length 정렬을 보장한다.
                    # 이후 로직은
                    # hit_length = len(hit_blocks_other_attn[0]) * self.other_block_size
                    # 를 가정하므로, 앞부분에 더미 블록을 삽입한다.
                    computed.extend([block_pool.null_block] * i)
                    computed.append(cached)
                break  # 마지막 일치만 필요합니다 - 조기 중지

        return computed_blocks

    def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> None:
        assert isinstance(self.kv_cache_spec, MambaSpec)

        # NOTE(tdoublep): 비동기 스케줄링에서는 num_computed_tokens에
        # 이후 거부될 수도 있는 이전 스텝 draft 토큰이 포함될 수 있다.
        # 시퀀스 진행도를 과대평가해 필요한 블록을 너무 일찍 해제하지 않도록
        # speculative 블록 수만큼 보수적으로 차감한다.
        num_computed_tokens = max(0, num_computed_tokens - self.num_speculative_blocks)

        super().remove_skipped_blocks(request_id, num_computed_tokens)
        if self.mamba_cache_mode == "align":
            # `last_state_block_idx`는 두 단계 전에 할당된 블록 인덱스를 나타냅니다.
            # 이전 단계에서 할당된 블록은 Mamba 상태를 복사하는 데 사용됩니다.
            # 현재 단계에서 새로 할당된 블록으로 상태를 옮긴 뒤,
            # 이전 상태 블록은 더 이상 필요하지 않으므로 해제한다.
            last_state_block_idx = self.last_state_block_idx.get(request_id)
            # prefill 중에는 블록이 비연속일 수 있어 인덱스로 정확히 찾아
            # 해당 블록만 해제하고 null block으로 치환한다.
            if (
                last_state_block_idx is not None
                and last_state_block_idx
                < cdiv(num_computed_tokens, self.block_size) - 1
            ):
                blocks = self.req_to_blocks[request_id]
                if blocks[last_state_block_idx] != self._null_block:
                    self.block_pool.free_blocks([blocks[last_state_block_idx]])
                    blocks[last_state_block_idx] = self._null_block

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        Cascade attention은 Mamba에서 지원하지 않는다.
        """
        return 0

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if (
            len(new_computed_blocks) > 0
            and new_computed_blocks[-1].block_hash in self.cached_blocks_this_step
        ):
            # Mamba는 같은 스텝에서 다른 요청이 만든 블록에 의존하면 안 된다.
            # 이번 스텝 스케줄을 막기 위해 의도적으로 과대한 값을 반환한다.
            return self.block_pool.num_gpu_blocks + 1
        if self.mamba_cache_mode != "align":
            # 선형 attention + speculative decoding(MTP/EAGLE)용으로
            # `num_speculative_blocks`를 추가 할당한다.
            if self.num_speculative_blocks > 0:
                num_tokens += (
                    self.kv_cache_spec.block_size * self.num_speculative_blocks
                )
            return super().get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_tokens_main_model,
            )
        else:
            # align 모드에서는 speculative 토큰용 블록을 별도 할당하지 않는다.
            # x * block_size 토큰이 배정되면 num_tokens는
            # x * block_size + num_lookahead_tokens 형태가 되어 정렬이 깨질 수 있다.
            # 현재 draft 모델에는 Mamba 레이어가 없으므로 speculative 토큰은 무시한다.
            num_tokens = num_tokens_main_model

            # NOTE(tdouble): 나중에 거부될 draft 토큰이 섞일 수 있어
            # 필요한 블록 수를 보수적으로(과대) 추정한다.
            num_required_blocks = (
                cdiv(num_tokens, self.block_size) + self.num_speculative_blocks
            )
            num_new_blocks = (
                num_required_blocks
                - len(new_computed_blocks)
                - len(self.req_to_blocks[request_id])
            )
            if num_new_blocks > 0:
                if request_id in self._allocated_block_reqs:
                    # 기존 요청은 이전 스텝 speculative 블록을 재사용할 수 있어
                    # 최대 1개만 추가로 필요하다.
                    num_new_blocks = 1
                else:
                    # 첫 prefill은 실행 상태 블록 1개 + speculative 블록을 잡는다.
                    num_new_blocks = 1 + self.num_speculative_blocks

            num_evictable_computed_blocks = self._get_num_evictable_blocks(
                new_computed_blocks
            )
            return num_new_blocks + num_evictable_computed_blocks

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.mamba_cache_mode != "align":
            # 선형 attention + speculative decoding(MTP/EAGLE)용으로
            # `num_speculative_blocks`를 추가 할당한다.
            if self.num_speculative_blocks > 0:
                num_tokens += self.block_size * self.num_speculative_blocks
            return super().allocate_new_blocks(
                request_id, num_tokens, num_tokens_main_model
            )
        else:
            # align 모드에서는 speculative 토큰용 블록을 별도 할당하지 않는다.
            # x * block_size 토큰이 배정되면 num_tokens는
            # x * block_size + num_lookahead_tokens 형태가 되어 정렬이 깨질 수 있다.
            # 현재 draft 모델에는 Mamba 레이어가 없으므로 speculative 토큰은 무시한다.
            num_tokens = num_tokens_main_model
            req_blocks: list[KVCacheBlock] = self.req_to_blocks[request_id]
            # NOTE(tdouble): 나중에 거부될 draft 토큰이 섞일 수 있어
            # 필요한 블록 수를 보수적으로(과대) 추정한다.
            num_required_blocks = (
                cdiv(num_tokens, self.block_size) + self.num_speculative_blocks
            )
            if num_required_blocks == len(req_blocks):
                return []
            else:
                assert num_required_blocks > len(req_blocks), (
                    "num_required_blocks "
                    f"{num_required_blocks} < len(req_blocks) {len(req_blocks)}"
                )
                prev_block_len = len(req_blocks)
                blocks_allocated = request_id in self._allocated_block_reqs
                # 마지막 상태 블록을 기록합니다.
                if blocks_allocated:
                    # 항상 마지막 (1 + num_speculative_blocks) 블록을 기준으로
                    # 상태 블록 인덱스를 기록한다.
                    self.last_state_block_idx[request_id] = (
                        prev_block_len - 1 - self.num_speculative_blocks
                    )
                elif prev_block_len > 0:
                    # 새 요청이 prefix cache hit로 시작한 경우,
                    # 마지막 기존 블록에 상태가 있다.
                    self.last_state_block_idx[request_id] = prev_block_len - 1

                num_skipped_blocks = (
                    num_required_blocks - self.num_speculative_blocks - 1
                )
                # null 블록
                if prev_block_len < num_skipped_blocks:
                    req_blocks.extend(
                        [
                            self._null_block
                            for _ in range(prev_block_len, num_skipped_blocks)
                        ]
                    )

                if blocks_allocated:
                    # 이 단계에서 이전 추측 블록을 재사용합니다.
                    for block_idx in range(
                        prev_block_len - self.num_speculative_blocks, prev_block_len
                    ):
                        if block_idx < num_skipped_blocks:
                            req_blocks.append(req_blocks[block_idx])
                            req_blocks[block_idx] = self._null_block
                        else:
                            break
                num_new_blocks = num_required_blocks - len(req_blocks)
                if blocks_allocated:
                    assert num_new_blocks <= 1
                else:
                    assert num_new_blocks <= self.num_speculative_blocks + 1
                new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
                req_blocks.extend(new_blocks)
                self._allocated_block_reqs.add(request_id)
                return req_blocks[prev_block_len:]

    def free(self, request_id: str) -> None:
        if self.mamba_cache_mode == "align":
            self._allocated_block_reqs.discard(request_id)
            self.last_state_block_idx.pop(request_id, None)
        super().free(request_id)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Mamba 상태 계산에서 더 이상 필요 없는 토큰 수를 반환한다.
        Mamba는 마지막 계산 토큰의 상태만 유지하면 되므로
        `num_computed_tokens - 1`을 반환한다.
        """
        return num_computed_tokens - 1

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        num_cached_blocks_before = self.num_cached_block.get(request.request_id, 0)
        super().cache_blocks(request, num_tokens)
        num_cached_blocks_after = self.num_cached_block.get(request.request_id, 0)
        if num_cached_blocks_after > num_cached_blocks_before:
            for block in self.req_to_blocks[request.request_id][
                num_cached_blocks_before:num_cached_blocks_after
            ]:
                if block.is_null:
                    continue
                assert block.block_hash is not None
                self.cached_blocks_this_step.add(block.block_hash)

    def new_step_starts(self) -> None:
        self.cached_blocks_this_step.clear()


class CrossAttentionManager(SingleTypeKVCacheManager):
    """인코더-디코더 모델의 교차 주의 KV 캐시에 대한 관리자를 반환합니다."""

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        # cross attention은 요청 간 공유 캐싱을 하지 않으므로
        # `new_computed_blocks`는 항상 비어 있어야 한다.
        assert len(new_computed_blocks) == 0

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        # cross-attention은 요청별 캐시만 사용하므로
        # prefix cache 경로인 이 메서드는 호출되면 안 된다.
        raise ValueError("Should not be called as prefix caching is disabled.")

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        # Cross-attention 블록은 요청별 인코더 상태
        # 를 포함하며 다른 요청 간에 공유되지 않는다.
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, CrossAttentionSpec), (
            "CrossAttentionManager can only be used for cross-attention groups"
        )
        # Cross-attention은 다음과 같은 이유로 접두사 캐싱의 이점을 얻지 못합니다. 
        # 1. 인코더 상태는 요청마다 고유합니다(다른 오디오/이미지
        # 입력)
        # 점진적으로
        # 3. 서로 다른 멀티모달 입력 사이에 재사용 가능한 접두사가 존재하지 않습니다.
        # 캐시 히트가 없음을 나타내기 위해 빈 블록을 반환합니다.
        raise NotImplementedError("CrossAttentionManager does not support caching")


class SinkFullAttentionManager(FullAttentionManager):
    def __init__(
        self,
        kv_cache_spec: SinkFullAttentionSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ):
        super().__init__(
            kv_cache_spec,
            block_pool,
            enable_caching,
            kv_cache_group_id,
            dcp_world_size,
            pcp_world_size,
        )
        sink_len = kv_cache_spec.sink_len
        assert sink_len is not None and sink_len > 0 and sink_len % self.block_size == 0
        num_sink_block = sink_len // self.block_size
        self.sink_blocks = self.block_pool.free_block_queue.popleft_n(num_sink_block)


spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    FullAttentionSpec: FullAttentionManager,
    MLAAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
    MambaSpec: MambaManager,
    CrossAttentionSpec: CrossAttentionManager,
    SinkFullAttentionSpec: SinkFullAttentionManager,
}


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec, **kwargs
) -> SingleTypeKVCacheManager:
    manager_class = spec_manager_map[type(kv_cache_spec)]
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager
