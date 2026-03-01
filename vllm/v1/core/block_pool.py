# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    generate_block_hash_extra_keys,
    get_block_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashToBlockMap:
    """
    접두사 캐싱에 사용되는 블록 캐시입니다. 블록을 캐시합니다.
    해시에서 블록 또는 여러 블록으로 직접
    (예: {block_hash: KVCacheBlocks})
    - 대부분 block_hash는 단일 KVCacheBlock 및 KVCacheBlock에 매핑됩니다.
        단순히 KVCacheBlock일 뿐입니다.
    - 그렇지 않으면 KVCacheBlocks는 {block_id: KVCacheBlock}의 dict입니다.

    캐시된 블록은 사용할 수 있는 블록 해시가 포함된 전체 블록입니다.
    접두사 캐싱의 경우.
    캐시된 블록은 요청을 실행하거나
    잠재적으로 퇴거될 수 있는 free_block_queue.

    참고 #1: 현재 캐시에 있는 블록의 중복을 제거하지 않습니다.
    즉, 블록이 가득 차서 캐시되면 확인하지 않습니다.
    캐시에 동일한 블록이 이미 있는 경우 이는 다음과 같습니다.
    우리는 할당된 블록 ID가 변경되지 않도록 하려고 합니다.
    블록 테이블은 추가 전용입니다.
    참고 #2: GC 비용을 줄이기 위해 Union 유형이 도입되었습니다.
    내부 dict에서.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        주어진 블록 해시 키를 가진 모든 블록을 가져옵니다.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        KVCacheBlock을 캐시에 삽입합니다.
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # 열쇠를 찾지 못한 경우, 열쇠에 블록 하나를 부착하세요.
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # 동일한 키를 가진 블록이 있으면 원래 블록을 병합합니다.
            # 그리고 새 블록을 사전으로
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # 이미 사전인 경우 블록을 삽입하기만 하면 됩니다.
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        block_hash가 존재하는지 확인하고 캐시에서 block_id를 팝합니다.
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash를 캐시에서 찾을 수 없습니다.
            return None
        # TODO(Jialin): 키가 발견되면 block_id가 항상 존재해야 합니다.
        # 블록으로. 현재는 안전을 위해 원래 동작을 유지합니다.
        #
        # block_id == block.block_id 어설션을 추가하고
        # 대신 후속 조치로 del 블록[block_id]를 사용하세요.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # 단일 블록 ID가 일치하지 않으면
            # 차단(거의 발생하지 않음)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # 블록 사전에서 block_id를 팝하려고 시도하고, 사전이 여전히 있는 경우
            # 블록이 포함된 경우 캐시에 다시 넣습니다.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    """KVCacheBlock을 관리하는 BlockPool입니다.
    kv 캐시 블록을 할당, 해제 및 캐시하는 방법을 제공합니다. 그만큼
    free_block_queue는 사용 가능한 블록을 제거 순서대로 저장합니다.
    할당, 무료 및 캐시 제거. cashed_block_hash_to_block
    캐시된 블록 찾기를 지원하기 위해 블록 해시와 캐시된 블록 사이를 매핑합니다.
    블록 해시로.

    인수:
        num_gpu_blocks: 풀의 블록 수입니다.
        활성화_caching: 접두사 캐싱을 활성화할지 여부입니다.
        hash_block_size: 블록 해시가 계산되는 블록 크기입니다.
            실제 블록 크기는 일반적으로 hash_block_size와 동일하지만 경우에 따라
            서로 다른 KV 캐시 그룹이 서로 다른 블록 크기를 갖는 경우
            실제 블록 크기는 hash_block_size의 배수일 수 있습니다.
        활성화_kv_cache_events: kv 캐시 이벤트 활성화 여부.
        metrics_collector: 블록 상주 추적을 위한 선택적 지표 수집기입니다.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # 모든 kv-cache 블록.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # 이중 링크를 구성하고 조작하는 자유 블록 큐
        # 사용 가능한 블록 목록(캐싱이 있을 때 제거 후보 포함)
        # 활성화됨).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # 블록 조회용 캐시
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # block_id=0으로 자리 표시자 블록을 나타냅니다.
        # null_block의 ref_cnt는 유지되지 않으므로 특별한 주의가 필요합니다.
        # 그것을 해제하지 마십시오.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """각 그룹의 블록 해시로 캐시된 블록을 가져옵니다.
        `kv_cache_group_ids` 또는 그룹에 대한 캐시가 누락된 경우 없음입니다.
        중복된 블록이 있으면 캐시의 첫 번째 블록을 반환합니다.

        인수:
            block_hash: 블록의 해시 값입니다.
            kv_cache_group_ids: KV 캐시 그룹의 ID입니다.

        보고:
            캐시된 블록이 존재하는 경우 블록이 존재하거나 없음입니다.
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """접두사 캐싱을 위해 전체 블록 목록을 캐시합니다.
        이 함수는 블록 해시를 갖게 될 블록 목록을 가져옵니다.
        업데이트되고 캐시될 메타데이터입니다. 요청이 주어지면 업데이트됩니다.
        각 블록에 대한 메타데이터를 저장하고
        `cached_block_hash_to_block` 맵에 저장합니다.
        블록 해시 값은 요청 객체에 의해 즉시 계산됩니다.
        토큰이 생성될 때와 새 토큰이 추가될 때.

        인수:
            request: 블록을 캐시하라는 요청입니다.
            블록: 요청의 모든 블록입니다.
            num_cached_blocks: 이미 캐시된 블록 수입니다.
            num_full_blocks: 가득 차서 채워야 하는 블록 수
                이 기능 후에 캐시됩니다.
            block_size: 각 블록의 토큰 수입니다.
            kv_cache_group_id: KV 캐시 그룹의 ID입니다.
        """
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        if block_size == self.hash_block_size:
            # 일반적인 경우.
            block_hashes: BlockHashList = request.block_hashes
        else:
            # block_size는 hash_block_size의 배수입니다. 이런 경우가 발생합니다.
            # KV 캐시 그룹마다 블록 크기가 다릅니다.
            assert block_size % self.hash_block_size == 0
            # 다음을 사용하여 block_size의 세분성에서 block_hash를 다시 계산합니다.
            # 원래 block_hashes(hash_block_size 단위).
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            # 다음과 같이 희박한 주의를 활성화하는 경우 일부 블록은 null 블록이 될 수 있습니다.
            # 슬라이딩 윈도우 어텐션 또는 접두사 캐싱이 포함된 Mamba 모델
            # 정렬 모드. 여기서는 null 블록을 건너뜁니다.
            if blk.is_null:
                continue
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # 전체 블록을 업데이트하고 캐시에 추가했습니다.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            # 캐시되는 블록의 토큰 범위 계산
            start_token_idx = num_cached_blocks * block_size
            end_token_idx = num_full_blocks * block_size

            # 각 블록에 대해 개별적으로 추가 키를 생성합니다.
            # 각 블록은 서로 다른 extra_key를 가질 수 있습니다(예: 서로 다른 MM
            # 기능 또는 첫 번째 블록에 대해서만 캐시_솔트).
            # new_hashes의 길이와 일치하도록 null 블록을 건너뜁니다.
            extra_keys_list: list[tuple[Any, ...] | None] = []
            curr_mm_idx = 0
            for i in range(num_cached_blocks, num_full_blocks):
                if blocks[i].is_null:
                    continue
                block_start = i * block_size
                block_end = block_start + block_size
                extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                    request, block_start, block_end, curr_mm_idx
                )
                extra_keys_list.append(extra_keys)

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[start_token_idx:end_token_idx],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                    lora_name=request.lora_request.name
                    if request.lora_request
                    else None,
                    extra_keys=extra_keys_list if extra_keys_list else None,
                )
            )

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """무료 블록 풀에서 새 블록을 가져옵니다.

        이 함수에서는 블록 캐시를 확인하지 않습니다.

        인수:
            num_blocks: 할당할 블록 수입니다.

        보고:
            새로운 블록의 목록입니다.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # 목록을 한 번만 반복하기 위해 코드를 약간 복제했습니다.
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        블록이 `cached_block_hash_to_block`에 캐시되어 있으면 해당 해시를 재설정합니다.
        메타데이터를 제거하고 캐시에서 제거합니다.

        인수:
            block: 퇴거할 블록입니다.

        보고:
            블록이 제거되면 True이고, 그렇지 않으면 False입니다.
        """
        # 누출을 방지하려면 먼저 측정항목 추적을 정리하세요.
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)

        block_hash = block.block_hash
        if block_hash is None:
            # 블록에 해시가 없으므로 제거가 필요하지 않습니다.
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # 캐시된_블록_해시_to_block에서 블록을 찾을 수 없습니다.
            # 퇴거는 필요하지 않습니다
            return False

        block.reset_hash()

        if self.enable_kv_cache_events:
            # FIXME(첸): `hash_value`를 반환해야 할지 잘 모르겠습니다.
            # 또는 `(hash_value, group_id)`를 입력하세요. 하지만 지금은 괜찮으니까
            # kv 캐시 이벤트가 발생하면 하이브리드 kv 캐시 관리자를 비활성화합니다.
            # 활성화되어 있으므로 그룹이 하나만 있습니다.
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                )
            )
        return True

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """블록을 터치하면 참조 카운트가 1씩 증가하고 제거될 수도 있습니다.
        무료 대기열의 블록입니다. 블록이 부딪혔을 때 사용됩니다.
        동일한 접두사를 가진 다른 요청.

        인수:
            블록: 터치할 블록 목록입니다.
        """
        for block in blocks:
            # ref_cnt=0은 이 블록이 사용 가능 목록에 있음을 의미합니다(예: 제거
            # 후보)이므로 삭제하세요.
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """블록 목록을 해제합니다. 블록은 해당 블록에 따라 주문되어야 합니다.
        퇴거 우선순위, 첫 번째 블록이 먼저 퇴거됩니다.

        인수:
            order_blocks: 퇴거에 따라 정렬된 해제할 블록 목록
                우선 사항.
        """
        # 여러 패스를 허용하도록 반복 가능 항목을 구체화합니다.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n(
            [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
        )

    def evict_blocks(self, block_ids: set[int]) -> None:
        """블록 ID를 기준으로 접두사 캐시에서 블록을 제거합니다.

        현재 캐시된(해시가 있는) 블록만 제거합니다. 블록
        ref_cnt > 0이면 블록 풀에서 해제되지 않고 제거만 됩니다.
        접두사 캐시 해시 테이블에서.

        인수:
            block_ids: 캐시에서 제거할 블록 ID 세트입니다.
        """
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]
            self._maybe_evict_cached_block(block)

    def reset_prefix_cache(self) -> bool:
        """접두사 캐시를 재설정합니다. 이 기능은 RLHF에서 사용될 수 있습니다
        가중치가 업데이트된 후 잘못된 접두사 캐싱으로 이동합니다.
        또는 벤치마킹을 위해 접두사 캐싱 상태를 재설정하는 데 사용됩니다.

        보고:
            bool: 접두사 캐시가 성공적으로 재설정되면 true,
            그렇지 않으면 거짓입니다.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # null 블록은 항상 사용된 것으로 표시됩니다.
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # 새로운 블록이 도달하지 않도록 모든 해시를 제거하십시오.
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # 모든 블록에서 모든 해시를 제거합니다.
        for block in self.blocks:
            block.reset_hash()

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """풀의 사용 가능한 블록 수를 가져옵니다.

        보고:
            사용 가능한 블록 수입니다.
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """KV 캐시 사용량을 가져옵니다.

        보고:
            KV 캐시 사용량(0.0에서 1.0 사이)입니다.
        """

        # null 블록을 설명하려면 1을 뺍니다.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """모든 이벤트를 원자적으로 취하고 대기열을 지웁니다.

        보고:
            KV 캐시 이벤트 목록입니다.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
