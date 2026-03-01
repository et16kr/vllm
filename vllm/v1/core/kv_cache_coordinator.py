# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import lcm

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
)
from vllm.v1.request import Request


class KVCacheCoordinator(ABC):
    """
    다양한 KV 캐시 그룹의 KV 캐시를 조정합니다.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching

        self.block_pool = BlockPool(
            kv_cache_config.num_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )

        # Eagle이 활성화된 경우 find_longest_cache_hit에 대한 특별한 처리가 필요합니다.
        self.use_eagle = use_eagle
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:
        """
        요청에 할당되는 데 필요한 블록 수를 가져옵니다.

        인수:
            request_id: 요청 ID입니다.
            num_tokens: 슬롯이 필요한 총 토큰 수(포함)
                이미 할당된 토큰).
            new_computed_blocks: 새로 계산된 블록이 방금 도달했습니다.
                접두사 캐싱.
            num_encoder_tokens: 할당할 인코더 토큰 개수
                교차주의를 위한 블록.
            total_computed_tokens: 로컬 및 외부 토큰을 모두 포함합니다.
            num_tokens_main_model: 메인 모델의 토큰 수(일명 target
                사양 디코드의 모델). 사양 디코드가 없으면 num_tokens입니다.
                사양 디코드의 경우 num_tokens - num_lookahead_tokens입니다.

        보고:
            할당할 블록 수입니다.
        """
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            if isinstance(manager, CrossAttentionManager):
                # 교차주의를 위해 단일 정적 할당을 발행합니다.
                # 인코더 입력 토큰 수에 따른 블록 수입니다.
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id, num_encoder_tokens, [], 0, num_encoder_tokens
                )
            else:
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[i],
                    total_computed_tokens,
                    num_tokens_main_model,
                )
        return num_blocks_to_allocate

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        요청에 새로운 계산된 블록을 추가합니다. 선택적으로 새 할당
            외부 계산 토큰에 대한 블록(있는 경우)

        인수:
            request_id: 요청 ID입니다.
            new_computed_blocks: 새로 계산된 블록이 방금 도달했습니다.
                접두사 캐시.
            num_local_computed_tokens: 로컬 계산 토큰 수입니다.
            num_external_computed_tokens: 외부 계산 토큰 수입니다.
        """
        for i, manager in enumerate(self.single_type_managers):
            manager.allocate_new_computed_blocks(
                request_id,
                new_computed_blocks[i],
                num_local_computed_tokens,
                num_external_computed_tokens,
            )

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        요청에 최소한 'num_tokens'를 제공하도록 새 블록을 할당합니다.
        토큰 슬롯.

        인수:
            request_id: 요청 ID입니다.
            num_tokens: 슬롯이 필요한 총 토큰 수(포함)
                이미 할당된 토큰).
            num_tokens_main_model: 메인 모델의 토큰 수(일명 target
                사양 디코드의 모델). 사양 디코드가 없으면 num_tokens입니다.
                사양 디코드의 경우 num_tokens - num_lookahead_tokens입니다.
            num_encoder_tokens: 할당할 인코더 토큰 개수
                교차주의를 위한 블록.

        보고:
            새로 할당된 블록입니다.
        """
        return tuple(
            manager.allocate_new_blocks(
                request_id,
                num_encoder_tokens
                if isinstance(manager, CrossAttentionManager)
                else num_tokens,
                num_tokens_main_model,
            )
            for manager in self.single_type_managers
        )

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """
        요청에 대한 블록을 캐시합니다.

        인수:
            요청: 요청입니다.
            num_computed_tokens: 총 토큰 수
                캐시해야 하는 것
                (이미 캐시된 토큰 포함)
        """
        for manager in self.single_type_managers:
            manager.cache_blocks(request, num_computed_tokens)

    def free(self, request_id: str) -> None:
        """
        요청에 대한 블록을 해제합니다.

        인수:
            request_id: 요청 ID입니다.
        """
        for manager in self.single_type_managers:
            manager.free(request_id)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """
        할당된 모든 요청에 ​​대한 공통 접두사 블록 수를 가져옵니다.
        각 kv 캐시 그룹에 대한 KV 캐시.

        인수:
            running_request_id: 실행 중인 요청의 요청 ID입니다.
                공통 접두사 블록을 식별합니다.

        보고:
            list[int]: 각 kv 캐시 그룹에 대한 공통 접두사 블록 수입니다.
        """
        return [
            manager.get_num_common_prefix_blocks(running_request_id)
            for manager in self.single_type_managers
        ]

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """
        '블록'에서 더 이상 필요하지 않은 블록을 제거하고 교체합니다.
        null_block으로 제거된 블록.

        인수:
            request_id: 요청 ID입니다.
            total_computed_tokens: 계산된 토큰의 총 개수입니다.
                로컬 계산 토큰 및 외부 계산 토큰.
        """
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(request_id, total_computed_tokens)

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """
        요청에 대한 블록을 가져옵니다.
        """
        return tuple(
            manager.req_to_blocks.get(request_id) or []
            for manager in self.single_type_managers
        )

    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        pass

    def new_step_starts(self) -> None:
        """새 단계가 시작될 때 호출됩니다."""
        for manager in self.single_type_managers:
            manager.new_step_starts()


class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    """
    접두사 캐싱이 비활성화되거나 지원되지 않는 경우 사용할 KV 캐시 코디네이터입니다.
    UnitaryKVCacheCoordinator 및 HybridKVCacheCoordinator와 달리,
    임의 개수의 KV 캐시 그룹(0개 그룹 포함)을 지원합니다.
    접두사 캐싱과 관련된 기능을 구현하지 않습니다.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            use_eagle,
            False,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        self.num_single_type_manager = len(self.single_type_managers)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(self.num_single_type_manager)
        )
        return blocks, 0


class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """
    KV 캐시 그룹이 하나만 있는 모델을 위한 KV 캐시 코디네이터입니다. 이것은
    KV 캐시 유형이 하나만 있는 모델의 경우(예: 모든 Attention 레이어 사용)
    전체 주의 또는 모든 주의 레이어는 슬라이딩 윈도우 주의를 사용합니다.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size > 1:
            self.block_size *= dcp_world_size
        if pcp_world_size > 1:
            self.block_size *= pcp_world_size
        # Mamba만 사용하는 모델의 경우 block_size는 max_model_len으로 설정됩니다.
        # 접두사 캐싱이 비활성화되고 hash_block_size 검증을 건너뜁니다.
        assert not enable_caching or (hash_block_size == self.block_size), (
            "UnitaryKVCacheCoordinator assumes hash_block_size == block_size"
        )
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group"
        )

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        hit_blocks = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
            alignment_tokens=self.block_size,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
        )
        return hit_blocks, len(hit_blocks[0]) * self.block_size


class HybridKVCacheCoordinator(KVCacheCoordinator):
    """
    여러 KV 캐시 유형을 갖춘 하이브리드 모델용 KV 캐시 코디네이터
    따라서 여러 kv 캐시 그룹.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        # hash_block_size: 블록 해시를 계산하는 데 사용되는 블록 크기입니다.
        # 실제 블록 크기는 일반적으로 hash_block_size와 동일하지만
        # KV 캐시 그룹마다 블록 크기가 다르므로 실제 블록 크기는
        # hash_block_size의 배수일 수 있습니다.
        self.hash_block_size = hash_block_size
        assert all(
            g.kv_cache_spec.block_size % hash_block_size == 0
            for g in kv_cache_config.kv_cache_groups
        ), "block_size must be divisible by hash_block_size"
        assert dcp_world_size == 1, "DCP not support hybrid attn now."
        assert pcp_world_size == 1, "PCP not support hybrid attn now."
        self.verify_and_split_kv_cache_groups()

    def verify_and_split_kv_cache_groups(self) -> None:
        """
        효율적인 일괄 처리를 위해 KV 캐시 그룹을 사양 유형별로 그룹화합니다.
        캐시 적중 조회 중.
        """
        attention_groups: list[
            tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]
        ] = []

        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec

            # 동일한 사양을 가진 기존 그룹을 찾아보세요.
            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls, (
                        "Expected same manager class for identical KV cache specs."
                    )
                    group_ids.append(i)
                    break
            else:
                attention_groups.append((spec, [i], manager_cls))

        assert len(attention_groups) > 1, (
            "HybridKVCacheCoordinator requires at least two attention groups."
        )

        # 완전한 주의를 최우선으로 생각하십시오. 효율적인 왼쪽에서 오른쪽 스캔은 다음을 제공합니다.
        # 초기 경계가 더 엄격해져서 후속 그룹의 작업이 줄어듭니다.
        self.attention_groups = sorted(
            attention_groups,
            key=lambda x: not isinstance(x[0], FullAttentionSpec),
        )

        # 모든 Attention 유형의 블록 크기에 대한 LCM입니다.
        # 캐시 적중 길이는 블록 크기의 LCM의 배수여야 합니다.
        # 캐시 적중 길이가 블록 크기의 배수인지 확인하십시오.
        # 각 주의 유형. 부분적인 지원을 하지 않기 때문에 이것을 요구합니다.
        # 아직 블록 캐시 히트가 발생하지 않았습니다.
        block_sizes = [spec.block_size for spec, _, _ in attention_groups]
        self.lcm_block_size = lcm(*block_sizes)

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        반복 고정 소수점 알고리즘을 사용하여 가장 긴 캐시 적중을 찾습니다.

        각 관심 유형은 현재 후보 길이를 수락하거나
        그것을 줄입니다. 어떤 유형이든 길이를 줄이면 전체 검사를 다시 시작합니다.
        유형. 이는 길이가 단조롭게 감소하고 다음과 같기 때문에 수렴됩니다.
        아래는 0으로 제한됩니다.

        인수:
            block_hashes: 요청의 블록 해시입니다.
            max_cache_hit_length: 캐시 적중의 최대 길이입니다.

        보고:
            다음을 포함하는 튜플:
                - 각 단일 유형 관리자에 대한 캐시 적중 블록의 튜플입니다.
                - 가장 긴 캐시 히트의 토큰 수입니다.
        """

        def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
            if kv_cache_spec.block_size == self.hash_block_size:
                return block_hashes
            return BlockHashListWithBlockSize(
                block_hashes, self.hash_block_size, kv_cache_spec.block_size
            )

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups

        # 단순 하이브리드(1개의 전체 속성 + 1개의 기타): 한 번의 반복으로 충분합니다.
        # 전체 attn이 존재하는 경우 항상 첫 번째입니다. 이렇게 하면 EAGLE 방울이 방지됩니다.
        # 전체 참여가 아닌 그룹에 여러 번 적용됩니다.
        # FIXME(yifan): 단, 다중 속성을 갖는 복잡한 하이브리드 모델의 경우
        # 그룹에서는 여전히 EAGLE 나선형 블록 삭제 문제가 있습니다. 보다
        # 문제 https://github.com/vllm-project/vllm/issues/32802의 토론.
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0][0], FullAttentionSpec
        )

        while True:
            curr_hit_length = hit_length

            for spec, group_ids, manager_cls in self.attention_groups:
                is_full_attn = isinstance(spec, FullAttentionSpec)

                # 전체 주의: 캐시된 블록 재사용(하향 폐쇄 속성)
                cached_blocks = hit_blocks_by_group[group_ids[0]]
                if is_full_attn and cached_blocks is not None:
                    # 완전한 주의를 끌기 위해서는 캐시 적중만 계산하면 됩니다.
                    # 길이는 한 번. 두 번째 반복부터 시작하면
                    # curr_hit_length가 다른 그룹에 의해 줄어들면 간단히 다음과 같이 할 수 있습니다.
                    # 첫 번째 (curr_hit_length // block_size) 블록을 유지하십시오.
                    # 마지막 반복.
                    num_blocks = curr_hit_length // spec.block_size
                    curr_hit_length = num_blocks * spec.block_size
                else:
                    hit_blocks = manager_cls.find_longest_cache_hit(
                        block_hashes=_get_block_hashes(spec),
                        max_length=curr_hit_length,
                        kv_cache_group_ids=group_ids,
                        block_pool=self.block_pool,
                        kv_cache_spec=spec,
                        use_eagle=self.use_eagle,
                        alignment_tokens=self.lcm_block_size,
                    )
                    curr_hit_length = len(hit_blocks[0]) * spec.block_size
                    for group_id, blocks in zip(group_ids, hit_blocks):
                        hit_blocks_by_group[group_id] = blocks

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            # 단순 하이브리드: 한 번의 반복 후 종료
            if is_simple_hybrid:
                break

        # 전체 주의 블록을 최종 hit_length로 자릅니다(있는 경우).
        spec, group_ids, _ = self.attention_groups[0]
        if isinstance(spec, FullAttentionSpec):
            num_blocks = hit_length // spec.block_size
            for group_id in group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]

        return tuple(
            blocks if blocks is not None else [] for blocks in hit_blocks_by_group
        ), hit_length


def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    hash_block_size: int,
    metrics_collector: KVCacheMetricsCollector | None = None,
) -> KVCacheCoordinator:
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    return HybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        hash_block_size=hash_block_size,
        metrics_collector=metrics_collector,
    )
