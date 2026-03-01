# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    """
    KVCacheManager의 할당 결과를 담는 객체.
    Scheduler와 KVCacheManager 사이의 인터페이스 역할을 하며,
    스케줄러가 KVCacheManager 내부 자료구조에 직접 의존하지 않도록 한다.
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    `blocks[i][j]`는 i번째 kv_cache_group의 j번째 토큰 블록을 뜻한다.
    토큰 블록 축을 바깥 차원으로 두지 않는 이유는,
    현재는 그룹마다 블록 수가 같지만 향후 그룹별 block_size가 달라지면
    이 가정이 깨질 수 있기 때문이다.

    단일 타입 KVCacheBlocks는 다음 형태 중 하나다.
    - `list[KVCacheBlock]` (블록이 1개 이상인 경우)
    - 빈 튜플 (해당 요청에 KVCacheBlock이 없는 경우)
      (GC 오버헤드 절감을 위해 KVCacheManager에 빈 객체를 미리 만들어 둔다)
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """두 KVCacheBlocks 인스턴스를 결합한다."""
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]: ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> tuple[list[int], ...] | None: ...

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        """
        KVCacheBlocks 인스턴스를 block_id 목록으로 변환한다.

        반환:
            tuple[list[int], ...]: 다음 구조의 튜플.
                - 바깥 튜플: KV cache group 단위
                - 안쪽 리스트: 해당 그룹 블록의 block_id 목록
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """해시가 없는 블록들의 block_id를 반환한다."""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]

    def new_empty(self) -> "KVCacheBlocks":
        """
        블록 없이 새 KVCacheBlocks 인스턴스를 생성합니다.
        """
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: prefix cache stats 생성을 log_stats에 더 엄격히 연동할 수 있다.
        # 현재도 log_stats가 켜진 경우 향후 노출 가능한 구성 포인트가 있어
        # 관련 주석을 유지한다.
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

        # 블록이 없는 KVCacheBlocks를 미리 만들어 둔다.
        # 호출자는 새 객체를 직접 생성하지 말고 create_kv_cache_blocks를 통해
        # 이를 재사용하여 GC 오버헤드를 줄여야 한다.
        #
        # 빈 KVCacheBlocks가 불변이 되도록 중첩 튜플을 사용한다.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

    @property
    def usage(self) -> float:
        """KV 캐시 사용량을 가져옵니다.

        반환:
            KV 캐시 사용량(0.0과 1.0 사이).
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """접두사 캐시 통계를 가져오고 재설정합니다.

        반환:
            현재 접두사 캐싱 통계 또는 로깅이 비활성화된 경우 없음.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """요청의 계산 완료(캐시 히트) 블록을 반환한다.
        계산 완료 블록은 항상 full block이어야 한다.

        인수:
            request: 계산 완료 블록을 조회할 요청.

        반환:
            다음을 포함하는 튜플:
                - 요청의 계산 완료 블록들
                - 계산 완료 토큰 수
        """
        # prefix caching이 꺼져 있거나, 요청이 kv cache read skip으로 표시되면
        # prefix cache hit 탐색을 생략한다.
        # (예: prompt logprobs 필요 요청, all-pooling 모델 요청)
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # NOTE: 모든 토큰이 캐시에 있더라도 logits 계산을 위해 마지막 토큰은
        # 반드시 재계산해야 하므로, 최대 캐시 히트 길이는 prompt_length - 1이다.
        # allocate_slots()는 block-size 정렬된 num_computed_tokens를 요구하므로
        # 마지막 1토큰 대신 블록 단위 재계산이 발생할 수 있다.
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=num_new_computed_tokens,
                preempted=request.num_preemptions > 0,
            )

        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> KVCacheBlocks | None:
        """새 토큰을 붙일 요청에 대해 KV 슬롯을 할당한다.

        인수:
            request: 슬롯을 할당할 요청.
            num_new_tokens: 새로 할당/계산할 토큰 수.
            num_new_computed_tokens: 방금 prefix cache hit로 계산 완료된 토큰 수
                (외부 계산 토큰 제외).
            new_computed_blocks: 위 계산 완료 토큰에 대응하는 캐시 블록들
                (kv cache group별 튜플).
            num_lookahead_tokens: speculative decoding용 lookahead 토큰 수.
                (예: Eagle 같은 proposer에서 사용)
            num_external_computed_tokens: vLLM 내부가 아닌 커넥터 쪽에
                캐시되어 있는 외부 계산 토큰 수.
            delay_cache_blocks: 블록 캐시 반영을 지연할지 여부.
                P/D에서 KV 전송 완료가 다음 스텝에 일어나는 경우 사용한다.
            num_encoder_tokens: encoder-decoder 모델(예: Whisper)에서
                cross-attention용으로 할당할 encoder 토큰 수.
                decoder-only 모델이면 0이어야 한다.

        블록 배치:
        ```
        ----------------------------------------------------------------------
        | < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
        ----------------------------------------------------------------------
                                                  |   < to be computed > |
        ----------------------------------------------------------------------
                                  |            < to be allocated > |
        ----------------------------------------------------------------------
                                  | < to be cached (roughly, |
                                  | details below)>          |
        ----------------------------------------------------------------------
        | vLLM 또는 connector의 prefix-cached 토큰 |
        | 슬라이딩 윈도우 밖이면 안전하게 제거 가능 |
        ----------------------------------------------------------------------
        |   < vLLM에 의해 캐시됨 > | 캐시되지 않음 |
                                  | vLLM이지만 |
        | ref_cnt 증가됨 | ref_cnt 아직 미증가 | connector에 의해 캐시됨 |
        ----------------------------------------------------------------------
        ```

        약어:

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        ext_comp  = num_external_computed_tokens (connector 캐시)
        new       = num_new_tokens (미검증 draft 토큰 포함)
        lookahead = num_lookahead_tokens
        ```

        NOTE: new 토큰에 검증/미검증 draft가 섞여 있을 수 있으므로,
        캐시는 검증 완료 토큰만 반영하도록 `request.num_tokens`로 상한을 둔다.

        할당은 세 단계로 진행된다.
        - `comp` 구간의 불필요 블록을 해제하고 free block 여유를 확인
          (부족하면 None 반환)
        - prefix 토큰(`comp + new_comp + ext_comp`) 처리:
            - 불필요 블록 해제(예: 슬라이딩 윈도우 바깥)
            - 윈도우 내부 `ext_comp`용 새 블록 할당
        - 실제 계산 대상(`new + lookahead`)용 새 블록 할당

        반환:
            새로 할당된 블록 목록.
        """
        # KV를 비동기 로드하는 경우, 새로 계산할 토큰은 0이지만
        # 외부 계산 토큰 슬롯은 계속 할당해야 할 수 있다.
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # 로컬 계산 완료 토큰 = 기존 계산 완료 + 이번 prefix cache hit
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )
        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens,
            self.max_model_len,
        )

        # attention 계산에서 건너뛰는 블록(예: sliding window 바깥)을 먼저 해제한다.
        # 이후 새 블록 할당이 실패하더라도 이 정리는 수행해 두는 편이 유리하다.
        # (추가 축출 블록 수를 줄일 수 있음)
        self.coordinator.remove_skipped_blocks(
            request.request_id, total_computed_tokens
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # 새 블록 할당 불가
            return None

        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # 지금까지의 요청 블록에 새 계산 블록을 먼저 반영해 두어
            # 이후 새 블록 할당 실패 시 상태 불일치를 피한다.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )

        # P/D: 원격 수신이 필요한 경우 블록 캐시 반영을 지연한다.
        # 이 경우 로컬 캐시 상태만 우선 갱신한다.
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): 캐시 반영 상한은
        # num_local_computed_tokens + num_external_computed_tokens + num_new_tokens
        # 이지만, 거부될 수 있는 draft 같은 비확정 토큰은 제외해야 한다.
        # 따라서 `request.num_tokens`로 상한을 걸어 확정 토큰만 캐시한다.
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    def free(self, request: Request) -> None:
        """요청에 할당된 블록을 해제한다.
        캐싱 활성화 시 tail 블록이 먼저 축출되도록 역순으로 해제한다.

        인수:
            request: 블록을 해제할 요청.
        """
        self.coordinator.free(request.request_id)

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """`blocks`에서 더 이상 필요 없는 블록을 제거하고
        제거 위치를 null_block으로 대체한다.

        인수:
            request_id: 요청 ID.
            total_computed_tokens: 계산 완료 토큰 총수
                (로컬 계산 + 외부 계산 포함).
        """
        self.coordinator.remove_skipped_blocks(request_id, total_computed_tokens)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """block_id 기준으로 prefix cache에서 블록을 제거한다.

        인수:
            block_ids: 캐시에서 제거할 block_id 집합.
        """
        self.block_pool.evict_blocks(block_ids)

    def reset_prefix_cache(self) -> bool:
        """prefix cache를 재설정한다.
        RLHF에서 가중치 업데이트 후 캐시 무효화, 또는 벤치마크 초기화에 사용된다.

        반환:
            bool: 재설정 성공 시 True, 실패 시 False.
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """각 KV cache group별 공통 prefix block 개수를 계산한다.

        실행 중인 요청 하나를 기준으로 그 요청의 블록을 순회하며,
        할당된 KV cache를 가진 모든 요청이 공유하는 블록(ref_cnt가 req_to_blocks
        엔트리 수와 같은 블록)을 공통 prefix block으로 본다.

        NOTE(woosuk): KV cache가 할당된 요청 수는 현재 스텝에서 스케줄된 요청 수보다
        크거나 같을 수 있다. KV cache 할당 여부는
        1) 요청이 아직 종료되지 않았고
        2) 요청이 블록을 아직 해제하지 않았다는 사실만 의미하기 때문이다.

        따라서 "현재 스텝에서 스케줄된 요청들"은 모두 KV cache를 갖지만,
        KV cache를 가진 요청이 모두 스케줄된 것은 아닐 수 있다.

        이 때문에 스케줄된 요청들끼리는 공통 prefix를 공유하더라도,
        스케줄되지 않은 요청 중 비공유 요청이 섞여 있으면 결과가 0이 될 수 있다.
        현재는 이 케이스를 쉽게 구분하지 못하므로 0을 반환한다.

        인수:
            running_request_id: 실행 중인 요청의 요청 ID, 
                공통 prefix block 식별에 사용할 실행 중 요청 ID.

        반환:
            list[int]: 각 kv cache group의 공통 prefix block 개수.
        """
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    def take_events(self) -> list[KVCacheEvent]:
        """block pool에서 KV cache 이벤트를 가져온다.

        반환:
            KV cache 이벤트 목록.
        """
        return self.block_pool.take_events()

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """요청의 블록을 반환한다."""
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """요청의 block_id 목록을 반환한다."""
        return self.get_blocks(request_id).get_block_ids()

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """활성화된 경우 요청 블록을 캐시에 반영한다.

        인수:
            request: 캐시 반영 대상 요청.
            num_computed_tokens: 계산 완료 토큰 수.
                (이미 캐시된 토큰 + 이번에 캐시할 토큰 포함)
        """
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        # 비어 있지 않은 경우에만 새 KVCacheBlocks 객체를 생성한다.
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks

    def new_step_starts(self) -> None:
        """새 스케줄링 스텝 시작 시 호출된다."""
        self.coordinator.new_step_starts()
