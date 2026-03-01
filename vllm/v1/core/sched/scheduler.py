# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import numpy as np

from vllm import envs
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsReader,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set가 켜지면 완료된 요청 ID 집합을
        # EngineCoreOutputs에 별도로 포함한다.
        # 이는 멀티 엔진 환경에서 요청 수명을 효율적으로 추적할 때 사용된다.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        self.prev_step_scheduled_req_ids: set[str] = set()

        # 스케줄링 제약.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )

        # 스케줄러용 KVConnector를 생성한다.
        # 각 워커에는 role=WORKER인 대응 커넥터가 있다.
        # KV 커넥터는 P/D 및 오프로드를 위해 원격 KV를 푸시/풀합니다.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        if self.vllm_config.kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = (
                self.vllm_config.kv_transfer_config.kv_load_failure_policy
            )
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # req_id -> 요청
        self.requests: dict[str, Request] = {}
        # 스케줄링 정책
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # 요청에 대한 우선순위 대기열.
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # 이전 단계와 
        # 현재 단계. 이는 완료된
        # 요청에 대해 작업자에게 알리고 해당 요청에 대해 캐시된 상태를 해제할 수 있도록 하는 데 사용됩니다.
        # 이는 각 예약 단계가 끝날 때 플러시됩니다.
        self.finished_req_ids: set[str] = set()

        # 스트리밍 입력을 기다리는 요청에 대한 카운터입니다. 완료되지 않은 요청 수
        # 계산에 사용됩니다.
        self.num_waiting_for_streaming_input: int = 0

        # KV 커넥터: 비동기 KV 로드 또는 수신 중 요청
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # 인코더 관련.
        # 해당되는 경우 인코더 캐시 크기 계산
        self.supports_mm_inputs = mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )
        self.mm_budget = mm_budget = (
            MultiModalBudget(vllm_config, mm_registry)
            if self.supports_mm_inputs
            else None
        )

        # 참고: 텍스트 전용 encoder-decoder도
        # 구현 편의를 위해 멀티모달 인터페이스를 통해 다룬다.
        # 예: https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        self.encoder_cache_manager = (
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)
        )

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.uses_draft_model():
                self.num_lookahead_tokens = self.num_spec_tokens

        # KV 캐시 관리자를 생성합니다.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
            hash_block_size=self.block_size,
            metrics_collector=self.kv_metrics_collector,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER

        def has_mamba_layers(kv_cache_config: KVCacheConfig) -> bool:
            return any(
                isinstance(group_spec.kv_cache_spec, MambaSpec)
                for group_spec in kv_cache_config.kv_cache_groups
            )

        self.has_mamba_layers = has_mamba_layers(kv_cache_config)
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        if self.vllm_config.model_config.enable_return_routed_experts:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            self.routed_experts_reader = RoutedExpertsReader.create()

            assert len(kv_cache_config.kv_cache_groups) > 0, (
                "enable_return_routed_experts requires at least one kv cache group"
            )
            self.max_num_kv_tokens = (
                kv_cache_config.num_blocks // len(kv_cache_config.kv_cache_groups) + 1
            ) * self.block_size

            self.routed_experts_reader.attach_buffer(
                max_num_kv_tokens=self.max_num_kv_tokens,
                vllm_config=self.vllm_config,
            )

        self._pause_state: PauseState = PauseState.UNPAUSED

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        assert num_external_computed_tokens == 0, (
            "External KV connector is not verified yet"
        )
        num_computed_tokens = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # 아래 조건에서는 prefill 구간에 대해 block 정렬 분할을 적용한다.
        # - 재개 전 요청: num_computed_tokens < num_prompt_tokens
        # - 재개된 요청: num_computed_tokens < num_tokens - 1
        #   (`num_tokens - 1`은 일반 decode 구간을 제외하기 위한 기준)
        if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
            # Mamba 상태를 block 단위로 캐시하려면 `num_new_tokens`는
            # `block_size` 배수여야 한다.
            # 단, `num_new_tokens < block_size`이면 해당 상태는 캐시되지 않는다.
            # Eagle 모드에서는 FullAttn이 마지막 hit 블록을 제거하므로,
            # Mamba 캐시 누락 방지를 위해 마지막 청크 길이를 보정한다.
            block_size = self.cache_config.block_size
            last_cache_position = request.num_tokens - request.num_tokens % block_size
            # eagle prune
            if self.use_eagle:
                last_cache_position = max(last_cache_position - block_size, 0)
            num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
            if num_computed_tokens_after_sched < last_cache_position:
                # block_size 정렬
                num_new_tokens = num_new_tokens // block_size * block_size
            elif (
                num_computed_tokens
                < last_cache_position
                < num_computed_tokens_after_sched
            ):
                # 마지막 청크를 강제로 캐시합니다.
                num_new_tokens = last_cache_position - num_computed_tokens
            else:
                # 마지막 몇 개의 토큰을 미리 채웁니다.
                pass
        return num_new_tokens

    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on scheduling:
        # 스케줄러에는 "디코딩 단계" / "프리필 단계"가 분리되어 있지 않다.
        # 각 요청은 단지 num_computed_tokens와 num_tokens_with_spec를 가진다.
        # num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)
        # 매 스텝마다 스케줄러는 각 요청의 num_computed_tokens가
        # num_tokens_with_spec를 따라잡도록 토큰을 배정한다.
        # 이 방식은 chunked prefill, prefix caching, speculative decoding,
        # 그리고 향후 jump decoding 최적화까지 포괄할 수 있다.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # 일시 중지되면 요청을 예약하지 마십시오.
            token_budget = 0

        # Encoder 관련
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode 관련
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # 로깅용 타임스탬프
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        # 먼저 RUNNING 요청을 스케줄한다.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
                # 이는 (num_computed_tokens + 1) - (num_output_placeholders - 1)입니다.
                # 출력 자리 표시자도 계산된 토큰
                # 개수에 포함되어 있으므로 (num_output_placeholders - 1)을 빼서 초안
                # 토큰을 제거합니다. 
                # 모두 거부되더라도 추가 단계가 필요하지 않습니다.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling:
                # 이전 스텝에서 request.max_tokens에 도달했음이 확실하면
                # 불필요한 추가 스텝을 잡지 않는다.
                # partial draft token 스케줄링은 uniform decode 최적화를 방해하므로 피한다.
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # 입력 위치가 max_model_len을 넘지 않도록 제한한다.
            # (spec decoding 사용 시 특히 필요)
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )

            # Encoder 입력 스케줄링
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            if num_new_tokens == 0:
                # 아래 사유 중 하나로 요청을 스케줄할 수 없다.
                # 1) 배정할 새 토큰이 없음
                #    (a) PP>1에서 프롬프트 토큰은 모두 배정했지만 아직 종료 전
                #    (b) Async scheduling에서 max_total_tokens/max_model_len 도달
                # 2) encoder 예산 소진
                # 3) encoder cache 소진
                # 4) mamba cache mode="align" 하이브리드 모델에서
                #    block-aligned chunk 예산 부족
                # NOTE(woosuk): 여기서는 break가 아니라 continue를 사용해
                # FCFS를 엄격히 고수하지 않고, 더 낮은 우선순위 요청의
                # 스케줄 가능성을 남긴다.
                req_index += 1
                continue

            # 요청에 필요한 새 KV 블록을 스케줄한다.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # 요청 스케줄 가능
                        break

                    # 요청 스케줄 불가 -> 가장 낮은 우선순위 요청 선점
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                # 선점된 요청에 이 스텝에서 배정된 encoder 입력이 있으면
                                # encoder compute budget을 되돌린다.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # 더 이상 선점할 요청이 없으므로 현재 요청 스케줄 불가
                        break

            if new_blocks is None:
                # 현재 요청 스케줄 불가
                break

            # 요청 스케줄 확정
            scheduled_running_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decoding 관련
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # 필요 시 다음 스텝 전에 update_draft_token_ids에서
                # 새 spec token을 설정한다.
                request.spec_token_ids = []

            # Encoder 관련
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # 인코더 캐시를 할당합니다.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # scheduled_running_reqs의 LoRA 집합
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # 다음으로 WAITING 요청을 스케줄한다.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            # 임시 RequestQueue에 "이번에 건너뛴 WAITING 요청"을 모아두고
            # 나중에 waiting 큐 앞쪽으로 되돌린다.
            skipped_waiting_requests = create_request_queue(self.policy)

            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()
                request_id = request.request_id

                # KVTransfer: 원격 KV를 아직 기다리는 요청은 건너뛴다.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        if request.num_preemptions:
                            # 새 요청이 아니라, 재개되는 선점 요청으로 처리
                            request.status = RequestStatus.PREEMPTED
                        else:
                            request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Structured output 요청이 FSM 컴파일 대기 중이면 건너뛴다.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Streaming: 다음 스트리밍 요청을 기다리는 상태면 건너뛴다.
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    assert not request.streaming_queue
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                # 요청 추가 시 max_loras 제약을 만족하는지 확인
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # 배정 시 max_loras를 초과하므로 건너뜀
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0

                # 이미 캐시된 토큰 조회
                if request.num_computed_tokens == 0:
                    # 로컬 cache hit 조회
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    # KVConnector 사용 시 외부 cache hit 조회
                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            # 요청을 예약할 수 없습니다. 왜냐하면
                            # KVConnector가 결정할 수 없었기 때문입니다.
                            # 일치 토큰 수.
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # 계산된 총 토큰 수(로컬 + 외부).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                else:
                    # KVTransfer: 비동기 KV 수신 완료 요청은
                    # num_computed_tokens > 0일 수 있다.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    # KVTransfer: 원격 KV를 로드하고 새 작업에 할당하지 않습니다.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    # 스케줄링 대상 토큰 수.
                    # 출력 토큰이 있는 재개 요청까지 포함하려고
                    # `request.num_prompt_tokens` 대신 `request.num_tokens`를 사용한다.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # pooler 요청이 chunk되는 것을 막기 위해,
                    # chunked prefill은 명시적으로 켜져 있어야 한다.
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # chunked prefill이 꺼져 있으면 여기서 중단한다.
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # 인코더 입력을 예약합니다.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # 요청을 스케줄할 수 없습니다.
                            break

                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        break

                # P/D 분리 + speculative decoding 조합에서
                # 로컬/원격 블록 수가 잠시 어긋나는 극단 케이스를 보정한다.
                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                # cross-attention 블록 할당 필요 여부를 계산한다.
                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    # 요청을 스케줄할 수 없습니다.

                    # 참고: 스케줄 실패 시 encoder cache 할당도 롤백해야 한다.
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # KVTransfer: 커넥터는 이 정보로 해당 요청의 로드 필요 여부를 판단한다.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                # 위에서 break되지 않았다면 요청은 waiting에서 pop해 확정한다.
                request = self.waiting.pop_request()
                if load_kv_async:
                    # 비동기 로드 시에는 메모리만 확보하고
                    # 요청 상태를 WAITING_FOR_REMOTE_KVS로 바꿔 다시 대기시킨다.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # 접두사 캐시된 토큰 수를 계산합니다.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # 인코더 관련.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    # 인코더 캐시를 할당합니다.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # 외부 로드 인코더 캐시에 할당
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)

            # 건너뛴 요청을 대기 대기열의 헤드에 다시 넣습니다.
            if skipped_waiting_requests:
                self.waiting.prepend_requests(skipped_waiting_requests)

        # 일정 제약 조건이 충족되는지 확인합니다.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # RUNNING 대기열의 일부 요청은 이 단계에서 일정이 지정되지 않을 수 있으므로
        # 예약된 요청 수는 len(self.running)보다 작을 수 있다.
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # 실행 중인 대기열의 모든 요청 중에서 가장 긴 공통 접두사를 가져옵니다.
        # 이는 잠재적으로 계단식 주의에 사용될 수 있습니다.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        # 스케줄러 출력을 구성합니다.
        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # 이 단계에서 예약된 요청 ID를 기록합니다.
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            # done_req_ids에는 "이번 스텝에 새로 스케줄된 요청"이 아니라
            # 직전 스텝 이후 지금까지 완료된 요청 ID가 들어간다.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
        )

        # NOTE(Kuntai): connector metadata는 여러 목적을 갖는다.
        # 1. KV 캐시 저장소 계획
        # 2. KV 캐시 load/save 작업을 하나의 opaque 객체로 정리
        # 3. 커넥터의 내부 상태를 지웁니다.
        if self.connector is not None:
            meta: KVConnectorMetadata = self.connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.kv_connector_metadata = meta

        # ECConnector용 메타데이터를 구성한다.
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """요청을 선점하고 대기 대기열에 다시 넣습니다.

        참고: 요청은 이 메서드 밖에서 running 큐에서 pop되어야 한다.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # 요청을 waiting 큐로 되돌린다.
        self.waiting.prepend_request(request)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # 스케줄된 요청들의 num_computed_tokens를 선반영한다.
        # 1) 현재 스텝 SchedulerOutput에는 "원래 스케줄된 토큰 수"가 필요하고,
        # 2) num_computed_tokens를 여기서 올려야 다음 스텝 재스케줄링이 가능하다.
        # 3) speculative 토큰 거부 등 후처리는 update_from_output에서 보정한다.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )

            # _free_encoder_inputs는 num_computed_tokens를 기준으로 동작한다.
            # speculative decoding에서 값이 나중에 조정될 수 있지만,
            # encoder 입력은 출력이 아닌 prompt 영역이므로 여기서 호출해도 안전하다.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # 완료 요청 ID 집합 초기화.
        # 참고: 기존 set 객체를 clear하면 외부 참조(scheduler_output)에 영향이 있어
        # 새 set으로 교체한다.
        self.finished_req_ids = set()

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """
        다음 스트리밍 업데이트로 대기 세션을 업데이트합니다.

        이전 입력 청크에서 마지막으로 샘플링된 출력 토큰을 삭제합니다.
        """

        # 현재 스트리밍 입력 처리: 계산 완료된 출력 토큰만 유지하고
        # 마지막 샘플링 토큰은 폐기한다.
        num_computed_tokens = session.num_computed_tokens
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : num_computed_tokens
        ]
        del session._all_token_ids[num_computed_tokens:]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # 보관된 출력 토큰으로 프롬프트 확장.
        session.prompt_token_ids.extend(kept_output_tokens)

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # 새 토큰에 대한 블록 해시 업데이트.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # 참고: PP+비동기 스케줄링에서는 직접 GPU
            # 브로드캐스트 경로(`input_batch.prev_sampled_token_ids`)를 통해 토큰 ID를 소비하므로 
            # 생략할 수 있습니다. 이 페이로드.
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # PP를 사용할 때 스케줄러는 샘플링된 토큰을 다시 보냅니다. 
                # 첫 번째 
                # 단계 작업자와 마지막 단계 작업자 사이에 직접적인 통신이 없기 때문입니다. 그렇지 않으면 모델 실행기
                # 가 샘플링 토큰을 캐시하므로
                # 샘플링된 토큰을 다시 보낼 필요가 없습니다.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            scheduled_in_prev_step = req_id in self.prev_step_scheduled_req_ids
            if idx >= num_running_reqs:
                assert not scheduled_in_prev_step
                resumed_req_ids.add(req_id)
            if not scheduled_in_prev_step:
                all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        현재 스텝에서 처리할 encoder 입력을 결정하고,
        그 결과에 맞춰 `num_new_tokens`와 encoder 예산을 조정한다.

        encoder 입력은 아래 조건을 모두 만족하면 스케줄된다.
        - 이번 스텝 계산 토큰 구간과 겹침:
          [num_computed_tokens, num_computed_tokens + num_new_tokens)
        - 아직 계산되지 않았고 encoder cache에도 없음
        - 원격 encoder cache(ECConnector)에도 없음
        - encoder token budget이 충분함
        - encoder cache에 저장 공간이 있음

        cache/budget 제약 때문에 특정 encoder 입력을 스케줄할 수 없으면,
        그 입력 직전까지만 decoder 토큰이 계산되도록 `num_new_tokens`를 줄인다.

        참고: num_computed_tokens에는 로컬 cache hit와
        외부 cache hit(KVConnector 경유)가 모두 포함된다.
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # 스케줄러는 request 단위로 동작하므로,
        # request 하나에 encoder 입력이 여러 개인 경우를 위해
        # encoder 입력 단위 임시 추적기가 필요하다.
        mm_hashes_to_schedule = set()
        num_embeds_to_schedule = 0
        for i, mm_feature in enumerate(mm_features):
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            # 아래 두 구간이 겹치면 encoder 출력이 필요하다.
            # [num_computed_tokens, num_computed_tokens + num_new_tokens)
            # [start_pos, start_pos + num_encoder_tokens)
            if (
                start_pos
                >= num_computed_tokens + num_new_tokens + shift_computed_tokens
            ):
                # 이 단계에서는 인코더 입력이 필요하지 않습니다.
                break

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # encoder 입력은 이미 계산되었다고 본다.
                # 여기 계산은 일반 decoder 토큰 계산과 다르다.
                # encoder 출력은 decoder가 처리하는 토큰으로 변환되지 않아
                # num_computed_tokens에 직접 반영되지 않는다.
                # 대신 start_pos는 encoder 입력 계산 보장 지점을 뜻한다.
                # encoder-decoder에서는 decoder 실행 전에 encoder 계산이 끝나야 하므로
                # start_pos는 0이어야 한다. num_computed_tokens > 0이면 이미 encoder
                # 계산을 마쳤다고 볼 수 있어 여기서 건너뛴다.
                continue
            elif start_pos + num_encoder_tokens <= num_computed_tokens:
                # encoder 입력이 이미 계산되어 decoder KV cache에 있다.
                continue

            if not self.is_encoder_decoder:
                # 현재 encoder-decoder 모델에는 encoder cache를 아직 사용하지 않는다.
                if item_identifier in mm_hashes_to_schedule:
                    # 동일 encoder 입력이 현재 스텝에 이미 스케줄됨
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # encoder 입력이 이전 스텝에서 이미 계산/캐시됨
                    continue

            # encoder 입력 chunking이 금지되면 MM 항목을 부분 스케줄하지 않는다.
            # 스케줄 범위가 MM 입력 일부만 덮는 경우 MM 시작 직전으로 롤백한다.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                # EAGLE shift를 고려해 롤백해야 encoder cache miss를 막을 수 있다.
                # 롤백 후에도 start_pos 이전에서 멈추도록 보장한다.
                num_new_tokens = max(
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)
                )
                break
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # encoder cache가 가득 찼거나 encoder compute budget이 소진됨.
                # NOTE(woosuk): encoder는 보통 양방향 attention이므로
                # 입력 토큰 전체를 한 번에 처리한다고 가정한다.
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # encoder 입력 바로 직전까지 decoder 토큰만 스케줄한다.
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # 접두사 캐싱으로 인해 인코더 입력을 사용할 수 없더라도
                    # num_computed_tokens가 start_pos보다 클 수 있다.
                    # 이 경우 이번 스텝에는 토큰을 더 스케줄하지 않는다.
                    num_new_tokens = 0
                break

            # 현재 스케줄된 encoder placeholder 범위에서
            # 실제로 처리할 임베딩 수를 계산한다.
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos
            )
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # 인코더 자리표시자 토큰의 현재 범위에 임베딩이 없으면
            # 해당 인코더 입력은 건너뛴다.
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier
            ):
                mm_hashes_to_schedule.add(item_identifier)
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(item_identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # structured output을 사용하는 스케줄된 요청 ID를 모은다.
        # 비트마스크 행 순서는 이 리스트 순서를 따른다.
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # 로드 실패한 외부 계산 블록을 기준으로 영향을 받은 요청을 찾고,
            # 재계산이 필요하도록 계산 토큰 카운트를 조정한다.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids
            )

        # 참고(woosuk): len(num_scheduled_tokens)은 최대 1K 이상이 될 수 있으므로
        # 아래 루프는 병목이 될 수 있다. 루프 안의 고비용 연산을 최소화한다.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # KV 로드 실패로 인해 실패했거나 다시 예약된 요청을 건너뜁니다.
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # 모델 실행 중 요청이 중단되면(예: PP/async scheduling)
                # 여기 시점에 이미 완료 상태일 수 있다.
                # NOTE(Kuntai): Delay_free_blocks=True(비동기 KV 전송)에서는
                # 요청 객체가 None이 아닐 수 있으므로 is_finished()로 확인한다.
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens는 이번 스텝 처리 토큰 수를 뜻한다.
                # speculative 토큰이 거부되면 거부 개수만큼 되돌린다.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # 비동기 스케줄링의 경우 num_output_placeholders에는 
                # 예약된 사양 토큰 개수도 포함되므로 유사하게 조정됩니다.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status

            # 중지를 확인하고 요청 상태를 업데이트합니다.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:
                # pooling 요청은 출력이 생성되면 즉시 종료된다.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            routed_experts = None
            finish_reason = None
            if stopped:
                routed_experts = self._get_routed_experts(request)

                # _handle_stopped_request가 상태를 바꿀 수 있으므로
                # 먼저 finish reason을 캡처한다.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params = self._free_request(request)

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # 필요한 경우 샘플 logprobs를 추출합니다.
            if (
                request.sampling_params is not None
                and request.sampling_params.logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                ok = struct_output_request.grammar.accept_tokens(req_id, new_token_ids)
                if not ok:
                    logger.warning(
                        "Unexpected: grammar rejected tokens %s for request %s.",
                        new_token_ids,
                        req_id,
                    )

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # 이 요청에 대한 프롬프트 logprobs를 가져옵니다.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if (
                new_token_ids
                or pooler_output is not None
                or kv_transfer_params
                or stopped
            ):
                # 이 요청에 대한 EngineCoreOutput을 추가합니다.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                        num_external_computed_tokens=request.num_external_computed_tokens,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # 불변: EngineCore는 부분 사전 채우기 출력을 반환하지 않습니다.
                assert not prompt_logprobs_tensors

        # 실행 중인 대기열과 대기 중인 대기열에서 중지된 요청을 제거합니다.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # 이는 드문 경우이며 성능에 영향을 미칠 가능성이 거의 없습니다.
            self.waiting.remove_requests(stopped_preempted_reqs)

        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )

        # KV 커넥터: 완료된 KV 전송에 대한 상태를 업데이트합니다.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # KV 캐시 관리자에서 KV 캐시 이벤트를 수집합니다.
        events = self.kv_cache_manager.take_events()

        # 커넥터에서 KV 캐시 이벤트를 수집합니다.
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # 수집된 KV 캐시 이벤트를 게시한다.
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # 이번 스텝 출력이 있는 모든 클라이언트에 대해 EngineCoreOutputs를 구성한다.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # 마지막 출력 전송 이후 완료된 요청 ID를 포함한다.
            for client_index, finished_set in finished_req_ids.items():
                # 이 클라이언트의 EngineCoreOutputs에 완료된 요청 세트를 설정합니다.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats
            )
        ) is not None:
            # 프런트엔드 중 하나만 통계를 반환합니다.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # 요청이 없더라도 통계를 반환해야 합니다.
                # 이번 스텝 출력을 만들기 위해 빈 객체를 생성한다.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def _handle_stopped_request(self, request: Request) -> bool:
        """완료되면 True를 반환합니다(재개 가능한 요청의 경우 False일 수 있음)."""
        if not request.resumable:
            return True

        if request.streaming_queue:
            update = request.streaming_queue.popleft()
            if update is None:
                # 스트리밍 요청이 완료되었습니다.
                return True
            self._update_request_as_session(request, update)
        else:
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            self.num_waiting_for_streaming_input += 1

        self.waiting.add_request(request)
        return False

    def _get_routed_experts(self, request: Request) -> np.ndarray | None:
        if not self.vllm_config.model_config.enable_return_routed_experts:
            return None

        kv_blocks = self.kv_cache_manager.get_blocks(request.request_id)
        block_ids = kv_blocks.get_block_ids()[0]
        num_tokens = request.num_tokens - 1

        # 컴퓨팅 슬롯 매핑
        block_ids_array = np.array(block_ids, dtype=np.int32)
        num_blocks = len(block_ids)
        block_size = self.block_size

        # 블록 오프셋 생성
        block_offsets = np.arange(0, block_size)

        # 컴퓨팅 슬롯 매핑: 슬롯 = block_id * block_size + offset
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_array.reshape((num_blocks, 1)) * block_size
        ).flatten()[:num_tokens]

        return self.routed_experts_reader.get_routed_experts(indices=slot_mapping)

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        # 생성 토큰을 반영하고 stop 조건을 확인한다.
        # 요청이 아직 prefill 중이면 모델 러너가 빈 토큰 ID를 줄 수 있다.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # 중지 및 업데이트 요청 상태를 확인합니다.
            # 이는 EngineCoreOutput을 만들기 전에 호출해야 합니다.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # 필요한 경우 새 토큰을 자릅니다.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # 최적화: 빈 set이면 list(...) 변환을 피한다.
        if not cached_encoder_input_ids:
            return

        # 순회 중 set 변경을 피하려고 list(...)로 복사해 순회한다.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # Whisper를 사용하면 단일 토큰을 생성하자마자
                # encoder 입력 처리가 끝났다고 볼 수 있다.
                # cross-attention KV는 이미 계산/캐시된 상태다.
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif start_pos + num_tokens <= request.num_computed_tokens:
                # encoder 출력이 이미 처리/저장된 상태다.
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # 요청이 이미 완료되었을 수 있으므로 건너뛴다.
                continue

            if request.is_prefill_chunk:
                # prefill chunk에서는 draft 토큰을 무시한다.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # 새로 생성된 사양 토큰 ID를 요청에 추가합니다.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # 유형: 무시[union-attr]
            request.spec_token_ids = spec_token_ids

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # 요청이 이미 완료되었을 수 있으므로 건너뛴다.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # 초안을 예정된 사양 토큰 수로 자릅니다.
            # (예를 들어 청크 사전 채우기 사례에 필요함).
            del spec_token_ids[orig_num_spec_tokens:]
            # 문법 제약을 다시 적용한다.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                assert metadata is not None and metadata.grammar is not None
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
            # 사양 토큰의 원래 수를 패드합니다.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_request_counts(self) -> tuple[int, int]:
        """반환 (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting)

    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # 다음 입력 청크(또는 완료된 센티널)를 대기열에 넣습니다.
                existing.streaming_queue.append(update)
            elif update is not None:
                # 다음 입력 청크를 시작합니다.
                self._update_request_as_session(existing, update)
            else:
                # 스트리밍 입력 세션이 완료되었습니다.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            if request.resumable:
                request.streaming_queue = deque()
            self.waiting.add_request(request)
            self.requests[request.request_id] = request
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        """스케줄러 외부에서 종료 신호를 처리합니다.

        예를 들어 API 서버는 다음과 같은 경우 요청을 중단할 수 있습니다. 클라이언트
        연결이 끊어집니다.

        request_ids가 None이면 모든 요청이 완료됩니다.

        반환:
            중단된 요청에 대한 (req_id, client_index)의 튜플입니다. 않을 것이다
            이미 완료된 항목을 포함합니다.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # 첫 번째 통과: 대기열에서 제거할 요청을 수집합니다.
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # 잘못된 요청 ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                waiting_requests_to_remove.append(request)

        # 대기열에서 모든 요청을 한 번에 제거합니다. 효율성 향상을 위해
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        # 두 번째 패스: 상태 설정 및 요청 해제
        for request in valid_requests:
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)

        return [(r.request_id, r.client_index) for r in valid_requests]

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        assert request.is_finished()

        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        self._pause_state = pause_state

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        if self._pause_state == PauseState.PAUSED_NEW:
            return len(self.running)
        num_waiting = len(self.waiting) - self.num_waiting_for_streaming_input
        return num_waiting + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """KV 접두사 캐시를 재설정합니다.

        reset_running_requests가 True이면 실행 중인 모든 요청이
        선점되고 대기 대기열로 이동됩니다.
        그렇지 않으면 KV 캐시를 쓰는 실행 중 요청이 없을 때만 재설정한다.
        """
        if reset_running_requests:
            # 로깅용입니다.
            timestamp = time.monotonic()
            # 모든 running 요청을 waiting으로 되돌려 KV를 무효화한다.
            # 이렇게 하면 모든 KV 블록의 ref_cnt를 0으로 만들 수 있어
            # prefix cache reset 성공 여부를 확인할 수 있다.
            # 역순 선점으로 waiting 큐 재삽입 시 FIFO 순서를 맞춘다.
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp)
                # NOTE(zhuohan): async scheduling에서는 최신 출력 토큰을 즉시 버려
                # 중복 반복 출력을 방지한다.
                request.num_output_placeholders = 0
                request.discard_latest_async_tokens = True

            # 같은 스텝에서 선점+재개를 강제하므로,
            # 이 요청들이 이전 스텝에 스케줄되지 않았던 것처럼 처리되게
            # prev_step_scheduled_req_ids를 비운다.
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            logger.warning("reset_connector called but no KV connector is configured.")
            return False

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def reset_encoder_cache(self) -> None:
        """인코더 캐시를 재설정하여 캐시된 모든 인코더 출력을 무효화합니다.

        이 메서드는 모델 가중치가 업데이트될 때 호출되어
        오래된 비전 임베딩이 재사용되지 않도록 해야 합니다.
        """
        self.encoder_cache_manager.reset()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            encoder_cache_usage=self._get_encoder_cache_usage(),
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    def _get_encoder_cache_usage(self) -> float:
        """인코더 캐시 사용량을 분수(0.0~1.0)로 가져옵니다."""
        ecm = self.encoder_cache_manager
        if ecm.cache_size == 0:
            return 0.0
        used_slots = ecm.cache_size - ecm.num_free_slots
        return used_slots / ecm.cache_size

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

    ########################################################################
    # KV 커넥터 관련 메서드
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        해당하는 경우 KV 커넥터 request_finished() 메서드를 호출합니다.

        포함할 선택적 kv 전송 매개변수를 반환합니다.
        요청 출력.
        """
        if self.connector is None:
            return False, None

        # connector에 block table을 넘기기 전에,
        # window 밖 prefix block을 먼저 해제한다.
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            total_computed_tokens=request.num_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): 모든 connector가 HMA를 지원하도록 강제한 뒤에는
            # 이 코드 경로는 제거되어야 한다.
            # 이 경로에서는 hybrid memory allocator가 이미 꺼져 있어야 하므로
            # 여기서 한 번 더 확인한다.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV connector 관점에서 request_id 수신 완료 여부를 확인한다.

        finished_recving_kv_req_ids는 이전 steps()의 update_from_output 시점에
        worker 쪽 connector가 채워둔다.

        KV 전송 준비가 끝나면 블록을 캐시하고,
        요청 상태를 WAITING_FOR_REMOTE_KV에서 WAITING으로 되돌린다.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        if request.request_id in self.failed_recving_kv_req_ids:
            # KV 로드 실패 요청이다. num_computed_tokens는
            # _update_requests_with_invalid_blocks에서 이미 조정됐다.
            if request.num_computed_tokens:
                # 유효한 계산 토큰을 캐시합니다.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # 유효한 계산 토큰이 없습니다. 할당된 블록을 해제합니다.
                # 재시도 시 로컬 캐시 적중이 발생할 수 있습니다.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # 블록 수신이 끝났으므로 실제 캐시 반영을 수행한다.
            (block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)
            num_computed_tokens = len(block_ids) * self.block_size
            # 요청 길이가 1블록 미만인 경우도 처리한다.
            num_computed_tokens = min(num_computed_tokens, request.num_tokens)
            if num_computed_tokens == request.num_tokens:
                num_computed_tokens -= 1
            # 캐싱이 활성화된 경우 블록을 캐시합니다.
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

            # 예약을 위한 요청 상태를 업데이트합니다.
            request.num_computed_tokens = num_computed_tokens

        # 준비가 되었음을 반환합니다.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV 커넥터: 출력을 기반으로 스케줄러 상태를 업데이트합니다.

        작업자 측 커넥터는 done_recving을 추가하고
        done_sending 요청을 출력으로 보냅니다.
        * done_sending인 경우: 블록을 해제합니다.
        # if done_recving: 상태에 추가하여 할 수 있도록 합니다.
            다음 단계에서 요청을 예약하세요.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV 커넥터:: 마지막 단계에서 수신 상태를 업데이트하고 상태를 보냅니다.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        잘못된 KV 캐시 블록의 영향을 받은 요청을 식별하고 업데이트합니다.

        이 메서드는 지정된 요청을 스캔하고 유효하지 않은 블록이 있는 요청을 감지하고
         `num_computed_tokens`를 유효한 가장 긴 접두사로 조정합니다.
        관찰성을 위해 영향을 받는 모든 요청에서 다시 계산해야 하는
        총 토큰 수도 누적합니다.

        인수:
            requests: invalid block 영향을 스캔할 요청 집합.
            invalid_block_ids: 유효하지 않은 블록 ID 집합.
            evict_blocks: 캐시에서 제거할 블록을 수집할지 여부.
                (아직 캐시되지 않은 비동기 요청은 False)

        반환:
            튜플:
                - affected_req_ids (set[str]): invalid block 영향을 받은 요청 ID.
                - total_affected_tokens (int): 전체 요청에서 재계산이 필요한 총 토큰 수.
                - blocks_to_evict (set[int]): invalid block 및 하류 의존 블록 포함
                  캐시 제거 대상 블록 ID 집합.
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # invalid block이 여러 요청에 공유된 경우,
        # 첫 번째 요청에서만 실제 재계산 표시를 하고 나머지는 공유 처리한다.
        # 이 집합은 이미 재계산 대상으로 표시한 block_id를 추적한다.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): 하이브리드 메모리 할당자에 대한 지원 추가
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # 외부 계산 토큰 포함 케이스 처리.
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # 비동기 로드 요청은 실제로 계산 완료된 블록까지만 본다.
                # 실패 재수신 요청이면 num_computed_tokens가 이미 보정돼 있다.
                req_num_computed_tokens = (
                    request.num_computed_tokens
                    if req_id in self.failed_recving_kv_req_ids
                    else len(req_block_ids) * self.block_size
                )
            else:
                # 동기 로드는 실패 보정을 이미 반영했으므로 cached 토큰 기준을 사용한다.
                req_num_computed_tokens = request.num_cached_tokens

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # 이전 요청과 공유된 invalid block이며,
                    # 이미 재계산 대상으로 표시된 상태다.
                    # 따라서 이 요청 재스케줄 시 해당 블록은 계산됨으로 간주 가능하다.
                    # 현재 이 로직은 동기 로드에만 적용된다(비동기 공유 미지원).
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # 이 요청은 이미 첫 invalid block을 처리해
                    # num_computed_tokens를 갱신했다.
                    continue

                marked_invalid_block = True
                # 첫 번째 실패한 블록에서 계산된 토큰을 자릅니다.
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens
                request.num_external_computed_tokens -= num_affected_tokens
                # 잘못된 블록 및 모든 다운스트림 종속 블록을 수집합니다.
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # 이 요청의 invalid block은 모두 공유 블록이므로
                    # 재계산은 이전 요청이 담당한다.
                    # 따라서 cached 토큰까지만 계산된 것으로 롤백한다.
                    # 현재 이 로직은 동기 로드에만 적용된다(비동기 공유 미지원).
                    total_affected_tokens += (
                        request.num_computed_tokens - request.num_cached_tokens
                    )
                    request.num_computed_tokens = request.num_cached_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        """
        invalid KV 캐시 블록의 영향을 받은 요청들을 처리한다.

        반환:
            update_from_output 메인 루프에서 건너뛸 영향을 받는 요청 ID 집합입니다.
        """
        should_fail = not self.recompute_kv_load_failures

        # 비동기 KV 로드 처리(아직 캐시되지 않음, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs, invalid_block_ids, evict_blocks=False
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # 동기화 로드 처리(캐시될 수 있음, 제거를 위해 블록 수집)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # 캐시에서 유효하지 않은 블록 및 다운스트림 종속 블록 제거
        # 재계산 정책을 사용하지 않는 경우에만(블록은 다시 계산됨
        # 이를 공유하는 다른 요청에서 재사용됨)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # 로드가 완료되면 재시도하도록 KV 로드 실패가 포함된 비동기 요청을 표시
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # update_from_output에서 건너뛸 동기화 영향을 받은 ID 반환
        return sync_failed_req_ids
