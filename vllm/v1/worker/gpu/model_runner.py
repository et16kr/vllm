# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
참고: 이 파일의 코딩 스타일 가이드:
이 모델 러너는 텍스트/멀티모달, 생성/임베딩, 공개/비공개를 포함한
모든 모델이 함께 사용합니다. 따라서 이 파일에는 모든 모델에 공통인
코드만 포함되어야 합니다. 모델별 동작은 해당 모델 전용 파일에
배치해야 합니다.

즉:
* 이 파일을 변경할 때는 매우 신중해야 합니다. 안정적으로 유지되어야 합니다.
* 새 줄을 추가할 때는 그보다 더 신중해야 합니다. 최소한으로 유지되어야 합니다.

공통 기능(예: 다양한 병렬화 모드)이라도 이 경로에 복잡도를 넣지 마세요.
기능이 덜 일반적일수록 더 숨겨져야 합니다. 기능별 로직을 여기에 직접
삽입하기보다, 다른 곳에 정의된 유틸리티 함수를 호출하는 방식을 우선하세요.
"""

import functools
import gc
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
    prepare_communication_buffer_for_model,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility
from vllm.v1.worker.gpu.async_utils import AsyncOutput, AsyncPoolingOutput
from vllm.v1.worker.gpu.attn_utils import (
    build_slot_mappings_by_layer,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.dp_utils import (
    get_cudagraph_and_dp_padding,
    make_num_tokens_across_dp,
)
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    get_num_sampled_and_rejected,
    post_update,
    post_update_pool,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.kv_connector import (
    NO_OP_KV_CONNECTOR,
    KVConnector,
    get_kv_connector,
)
from vllm.v1.worker.gpu.lora_utils import LoraState
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states import init_model_state
from vllm.v1.worker.gpu.pool.pooling_runner import PoolingRunner
from vllm.v1.worker.gpu.pp_utils import pp_broadcast, pp_receive
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    set_eagle3_aux_hidden_state_layers,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sample import rejection_sample
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

logger = init_logger(__name__)


class GPUModelRunner(LoRAModelRunnerMixin):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        self.device = device
        self.dtype = self.model_config.dtype
        self.kv_cache_dtype = self.dtype
        if self.cache_config.cache_dtype != "auto":
            # 양자화된 KV 캐시.
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype
            ]

        self.vocab_size = self.model_config.get_vocab_size()
        self.max_model_len = self.model_config.max_model_len
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs

        self.use_async_scheduling = self.scheduler_config.async_scheduling
        self.output_copy_stream = torch.cuda.Stream(self.device)
        self.output_copy_event = torch.cuda.Event()

        # 파이프라인 병렬화.
        self.pp_size = self.parallel_config.pipeline_parallel_size
        self.use_pp = self.pp_size > 1
        if self.use_pp:
            self.is_first_pp_rank = get_pp_group().is_first_rank
            self.is_last_pp_rank = get_pp_group().is_last_rank
        else:
            self.is_first_pp_rank = True
            self.is_last_pp_rank = True

        # 디코드 컨텍스트 병렬화.
        self.dcp_size = self.parallel_config.decode_context_parallel_size
        self.use_dcp = self.dcp_size > 1
        self.dcp_rank = get_dcp_group().rank_in_group if self.use_dcp else 0
        self.cp_interleave = self.parallel_config.cp_kv_cache_interleave_size

        # 멀티모달
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            self.model_config
        )
        self.encoder_cache = None
        if self.supports_mm_inputs and self.is_first_pp_rank:
            self.encoder_cache = EncoderCache()

        self.speculator = None
        self.num_speculative_steps = 0
        self.use_aux_hidden_state_outputs = False
        if self.speculative_config is not None:
            self.num_speculative_steps = self.speculative_config.num_speculative_tokens
            if self.is_last_pp_rank:
                self.speculator = init_speculator(self.vllm_config, self.device)

            if self.speculative_config.method == "eagle3":
                # EAGLE3는 타깃 모델 출력의 보조 hidden state가 필요할 수 있음.
                self.use_aux_hidden_state_outputs = True
                if self.pp_size > 1:
                    raise ValueError("EAGLE3 with pipeline parallel is not supported.")

        # 초안 토큰 전파 - spec-dec + 구조화 출력용.
        self.draft_tokens_handler = DraftTokensHandler(self.device)

        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )
        self.sampler = Sampler(
            max_num_reqs=self.max_num_reqs,
            vocab_size=self.vocab_size,
            device=self.device,
            req_states=self.req_states,
            logprobs_mode=self.model_config.logprobs_mode,
            num_speculative_tokens=self.num_speculative_steps + 1,
        )
        self.prompt_logprobs_worker = PromptLogprobsWorker(self.max_num_reqs)

        # CUDA 그래프.
        self.cudagraph_manager = CudaGraphManager(
            self.vllm_config,
            self.use_aux_hidden_state_outputs,
            self.device,
        )
        # 구조화 출력 워커.
        self.structured_outputs_worker = StructuredOutputsWorker(
            max_num_logits=self.max_num_reqs * (self.num_speculative_steps + 1),
            vocab_size=self.vocab_size,
            device=self.device,
        )
        # LoRA 관련 워커.
        self.lora_state = LoraState(max_num_reqs=self.max_num_reqs)
        # 설정된 경우 KV 커넥터.
        self.kv_connector: KVConnector = NO_OP_KV_CONNECTOR

        # 풀링 모델.
        self.is_pooling_model = self.model_config.runner_type == "pooling"
        self.pooling_runner: PoolingRunner | None = None

        # execute_model에서 이후 sample_tokens 호출로 상태를 전달하기 위한 용도.
        self.execute_model_state: tuple | None = None

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        self.req_states.max_model_len = max_model_len

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks: list[SupportedTask] = []
        if self.model_config.runner_type == "generate":
            tasks.append("generate")
        if self.pooling_runner is not None:
            tasks.extend(self.pooling_runner.get_supported_pooling_tasks())
        return tuple(tasks)

    def load_model(self, *args, **kwargs) -> None:
        time_before_load = time.perf_counter()
        with DeviceMemoryProfiler() as m:
            model_loader = get_model_loader(self.vllm_config.load_config)
            logger.info("Loading model from scratch...")

            self.model = model_loader.load_model(
                vllm_config=self.vllm_config,
                model_config=self.vllm_config.model_config,
            )
            if self.lora_config:
                self.model = self.load_lora_model(
                    self.model, self.vllm_config, self.device
                )

            if self.use_aux_hidden_state_outputs:
                assert self.speculative_config is not None
                set_eagle3_aux_hidden_state_layers(self.model, self.speculative_config)
            if self.speculator is not None:
                self.speculator.load_model(self.model)
        time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory
        logger.info(
            "Model loading took %s GiB and %.6f seconds",
            format_gib(m.consumed_memory),
            time_after_load - time_before_load,
        )

        prepare_communication_buffer_for_model(self.model)
        if self.speculator is not None:
            prepare_communication_buffer_for_model(self.speculator)

        # 모델이 필요한 컴포넌트 초기화.
        self.model_state = init_model_state(
            self.vllm_config, self.model, self.encoder_cache, self.device
        )
        if self.is_pooling_model:
            self.pooling_runner = PoolingRunner(self.model)

    def get_model(self) -> nn.Module:
        return self.model

    @functools.cached_property
    def main_stream(self) -> torch.cuda.Stream:
        # 조회 오버헤드를 피하기 위해 기본 CUDA 스트림을 캐시.
        return torch.cuda.current_stream(self.device)

    def get_kv_cache_spec(self):
        return get_kv_cache_spec(self.vllm_config)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
        ]

        self.block_tables = BlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_model_len=self.max_model_len,
            device=self.device,
            cp_size=self.dcp_size,
            cp_rank=self.dcp_rank,
            cp_interleave=self.cp_interleave,
        )

        self.attn_backends, self.attn_groups = init_attn_backend(
            self.kv_cache_config, self.vllm_config, self.device
        )
        check_attention_cp_compatibility(self.vllm_config)
        if self.speculator is not None:
            # HACK(woosuk)
            self.speculator.set_attn(
                self.kv_cache_config,
                self.attn_groups,
                self.block_tables,
            )

        self.kv_caches: list[torch.Tensor] = []
        kv_caches_dict = init_kv_cache(
            self.kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.attn_backends,
            self.device,
        )
        self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)

    @torch.inference_mode()
    def _dummy_run(
        self, num_tokens: int, *args, skip_attn: bool = True, **kwargs
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # 더미 스케줄러 출력 생성.
        num_reqs = min(num_tokens, self.max_num_reqs)
        num_tokens_per_request = [num_tokens // num_reqs] * num_reqs
        num_tokens_per_request[-1] += num_tokens % num_reqs
        assert sum(num_tokens_per_request) == num_tokens
        num_scheduled_tokens = {
            f"_dummy_req_{i}": n for i, n in enumerate(num_tokens_per_request)
        }
        dummy_scheduler_output = SchedulerOutput.make_empty()
        dummy_scheduler_output.total_num_scheduled_tokens = num_tokens
        dummy_scheduler_output.num_scheduled_tokens = num_scheduled_tokens

        # 더미 실행에서는 KVConnector 사용을 비활성화.
        self.kv_connector.set_disabled(True)

        # 첫 번째가 아닌 PP rank를 위해 더미 intermediate_tensors 생성.
        intermediate_tensors = None
        if not self.is_first_pp_rank:
            intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=num_tokens,
                dtype=self.model_config.dtype,
                device=self.device,
            )

        # 모델 실행.
        self.execute_model(
            dummy_scheduler_output,
            intermediate_tensors=intermediate_tensors,
            dummy_run=True,
            skip_attn_for_dummy_run=skip_attn,
        )
        self.kv_connector.set_disabled(False)

        # 마지막이 아닌 PP rank는 샘플링용 출력을 생성하지 않음.
        if not self.is_last_pp_rank:
            return None, None

        assert self.execute_model_state is not None
        input_batch, _, _, _, hidden_states, _, _ = self.execute_model_state
        self.execute_model_state = None
        assert hidden_states is not None  # Last PP rank always has hidden_states
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        return hidden_states, sample_hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        num_reqs = hidden_states.shape[0]
        logits = self.model.compute_logits(hidden_states)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
        pos = torch.zeros(num_reqs, dtype=torch.int64, device=self.device)
        dummy_input_ids = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        expanded_local_pos = torch.zeros(
            num_reqs, dtype=torch.int32, device=self.device
        )
        # NOTE(woosuk): 초기 메모리 프로파일링 중에는 sampler가 top_k, top_p,
        # logprobs를 생략할 수 있어 실제 실행 가능 시점보다 GPU 메모리를 더 적게
        # 사용할 수 있음.
        self.sampler(
            logits,
            idx_mapping,
            idx_mapping_np,
            idx_mapping_np,
            pos,
            dummy_input_ids,
            expanded_local_pos,
        )

    @torch.inference_mode()
    def _dummy_pooler_run(self, hidden_states: torch.Tensor) -> None:
        assert self.pooling_runner is not None
        self.pooling_runner.dummy_pooler_run(hidden_states)

    @torch.inference_mode()
    def profile_run(self) -> None:
        hidden_states, sample_hidden_states = self._dummy_run(
            self.max_num_tokens, skip_attn=True
        )

        # 마지막 PP rank에서만 sampler/pooler 실행 (그 외 rank는 None 반환).
        if self.is_last_pp_rank:
            assert sample_hidden_states is not None
            if self.pooling_runner is None:
                self._dummy_sampler_run(sample_hidden_states)
            else:
                self._dummy_pooler_run(hidden_states)

            if self.speculator is not None:
                num_tokens_across_dp = make_num_tokens_across_dp(
                    self.parallel_config.data_parallel_size, self.max_num_tokens
                )
                self.speculator.run_model(
                    self.max_num_tokens,
                    attn_metadata=None,
                    slot_mappings=None,
                    num_tokens_across_dp=num_tokens_across_dp,
                )

        torch.cuda.synchronize()
        del hidden_states, sample_hidden_states
        gc.collect()

    def reset_mm_cache(self) -> None:
        if self.encoder_cache is not None:
            self.encoder_cache.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        if self.encoder_cache is not None:
            self.encoder_cache.reset_encoder_cache()

    def _get_num_input_tokens(self, num_scheduled_tokens: int) -> int:
        # SP는 아직 지원되지 않음.
        return num_scheduled_tokens

    @torch.inference_mode()
    def capture_model(self) -> int:
        if not self.cudagraph_manager.needs_capture():
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        # TODO (zhanqiu): PP에 대한 CUDA 그래프 지원.
        if self.use_pp:
            logger.warning_once(
                "Skipping CUDA graph capture because pipeline parallel is "
                "enabled. Pipeline parallel is currently eager-only.",
            )
            return 0

        start_time = time.perf_counter()
        gc.collect()
        torch.cuda.empty_cache()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        with self.maybe_setup_dummy_loras(self.lora_config):
            self.cudagraph_manager.capture(
                model=self.model,
                model_state=self.model_state,
                input_buffers=self.input_buffers,
                block_tables=self.block_tables,
                attn_groups=self.attn_groups,
                kv_cache_config=self.kv_cache_config,
                has_lora=self.lora_config is not None,
            )
            if self.speculator is not None:
                self.speculator.capture_model()

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # 보통 5~20초가 소요됨.
        logger.info(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
        )
        return cuda_graph_size

    def warmup_for_prefill(self) -> None:
        # FlashInfer에서는 JIT 컴파일을 유도하기 위해 더미 prefill 실행이 필요함.
        if all("FLASHINFER" in b.get_name() for b in self.attn_backends.values()):
            self._dummy_run(self.max_num_tokens, skip_attn=False)
            torch.cuda.synchronize()

    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        finished_req_ids = scheduler_output.finished_req_ids
        preempted_req_ids = scheduler_output.preempted_req_ids
        if preempted_req_ids:
            finished_req_ids = finished_req_ids.union(preempted_req_ids)
        for req_id in finished_req_ids:
            self.req_states.remove_request(req_id)
            if self.encoder_cache is not None:
                self.encoder_cache.remove_request(req_id)
            self.prompt_logprobs_worker.remove_request(req_id)
            self.lora_state.remove_request(req_id)

    def free_states(self, scheduler_output: SchedulerOutput) -> None:
        if self.encoder_cache is not None:
            for mm_hash in scheduler_output.free_encoder_mm_hashes:
                self.encoder_cache.free_encoder_cache(mm_hash)

    def add_requests(self, scheduler_output: SchedulerOutput) -> None:
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.prompt_token_ids is not None
            assert new_req_data.prefill_token_ids is not None
            req_id = new_req_data.req_id
            prompt_len = len(new_req_data.prompt_token_ids)
            self.req_states.add_request(
                req_id=req_id,
                prompt_len=prompt_len,
                all_token_ids=new_req_data.prefill_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
            )
            req_index = self.req_states.req_id_to_index[req_id]

            if self.encoder_cache is not None:
                self.encoder_cache.add_request(req_id, new_req_data.mm_features)

            self.model_state.add_request(req_index, new_req_data)
            self.block_tables.append_block_ids(
                req_index, new_req_data.block_ids, overwrite=True
            )
            self.lora_state.add_request(req_id, req_index, new_req_data.lora_request)

            if new_req_data.sampling_params is not None:
                self.sampler.add_request(
                    req_index, prompt_len, new_req_data.sampling_params
                )
                self.prompt_logprobs_worker.add_request(
                    req_id, req_index, new_req_data.sampling_params
                )

        if scheduler_output.scheduled_new_reqs:
            self.req_states.apply_staged_writes()
            self.sampler.apply_staged_writes()
            self.model_state.apply_staged_writes()

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        # 기존 요청에 새 블록 추가.
        reqs = scheduler_output.scheduled_cached_reqs
        for req_new_block_ids, req_id in zip(reqs.new_block_ids, reqs.req_ids):
            if req_new_block_ids is not None:
                req_index = self.req_states.req_id_to_index[req_id]
                self.block_tables.append_block_ids(
                    req_index, req_new_block_ids, overwrite=False
                )

    def prepare_inputs(
        self, scheduler_output: SchedulerOutput, num_tokens_after_padding: int
    ) -> InputBatch:
        num_tokens = scheduler_output.total_num_scheduled_tokens
        assert num_tokens > 0
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)

        # 먼저 decode, 그다음 prefill.
        # batch_idx -> req_id
        req_ids = sorted(num_tokens_per_req, key=num_tokens_per_req.get)  # type: ignore[arg-type]
        numtoks_iter = map(num_tokens_per_req.get, req_ids)
        num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)

        idx_mapping_iter = map(self.req_states.req_id_to_index.get, req_ids)
        idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.int32, count=num_reqs)
        idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        # 각 요청의 초안 토큰 수 계산.
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        if not draft_tokens:
            # 초안 토큰이 스케줄되지 않음 (일반적인 경우).
            total_num_draft_tokens = 0
            total_num_logits = num_reqs
            cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(
                num_reqs + 1, device=self.device, dtype=torch.int32
            )
            expanded_idx_mapping = idx_mapping
            expanded_local_pos = torch.zeros(
                num_reqs, dtype=torch.int32, device=self.device
            )
        else:
            num_draft_tokens = np.array(
                [len(draft_tokens.get(req_id, ())) for req_id in req_ids],
                dtype=np.int32,
            )
            total_num_draft_tokens = int(num_draft_tokens.sum())
            total_num_logits = num_reqs + total_num_draft_tokens

            num_logits = num_draft_tokens + 1
            cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
            cu_num_logits_np[0] = 0
            np.cumsum(num_logits, out=cu_num_logits_np[1:])
            cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

            max_expand_len = self.num_speculative_steps + 1
            expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                idx_mapping, total_num_logits, cu_num_logits, max_expand_len
            )

        # query_start_loc 계산.
        query_start_loc_np = np.empty(self.max_num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
        # full CUDA 그래프 모드를 위한 패딩.
        # FA3 같은 일부 attention backend는 query_start_loc가 비감소여야 함.
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)
        query_start_loc_np = query_start_loc_np[: num_reqs + 1]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]

        # prefill 토큰이 있으면 준비.
        if self.req_states.any_prefills(idx_mapping_np):
            prepare_prefill_inputs(
                self.input_buffers.input_ids,
                self.req_states.next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
            )

        # positions와 seq_lens 준비.
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs]

        dcp_local_seq_lens = None
        if self.use_dcp:
            # dcp 로컬 seq_lens 준비.
            prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                self.input_buffers.seq_lens,
                num_reqs,
                self.dcp_size,
                self.dcp_rank,
                self.cp_interleave,
            )
            dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens[:num_reqs]

        # 일부 입력 토큰 ID는 마지막 샘플링 토큰과 초안 토큰에서 직접 읽음.
        # 또한 토큰 샘플링에 사용할 logits 인덱스를 계산.
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
        )

        return InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            dcp_local_seq_lens=dcp_local_seq_lens,
            input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
            positions=self.input_buffers.positions[:num_tokens_after_padding],
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
        )

    def prepare_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        # 블록 테이블: num_kv_cache_groups x [num_reqs, max_num_blocks]
        block_tables = self.block_tables.gather_block_tables(input_batch.idx_mapping)
        # 슬롯 매핑 계산: [num_kv_cache_groups, num_tokens]
        slot_mappings = self.block_tables.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.positions,
        )
        return block_tables, slot_mappings

    def prepare_dummy_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        block_tables = self.block_tables.get_dummy_block_tables(input_batch.num_reqs)
        slot_mappings = self.block_tables.get_dummy_slot_mappings(
            input_batch.num_tokens
        )
        return block_tables, slot_mappings

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        sample_pos = input_batch.positions[input_batch.logits_indices]
        input_ids = input_batch.input_ids[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        if grammar_output is not None:
            # logits에 grammar 비트마스크를 in-place로 적용.
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )

        # 토큰 샘플링 및 (필요 시) logprobs 계산.
        sampler_output = self.sampler(
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            input_batch.cu_num_logits_np,
            sample_pos,
            input_ids,
            input_batch.expanded_local_pos,
        )

        if input_batch.num_draft_tokens == 0:
            # 초안 토큰 없음 (일반적인 경우).
            num_sampled = torch.ones(
                input_batch.num_reqs, dtype=torch.int32, device=self.device
            )
        else:
            # spec decoding용 rejection sampling.
            sampled_tokens, num_sampled = rejection_sample(
                sampler_output.sampled_token_ids,
                input_ids,
                input_batch.cu_num_logits,
                self.num_speculative_steps,
            )
            sampler_output.sampled_token_ids = sampled_tokens

        # 샘플링/거절된 토큰 수 계산.
        # chunked prefill에서는 num_sampled와 num_rejected가 모두 0.
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len.gpu,
        )
        return sampler_output, num_sampled, num_rejected

    def postprocess(
        self,
        input_batch: InputBatch,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        # 계산된 토큰 수 업데이트.
        post_update(
            input_batch.idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.last_sampled_tokens,
            self.sampler.penalties_state.output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
        )

        # 계산된 prefill 토큰 수 업데이트.
        idx_mapping_np = input_batch.idx_mapping_np
        computed_prefill = self.req_states.num_computed_prefill_tokens
        computed_prefill[idx_mapping_np] += input_batch.num_scheduled_tokens
        np.minimum(
            computed_prefill, self.req_states.prefill_len.np, out=computed_prefill
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        if not dummy_run:
            # 요청 상태 업데이트.
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self.block_tables.apply_staged_writes()
            if scheduler_output.total_num_scheduled_tokens == 0:
                # 모델 실행 불필요.
                empty_output = self.kv_connector.no_forward(scheduler_output)
                return empty_output

        # 로컬 cudagraph 모드와 크기 계산.
        local_cudagraph_mode, local_cudagraph_size = (
            self.cudagraph_manager.get_cudagraph_runtime_mode(
                num_reqs=len(scheduler_output.num_scheduled_tokens),
                num_tokens=scheduler_output.total_num_scheduled_tokens,
                max_query_len=max(scheduler_output.num_scheduled_tokens.values()),
            )
        )

        # DP 동기화: num_tokens + cudagraph_size + cudagraph_mode
        num_tokens_after_padding, num_tokens_across_dp, synced_cudagraph_mode = (
            get_cudagraph_and_dp_padding(
                scheduler_output.total_num_scheduled_tokens,
                local_cudagraph_size,
                local_cudagraph_mode.value,
                self.parallel_config.data_parallel_size,
                self.parallel_config.data_parallel_rank,
            )
        )
        cudagraph_runtime_mode = CUDAGraphMode(synced_cudagraph_mode)
        if num_tokens_after_padding == 0:
            # 모든 DP rank에서 실행할 토큰이 0개.
            empty_output = self.kv_connector.no_forward(scheduler_output)
            return empty_output

        if not dummy_run:
            # 일반적인 경우.
            # 모든 입력을 준비해 입력 버퍼로 복사.
            input_batch = self.prepare_inputs(
                scheduler_output, num_tokens_after_padding
            )
            block_tables, slot_mappings = self.prepare_attn(input_batch)

            if self.lora_config:
                # LoRA 어댑터 활성화.
                lora_inputs = self.lora_state.make_lora_inputs(
                    input_batch.req_ids,
                    input_batch.idx_mapping_np,
                    input_batch.num_scheduled_tokens,
                )
                self._set_active_loras(*lora_inputs)
        else:
            # 실제 실행 토큰 없음. DP 또는 메모리 프로파일링용 더미 실행.
            num_reqs = min(num_tokens_after_padding, self.max_num_reqs)
            input_batch = InputBatch.make_dummy(
                num_reqs=num_reqs,
                num_tokens=num_tokens_after_padding,
                input_buffers=self.input_buffers,
                device=self.device,
            )
            if not skip_attn_for_dummy_run:
                block_tables, slot_mappings = self.prepare_dummy_attn(input_batch)
            else:
                block_tables = None
                slot_mappings = None
            # FIXME(woosuk): LoRA 워밍업 수정.

        attn_metadata = None
        slot_mappings_by_layer = None
        if not (dummy_run and skip_attn_for_dummy_run):
            assert slot_mappings is not None
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_mappings, self.kv_cache_config
            )
            assert block_tables is not None
            attn_metadata = self.model_state.prepare_attn(
                input_batch,
                block_tables,
                slot_mappings,
                self.attn_groups,
                self.kv_cache_config,
            )

        inputs_embeds = None
        if self.supports_mm_inputs and self.is_first_pp_rank and not dummy_run:
            # (필요 시) MM 인코더를 실행하고 멀티모달 임베딩을 가져옴.
            # 멀티모달 임베딩은 첫 번째 PP rank에서만 준비.
            inputs_embeds = self.model_state.get_mm_embeddings(
                scheduler_output.scheduled_encoder_inputs,
                input_batch,
                self.req_states,
            )

        model_inputs = {
            "input_ids": input_batch.input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": inputs_embeds,
            # NOTE: `prepare_inputs`가 반환한 값이 위 기본값을 덮어씀.
            **self.model_state.prepare_inputs(input_batch, self.req_states),
        }
        if not self.is_first_pp_rank:
            # 첫 번째가 아닌 PP rank에 맞게 갱신.
            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = None
            model_inputs["intermediate_tensors"] = intermediate_tensors

        # 모델 실행.
        if cudagraph_runtime_mode == CUDAGraphMode.FULL:
            # FULL 모드에서는 명시적 cudagraph replay 사용.
            # NOTE(woosuk): 입력 텐서는 이미 CUDA 그래프 입력 버퍼로 복사되어
            # 있으므로 여기서 별도로 전달할 필요가 없음.
            self.kv_connector.pre_forward(scheduler_output)
            model_output = self.cudagraph_manager.run_fullgraph(
                input_batch.num_tokens_after_padding
            )
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
            else:
                hidden_states = model_output
                aux_hidden_states = None
        else:
            # piecewise/eager 모드에서는 model()을 직접 호출.
            batch_descriptor = BatchDescriptor(
                num_tokens=input_batch.num_tokens_after_padding,
                has_lora=self.lora_config is not None,
            )

            with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=input_batch.num_tokens_after_padding,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=batch_descriptor,
                slot_mapping=slot_mappings_by_layer,
            ):
                self.kv_connector.pre_forward(scheduler_output)
                model_output = self.model(**model_inputs)
                if self.use_aux_hidden_state_outputs:
                    hidden_states, aux_hidden_states = model_output
                else:
                    hidden_states = model_output
                    aux_hidden_states = None

        kv_connector_output = self.kv_connector.post_forward(scheduler_output)
        self.execute_model_state = (
            input_batch,
            model_inputs,
            attn_metadata,
            slot_mappings_by_layer,
            hidden_states,
            aux_hidden_states,
            kv_connector_output,
        )

        if not self.is_last_pp_rank:
            # 마지막이 아닌 PP rank: 전송용 IntermediateTensors 반환.
            assert isinstance(hidden_states, IntermediateTensors)
            hidden_states.kv_connector_output = kv_connector_output
            return hidden_states
        # 마지막 rank(또는 PP 미사용): hidden_states는 샘플링용 텐서.
        assert isinstance(hidden_states, torch.Tensor)
        return None

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> AsyncOutput | ModelRunnerOutput | None:
        if self.execute_model_state is None:
            # 직전 execute_model 호출이 실패한 경우.
            return None
        (
            input_batch,
            model_inputs,
            attn_metadata,
            slot_mappings_by_layer,
            hidden_states,
            aux_hidden_states,
            kv_connector_output,
        ) = self.execute_model_state
        self.execute_model_state = None

        if not self.is_last_pp_rank:
            # 마지막이 아닌 PP rank: 이 rank는 최종 hidden state 대신
            # IntermediateTensors를 생성하므로 hidden_states가 None임.
            # 마지막 rank에서 브로드캐스트된 샘플링 토큰을 받아 로컬 상태 업데이트.
            sampled, num_sampled, num_rejected = pp_receive(
                input_batch.num_reqs, max_sample_len=self.num_speculative_steps + 1
            )
            self.postprocess(input_batch, sampled, num_sampled, num_rejected)
            return None

        # 마지막 rank: 토큰 샘플링.
        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )

        if self.use_pp:
            # 마지막이 아닌 PP rank로 브로드캐스트 (spec decode 멀티 토큰 처리).
            pp_broadcast(sampler_output.sampled_token_ids, num_sampled, num_rejected)

        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
            self.model.compute_logits,
            hidden_states,
            input_batch,
            self.req_states.all_token_ids.gpu,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.prompt_len.np,
            self.req_states.prefill_len.np,
            self.req_states.num_computed_prefill_tokens,
        )

        # 모델 러너 출력 준비.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            # NOTE(woosuk): 이 모델 러너에서는 req_id_to_index를 사용하지 않음.
            # 기존 모델 러너/스케줄러와의 호환성만을 위해 유지.
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,  # type: ignore
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
            kv_connector_output=kv_connector_output,
        )
        async_output = AsyncOutput(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=num_sampled,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
            copy_event=self.output_copy_event,
        )

        # 결과를 후처리하고 요청 상태를 업데이트.
        # NOTE: 의도적으로 AsyncOutput 생성 이후에 수행하여
        # postprocess 호출 전 `copy_event`가 기록되도록 보장함.
        # 이 순서는 비동기 D2H 복사가 postprocess 완료를 기다릴 필요가 없어서
        # 지연 시간을 약간 줄일 수 있음.
        self.postprocess(
            input_batch, sampler_output.sampled_token_ids, num_sampled, num_rejected
        )
        if self.speculator is not None:
            draft_tokens = self.speculator.propose(
                input_batch,
                attn_metadata,
                slot_mappings_by_layer,
                hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                self.req_states.last_sampled_tokens,
                self.req_states.next_prefill_tokens,
                self.sampler.sampling_states.temperature.gpu,
                self.sampler.sampling_states.seeds.gpu,
            )
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
            self.draft_tokens_handler.set_draft_tokens(input_batch, draft_tokens)

        if self.use_async_scheduling:
            return async_output
        return async_output.get_output()

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.draft_tokens_handler.get_draft_tokens()

    @torch.inference_mode()
    def pool(self) -> AsyncPoolingOutput | ModelRunnerOutput | None:
        if self.execute_model_state is None:
            # 직전 execute_model 호출이 실패한 경우.
            return None

        input_batch, _, _, _, hidden_states, _, kv_connector_output = (
            self.execute_model_state
        )
        self.execute_model_state = None

        if not self.is_last_pp_rank:
            self.postprocess_pool(input_batch)
            return None

        assert self.pooling_runner is not None
        pooler_output, is_valid = self.pooling_runner.pool(
            hidden_states, input_batch, self.req_states
        )
        self.postprocess_pool(input_batch)

        # 모델 러너 출력 생성.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            kv_connector_output=kv_connector_output,
        )
        async_output = AsyncPoolingOutput(
            model_runner_output=model_runner_output,
            pooler_output=pooler_output,
            is_valid=is_valid,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
            copy_event=self.output_copy_event,
        )
        if self.use_async_scheduling:
            return async_output
        return async_output.get_output()

    def postprocess_pool(self, input_batch: InputBatch) -> None:
        # 계산된 토큰 수 업데이트.
        post_update_pool(
            input_batch.idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            input_batch.query_start_loc,
        )

        # 계산된 prefill 토큰 수 업데이트.
        idx_mapping_np = input_batch.idx_mapping_np
        computed_prefill = self.req_states.num_computed_prefill_tokens
        computed_prefill[idx_mapping_np] += input_batch.num_scheduled_tokens
        np.minimum(
            computed_prefill, self.req_states.prefill_len.np, out=computed_prefill
        )
