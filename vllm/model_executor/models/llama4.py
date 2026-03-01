# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
# All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HuggingFace 가중치와 호환되는 추론 전용 LLaMA 모델."""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import Llama4TextConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import (
    Attention,
    ChunkedLocalAttention,
)
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform
from vllm.utils.torch_utils import is_torch_equal_or_newer

from .llama import LlamaForCausalLM, LlamaMLP, LlamaModel
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    fast_topk,
    is_pp_missing_parameter,
)

logger = init_logger(__name__)


class Llama4MoE(nn.Module):
    @staticmethod
    def custom_routing_function(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_scores, router_indices = fast_topk(gating_output, topk, dim=-1)
        # 관례적으로 router score는 float로 취급한다.
        router_scores = torch.sigmoid(router_scores.float())
        return (router_scores, router_indices.to(torch.int32))

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()
        self.top_k = config.num_experts_per_tok
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()

        intermediate_size_moe = config.intermediate_size
        self.router = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.router",
        )

        self.shared_expert = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size_moe,
            hidden_act="silu",
            quant_config=quant_config,
            bias=False,
            prefix=f"{prefix}.shared_expert",
            reduce_results=False,
            disable_tp=self.is_sequence_parallel,
        )

        # 로드 밸런싱 설정.
        eplb_config = parallel_config.eplb_config if parallel_config else None
        self.enable_eplb = parallel_config.enable_eplb if parallel_config else False
        self.n_redundant_experts = (
            eplb_config.num_redundant_experts if eplb_config else 0
        )

        self.n_routed_experts: int = config.num_local_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_shared_experts: int = 1
        self.n_local_experts: int = config.num_local_experts
        self.n_physical_experts = self.n_local_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_expert,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            custom_routing_function=Llama4MoE.custom_routing_function,
            intermediate_size=intermediate_size_moe,
            apply_router_weight_on_input=True,
            reduce_results=False,
            renormalize=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            is_sequence_parallel=self.is_sequence_parallel,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
        )

    def forward(self, hidden_states):
        num_tokens = hidden_states.shape[0]
        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        router_logits, _ = self.router(hidden_states)

        shared_out, routed_out = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        experts_out = routed_out + shared_out

        if self.is_sequence_parallel:
            experts_out = tensor_model_parallel_all_gather(experts_out, 0)
            experts_out = experts_out[:num_tokens]
        elif self.tp_size > 1:
            experts_out = self.experts.maybe_all_reduce_tensor_model_parallel(
                experts_out
            )

        return experts_out


class Llama4Attention(nn.Module):
    def __init__(
        self,
        config: Llama4TextConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        self.no_rope_layers = config.no_rope_layers
        self.nope = self.no_rope_layers[self.layer_idx] == 0
        self.use_qk_norm = config.use_qk_norm and not self.nope
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # KV head 수가 TP 크기보다 크므로,
            # KV head를 여러 tensor parallel GPU에 분할한다.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # KV head 수가 TP 크기보다 작으므로,
            # KV head를 여러 tensor parallel GPU에 복제한다.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn_temperature_tuning = self.nope and config.attn_temperature_tuning

        self.floor_scale = getattr(config, "floor_scale", 8192.0)
        self.attn_scale = getattr(config, "attn_scale", 0.1)
        self.max_position_embeddings = max_position_embeddings
        self.n_rep = self.num_heads // self.num_kv_heads
        self.qk_norm = (
            RMSNorm(
                hidden_size=self.head_dim,
                eps=config.rms_norm_eps,
                has_weight=False,
                dtype=torch.float32,
            )
            if self.use_qk_norm
            else None
        )
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        self.rotary_emb = (
            get_rope(
                self.head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=is_neox_style,
            )
            if not self.nope
            else None
        )

        use_chunked_local_attn = not self.nope and config.attention_chunk_size
        attn_cls = ChunkedLocalAttention if use_chunked_local_attn else Attention
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **(
                {"attention_chunk_size": config.attention_chunk_size}
                if use_chunked_local_attn
                else {}
            ),
        )

    def _get_attn_scale(self, positions: torch.Tensor) -> torch.Tensor:
        floor = torch.floor((positions + 1.0) / self.floor_scale)
        attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0

        return attn_scale.unsqueeze(-1)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

        if self.qk_norm is not None:
            # 정규화는 head_dim 차원에 적용한다. 나머지 차원은
            # custom rms_norm CUDA 커널을 지원하기 위해
            # 하나의 차원으로 펼친다.
            q = q.reshape(-1, self.head_dim)
            q = self.qk_norm(q.float()).reshape(-1, self.q_size).to(q.dtype)
            k = k.reshape(-1, self.head_dim)
            k = self.qk_norm(k.float()).reshape(-1, self.kv_size).to(k.dtype)

        # NoPE 레이어에는 temperature tuning(https://arxiv.org/abs/2501.19399)을
        # 적용한다. 추론 시 사용하는 temperature tuning 함수는
        # 짧은 컨텍스트에는 영향을 주지 않으면서
        # 매우 긴 컨텍스트에서 동작하도록 맞춤화되어 있다.
        # https://arxiv.org/abs/2501.19399
        #
        # Temperature tuning은 rotary/QK norm 이후, attention 이전에 적용한다.
        if self.attn_temperature_tuning and self.nope:
            attn_scale = self._get_attn_scale(positions)
            q = (q * attn_scale).to(q.dtype)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Llama4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: Llama4TextConfig | None = None,
    ) -> None:
        super().__init__()

        config = config or vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_idx = extract_layer_index(prefix)
        self.global_layer = config.no_rope_layers[self.layer_idx] == 0
        self.hidden_size = config.hidden_size
        max_position_embeddings = config.max_position_embeddings

        self.self_attn = Llama4Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=False,
            bias_o_proj=False,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        is_moe_layer = (
            config.interleave_moe_layer_step > 0
            and (self.layer_idx + 1) % config.interleave_moe_layer_step == 0
        )
        if is_moe_layer:
            self.feed_forward = Llama4MoE(
                vllm_config=vllm_config,
                prefix=f"{prefix}.feed_forward",
            )
        else:
            self.feed_forward = LlamaMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size_mlp,
                hidden_act="silu",
                quant_config=quant_config,
                bias=False,
                prefix=f"{prefix}.feed_forward",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 셀프 어텐션
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # 완전연결층
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Llama4Model(LlamaModel):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[Llama4DecoderLayer] = Llama4DecoderLayer,
    ):
        self.num_experts = vllm_config.model_config.hf_config.num_local_experts
        self.n_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

    def load_moe_expert_weights(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
        loaded_params: set[str],
        expert_params_mapping: list[tuple[str, str, int, str]],
        fused: bool = True,
    ) -> bool:
        """
        MoE 전문가 가중치를 로드한다.

        Args:
            name: 로드할 가중치 이름.
            loaded_weight: 로드할 가중치 텐서.
            params_dict: 모듈 파라미터 딕셔너리.
            loaded_params: 이미 로드된 파라미터 이름 집합.
            expert_params_mapping: 전문가 파라미터 매핑. 반드시
                SharedFusedMoE.make_expert_params_mapping()으로 생성되어야 한다.
            fused: 전문가 가중치가 단일 텐서로 fuse되어 있는지 여부.
                False이면 전문가별 개별 텐서여야 한다.
                fused=True인 경우 loaded_weight의 shape은 다음을 따른다:
                gate/up/down proj: [num_experts, hidden_in, hidden_out]
                router 등 기타: [hidden_out, hidden_in]
                fused=False인 경우 loaded_weight의 shape:
                [hidden_out, hidden_in]

        Returns:
            loaded_weight가 MoE 가중치이며 전문가 가중치 로딩에 성공하면 True,
            그렇지 않으면 False.
        """

        # MoE 전문가 가중치 로딩 성공 여부.
        expert_param_loaded = False

        # fused=True이면 로드된 가중치 레이아웃이
        # [num_experts, hidden_in, hidden_out]이므로, 마지막 두 차원을
        # transpose해 파라미터 기대 레이아웃에 맞춘다.
        if fused and loaded_weight.ndim == 3:
            loaded_weight = loaded_weight.transpose(-1, -2)

            # gate_proj와 up_proj가 단일 텐서로 fuse된 경우,
            # hidden_out 차원 기준으로 두 텐서 튜플로 분리한다.
            if "experts.gate_up_proj" in name:
                loaded_weight = loaded_weight.chunk(2, dim=-2)

        # 모든 전문가 파라미터를 순회하면서
        # 가중치 이름이 일치하면 로드한다.
        for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
            # 반복 간 원본 수정 방지를 위해 loaded_weight의 뷰를 사용한다.
            new_loaded_weight = loaded_weight

            # 전문가 가중치가 단일 텐서로 fuse된 경우, 기대 가중치 이름에서
            # 전문가 인덱스를 제거한다.
            if fused:
                # e_str과 proj_str 사이 문자열이 전문가 인덱스다.
                e_str, _, proj_str, _ = weight_name.split(".")
                weight_name = f"{e_str}.{proj_str}"
                param_name = f"{param_name}weight"

            # 현재 가중치가 MoE 가중치가 아니면 건너뛴다.
            if weight_name not in name:
                continue

            # 가중치 이름을 파라미터 이름으로 치환한다.
            full_param_name = name.replace(weight_name, param_name)

            # 현재 PP(pipeline parallel) rank에 존재하지 않는 파라미터면
            # 건너뛴다.
            if is_pp_missing_parameter(name, self):
                continue

            # 현재 가중치가 bias이고 해당 파라미터가 없으면 건너뛴다.
            if (
                name.endswith(".bias") or name.endswith("_bias")
            ) and name not in params_dict:
                continue

            param = params_dict[full_param_name]
            weight_loader = param.weight_loader

            if fused:
                # 파라미터가 w13 통합 형태라면 대응 가중치는 튜플이므로,
                # shard id("w1" 또는 "w3")에 따라 올바른 가중치를 선택한다.
                if "w13" in full_param_name:
                    assert shard_id in ["w1", "w3"]
                    shard_idx = 0 if shard_id == "w1" else 1
                    new_loaded_weight = new_loaded_weight[shard_idx]

                # EP(expert parallel)가 활성화된 경우, expert_id를 현재 EP rank의
                # 시작 전문가 인덱스로 갱신하고 해당 전문가 가중치만 추출한다.
                layer_idx = extract_layer_index(name)
                expert_map = self.layers[layer_idx].feed_forward.experts.expert_map
                if expert_map is not None:
                    local_expert_indices = (
                        (expert_map != -1)
                        .nonzero()
                        .flatten()
                        .to(new_loaded_weight.device)
                    )
                    # 구버전 PyTorch의 FP8 CPU 인덱싱 우회 처리:
                    # https://github.com/vllm-project/vllm/issues/32862
                    is_fp8_dtype = new_loaded_weight.dtype == (
                        current_platform.fp8_dtype()
                    ) or (
                        new_loaded_weight.dtype.is_floating_point
                        and new_loaded_weight.element_size() == 1
                    )
                    if (
                        new_loaded_weight.device.type == "cpu"
                        and is_fp8_dtype
                        and not is_torch_equal_or_newer("2.11.0")
                    ):
                        # PyTorch < 2.11은 CPU float8 인덱싱을 지원하지 않는다.
                        new_loaded_weight = new_loaded_weight.to(torch.float16)[
                            local_expert_indices
                        ].to(new_loaded_weight.dtype)
                    else:
                        new_loaded_weight = new_loaded_weight[local_expert_indices]
                    expert_id = local_expert_indices[0].item()
            else:
                # TODO: non-fused 가중치에 대한 EP 지원 추가
                pass

            # 대응 shard id와 expert id를 사용해 가중치를 모듈 파라미터에 로드한다.
            weight_loader(
                param,
                new_loaded_weight,
                full_param_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            loaded_params.add(full_param_name)
            expert_param_loaded = True

        return expert_param_loaded

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # 파라미터 이름을 shard 이름 및 shard id로 매핑한다.
        stacked_params_mapping = [
            # (파라미터 이름, shard 이름, shard id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        # 전문가 가중치가 단일 텐서로 fuse되어 있는지 나타낸다.
        fused_experts_params = False
        # 전문가 가중치가 단일 텐서로 fuse되지 않은 경우의
        # 전문가 파라미터 매핑.
        expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.num_experts,
            num_redundant_experts=self.n_redundant_experts,
        )
        # 전문가 가중치가 단일 텐서로 fuse된 경우의
        # 전문가 파라미터 매핑.
        expert_params_mapping_fused = SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_up_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="gate_up_proj",
            num_experts=1,
        )
        # 모듈의 모든 파라미터.
        params_dict = dict(self.named_parameters())
        # 로드가 완료된 모듈 파라미터 집합.
        loaded_params: set[str] = set()

        # 모든 가중치를 순회하며 모듈 파라미터에 로드한다.
        for name, loaded_weight in weights:
            # 이름에 전문가 인덱스 없이 "experts.gate_up_proj" 또는
            # "experts.down_proj"가 포함되면, 전문가 전체가 단일 텐서로
            # fuse된 가중치로 간주한다.
            if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                fused_experts_params = True
                expert_params_mapping = expert_params_mapping_fused

            # KV 캐시 양자화 스케일이 존재하고 현재 이름이 그중 하나에 해당하면
            # 해당 스케일을 로드한다.
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            # stacked_params_mapping을 순회하며 현재 가중치가 stacked 파라미터인지
            # 확인한다. 해당하면 대응 shard id로 로드한다.
            # MoE 가중치는 아래 else 블록에서 별도로 처리한다.
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # 현재 가중치가 stacked 파라미터가 아니거나 MoE 가중치면 건너뛴다.
                if weight_name not in name or "experts" in name:
                    continue

                # ModelOpt 체크포인트에서는 KV 캐시 scale을 제외한
                # self_attn weight/weight_scale 이름을 재매핑한다.
                if not (
                    name.endswith((".k_scale", ".v_scale")) and "self_attn" in name
                ):
                    name = name.replace(weight_name, param_name)

                # 현재 PP(pipeline parallel) rank에 해당 파라미터가 없으면 건너뛴다.
                if is_pp_missing_parameter(name, self):
                    continue

                # ModelOpt 체크포인트용 KV 캐시 scale 이름 재매핑.
                # TODO: ModelOpt에서 get_cache_scale()을 구현해
                #       KV 캐시 scale 이름 재매핑을 그쪽에서 처리해야 한다.
                if name.endswith("scale"):
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                # 대응 shard id로 가중치를 로드하고,
                # for 루프 및 else 블록 처리를 종료한다.
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)

                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)

                loaded_params.add(name)
                break

            # 일반(non-stacked) 가중치와 MoE 가중치를 처리한다.
            else:
                # 먼저 load_moe_expert_weights로 MoE 가중치 로드를 시도한다.
                # 성공하면 다음 가중치로 진행한다.
                if self.load_moe_expert_weights(
                    name,
                    loaded_weight,
                    params_dict,
                    loaded_params,
                    expert_params_mapping,
                    fused=fused_experts_params,
                ):
                    continue

                # 현재 PP(pipeline parallel) rank에 해당 파라미터가 없으면 건너뛴다.
                if is_pp_missing_parameter(name, self):
                    continue

                # 전문가별 패턴과 매칭되지 않는 평탄(flat) expert scale 파라미터를
                # 처리한다. 즉, 전문가 전체에 대해 scale 텐서 하나를 쓰는 경우다.
                scale_names = [
                    "w13_input_scale",
                    "w13_weight_scale",
                    "w2_input_scale",
                    "w2_weight_scale",
                ]
                if "experts." in name and any(
                    scale_name in name for scale_name in scale_names
                ):
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )

                    # weight loader가 특수 MoE 로딩을 지원하면
                    # 비용이 큰 런타임 리플렉션을 피하기 위해 이를 사용한다.
                    if getattr(weight_loader, "supports_moe_loading", False):
                        # 가중치 이름을 대응 shard id로 매핑한다.
                        shard_id = "w2" if "w2_" in name else "w1"

                        # weight scale이 3차원 FP8 block scale
                        # [num_experts, hidden_in, hidden_out]이면 transpose한다.
                        if (
                            name.endswith("weight_scale")
                            and loaded_weight.dtype == torch.float8_e4m3fn
                            and loaded_weight.ndim == 3
                        ):
                            loaded_weight = loaded_weight.transpose(-1, -2)

                        # 대응 shard id, expert id로 가중치를 모듈 파라미터에 로드한다.
                        weight_loader(
                            param, loaded_weight, name, shard_id=shard_id, expert_id=0
                        )

                    else:
                        # 일반 weight loader 사용
                        # (param.weight_loader와 default_weight_loader 모두 지원).
                        weight_loader(param, loaded_weight)

                    loaded_params.add(name)
                    continue

                # 일반(non-stacked, non-MoE) 가중치 처리.
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        # 최종적으로 로드된 파라미터 집합을 반환한다.
        return loaded_params


class Llama4ForCausalLM(LlamaForCausalLM, MixtureOfExperts):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # generation config를 기반으로 temperature tuning 설정을 갱신한다.
        gen_config = vllm_config.model_config.try_get_generation_config()
        gen_config.update(vllm_config.model_config.override_generation_config)
        # max_model_len > 32K이면 기본값으로 temperature tuning을 활성화한다.
        default_attn_temperature_tuning = vllm_config.model_config.max_model_len > 32768
        vllm_config.model_config.hf_config.attn_temperature_tuning = gen_config.get(
            "attn_temperature_tuning", default_attn_temperature_tuning
        )

        super().__init__(
            vllm_config=vllm_config, prefix=prefix, layer_type=Llama4DecoderLayer
        )
        # MoE 하이퍼파라미터 설정.
        self.set_moe_parameters()

    def set_moe_parameters(self):
        self.expert_weights = []

        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, Llama4DecoderLayer)
            if isinstance(layer.feed_forward, Llama4MoE):
                # 앞부분은 dense 레이어일 수 있으므로 마지막 MoE 레이어를 기준으로 사용한다.
                example_moe = layer.feed_forward
                self.moe_layers.append(layer.feed_forward.experts)

        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            logger.warning("No Llama4MoE layer found in model.layers.")
        else:
            self.num_moe_layers = len(self.moe_layers)
            self.num_expert_groups = 1
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            if isinstance(layer.feed_forward, Llama4MoE):
                moe = layer.feed_forward
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[Llama4DecoderLayer] = Llama4DecoderLayer,
    ):
        return Llama4Model(
            vllm_config=vllm_config, prefix=prefix, layer_type=layer_type
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        weights = [
            self.permute_qk_weight_for_rotary(name, loaded_weight)
            for name, loaded_weight in weights
        ]
        return loader.load_weights(weights)

    def permute_qk_weight_for_rotary(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> tuple[str, torch.Tensor]:
        modules = name.split(".")
        # rotary embedding에 맞게 Q/K 가중치와 대응 scale을 permute한다.
        # 이 경로는 modelopt 및 compressed-tensors 체크포인트에서 검증되었고,
        # per-tensor, per-group(예: GPTQ), per-channel 양자화 스킴을 지원한다.
        # 참고: per-block(예: DeepSeek 128x128) 양자화에서는 permute가 사실상 어렵다.
        # per-block 양자화라면 q/k_proj를 양자화하지 않는 방안을 고려한다.
        is_weight = modules[-1] in ("weight", "weight_packed")
        is_weight_scale = (
            modules[-1] == "weight_scale"
            and loaded_weight.numel() > 1  # per-tensor scale은 permute가 필요 없다.
        )
        is_k_proj = "wk" in modules or "k_proj" in modules
        is_q_proj = "wq" in modules or "q_proj" in modules

        if (is_weight or is_weight_scale) and (is_k_proj or is_q_proj):
            original_ndim = loaded_weight.ndim
            if original_ndim == 1:
                loaded_weight = loaded_weight.unsqueeze(-1)

            f_out, f_in = loaded_weight.shape
            n_heads = (
                self.config.num_key_value_heads
                if is_k_proj
                else self.config.num_attention_heads
            )
            loaded_weight = (
                loaded_weight.view(n_heads, f_out // n_heads // 2, 2, f_in)
                .transpose(1, 2)
                .reshape(f_out, f_in)
            )

            if original_ndim == 1:
                loaded_weight = loaded_weight.squeeze(-1)

        return name, loaded_weight
