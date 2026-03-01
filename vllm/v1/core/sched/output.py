# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request
else:
    ECConnectorMetadata = object
    KVConnectorMetadata = object
    LoRARequest = object
    MultiModalFeatureSpec = object
    PoolingParams = object
    SamplingParams = object
    Request = object


@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: "torch.Tensor | None" = None

    # v2 Model Runner에서만 사용된다.
    prefill_token_ids: list[int] | None = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "NewRequestData":
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            prefill_token_ids=prefill_token_ids,
        )

    def __repr__(self) -> str:
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"prefill_token_ids={self.prefill_token_ids},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )

    # 프롬프트 관련 민감 데이터를 숨긴 __repr__ 버전.
    def anon_repr(self) -> str:
        prompt_token_ids_len = (
            len(self.prompt_token_ids) if self.prompt_token_ids is not None else None
        )
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        prefill_token_ids_len = (
            len(self.prefill_token_ids) if self.prefill_token_ids is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={prompt_token_ids_len},"
            f"prefill_token_ids_len={prefill_token_ids_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )


@dataclass
class CachedRequestData:
    req_ids: list[str]
    # resumed_req_ids에 없는 요청은 기존 블록 뒤에 new_block_ids를 append한다.
    # resumed_req_ids에 있는 요청은 new_block_ids를 전체 블록 테이블로 사용한다.
    resumed_req_ids: set[str]
    # NOTE(woosuk): new_token_ids는 pipeline parallelism에서만 사용한다.
    # PP를 쓰지 않으면 빈 리스트다.
    new_token_ids: list[list[int]]
    # 직전 스텝에 스케줄되지 않은 요청만 connector로 전체 토큰 ID를 보낸다.
    # 이전 스텝에 이미 스케줄된 요청은 제외한다.
    all_token_ids: dict[str, list[int]]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]

    # 토큰 ID를 난독화한 dataclass 표현.
    def anon_repr(self) -> str:
        new_token_ids_lens = [len(toks) for toks in self.new_token_ids]
        all_token_ids_lens = {
            req_id: len(toks) for req_id, toks in self.all_token_ids.items()
        }
        return (
            f"CachedRequestData("
            f"req_ids={self.req_ids},"
            f"resumed_req_ids={self.resumed_req_ids},"
            f"new_token_ids_lens={new_token_ids_lens},"
            f"all_token_ids_lens={all_token_ids_lens},"
            f"new_block_ids={self.new_block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"num_output_tokens={self.num_output_tokens}"
            f")"
        )

    def __repr__(self) -> str:
        return self.anon_repr()

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """req_id -> num_output_tokens 매핑을 O(1) 조회용으로 캐시한다.

        CachedRequestData 인스턴스는 스케줄링 반복마다 새로 생성되고,
        반복 중에 변경되지 않으므로 캐시해도 안전하다.
        """
        return dict(zip(self.req_ids, self.num_output_tokens))

    def is_context_phase(self, req_id: str) -> bool:
        num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
        return num_output_tokens is not None and num_output_tokens == 0

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


@dataclass
class SchedulerOutput:
    # 처음 스케줄되는 요청 목록.
    # 요청 데이터는 워커 프로세스별로 캐시하므로
    # 매 스케줄링 스텝마다 다시 보낼 필요가 없다.
    scheduled_new_reqs: list[NewRequestData]
    # 이전에 스케줄된 적이 있는 요청 목록.
    # 해당 요청 데이터는 워커 측에 이미 캐시되어 있으므로
    # 통신량을 줄이기 위해 diff만 전송한다.
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens 매핑
    # 각 요청에 예약된 토큰 수.
    num_scheduled_tokens: dict[str, int]
    # 모든 요청에 스케줄된 토큰 총합.
    # sum(num_scheduled_tokens.values())와 동일.
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids 매핑
    # spec decode 토큰이 없는 요청은 딕셔너리에 포함되지 않는다.
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> 처리해야 할 encoder 입력 인덱스.
    # 예: [0, 1]이면 현재 스텝에서 해당 요청의 0, 1번 입력(예: 이미지)을
    # encoder가 처리해야 함을 의미한다.
    scheduled_encoder_inputs: dict[str, list[int]]
    # 각 KV 캐시 그룹 기준 공통 접두사 블록 수.
    # cascade attention에 활용될 수 있다.
    num_common_prefix_blocks: list[int]

    # 이전 스텝 이후 완료된 요청 ID.
    # 워커가 해당 요청의 캐시 상태를 해제할 때 사용된다.
    finished_req_ids: set[str]
    # encoder 출력과 연관된 mm_hash 목록.
    # encoder cache 해제에 사용된다.
    free_encoder_mm_hashes: list[str]

    # 이번 스텝에서 선점된 요청 ID.
    # v2 Model Runner에서만 사용된다.
    preempted_req_ids: set[str] | None = None

    # 스케줄된 요청 중 structured output 사용 여부.
    # 비동기 스케줄링일 때만 설정된다.
    has_structured_output_requests: bool = False

    # 문법 비트마스크 계산에 필요한 출력 토큰이 모두 준비됐는지 여부.
    pending_structured_output_tokens: bool = False

    # speculative decoding 수락률 계산 보정용.
    num_invalid_spec_tokens: dict[str, int] | None = None

    # KV 캐시 커넥터 메타데이터.
    kv_connector_metadata: KVConnectorMetadata | None = None

    # EC 캐시 커넥터 메타데이터
    ec_connector_metadata: ECConnectorMetadata | None = None

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )


@dataclass
class GrammarOutput:
    # 구조화된 출력 요청의 ID.
    structured_output_request_ids: list[str]
    # 비트마스크 행 순서는 structured_output_request_ids와 동일하다.
    grammar_bitmask: "npt.NDArray[np.int32]"
