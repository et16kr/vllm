# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.metrics.stats import SchedulerStats
    from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager


class PauseState(enum.IntEnum):
    """스케줄러 일시 중지 상태입니다.

    - UNPAUSED: 정상 작업
    - PAUSE_NEW: 새 요청이 예약되지 않으며 이미 실행 중인 상태인
                 요청이 예약됩니다.
    - PAUSE_ALL: 예약된 요청이 없습니다
    """

    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2


class SchedulerInterface(ABC):
    @abstractmethod
    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def schedule(self) -> "SchedulerOutput":
        """이 예약 단계에서 처리할 요청을 예약합니다.

        예약 결정은 반복 수준에서 이루어집니다. 각 스케줄링
        단계는 모델의 단일 순방향 전달에 해당합니다. 따라서 이
        메서드는 엔진의 바쁜 루프에 의해 반복적으로 호출됩니다.

        기본적으로 스케줄러는 이
        스케줄링 단계에서 각 요청에 대해 처리할 토큰 수를 지정하는 {req_id: num_tokens}
         사전을 생성합니다. 예를 들어, num_tokens는 새 요청에 대한 프롬프트 토큰 수
        만큼 클 수 있고, 다음 요청에 대해서는 1일 수 있습니다.
        새로운 토큰을 하나씩 자동 회귀적으로 생성합니다. 그렇지 않은 경우
        청크 미리 채우기, 접두사 캐싱,
        추측 디코딩 등의 경우 그 사이 어딘가에 있을 수 있습니다.

        또한 스케줄러는 각 요청
        또는 일괄 처리에 대한 유용한 데이터도 반환합니다. 모델 실행기는 
        모델에 대한 입력을 준비하는 데 이 정보를 사용합니다.

        반환:
            예약된
            요청에 대한 정보가 포함된 SchedulerOutput 객체.
        """
        raise NotImplementedError

    @abstractmethod
    def get_grammar_bitmask(
        self, scheduler_output: "SchedulerOutput"
    ) -> "GrammarOutput | None":
        raise NotImplementedError

    @abstractmethod
    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, "EngineCoreOutputs"]:
        """모델 실행기 출력을 기반으로 스케줄러 상태를 업데이트합니다.

        이 메서드는 모델 실행기가 예약된
        요청을 처리한 후에 호출됩니다. 모델 실행기 출력에는 생성된 토큰 ID, 다음 단계에 대한 초안
        토큰 ID 등이 포함됩니다. 스케줄러는 이 정보를 사용하여 
        상태를 업데이트하고, 완료된 요청을 확인하고, 각 요청에 대해 출력
        을 반환합니다.

        반환:
            클라이언트 인덱스의 사전은 해당 요청에서 발생하는 각 요청에 대한 
            출력을 포함하는 EngineCoreOutputs 객체에 대한 딕셔너리입니다. client.
        """
        raise NotImplementedError

    @abstractmethod
    def update_draft_token_ids(self, draft_token_ids: "DraftTokenIds") -> None:
        """필요한 경우 새로 생성된 초안 토큰 ID로 요청을 업데이트하고
        구조화된 출력 문법 검증을 적용합니다.

        인수:
            draft_token_ids: 각 요청에 대한 입력 초안 토큰 ID입니다.
        """
        raise NotImplementedError

    @abstractmethod
    def update_draft_token_ids_in_output(
        self, draft_token_ids: "DraftTokenIds", scheduler_output: "SchedulerOutput"
    ) -> None:
        """새로 생성된 초안 토큰 ID로 스케줄러 출력을 업데이트하고
        구조화된 출력 문법 검증을 적용합니다.

        인수:
            draft_token_ids: 각 요청에 대한 입력 초안 토큰 ID입니다.
            scheduler_output: 지정된 Scheduler_output을 업데이트합니다
                 해당 초안 토큰 ID.
        """
        raise NotImplementedError

    @abstractmethod
    def add_request(self, request: "Request") -> None:
        """스케줄러의 내부 대기열에 새 요청을 추가합니다.

        인수:
            request: 새 요청이 추가됩니다.
        """
        raise NotImplementedError

    @abstractmethod
    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: "RequestStatus",
    ) -> list[tuple[str, int]]:
        """스케줄러의 내부 대기열에서 요청을 완료합니다. 요청하는 경우
        대기열에 없으면 이 메서드는 해당 요청에 대해 아무 작업도 수행하지 않습니다.

        이 메서드는 두 가지 경우에 호출됩니다.
        1. 클라이언트가 요청을 중단한 경우.
        2. 프런트엔드 프로세스가 이후 요청의 중지 문자열을 감지한 경우
           생성된 토큰을 토큰화 해제합니다.

        인수:
            request_ids: 단일 또는 요청 ID 목록, 또는 모두 완료하려면 None입니다.
            done_status: 해당 요청의 완료 상태입니다.

        반환:
            중단된 요청에 대한 (req_id, client_index)의 튜플입니다. 않을 것이다
            이미 완료된 항목을 포함합니다.
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        """스케줄러의 내부 대기열에 있는 완료되지 않은 요청 수입니다."""
        raise NotImplementedError

    def has_unfinished_requests(self) -> bool:
        """스케줄러에 완료되지 않은 요청이 있는 경우 True를 반환합니다.
        내부 대기열."""
        return self.get_num_unfinished_requests() > 0

    @abstractmethod
    def has_finished_requests(self) -> bool:
        """삭제해야 하는 완료된 요청이 있는 경우 True를 반환합니다.
        참고: 이는 `not self.has_unfinished_requests()`와 다릅니다.

        스케줄러는 
        이전 단계에서 완료된 요청의 내부 목록을 유지합니다. 이 목록은 다음 단계에서 Schedule() 호출에서 반환되며,
        완료된 요청에 대해 캐시된 상태를 지우기 위해 다음 단계에서 모델 실행자에게 전송됩니다
        .

        이 메서드는 완료된 요청의 내부 목록이
        비어 있지 않은지 확인합니다. 이 정보는 DP 주의에 유용합니다.
        """
        raise NotImplementedError

    def has_requests(self) -> bool:
        """완료되지 않은 요청이 있거나 완료된 요청이 있는 경우 True를 반환합니다.
        SchedulerOutputs에 아직 반환되지 않았습니다."""
        return self.has_unfinished_requests() or self.has_finished_requests()

    @property
    @abstractmethod
    def pause_state(self) -> PauseState:
        """스케줄러의 현재 일시 중지 상태입니다."""
        raise NotImplementedError

    @abstractmethod
    def set_pause_state(self, pause_state: PauseState) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """KV 캐시에 대한 접두사 캐시를 재설정합니다.

        이는 특히 모델 가중치가 다음과 같은 경우에 필요합니다. live-updated.

        인수:
            reset_running_requests: True이면 실행 중인 모든 요청이 
                선점되고 대기 대기열로 이동됩니다. 그렇지 않은 경우 이 메서드는
                실행 중인 요청이 없는 경우에만 KV 접두사 캐시를 재설정합니다.
                KV 캐시를 사용합니다.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_encoder_cache(self) -> None:
        """인코더 캐시를 재설정하여 캐시된 모든 인코더 출력을 무효화합니다.

        이 메서드는 모델 가중치가 업데이트될 때 호출되어
        오래된 비전 임베딩이 재사용되지 않도록 해야 합니다.
        """
        raise NotImplementedError

    @abstractmethod
    def get_request_counts(self) -> tuple[int, int]:
        """반환 (num_running_reqs, num_waiting_reqs)."""
        raise NotImplementedError

    @abstractmethod
    def make_stats(self) -> "SchedulerStats | None":
        """로깅을 위한 SchedulerStats 객체를 만듭니다.

        SchedulerStats 객체는 모든 예약 단계에 대해 생성됩니다.
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """스케줄러를 종료합니다."""
        raise NotImplementedError

    def get_kv_connector(self) -> "KVConnectorBase_V1 | None":
        return None
