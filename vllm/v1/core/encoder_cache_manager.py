# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import SchedulerConfig

logger = init_logger(__name__)


class EncoderCacheManager:
    """vLLM V1에서 다중 모달 모델에 대한 인코더 출력 캐싱을 관리합니다.

    EncoderCacheManager는 다중 모드 인코더 출력의 수명 주기를 처리합니다.
    (예: 요청 처리 중 이미지의 비전 임베딩) 그것
    인코더 출력을 다시 계산하지 않도록 메모리 인식 캐싱을 제공합니다.
    동일한 다중 모드 입력이 요청 처리의 여러 단계에 나타납니다.

    이 관리자는 다음과 같은 경우에 특히 중요합니다.
    - 이미지 인코더 출력이 있는 비전 언어 모델(예: LLaVA)
      캐시된
    - 인코더 계산 비용이 많이 들고
      캐시 가능

    캐시는 개별 다중 모드 입력 항목의 세분성에서 작동합니다.
    요청 내에서 세분화된 메모리 관리가 가능하고
    다중 모드 입력의 청크 처리.

    동일한 다중 모드 데이터의 임베딩을 공유하기 위해 캐시가 활성화되었습니다.
    서로 다른 요청 간의 항목(해시 값으로 식별됨)
    무료가 없는 할당 시간에 퇴거가 발생합니다.
    새로운 임베딩을 위한 공간.
    참조된 요청이 없는 가장 오래된 캐시된 임베딩이 먼저 제거됩니다.

    참고: EncoderCacheManager는 다중 모드 임베딩 수준에서 작동합니다.
    인코더 토큰(즉, 다중 모드 데이터를 나타내는 모든 토큰) 대신
    입력 순서에서). 이는 다중 모드 사이에 있는 모든 중단/텍스트 토큰을 의미합니다.
    캐시 크기 및 개수와 관련하여 임베딩은 고려되지 않습니다.
    무료 슬롯 수.

    인수:
        캐시_크기: 캐시의 크기를 제한하며, 개수로 측정됩니다.
                    입력 시퀀스의 인코더 임베딩.

    속성:
        캐시_크기: 인코더 임베딩의 총 캐시 용량입니다.
        num_free_slots: 인코더 임베딩에서 현재 사용 가능한 캐시 용량입니다.
        num_freeable_slots: 즉시 회수할 수 있는 용량
            참조가 0인 항목을 제거합니다(인코더 임베딩에서).
        캐시됨: mm_hash에서 현재 요청 ID 집합으로 매핑
            캐시된 항목을 참조하세요. 세트가 비어 있으면 항목이 존재하는 것입니다.
            그러나 어떤 요청에서도 참조되지 않으며 다음 사항에 적합합니다.
            교정.
        freeable: 항목을 나타내는 튜플 목록(mm_hash, num_encoder_embeds)
            현재 실행 중인 요청이 필요하지 않고 해제될 수 있는
            필요할 때 공간을 확보하십시오.
        freed: 이후 실제로 제거된 mm_hash 문자열 목록
            get_freed_mm_hashes()에 대한 마지막 호출입니다. 이 목록은 반환 시 지워집니다.
    """

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.num_freeable_slots = cache_size

        # mm_data의 mm_hash => mm_data를 참조하는 요청의 ID
        self.cached: dict[str, set[str]] = {}

        # mm_data의 mm_hash => mm_data의 num_encoder_embeds
        self.freeable: OrderedDict[str, int] = OrderedDict()
        self.freed: list[str] = []

    def reset(self) -> None:
        """인코더 캐시를 초기 상태로 재설정합니다.

        이렇게 하면 캐시된 인코더 출력이 모두 지워지고 용량 추적이 재설정됩니다.
        오래된 임베딩을 무효화하기 위해 모델 가중치가 업데이트될 때 호출됩니다.
        """
        self.cached.clear()
        self.freeable.clear()
        self.freed.clear()
        self.num_free_slots = self.cache_size
        self.num_freeable_slots = self.cache_size

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        """특정 멀티모달 입력에 대한 인코더 출력이 캐시되었는지 확인하세요.

        인코더 출력이 캐시된 경우 `cached`를 업데이트하여 요청 ID를 추가하세요.
        캐시된 인코더 출력을 참조하는 요청 ID 집합에 적용됩니다.
        이전에 어떤 요청에서도 인코더 출력을 참조하지 않은 경우
        이에 따라 'freeable' 및 'num_freeable_slots'를 업데이트하세요.

        인수:
            request: 멀티모달 입력이 포함된 요청입니다.
            input_id: 요청 내 다중 모달 입력의 인덱스

        보고:
            이 입력에 대한 인코더 출력이 이미 캐시된 경우 참입니다.
        """
        mm_hash = request.mm_features[input_id].identifier
        # 전혀 캐시되지 않음
        if mm_hash not in self.cached:
            return False

        # 캐시되었지만 현재 어떤 요청에서도 참조되지 않음
        if not self.cached[mm_hash]:
            num_encoder_embeds = self.freeable.pop(mm_hash)
            self.num_freeable_slots -= num_encoder_embeds

        self.cached[mm_hash].add(request.request_id)
        return True

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        """다중 모드 입력을 위한 충분한 캐시 공간이 있는지 확인하세요.
        있는 경우 True를 반환하고 EncoderCacheManager 상태를 업데이트합니다.

        `num_free_slots`에 여유 공간이 충분하지 않지만
        `num_freeable_slots`에 회수 가능한 공간이 충분하면 항목은
        까지 `freeable`에서 제거됩니다(`freed`에 mm_hash가 추가됨).
        사용 가능한 공간이 충분한 경우 이 메서드는 True를 반환합니다.
        오래된 항목이 먼저 제거됩니다.

        요청된 토큰 수가 둘 다 초과하는 경우에만 False를 반환합니다.
        무료 용량과 회수 가능 용량이 결합되었습니다.

        인수:
            요청: 다중 모드 입력이 포함된 요청입니다.
            input_id: 요청 내 다중 모달 입력의 인덱스입니다.
            인코더_compute_budget: 허용되는 인코더 임베딩 수
                이 메소드가 호출될 때 계산됩니다.
            num_embeds_to_schedule: 이미 예약된 인코더 임베딩 수
                이 메소드가 호출되면 캐시 공간이 할당됩니다.

        보고:
            이에 대한 인코더 출력을 보유할 만큼 충분한 용량이 있는 경우 참입니다.
            입력(아마도 `freeable` 항목을 회수한 후); 그렇지 않으면
            거짓.

        참고: 이 방법은 인코더에 물리적 메모리를 할당하지 않습니다.
        출력하지만 EncoderCacheManager의 상태만 표시됩니다.
        """
        num_embeds = request.get_num_encoder_embeds(input_id)

        # 컴퓨팅 예산이 충분하지 않습니다.
        if num_embeds > encoder_compute_budget:
            return False

        num_embeds += num_embeds_to_schedule

        # 충분한 여유 슬롯
        if num_embeds <= self.num_free_slots:
            return True

        # 회수 가능한 슬롯이 충분하지 않습니다.
        if num_embeds > self.num_freeable_slots:
            return False

        # 사용 가능한 슬롯은 충분하지 않지만 회수 가능한 슬롯은 충분합니다.
        # 참고: 여기서 제거가 발생하지만 실제 메모리는 해제되지 않습니다.
        # 모델 실행자가 스케줄러 출력을 통해 알림을 받을 때까지.
        while num_embeds > self.num_free_slots:
            mm_hash, num_free_embeds = self.freeable.popitem(last=False)
            del self.cached[mm_hash]
            self.freed.append(mm_hash)
            self.num_free_slots += num_free_embeds
        return True

    def allocate(self, request: Request, input_id: int) -> None:
        """다중 모드 입력의 인코더 출력을 위한 캐시 공간을 할당합니다.

        이는 인코더 출력을 저장하기 위한 캐시 공간을 예약합니다.
        지정된 다중 모드 입력. 실제 인코더 출력 저장은 다음에서 발생합니다.
        모델러너; 이 방법은 관리자의 장부를 업데이트합니다.

        메모:
            이 메서드는 can_allocate()가 동일한 입력에 대해 True를 반환했다고 가정합니다.
        """

        mm_hash = request.mm_features[input_id].identifier
        request_id = request.request_id
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()

        num_encoder_embeds = request.get_num_encoder_embeds(input_id)

        # 참고: 인코더 캐시에는 항상 인코더 입력을 위한 충분한 공간이 있어야 합니다.
        # can_allocate()에서 퇴거가 발생하기 때문에 예정된 것입니다.
        assert self.num_free_slots >= num_encoder_embeds
        assert self.num_freeable_slots >= num_encoder_embeds

        self.cached[mm_hash].add(request_id)
        self.num_free_slots -= num_encoder_embeds
        self.num_freeable_slots -= num_encoder_embeds

    def get_cached_input_ids(self, request: Request) -> set[int]:
        """요청에 대해 캐시된 모든 다중 모달 입력 ID를 가져옵니다.

        캐시 맵에 `mm_hash`가 존재하는 입력 ID 세트를 반환합니다.
        여기에는 현재 참조되지 않은 항목이 포함됩니다.
        '무료'에서); 그러한 항목에 대해 이 요청을 해제하는 것은
        작동하지 않습니다.
        """
        return {
            input_id
            for input_id in range(len(request.mm_features))
            if request.mm_features[input_id].identifier in self.cached
        }

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        """인코더 입력(`mm_data`)에 대한 요청 참조를 해제합니다.

        해당 `mm_hash`에 대한 참조 세트가 비어 있게 되면,
        항목은 `freeable`에 추가되고 `num_freeable_slots`는
        해당 입력에 대한 인코더 임베딩 수만큼 증가합니다.

        용량이 필요할 때까지 항목은 물리적으로 해제되지 않습니다(예:
        `할당 가능`).
        """
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        # mm_hash가 캐시에 없거나 req_id 세트가 비어 있습니다.
        if not self.cached.get(mm_hash, None):
            return
        self.cached[mm_hash].discard(req_id)
        if not self.cached[mm_hash]:
            num_encoder_embeds = request.get_num_encoder_embeds(input_id)
            self.freeable[mm_hash] = num_encoder_embeds
            self.num_freeable_slots += num_encoder_embeds

    def free(self, request: Request) -> None:
        """*요청*에 의해 보유된 모든 인코더 입력 캐시 참조를 해제합니다.

        캐시된 각 입력 ID에 대해 `free_encoder_input`이 호출됩니다.
        데이터는 future에 의해 제거가 실행될 때까지 메모리에 유지됩니다.
        'can_allocate'에 의해 호출된 할당을 시도합니다.

        일반적으로 요청이 완료, 취소 또는 중단될 때 호출됩니다.
        """
        input_ids = self.get_cached_input_ids(request)
        for input_id in input_ids:
            self.free_encoder_input(request, input_id)

    def get_freed_mm_hashes(self) -> list[str]:
        """최근에 해제된 인코더 캐시 항목 목록을 가져오고 지웁니다.

        보고:
            마지막 이후 실제로 제거된 mm_hash 문자열 목록
            스케줄러가 작업자에게 무엇을 알리기 위해 사용하는 호출입니다.
            인코더 출력은 캐시에서 제거될 수 있습니다. 내부
            이 호출 후에 목록이 지워집니다.
        """
        freed = self.freed
        self.freed = []
        return freed


def compute_mm_encoder_budget(
    scheduler_config: "SchedulerConfig",
    mm_max_toks_per_item: Mapping[str, int],
) -> tuple[int, int]:
    """모델 및 스케줄러를 기반으로 인코더 캐시 예산을 계산합니다.
    다중 모드 모델에 대한 구성입니다.

    인수:
        Scheduler_config: 스케줄러 구성입니다.
        mm_max_toks_per_item: 각 항목당 최대 토큰 수
            텍스트가 아닌 양식.

    보고:
        - 토큰 수로 측정된 인코더 실행을 위한 예산 계산
            입력 순서에서.
        - 토큰 수로 측정된 인코더 캐시 크기에 대한 공간 예산
            입력 순서에서.
    """

    if not mm_max_toks_per_item:
        logger.warning(
            "All non-text modalities supported by the model have been "
            "explicitly disabled via limit_mm_per_prompt. Encoder cache will "
            "not be initialized."
        )
        return 0, 0

    max_tokens_per_mm_item = max(mm_max_toks_per_item.values())

    if (
        scheduler_config.disable_chunked_mm_input
        and max_tokens_per_mm_item > scheduler_config.max_num_batched_tokens
    ):
        raise ValueError(
            "Chunked MM input disabled but max_tokens_per_mm_item "
            f"({max_tokens_per_mm_item}) is larger than max_num_batched_tokens"
            f" ({scheduler_config.max_num_batched_tokens}). Please increase "
            "max_num_batched_tokens."
        )

    encoder_compute_budget = max(
        scheduler_config.max_num_encoder_input_tokens, max_tokens_per_mm_item
    )
    encoder_cache_size = max(
        scheduler_config.encoder_cache_size, max_tokens_per_mm_item
    )

    return encoder_compute_budget, encoder_cache_size


# 참고(NickLucche): 인코더-디코더 모델에 대한 임시 구현은 다음과 같습니다.
# 일정 관리를 위해 관리자를 사용하세요. 인코더-디코더 모델은 결국
# 캐시를 활용하면 이 클래스는 다음과 같이 EncoderCacheManager로 접힐 것입니다.
# MM 모델과의 차이가 줄어듭니다.
class EncoderDecoderCacheManager(EncoderCacheManager):
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.allocated: list[str] = []
        self.to_free: list[str] = []

    def reset(self) -> None:
        """인코더 캐시를 초기 상태로 재설정합니다."""
        self.num_free_slots = self.cache_size
        self.allocated.clear()
        self.to_free.clear()

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        return False

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        # 컴퓨팅 예산이 충분하지 않습니다.
        if num_encoder_embeds > encoder_compute_budget:
            return False

        num_encoder_embeds += num_embeds_to_schedule
        # 충분한 여유 슬롯
        return num_encoder_embeds <= self.num_free_slots

    def allocate(self, request: Request, input_id: int) -> None:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        self.num_free_slots -= num_encoder_embeds

        mm_hash = request.mm_features[input_id].identifier
        self.allocated.append(mm_hash)

    def free(self, request: Request) -> None:
        for input_id in range(len(request.mm_features)):
            self.free_encoder_input(request, input_id)

    def get_cached_input_ids(self, request: Request) -> set[int]:
        return set(range(len(request.mm_features)))

    def get_freed_mm_hashes(self) -> list[str]:
        # enc-dec 모델에는 인코더 캐시가 사용되지 않으므로 여기서 항목을 해제할 수 있습니다.
        # 실제 프리는 모델이 실행되기 *전에* 러너에서 발생합니다.
        # 따라서 'freeable'은 항목을 해제한 후에만 항목을 해제하는 버퍼 역할을 합니다.
        # 'EncoderCacheManager'의 상태 전환을 모방하여 모델이 실행됩니다.
        to_free = self.to_free
        self.to_free = self.allocated
        self.allocated = []
        return to_free

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        self.num_free_slots += num_encoder_embeds
