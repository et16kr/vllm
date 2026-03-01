# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib

from vllm.v1.request import Request, RequestStatus


def remove_all(lst: list, items_to_remove: set) -> list:
    """항목 제거 세트에 있는 목록에서 모든 항목을 제거합니다.

    이 방법은 단일 항목을 제거하는 일반적인 경우에 최적화됩니다. item,
    여러 항목에 대한 목록 이해로 돌아갑니다.

    인수:
        lst: 항목을 제거할 목록
        items_to_remove: 제거할 항목 세트

    반환:
        수정된 원본 목록(단일 항목 제거의 경우) 또는
        새 목록(여러 항목 제거의 경우). 호출자는 
        반환 값을 사용해야 합니다.

    참고:
        단일 항목 제거의 경우 원래 목록을 내부에서 수정
        하여 반환합니다. 여러 항목의 경우 새 목록을 생성하고 반환합니다.
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # 단일 항목 제거를 위한 빠른 경로(가장 일반적인 경우)
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # 여러 항목의 경우 목록 이해를 사용합니다.
    return [item for item in lst if item not in items_to_remove]


def check_stop(request: Request, max_model_len: int) -> bool:
    assert not request.pooling_params

    sampling_params = request.sampling_params
    assert sampling_params is not None

    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    last_token_id = request.output_token_ids[-1]
    if last_token_id == sampling_params.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True
    return False
