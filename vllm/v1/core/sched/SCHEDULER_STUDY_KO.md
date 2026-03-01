# vLLM v1 Scheduler 학습 가이드 (scheduler.py 중심)

## 1) 왜 문서가 더 좋은가
스케줄러는 함수 간 상태 공유가 많아서, 코드 주석만 따라가면 전체 흐름이 끊기기 쉽습니다.

- 주석: 해당 줄의 의도 파악에 좋음
- Markdown: "어떤 함수가 어떤 순서로 호출되는지"를 큰 흐름으로 잡는 데 좋음

학습 단계에서는 문서로 흐름을 먼저 잡고, 코드에서 line-by-line로 검증하는 방식이 가장 빠릅니다.

---

## 2) 이 파일의 역할
`scheduler.py`는 엔진 코어에서 아래를 총괄합니다.

1. 요청 큐(`waiting`, `running`) 관리
2. 토큰 스케줄링(`schedule`)과 KV 슬롯 할당
3. 모델 출력 반영(`update_from_output`)과 요청 상태 전이
4. speculative decoding/structured output/멀티모달 encoder 입력 스케줄링
5. KV connector(원격 KV 로드/전송) 연동 및 오류 복구

---

## 3) Python 초심자용 "독립" 시작 구간
아래는 다른 복잡한 함수와 결합이 약해서 먼저 보기 좋습니다.

1. `pause_state`, `set_pause_state` (1711, 1714)
2. `get_request_counts` (1599)
3. `has_finished_requests` (1725)
4. `get_num_unfinished_requests` (1717)
5. `make_spec_decoding_stats` (1839)
6. `_get_encoder_cache_usage` (1831)

이 구간으로 필드/상태 이름에 익숙해진 뒤 핵심 경로로 들어가면 이해가 쉬워집니다.

---

## 4) 핵심 실행 경로 (반드시 이 순서로 보기)

### Step A. 초기화
대상: `__init__` (64)

확인할 것:
- 스케줄 제약: `max_num_running_reqs`, `max_num_scheduled_tokens`, `max_model_len`
- 큐/상태: `waiting`, `running`, `requests`, `finished_req_ids`
- 핵심 컴포넌트 생성: `KVCacheManager`, `EncoderCacheManager`, connector들
- speculative/eagle/mamba 모드 플래그

### Step B. 스케줄링 메인 루프
대상: `schedule` (317)

확인할 것:
- 1단계: `running` 요청 스케줄
- 2단계: `waiting` 요청 스케줄
- `token_budget` 감소 방식
- `allocate_slots` 실패 시 선점(preempt) 로직
- 결과물 조립: `SchedulerOutput`

### Step C. 멀티모달 encoder 입력 배정
대상: `_try_schedule_encoder_inputs` (1038)

확인할 것:
- 토큰 구간 겹침 판단
- encoder cache/budget 제약 시 `num_new_tokens` 축소 방식
- `disable_chunked_mm_input` 분기

### Step D. 모델 출력 반영
대상: `update_from_output` (1213)

확인할 것:
- 토큰 append + stop 판단
- speculative token 수락/거부 반영
- 완료 요청 처리 + output 생성
- KV 이벤트 수집 및 stats 구성

### Step E. 스케줄 직후 상태 선반영
대상: `_update_after_schedule` (908)

확인할 것:
- 왜 `num_computed_tokens`를 모델 실행 전에 선반영하는지
- `finished_req_ids`를 새 set으로 교체하는 이유

### Step F. 요청 생명주기 API
대상:
- `add_request` (1603)
- `finish_requests` (1625)
- `_free_request` / `_free_blocks` (1687, 1705)
- `_preempt_request` (887)

확인할 것:
- 상태 전이(`WAITING`, `RUNNING`, `PREEMPTED`, `FINISHED_*`)
- 스트리밍 요청의 `streaming_queue` 처리

### Step G. KV connector/오류 복구 경로
대상:
- `_connector_finished` (1871)
- `_update_waiting_for_remote_kv` (1902)
- `_update_from_kv_xfer_finished` (1946)
- `_update_requests_with_invalid_blocks` (1975)
- `_handle_invalid_blocks` (2078)

확인할 것:
- 원격 KV 수신 완료 시 상태 복귀
- invalid block 발생 시 재계산/실패 정책
- `recompute_kv_load_failures` 동작 차이

---

## 5) 함수 인덱스 (빠른 참조)

### A. 스케줄 핵심
- `_mamba_block_aligned_split` (271)
- `schedule` (317)
- `_make_cached_request_data` (978)

### B. 출력 반영/후처리
- `_update_after_schedule` (908)
- `update_from_output` (1213)
- `_update_request_with_output` (1500)
- `update_draft_token_ids` (1539)
- `update_draft_token_ids_in_output` (1561)
- `get_grammar_bitmask` (1189)

### C. 멀티모달/인코더
- `_try_schedule_encoder_inputs` (1038)
- `_free_encoder_inputs` (1517)
- `_get_encoder_cache_usage` (1831)

### D. 요청 생명주기
- `add_request` (1603)
- `finish_requests` (1625)
- `_preempt_request` (887)
- `_free_request` (1687)
- `_free_blocks` (1705)
- `_handle_stopped_request` (1458)
- `_update_request_as_session` (935)

### E. 제어/운영
- `pause_state` (1711)
- `set_pause_state` (1714)
- `get_num_unfinished_requests` (1717)
- `has_finished_requests` (1725)
- `reset_prefix_cache` (1728)
- `reset_connector_cache` (1771)
- `reset_encoder_cache` (1785)
- `make_stats` (1793)
- `make_spec_decoding_stats` (1839)
- `shutdown` (1858)

### F. KV connector 경로
- `get_kv_connector` (1868)
- `_connector_finished` (1871)
- `_update_waiting_for_remote_kv` (1902)
- `_update_from_kv_xfer_finished` (1946)
- `_update_requests_with_invalid_blocks` (1975)
- `_handle_invalid_blocks` (2078)

---

## 6) 함께 보면 좋은 파일 (scheduler 이해 필수)
1. `vllm/v1/core/sched/interface.py`
2. `vllm/v1/core/sched/output.py`
3. `vllm/v1/core/sched/request_queue.py`
4. `vllm/v1/core/kv_cache_manager.py`
5. `vllm/v1/core/single_type_kv_cache_manager.py`

---

## 7) 추천 학습 루틴 (실전)

1. `schedule`에서 `num_new_tokens`가 결정되는 모든 분기를 먼저 체크
2. 같은 요청이 `update_from_output`에서 어떤 상태로 끝나는지 역추적
3. `finish_requests`를 강제로 호출했을 때 어떤 큐에서 어떻게 제거되는지 확인
4. KV load failure 경로를 `_handle_invalid_blocks` 기준으로 별도 메모 작성

이 4개를 완료하면 scheduler의 80%는 구조적으로 잡힌 상태입니다.
