# vLLM v1 Core KV Cache 학습 계획서

## 1. 목표
이 문서의 목표는 `vllm/v1/core` 내부 KV cache 관련 코드를 **순서 있게 학습**해서 다음 질문에 답할 수 있게 만드는 것입니다.

1. 블록은 언제 생성/할당/해제/캐시되는가?
2. prefix cache hit는 어디서 계산되고 어떻게 반영되는가?
3. attention 타입별(full/sliding/chunked/mamba) 차이는 어디서 구현되는가?
4. scheduler는 KV cache와 어떤 계약(interface)으로 상호작용하는가?
5. remote KV(load/send) 실패 시 어떤 복구 경로를 타는가?

---

## 2. 학습 범위
주요 대상 경로:
- `vllm/v1/core/`

주요 파일:
1. `kv_cache_utils.py`
2. `block_pool.py`
3. `single_type_kv_cache_manager.py`
4. `kv_cache_coordinator.py`
5. `kv_cache_manager.py`
6. `sched/interface.py`
7. `sched/scheduler.py`
8. `kv_cache_metrics.py`
9. `encoder_cache_manager.py` (멀티모달/encoder cache 연동 보충)

보조 문서(이미 생성됨):
- `vllm/v1/core/KV_CACHE_UTILS_STUDY_KO.md`
- `vllm/v1/core/sched/SCHEDULER_STUDY_KO.md`

---

## 3. 추천 학습 순서
아래 순서가 의존성 기준으로 가장 안전합니다.

1. `kv_cache_utils.py`
2. `block_pool.py`
3. `single_type_kv_cache_manager.py`
4. `kv_cache_coordinator.py`
5. `kv_cache_manager.py`
6. `sched/interface.py`
7. `sched/scheduler.py`
8. `kv_cache_metrics.py`
9. `encoder_cache_manager.py`

핵심 이유:
- `utils`에서 해시/그룹/메모리 계산 개념을 잡고
- `block_pool`로 물리 블록 lifecycle을 이해한 뒤
- `single_type manager`와 `coordinator`로 attention 타입 분기를 이해하고
- 마지막에 `kv_cache_manager` + `scheduler`로 런타임 통합 흐름을 보는 구조입니다.

---

## 4. 2주(10세션) 학습 플랜

### Session 1: KV cache 용어/개념 정리
대상 파일:
- `kv_cache_utils.py` (해시/타입/큐 자료구조 파트)

목표:
1. `BlockHash`, `BlockHashWithGroupId`, `NONE_HASH` 의미 이해
2. `KVCacheBlock`, `FreeKVCacheBlockQueue` 동작 이해

완료 기준:
- `popleft/remove/append`가 왜 O(1)인지 설명 가능

### Session 2: request -> block hash 생성 경로
대상 파일:
- `kv_cache_utils.py` (`need_extra_keys` ~ `get_request_block_hasher`)

목표:
1. MM/LoRA/cache_salt/prompt_embeds가 해시에 어떻게 들어가는지 이해
2. full block만 해싱하는 이유 파악

완료 기준:
- "같은 token ids라도 다른 요청 조건이면 hash가 달라진다"를 코드로 설명 가능

### Session 3: 메모리/길이 추정 경로
대상 파일:
- `kv_cache_utils.py` (`_check_enough_kv_cache_memory` 이후)

목표:
1. `estimate_max_model_len` 이진 탐색 로직 이해
2. `get_num_blocks`, `get_max_concurrency_for_kv_cache_config` 계산식 이해

완료 기준:
- given memory에서 num_blocks/max_len이 어떻게 변하는지 수식으로 설명 가능

### Session 4: group/spec 구성 로직
대상 파일:
- `kv_cache_utils.py` (`get_kv_cache_groups`, `get_kv_cache_configs`)

목표:
1. uniform/hybrid/attention-free 분기 이해
2. worker별 config 투영(`_project_kv_cache_groups_to_worker`) 이해

완료 기준:
- PP 환경에서 왜 worker 투영이 필요한지 설명 가능

### Session 5: 블록 저장소(BlockPool)
대상 파일:
- `block_pool.py`

목표:
1. cached block map과 free queue 관계 이해
2. touch/cache/free/evict 흐름 이해

완료 기준:
- 블록이 "free -> allocated -> cached -> evicted"로 이동하는 경로를 추적 가능

### Session 6: attention 타입별 manager
대상 파일:
- `single_type_kv_cache_manager.py`

목표:
1. `FullAttentionManager`, `SlidingWindowManager`, `MambaManager` 차이 이해
2. `find_longest_cache_hit`, `remove_skipped_blocks`의 타입별 동작 이해

완료 기준:
- sliding/mamba에서 왜 null block/skip block 처리가 필요한지 설명 가능

### Session 7: coordinator 계층
대상 파일:
- `kv_cache_coordinator.py`

목표:
1. coordinator가 manager들을 어떻게 묶는지 이해
2. prefix hit 길이 정렬(alignment)과 eagle/mamba 예외 처리 이해

완료 기준:
- coordinator 없이 manager 직접 호출이 어려운 이유 설명 가능

### Session 8: 런타임 통합 관리자
대상 파일:
- `kv_cache_manager.py`

목표:
1. `get_computed_blocks -> allocate_slots -> cache_blocks -> free` 메인 경로 이해
2. `KVCacheBlocks` 구조와 scheduler 계약 확인

완료 기준:
- 요청 1개 기준 prefill/decode 한 스텝에서 블록 테이블이 어떻게 변하는지 설명 가능

### Session 9: scheduler 계약
대상 파일:
- `sched/interface.py`, `sched/scheduler.py`

목표:
1. scheduler가 KV cache manager를 언제/어떻게 호출하는지 이해
2. preempt/resume/remote KV 상태 전이 이해

완료 기준:
- `schedule`와 `update_from_output` 사이 상태 변화도를 그릴 수 있음

### Session 10: 보조 계층 + 복습
대상 파일:
- `kv_cache_metrics.py`, `encoder_cache_manager.py`

목표:
1. 메트릭 수집 포인트 파악
2. encoder cache와 scheduler 상호작용(멀티모달 경로) 이해

완료 기준:
- KV cache 장애/성능 이슈 분석 시 어떤 데이터부터 봐야 하는지 정리 가능

---

## 5. 매 세션 공통 체크리스트
각 세션마다 아래 5가지를 기록합니다.

1. 이 파일의 핵심 상태(state) 변수 5개
2. 외부에서 호출되는 public 함수 3개
3. 이 파일이 의존하는 파일 2개
4. 실패/예외 경로 2개
5. 성능 영향 포인트 2개

---

## 6. 코드 읽기 실전 팁

1. 먼저 함수 시그니처와 반환 타입만 훑고, 구현은 두 번째에 읽습니다.
2. `assert` 문은 설계 계약(contract)으로 보고 별도 메모합니다.
3. `RequestStatus` 변화는 반드시 타임라인으로 기록합니다.
4. 블록 수/토큰 수 변수(`num_*`)는 단위(토큰/블록/요청)를 붙여 메모합니다.
5. remote KV 경로는 동기/비동기를 분리해서 추적합니다.

---

## 7. 권장 산출물
학습이 끝나면 아래 산출물을 남기면 좋습니다.

1. "요청 1개 prefill -> decode -> finish" KV block 변화도 1장
2. "prefix hit miss 원인" 체크리스트
3. "remote KV load fail" 복구 플로우차트
4. "mamba align 모드" 예외 케이스 요약 1페이지

---

## 8. 빠른 시작(오늘 바로 시작)
오늘은 아래만 진행하면 됩니다.

1. `KV_CACHE_UTILS_STUDY_KO.md`의 Step A~C 완료
2. `block_pool.py`에서 `BlockHashToBlockMap`, `BlockPool` 클래스 시그니처 스캔
3. `single_type_kv_cache_manager.py`에서 manager 클래스 이름/역할만 맵으로 정리

이 3개를 끝내면 다음 세션부터 `kv_cache_manager.py` 이해 속도가 크게 올라갑니다.
