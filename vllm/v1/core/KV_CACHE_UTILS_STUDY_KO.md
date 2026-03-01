# vLLM v1 KV Cache 학습 가이드 (kv_cache_utils.py 중심)

## 1) Markdown vs 주석
결론: **학습용은 Markdown이 더 낫습니다.**

- Markdown: 파일 간 흐름, 함수 의존관계, 읽는 순서 정리에 적합
- 코드 주석: "왜 이렇게 구현했는가" 같은 유지보수 포인트를 짧게 남길 때 적합

즉, 지금처럼 구조를 공부할 때는 Markdown으로 큰 지도를 만들고, 코드 주석은 최소한으로 유지하는 방식이 가장 효율적입니다.

---

## 2) 이 파일의 역할
`kv_cache_utils.py`는 크게 아래를 담당합니다.

1. **블록 해시 계산/변환 유틸**
2. **free 블록 큐 자료구조**
3. **요청별 block hash 생성 로직** (멀티모달/LoRA/salt 포함)
4. **KV cache 메모리 계산 및 max_model_len 추정**
5. **KV cache group/spec 구성 및 워커별 config 생성**

---

## 3) Python이 익숙하지 않을 때, 먼저 볼 "독립" 구간
아래 순서는 다른 파일 의존성이 상대적으로 낮아 초반 진입이 쉽습니다.

1. `make_block_hash_with_group_id` (49)
2. `get_block_hash` (60)
3. `get_group_id` (65)
4. `KVCacheBlock` (108)
5. `FreeKVCacheBlockQueue` (156)
6. `hash_block_tokens` (519)
7. `BlockHashListWithBlockSize` (1524)

이 7개만 먼저 보면, 이후 스케줄러/KV 매니저 코드를 읽을 때 용어가 훨씬 쉬워집니다.

---

## 4) 단계별 Line-by-Line 학습 순서

### Step A. 해시/타입 기초 (49~89)
대상:
- `make_block_hash_with_group_id`
- `get_block_hash`
- `get_group_id`
- `maybe_convert_block_hash`
- `init_none_hash`

체크포인트:
- `BlockHash`와 `BlockHashWithGroupId`를 왜 분리했는지
- group id를 bytes 뒤 4바이트로 붙이는 이유
- `NONE_HASH`가 첫 블록 parent hash로 쓰이는 흐름

### Step B. 블록 메타데이터와 free 큐 (108~337)
대상:
- `KVCacheBlock`
- `FreeKVCacheBlockQueue`

체크포인트:
- `ref_cnt`, `is_null`, `block_hash`의 의미
- `popleft/remove/append`가 모두 O(1)인 이유(이중 연결 리스트)
- fake head/tail sentinel 패턴

### Step C. 요청별 extra key + block hash (357~598)
대상:
- `need_extra_keys`
- `_gen_mm_extra_hash_keys`
- `_gen_lora_extra_hash_keys`
- `_gen_prompt_embeds_extra_hash_keys`
- `generate_block_hash_extra_keys`
- `hash_block_tokens`
- `get_request_block_hasher`

체크포인트:
- 왜 MM/LoRA/cache_salt/prompt_embeds가 해시에 포함되는지
- "같은 토큰이라도 요청 조건이 다르면 다른 해시"가 되는 구조
- full block만 해싱하는 조건(`end_token_idx > num_tokens`)

### Step D. 메모리/길이 추정 (599~722)
대상:
- `_check_enough_kv_cache_memory`
- `max_memory_usage_bytes`
- `estimate_max_model_len`
- `check_enough_kv_cache_memory`

체크포인트:
- max_model_len 이진탐색 방식
- 메모리 부족 시 어떤 값으로 에러 메시지를 구성하는지

### Step E. 그룹/페이지/블록 수 계산 (723~1149)
대상:
- `create_kv_cache_group_specs`
- `is_kv_cache_spec_uniform`
- `get_max_concurrency_for_kv_cache_config`
- `may_override_num_blocks`
- `get_num_blocks`
- `get_uniform_page_size`
- `_get_kv_cache_groups_uniform_spec`
- `_get_kv_cache_groups_uniform_type`
- `is_kv_cache_page_size_uniform`
- `unify_kv_cache_spec_page_size`
- `is_kv_cache_type_attention_free`
- `_get_kv_cache_groups_uniform_page_size`
- `get_kv_cache_config_from_groups`
- `unify_hybrid_kv_cache_specs`
- `get_kv_cache_groups`

체크포인트:
- uniform / hybrid / attention-free 경로 분기
- 왜 page size 통일이 필요한지
- group 단위 텐서 공유(`KVCacheTensor.shared_by`) 개념

### Step F. 엔드투엔드 진입점 (1150~1518)
대상:
- `generate_scheduler_kv_cache_config`
- `_report_kv_cache_config`
- `_max_memory_usage_bytes_from_groups`
- `_estimate_max_model_len_from_groups`
- `_auto_fit_max_model_len`
- `_project_kv_cache_groups_to_worker`
- `get_kv_cache_configs`

체크포인트:
- 워커별 spec 병합 -> 전역 그룹 생성 -> 워커 투영 -> 공통 `num_blocks` 정렬
- PP(파이프라인 병렬) 환경에서 왜 투영이 필요한지

### Step G. 해시 블록 크기 변환기 (1524~끝)
대상:
- `BlockHashListWithBlockSize`

체크포인트:
- `hash_block_size -> target_block_size` 지연 변환(lazy)
- `scale_factor` 기반 인덱싱

---

## 5) 함수/클래스 인덱스 (빠른 참조)

### A. 해시 기본
- `make_block_hash_with_group_id` (49): block hash + group id 패킹
- `get_block_hash` (60): 패킹 키에서 block hash 추출
- `get_group_id` (65): 패킹 키에서 group id 추출
- `maybe_convert_block_hash` (70): 이벤트 로깅용 hash 타입 변환
- `init_none_hash` (89): 첫 블록용 seed hash 초기화

### B. 블록/큐 자료구조
- `KVCacheBlock` (108): 블록 메타데이터 단위
- `FreeKVCacheBlockQueue` (156): free 블록 이중 연결 리스트 큐
  - `popleft`, `popleft_n`, `remove`, `append`, `append_n`, `get_all_free_blocks`

### C. 요청별 해시 키 생성
- `need_extra_keys` (357)
- `_gen_mm_extra_hash_keys` (377)
- `_gen_lora_extra_hash_keys` (440)
- `_gen_prompt_embeds_extra_hash_keys` (454)
- `generate_block_hash_extra_keys` (480)
- `hash_block_tokens` (519)
- `get_request_block_hasher` (548)

### D. 메모리 검증/추정
- `_check_enough_kv_cache_memory` (599)
- `max_memory_usage_bytes` (636)
- `estimate_max_model_len` (643)
- `check_enough_kv_cache_memory` (695)

### E. 그룹/스펙/페이지 정리
- `create_kv_cache_group_specs` (723)
- `is_kv_cache_spec_uniform` (750)
- `get_max_concurrency_for_kv_cache_config` (773)
- `may_override_num_blocks` (792)
- `get_num_blocks` (806)
- `get_uniform_page_size` (823)
- `_get_kv_cache_groups_uniform_spec` (830)
- `_get_kv_cache_groups_uniform_type` (845)
- `is_kv_cache_page_size_uniform` (862)
- `unify_kv_cache_spec_page_size` (875)
- `is_kv_cache_type_attention_free` (915)
- `_get_kv_cache_groups_uniform_page_size` (920)
- `get_kv_cache_config_from_groups` (1013)
- `unify_hybrid_kv_cache_specs` (1088)
- `get_kv_cache_groups` (1150)

### F. 최종 설정 생성
- `generate_scheduler_kv_cache_config` (1186)
- `_report_kv_cache_config` (1206)
- `_max_memory_usage_bytes_from_groups` (1250)
- `_estimate_max_model_len_from_groups` (1285)
- `_auto_fit_max_model_len` (1320)
- `_project_kv_cache_groups_to_worker` (1385)
- `get_kv_cache_configs` (1419)

### G. 블록 크기 변환 헬퍼
- `BlockHashListWithBlockSize` (1524)

---

## 6) 이 파일 다음 추천 학습 순서
1. `vllm/v1/core/kv_cache_manager.py`
2. `vllm/v1/core/kv_cache_coordinator.py`
3. `vllm/v1/core/single_type_kv_cache_manager.py`
4. `vllm/v1/core/sched/scheduler.py`

이 순서로 보면 "설정 생성 -> 런타임 할당/해제 -> 스케줄링 연결"이 자연스럽게 이어집니다.
