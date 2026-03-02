# GPUModelRunner(v1) Deep Dive (학습용)

이 문서는 [`vllm/v1/worker/gpu/model_runner.py`](../../vllm/v1/worker/gpu/model_runner.py)
를 깊게 이해하기 위한 학습 노트입니다. 코드 한 줄씩 따라가기 전에,
"이 파일이 어떤 책임을 어디까지 갖고, 어떤 상태를 어떻게 움직이는지"를 먼저
머릿속에 구조화하는 데 초점을 둡니다.

[TOC]

## 1. 이 파일의 역할 한 줄 요약

`GPUModelRunner`는 **스케줄러가 정한 이번 step의 작업**을 받아서:

1. GPU 실행 입력으로 변환하고,
2. 모델 forward를 실행하고,
3. 샘플링(또는 풀링)하고,
4. 요청 상태를 다음 step용으로 갱신하는

v1 GPU worker의 핵심 실행기입니다.

상위 호출 흐름은 대략 다음과 같습니다.

```text
GPUWorker.execute_model()
  -> GPUModelRunner.execute_model()
GPUWorker.sample_tokens()
  -> GPUModelRunner.sample_tokens()  # generate 모델
  -> GPUModelRunner.pool()           # pooling 모델
```

관련 파일:

- [`vllm/v1/worker/gpu_worker.py`](../../vllm/v1/worker/gpu_worker.py)
- [`vllm/v1/worker/gpu/model_runner.py`](../../vllm/v1/worker/gpu/model_runner.py)
- [`vllm/v1/core/sched/output.py`](../../vllm/v1/core/sched/output.py)

## 2. 왜 이 파일이 "작고 안정적"이어야 하는가

파일 상단 docstring의 메시지는 매우 강합니다:

- 모든 모델(텍스트/멀티모달/생성/임베딩/공개/비공개)이 공용으로 사용한다.
- 모델별 분기는 이 파일이 아니라 각 기능/모델 전용 모듈로 밀어내야 한다.

즉, 이 파일은 "기능을 많이 담는 곳"이 아니라 **공통 실행 경로를 얇게 오케스트레이션**
하는 곳입니다. 실제 복잡도는 `input_batch.py`, `block_table.py`, `cudagraph_utils.py`,
sampler/spec_decode 관련 유틸로 분산됩니다.

## 3. 핵심 상태(State) 지도

`__init__`에서 만들어지는 주요 상태를 먼저 이해하면 이후 메서드가 훨씬 잘 읽힙니다.

| 상태 | 의미 | 주로 갱신되는 곳 |
|---|---|---|
| `req_states` | 요청별 길이/토큰/샘플링 상태를 담는 핵심 저장소 | `add_requests`, `postprocess` |
| `input_buffers` | step마다 재사용되는 입력 텐서 버퍼 | `prepare_inputs` |
| `block_tables` | 요청별 KV block 매핑 테이블 | `add_requests`, `update_requests`, `prepare_attn` |
| `model_state` | 모델별 입력/attention metadata 준비 로직 캡슐화 | `prepare_attn`, `execute_model` |
| `sampler` | logits -> token 샘플링 + logprobs 계산 | `sample` |
| `cudagraph_manager` | CUDA graph 캡처/런타임 모드 결정 | `capture_model`, `execute_model` |
| `kv_connector` | 외부 KV 전송/연동 훅 | `execute_model` 전후 |
| `execute_model_state` | `execute_model()`과 `sample_tokens()/pool()` 사이 임시 전달 상태 | `execute_model`, `sample_tokens`, `pool` |

보조 기능 상태:

- `lora_state`: 요청별 LoRA adapter 활성화 정보
- `encoder_cache`: 멀티모달 인코더 캐시 (첫 PP rank에서만)
- `speculator`, `draft_tokens_handler`: speculative decoding 관련 상태
- `structured_outputs_worker`: grammar bitmask 적용

## 4. 초기화/로딩 단계

## 4.1 `__init__`: 런타임 전략을 고정

여기서 다음 축이 결정됩니다.

- 병렬화 축: PP(`pipeline_parallel_size`), DCP(`decode_context_parallel_size`)
- 모델 타입 축: generate vs pooling
- 기능 축: LoRA, speculative decode, multimodal, async scheduling
- 메모리/실행 축: dtype, kv-cache dtype, CUDA stream/event

중요 포인트:

- `cache_dtype != "auto"`면 KV cache dtype을 별도로 강제합니다.
- speculative method가 `eagle3`이면 aux hidden state를 요구할 수 있으며,
  PP와 동시 사용은 금지(`ValueError`)됩니다.

## 4.2 `load_model`

`DeviceMemoryProfiler`로 모델 로드 메모리 사용량을 측정하면서:

1. 모델 로드
2. (옵션) LoRA 래핑
3. (옵션) speculator 모델 로드 연결
4. 통신 버퍼 준비
5. `model_state` 초기화
6. pooling 모델이면 `PoolingRunner` 생성

## 4.3 `initialize_kv_cache`

핵심 작업:

1. `BlockTables` 구성
2. attention backend 초기화
3. KV cache tensor 초기화
4. KV connector 설정

실제 step 실행 전에 KV/attention 경로의 뼈대를 완성하는 단계입니다.

## 5. step 실행의 큰 흐름

한 step은 두 단계로 쪼개집니다.

1. `execute_model(...)`:
   - 요청 상태 반영
   - 입력 준비
   - 모델 forward 수행
   - 샘플링/풀링에 필요한 중간 상태 저장
2. `sample_tokens(...)` 또는 `pool(...)`:
   - 출력 확정
   - 후처리/상태 업데이트

이를 의사코드로 보면:

```python
execute_model(scheduler_output):
    update_request_states()
    input_batch = prepare_inputs()
    attn = prepare_attn()
    hidden = run_model()
    stash_state_for_sampling(hidden, ...)

sample_tokens(grammar_output):
    state = pop_stashed_state()
    sampler_out = sample(state.hidden, grammar_output)
    postprocess_and_update_req_states(sampler_out)
    return model_runner_output
```

## 6. `execute_model` 상세 분해

## 6.1 요청 상태 반영

`dummy_run`이 아닌 일반 경로에서는 먼저 상태를 갱신합니다.

- `finish_requests`: 완료/선점 요청 제거
- `free_states`: encoder cache 해제
- `add_requests`: 신규 요청 등록 + sampler/model_state/lora 등록
- `update_requests`: 기존 요청 block table 증분 갱신

그리고 스케줄된 토큰 수가 0이면 즉시 `kv_connector.no_forward()`로 종료합니다.

## 6.2 cudagraph + DP 동기화

`CudaGraphManager.get_cudagraph_runtime_mode(...)`로 로컬 후보를 정한 뒤,
`get_cudagraph_and_dp_padding(...)`으로 DP rank 간 모드/패딩을 동기화합니다.

핵심 아이디어:

- rank마다 cudagraph 가능 여부가 다르면 전체를 eager로 fallback
- rank별 토큰 수를 맞추기 위해 패딩 길이를 합의
- 모든 rank 토큰이 0이면 no-op 반환

## 6.3 입력 준비 (`prepare_inputs`)

가장 중요한 전처리 단계입니다.

1. `num_scheduled_tokens`를 기준으로 요청 정렬
2. `req_id -> req_state_idx` 매핑(`idx_mapping`) 생성
3. speculative draft token 수를 반영해 logits 관련 인덱스 확장
4. `query_start_loc`, `positions`, `seq_lens` 생성
5. prefill이 남아 있으면 prompt/prefill token을 `input_ids`에 채움
6. decode의 경우 마지막 sampled token + draft token을 `input_ids`에 채움

여기서 생성된 `InputBatch`가 모델 실행의 단일 입력 패키지 역할을 합니다.

참고:
[`vllm/v1/worker/gpu/input_batch.py`](../../vllm/v1/worker/gpu/input_batch.py)

## 6.4 attention 준비 (`prepare_attn`)

- block table gather
- slot mapping 계산
- 이후 `model_state.prepare_attn(...)`에서 backend별 metadata 구성

`dummy_run`에서는 dummy block/slot 매핑을 사용합니다.

## 6.5 multimodal / LoRA 분기

- LoRA 활성화가 필요하면 요청별 adapter를 활성화
- 멀티모달 입력은 첫 PP rank에서만 encoder를 실행하고 `inputs_embeds` 준비

## 6.6 forward 실행

두 경로가 있습니다.

1. `CUDAGraphMode.FULL`: `cudagraph_manager.run_fullgraph(...)` 재생
2. 그 외: `set_forward_context(...)` 안에서 `self.model(**model_inputs)` 직접 호출

PP 비첫 rank는 `input_ids/inputs_embeds`를 비우고 `intermediate_tensors`를 입력받습니다.

실행 후에는 `kv_connector.post_forward(...)`까지 처리하고,
`execute_model_state`에 샘플링용 컨텍스트를 저장합니다.

PP 비마지막 rank는 `IntermediateTensors`를 반환해 다음 rank로 전달합니다.

## 7. `sample_tokens` 상세 분해

## 7.1 PP 비마지막 rank

비마지막 rank는 자체 샘플링을 하지 않습니다.

- 마지막 rank에서 broadcast된 sampled token을 수신
- `postprocess(...)`로 로컬 상태만 동일하게 갱신

## 7.2 PP 마지막 rank(또는 PP 미사용)

1. `sample(...)`에서 logits 계산 + sampler 호출
2. 필요 시 grammar bitmask 적용
3. draft token이 있으면 rejection sampling 수행
4. rank 간 동기화를 위해 sampled 결과를 broadcast (PP 사용 시)
5. prompt logprobs 계산
6. `ModelRunnerOutput` + `AsyncOutput` 생성

그리고 중요한 순서 제약이 있습니다:

- `AsyncOutput`을 먼저 만든 뒤 `postprocess`를 호출합니다.
- 이유: 비동기 D2H copy 이벤트를 먼저 기록해 지연 시간을 줄이기 위함입니다.

## 7.3 speculative proposal

`speculator`가 있으면 샘플링 결과를 바탕으로 다음 step용 draft token을 제안하고
`req_states.draft_tokens` 및 `draft_tokens_handler`에 저장합니다.

## 8. pooling 경로 (`pool`)

pooling 모델은 샘플링 대신:

1. `pooling_runner.pool(...)` 실행
2. `postprocess_pool(...)`로 길이 상태 갱신
3. `AsyncPoolingOutput`으로 비동기 CPU 복사

즉, generate 경로의 `sample_tokens`와 같은 위치를 pooling이 대체합니다.

## 9. 이 파일의 핵심 데이터 구조 연결

- 스케줄러 입력: `SchedulerOutput`
  - [`vllm/v1/core/sched/output.py`](../../vllm/v1/core/sched/output.py)
- step 입력 패키지: `InputBatch`
  - [`vllm/v1/worker/gpu/input_batch.py`](../../vllm/v1/worker/gpu/input_batch.py)
- step 출력 패키지: `ModelRunnerOutput`, `AsyncOutput`
  - [`vllm/v1/outputs.py`](../../vllm/v1/outputs.py)
  - [`vllm/v1/worker/gpu/async_utils.py`](../../vllm/v1/worker/gpu/async_utils.py)

학습 시에는 위 3개 타입 정의를 같이 열어두고 읽는 것이 가장 빠릅니다.

## 10. 병렬화 관점에서 보는 동작

## 10.1 PP (Pipeline Parallel)

- 첫 rank: 실제 토큰/임베딩 준비
- 중간 rank: `IntermediateTensors` 전달받아 연산
- 마지막 rank: 샘플링/풀링 및 최종 출력 생성

샘플링 결과는 마지막 rank에서 생성 후 다른 rank로 broadcast되어
모든 rank의 요청 상태가 일관되게 갱신됩니다.

## 10.2 DP (Data Parallel)

`get_cudagraph_and_dp_padding`이 rank 간 토큰 길이와 cudagraph 모드를 동기화합니다.
어느 한 rank라도 조건이 맞지 않으면 eager로 내려가서 전체 일관성을 유지합니다.

## 10.3 DCP (Decode Context Parallel)

`prepare_dcp_local_seq_lens`와 slot mapping 계산에서 rank별 로컬 시퀀스 범위를
구성합니다. 실제로는 "전역 시퀀스 길이"와 "이 rank가 담당하는 local view"를 함께
관리합니다.

## 11. 자주 헷갈리는 포인트

1. `prompt_len` vs `prefill_len`
   - `prefill_len`은 prompt + 일부 출력 토큰까지 포함될 수 있습니다.
2. `execute_model_state`
   - step 사이 임시 상태입니다. consume 후 즉시 `None`으로 되돌립니다.
3. `num_tokens` vs `num_tokens_after_padding`
   - 후자는 DP/cudagraph 정합성을 위한 패딩 반영 값입니다.
4. "모델 실행"과 "출력 확정" 분리
   - `execute_model`은 forward까지만, `sample_tokens/pool`에서 결과 확정.

## 12. 디버깅 체크리스트

증상이 있을 때 우선 확인할 포인트:

1. 출력이 비거나 샘플링이 안 됨
   - `scheduler_output.total_num_scheduled_tokens`
   - PP rank 위치(`is_last_pp_rank`)
2. cudagraph이 안 타는 것 같음
   - `capture_model()` 실행 여부
   - `get_cudagraph_runtime_mode()` 결과
   - DP 동기화 이후 mode가 eager로 강등됐는지
3. speculative decoding 이상
   - `input_batch.num_draft_tokens`
   - `rejection_sample` 결과와 `num_sampled/num_rejected`
4. 멀티모달 임베딩 이슈
   - 첫 PP rank에서만 encoder 경로가 실행되는지

## 13. 추천 읽기 순서

처음 읽을 때는 아래 순서가 효율적입니다.

1. `__init__` (상태 필드 지도 만들기)
2. `execute_model` (큰 흐름)
3. `prepare_inputs` / `prepare_attn` (입력/attention 준비)
4. `sample` / `sample_tokens` / `postprocess` (출력 확정과 상태 갱신)
5. `capture_model` / `profile_run` (성능 관련 경로)

그리고 보조 파일을 병행해서 보면 구조가 완성됩니다.

- [`vllm/v1/worker/gpu/input_batch.py`](../../vllm/v1/worker/gpu/input_batch.py)
- [`vllm/v1/worker/gpu/block_table.py`](../../vllm/v1/worker/gpu/block_table.py)
- [`vllm/v1/worker/gpu/cudagraph_utils.py`](../../vllm/v1/worker/gpu/cudagraph_utils.py)
- [`vllm/v1/worker/gpu/states.py`](../../vllm/v1/worker/gpu/states.py)

---

필요하면 다음 단계로, 같은 문서에 `execute_model` 한 step을 실제 입력 예시
(`num_scheduled_tokens`, `draft_tokens`, `query_start_loc`)로 숫자 단위 시뮬레이션해
"텐서 shape/값이 어떻게 변하는지"까지 추가할 수 있습니다.
