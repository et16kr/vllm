# vLLM Design 학습 가이드 (Korean)

이 문서는 `docs/design` 문서들을 기반으로 vLLM 설계를 체계적으로 학습하기 위한 가이드입니다.
단순히 읽는 순서만 제시하지 않고, 각 단계에서 무엇을 이해해야 하는지,
어떤 코드와 연결해 봐야 하는지, 어떤 실습을 하면 이해가 깊어지는지까지 포함합니다.

[TOC]

## 1. 이 가이드의 목표

학습 완료 시 다음을 설명할 수 있어야 합니다.

1. vLLM V1의 프로세스 구조(API 서버/엔진 코어/GPU 워커/DP 코디네이터)와 요청 흐름
2. KV 캐시(prefix caching 포함)와 스케줄링/실행 경로의 핵심 의사결정
3. `torch.compile`/CUDA Graphs의 역할 분리와 런타임 모드 선택 원리
4. MoE/멀티모달/플러그인 등 확장 포인트가 기존 실행 경로에 연결되는 방식
5. 운영 관점에서 성능/안정성 디버깅 시 어디를 먼저 확인해야 하는지

## 2. 권장 학습 순서 (필수 코어 경로)

아래 순서를 기본으로 권장합니다. 번호 순서대로 진행하면 개념 의존성이 잘 맞습니다.

1. [arch_overview_ko.md](arch_overview_ko.md)
2. [optimization_levels_ko.md](optimization_levels_ko.md)
3. [multiprocessing_ko.md](multiprocessing_ko.md)
4. [huggingface_integration_ko.md](huggingface_integration_ko.md)
5. [prefix_caching_ko.md](prefix_caching_ko.md)
6. [torch_compile_ko.md](torch_compile_ko.md)
7. [cuda_graphs_ko.md](cuda_graphs_ko.md)
8. [gpu_model_runner_v1_deep_dive_ko.md](gpu_model_runner_v1_deep_dive_ko.md)

학습 시간 가이드(대략):

- 입문(1~4): 3~5시간
- 핵심 성능(5~7): 4~7시간
- 실행기 심화(8): 3~6시간

## 3. 단계별 상세 학습 플랜

## 3.1 아키텍처 지도 잡기

문서:

- [arch_overview_ko.md](arch_overview_ko.md)

핵심 질문:

1. 오프라인(`LLM`)과 온라인(`vllm serve`) 엔트리포인트의 차이는 무엇인가?
2. TP/PP/DP 설정이 프로세스 수와 통신 경로를 어떻게 바꾸는가?
3. `LLMEngine`/`AsyncLLMEngine`/Worker/ModelRunner/Model 경계는 어디인가?

코드 연결 포인트:

- [llm.py](../../vllm/entrypoints/llm.py)
- [api_server.py](../../vllm/entrypoints/openai/api_server.py)
- [core.py](../../vllm/v1/engine/core.py)
- [gpu_worker.py](../../vllm/v1/worker/gpu_worker.py)

체크포인트:

- `vllm serve -tp=4`와 `vllm serve -tp=2 -dp=4`의 프로세스 구성을 손으로 설명할 수 있어야 합니다.
- API 서버와 엔진 코어가 ZMQ로 many-to-many 연결된 이유를 말할 수 있어야 합니다.

## 3.2 실행 환경 기본기(최적화 레벨 + 멀티프로세싱)

문서:

- [optimization_levels_ko.md](optimization_levels_ko.md)
- [multiprocessing_ko.md](multiprocessing_ko.md)

핵심 질문:

1. `-O0/-O1/-O2`를 무엇 기준으로 선택해야 하는가?
2. 왜 `spawn`/`fork` 선택이 라이브러리 사용 시(특히 `__main__` guard) 문제를 만들 수 있는가?
3. CUDA 선초기화가 왜 멀티프로세싱 시작 방식에 영향을 주는가?

코드 연결 포인트:

- [envs.py](../../vllm/envs.py)
- [scripts.py](../../vllm/scripts.py)
- [all_reduce_utils.py](../../vllm/distributed/device_communicators/all_reduce_utils.py)
- [api_server.py](../../vllm/entrypoints/openai/api_server.py)

실무 포인트:

- 개발/디버그는 `-O0` 또는 `-O1`, 운영 기본은 `-O2`를 기준으로 시작
- 라이브러리 임베드 시 `if __name__ == "__main__":` 가드 유무 먼저 점검

## 3.3 모델 입력 계층 이해(HF 통합)

문서:

- [huggingface_integration_ko.md](huggingface_integration_ko.md)

핵심 질문:

1. `model` 인자가 로컬 경로/허브 ID일 때 경로가 어떻게 갈리는가?
2. `model_type`, `architectures`, `--trust_remote_code`는 각각 어떤 위험/이점을 가지는가?
3. 토크나이저와 가중치 로딩 지점이 추론 파이프라인에 미치는 영향은 무엇인가?

코드 연결 포인트:

- [config.py](../../vllm/transformers_utils/config.py)
- [tokenizer.py](../../vllm/transformers_utils/tokenizer.py)
- [default_loader.py](../../vllm/model_executor/model_loader/default_loader.py)
- [registry.py](../../vllm/model_executor/models/registry.py)

체크포인트:

- 특정 모델이 "왜 로드 실패하는지"를 `config -> architecture -> registry` 순으로 추적할 수 있어야 합니다.

## 3.4 KV 캐시와 요청 재사용 메커니즘

문서:

- [prefix_caching_ko.md](prefix_caching_ko.md)

핵심 질문:

1. 왜 블록 해시에 parent hash와 block tokens를 함께 넣는가?
2. 왜 full block만 캐싱하는가?
3. free queue와 LRU eviction의 연결 관계는 무엇인가?
4. V1에서 duplicate block이 생기는 구조적 이유는 무엇인가?

코드 연결 포인트:

- [kv_cache_manager.py](../../vllm/v1/core/kv_cache_manager.py)
- [scheduler.py](../../vllm/v1/core/sched/scheduler.py)

권장 실습:

1. block size를 작게 둔 실험 환경에서 동일 prefix 요청을 연속 전송
2. cache hit/miss 시 지연 시간과 메모리 사용 패턴 관찰
3. `cache_salt` 유무에 따른 재사용 경계 비교

## 3.5 컴파일/그래프 실행 경로 이해

문서:

- [torch_compile_ko.md](torch_compile_ko.md)
- [cuda_graphs_ko.md](cuda_graphs_ko.md)
- (보강) [debug_vllm_compile.md](debug_vllm_compile.md)

핵심 질문:

1. `torch.compile`과 CUDA Graphs는 어떻게 다르고, 왜 둘을 분리해 생각해야 하는가?
2. `PIECEWISE`, `FULL`, `FULL_AND_PIECEWISE`는 어떤 배치에서 각각 유리한가?
3. dispatcher의 dispatch key(`BatchDescriptor`)가 런타임 결정을 어떻게 단순화하는가?

코드 연결 포인트:

- [cuda_graph.py](../../vllm/compilation/cuda_graph.py)
- [cudagraph_dispatcher.py](../../vllm/v1/cudagraph_dispatcher.py)
- [compiler_interface.py](../../vllm/compilation/compiler_interface.py)
- [model_runner.py](../../vllm/v1/worker/gpu/model_runner.py)

실무 디버깅 루틴:

1. 먼저 eager로 축소: `--enforce-eager` 또는 `-cc.mode=0`
2. 문제 재현 후 cudagraph만 on/off 분리 비교
3. compile cache 비활성화(`VLLM_DISABLE_COMPILE_CACHE=1`)로 캐시 원인 배제
4. debug 로그에서 어떤 그래프가 capture/replay되는지 확인

## 3.6 실행기 심화(핵심 코드 독해)

문서:

- [gpu_model_runner_v1_deep_dive_ko.md](gpu_model_runner_v1_deep_dive_ko.md)

코드:

- [model_runner.py](../../vllm/v1/worker/gpu/model_runner.py)

핵심 질문:

1. `execute_model`과 `sample_tokens/pool`의 2단계 분리가 왜 필요한가?
2. `req_states`, `block_tables`, `execute_model_state`는 각각 어떤 수명주기를 가지는가?
3. pipeline parallel 비마지막 rank에서 샘플링 동기화가 어떻게 이루어지는가?

추천 독해 순서:

1. `__init__`에서 상태 변수 지도 작성
2. `load_model` / `initialize_kv_cache`
3. `execute_model` 전체 흐름
4. `prepare_inputs` / `prepare_attn`
5. `sample_tokens` / `postprocess`

학습 산출물:

- "한 step에서 어떤 텐서/메타데이터가 어디서 생성되고 어디서 소비되는지"를 1페이지 다이어그램으로 정리

## 4. 심화 트랙 (선택)

코어 경로 이후, 관심 분야별로 아래를 추천합니다.

## 4.1 MoE/분산 성능 트랙

순서:

1. [moe_kernel_features.md](moe_kernel_features.md)
2. [fused_moe_modular_kernel.md](fused_moe_modular_kernel.md)
3. [dbo.md](dbo.md)
4. [hybrid_kv_cache_manager.md](hybrid_kv_cache_manager.md)

초점:

- all2all backend 선택 기준
- activation format(standard vs batched)
- DBO가 compute/communication overlap을 만드는 방식

## 4.2 멀티모달 트랙

순서:

1. [mm_processing.md](mm_processing.md)
2. [torch_compile_multimodal.md](torch_compile_multimodal.md)

초점:

- placeholder token과 실제 modal input 매핑
- multimodal encoder compile on/off의 비용과 이점

## 4.3 확장성/플러그인 트랙

순서:

1. [plugin_system.md](plugin_system.md)
2. [custom_op.md](custom_op.md)
3. [logits_processors.md](logits_processors.md)
4. [io_processor_plugins.md](io_processor_plugins.md)
5. [lora_resolver_plugins.md](lora_resolver_plugins.md)

초점:

- 다중 프로세스 환경에서 plugin 로딩 보장 방식
- 확장 포인트별 API 안정성/리스크 경계

## 4.4 운영/관측 트랙

순서:

1. [metrics.md](metrics.md)
2. [optimization_levels_ko.md](optimization_levels_ko.md)
3. [debug_vllm_compile.md](debug_vllm_compile.md)

초점:

- server-level metric과 request-level metric의 인과 관계
- 성능 이슈 발생 시 지표->설정->코드 경로 추적

## 5. 문서별 선후관계 요약

빠른 참조를 위해 선행 관계를 압축하면 아래와 같습니다.

- `arch_overview_ko` -> 모든 문서의 공통 선행
- `optimization_levels_ko`, `multiprocessing_ko` -> 실행/운영 관련 문서 선행
- `huggingface_integration_ko` -> 모델 로딩 실패/초기화 문제 분석 선행
- `prefix_caching_ko` -> 스케줄러/KV 캐시 성능 분석 선행
- `torch_compile_ko` -> `cuda_graphs_ko`, `debug_vllm_compile` 선행
- `gpu_model_runner_v1_deep_dive_ko` -> MoE/멀티모달 실행 경로 심화 선행

## 6. 권장 학습 루프 (읽기 -> 코드 -> 실험 -> 복기)

각 단계마다 아래 루프를 반복하면 이해가 급격히 깊어집니다.

1. 문서 읽기(핵심 주장 3개 추출)
2. 코드 대응(문서의 주장 각각이 코드 어디인지 링크 매핑)
3. 실험 1개(설정 하나만 바꿔 차이 관찰)
4. 복기(왜 차이가 났는지, 어떤 가정이 깨졌는지 기록)

복기 템플릿(권장):

```text
[주제]
- 문서 핵심 주장:
- 코드 근거 파일/함수:
- 실험 설정:
- 관찰 결과:
- 해석:
- 남은 질문:
```

## 7. 2주 집중 학습 예시 플랜

## Week 1: 코어 구조 + 성능 기본

1. Day 1: `arch_overview_ko`
2. Day 2: `optimization_levels_ko` + `multiprocessing_ko`
3. Day 3: `huggingface_integration_ko`
4. Day 4: `prefix_caching_ko`
5. Day 5: `torch_compile_ko`
6. Day 6: `cuda_graphs_ko`
7. Day 7: `debug_vllm_compile`로 복습/실험

## Week 2: 실행기 심화 + 선택 트랙

1. Day 8~9: `gpu_model_runner_v1_deep_dive_ko` + `model_runner.py` 독해
2. Day 10~11: MoE 또는 멀티모달 중 하나 선택
3. Day 12: plugin/extension 계층 훑기
4. Day 13: metrics/운영 관점 정리
5. Day 14: "내가 이해한 vLLM 실행 파이프라인" 2~3페이지 문서화

## 8. 자주 헷갈리는 포인트

1. `torch.compile`과 CUDA Graphs는 같은 것이 아닙니다.
   compile은 커널/그래프 최적화, cudagraph는 런타임 실행 오버헤드 절감입니다.

2. `FULL_AND_PIECEWISE`가 항상 절대적으로 좋은 것은 아닙니다.
   메모리/캡처 시간/백엔드 호환성을 함께 봐야 합니다.

3. prefix 캐시 hit율이 높아도 성능이 항상 선형 개선되지는 않습니다.
   스케줄링/큐잉/통신/샘플링 병목이 함께 작동합니다.

4. 라이브러리 임베드 환경의 multiprocessing 문제는 설정 하나로 끝나지 않습니다.
   `__main__` 가드, CUDA 초기화 시점, 실행 방식(CLI/라이브러리)을 같이 봐야 합니다.

## 9. 참고 문서 주의사항

- [paged_attention.md](paged_attention.md)는 문서 자체 경고처럼 historical 문서입니다.
  현재 코드 동작과 1:1 대응되는 최신 설계 문서로 보지 말고 배경 지식 용도로만 사용하세요.

- [attention_backends.md](attention_backends.md)는 자동 생성 문서입니다.
  기능 지원 매트릭스 확인용으로 쓰고, 설계 의도 파악은 관련 코드/설계 문서와 함께 보세요.

## 10. 마지막 체크리스트

아래 항목을 스스로 설명할 수 있으면 코어 설계 학습은 충분히 진행된 상태입니다.

1. 요청 1개가 API 서버에 들어와 토큰이 출력될 때까지의 주요 함수 경로
2. DP/TP/PP 설정 변화가 프로세스 수와 worker 역할에 미치는 영향
3. prefix caching의 해시 키 구성과 eviction 순서의 이유
4. `-O` 레벨과 `cudagraph_mode`를 어떤 기준으로 선택할지
5. compile/cudagraph 관련 문제를 축소 재현하는 디버깅 순서
