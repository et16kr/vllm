# 아키텍처 개요

이 문서는 vLLM 아키텍처를 개괄적으로 설명합니다.

[TOC]

## 엔트리포인트

vLLM은 시스템과 상호작용하기 위한 여러 엔트리포인트를 제공합니다.
아래 다이어그램은 이들 사이의 관계를 보여줍니다.

![Entrypoints Diagram](../assets/design/arch_overview/entrypoints.excalidraw.png)

### LLM 클래스

`LLM` 클래스는 오프라인 추론을 수행하기 위한 기본 Python 인터페이스를
제공합니다. 여기서 오프라인 추론이란 별도의 모델 추론 서버를 사용하지 않고
모델과 직접 상호작용하는 방식을 의미합니다.

다음은 `LLM` 클래스 사용 예시입니다.

??? code

    ```python
    from vllm import LLM, SamplingParams

    # Define a list of input prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The largest ocean is",
    ]

    # Define sampling parameters
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Initialize the LLM engine with the OPT-125M model
    llm = LLM(model="facebook/opt-125m")

    # Generate outputs for the input prompts
    outputs = llm.generate(prompts, sampling_params)

    # Print the generated outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```

더 자세한 API 정보는 API 문서의 [Offline Inference](../api/README.md#offline-inference)
섹션에서 확인할 수 있습니다.

`LLM` 클래스 구현 코드는 [vllm/entrypoints/llm.py](../../vllm/entrypoints/llm.py)에 있습니다.

### OpenAI 호환 API 서버

vLLM의 두 번째 주요 인터페이스는 OpenAI 호환 API 서버입니다.
이 서버는 `vllm serve` 명령으로 시작할 수 있습니다.

```bash
vllm serve <model>
```

`vllm` CLI 코드는 [vllm/entrypoints/cli/main.py](../../vllm/entrypoints/cli/main.py)에서 볼 수 있습니다.

경우에 따라 `vllm` CLI 대신 API 서버 엔트리포인트를 직접 실행하는 예를 볼 수 있습니다.
예를 들면 다음과 같습니다.

```bash
python -m vllm.entrypoints.openai.api_server --model <model>
```

!!! warning

    `python -m vllm.entrypoints.openai.api_server`는 더 이상 권장되지 않으며,
    향후 릴리스에서 지원이 중단될 수 있습니다.

해당 코드는 [vllm/entrypoints/openai/api_server.py](../../vllm/entrypoints/openai/api_server.py)에 있습니다.

API 서버에 대한 자세한 내용은
[OpenAI-Compatible Server](../serving/openai_compatible_server.md) 문서를 참고하세요.

## V1 프로세스 아키텍처

vLLM V1은 관심사를 분리하고 처리량을 극대화하기 위해 멀티 프로세스 아키텍처를 사용합니다.
이 구조를 이해하는 것은 배포 환경에서 CPU 리소스를 적절히 산정하는 데 중요합니다.
핵심 프로세스는 다음과 같습니다.

### API 서버 프로세스

API 서버 프로세스는 HTTP 요청(OpenAI 호환 API 등)을 처리하고,
입력 전처리(토크나이징, 멀티모달 데이터 로딩)를 수행하며,
결과를 클라이언트로 스트리밍합니다. 엔진 코어 프로세스와는 ZMQ 소켓으로 통신합니다.

기본값은 **API 서버 프로세스 1개**이지만,
데이터 병렬성을 사용하면 API 서버 수가 데이터 병렬 크기에 맞춰 자동으로 확장됩니다.
또한 `--api-server-count` 플래그로 수동 설정도 가능합니다.
각 API 서버는 many-to-many 토폴로지로 **모든** 엔진 코어와 ZMQ 연결을 맺기 때문에,
어떤 API 서버든 어떤 엔진 코어로든 요청을 라우팅할 수 있습니다.
각 API 서버 프로세스는 미디어 로딩을 위해 여러 CPU 스레드를 사용하며
(`VLLM_MEDIA_LOADING_THREAD_COUNT`로 제어, 기본값 8), 이 값은 환경변수로 조정할 수 있습니다.

관련 코드는 [vllm/entrypoints/openai/api_server.py](../../vllm/entrypoints/openai/api_server.py)와
[vllm/v1/utils.py](../../vllm/v1/utils.py)에 있습니다.

### 엔진 코어 프로세스

엔진 코어 프로세스는 스케줄러를 실행하고, KV 캐시를 관리하며,
GPU 워커 전반의 모델 실행을 조율합니다.
지속적으로 요청을 스케줄링하고 GPU 워커에 작업을 디스패치하는 busy loop를 실행합니다.

**데이터 병렬 rank당 엔진 코어 프로세스 1개**가 존재합니다.
예를 들어 `--data-parallel-size 4`라면 엔진 코어 프로세스는 4개입니다.

관련 코드는 [vllm/v1/engine/core.py](../../vllm/v1/engine/core.py)와
[vllm/v1/engine/utils.py](../../vllm/v1/engine/utils.py)에 있습니다.

### GPU 워커 프로세스

각 GPU는 전용 워커 프로세스가 관리합니다.
워커 프로세스는 모델 가중치를 로드하고, forward pass를 실행하며,
GPU 메모리를 관리합니다. 워커는 자신을 소유한 엔진 코어 프로세스와 통신합니다.

**GPU당 워커 프로세스 1개**가 존재합니다.
GPU 워커 총수는 엔진 코어당 `tensor_parallel_size x pipeline_parallel_size`와 같습니다.

관련 코드는 [vllm/v1/executor/multiproc_executor.py](../../vllm/v1/executor/multiproc_executor.py)와
[vllm/v1/worker/gpu_worker.py](../../vllm/v1/worker/gpu_worker.py)에 있습니다.

### DP 코디네이터 프로세스(조건부)

데이터 병렬성(`--data-parallel-size > 1`)을 사용하면,
추가 코디네이터 프로세스가 DP rank 간 로드 밸런싱을 관리하고
MoE 모델의 동기화된 forward pass를 조율합니다.

**DP 코디네이터 프로세스는 1개**이며,
데이터 병렬성이 활성화된 경우에만 생성됩니다.

관련 코드는 [vllm/v1/engine/coordinator.py](../../vllm/v1/engine/coordinator.py)에 있습니다.

### 프로세스 수 요약

`N`개의 GPU, 텐서 병렬 크기 `TP`, 데이터 병렬 크기 `DP`, API 서버 수 `A`인 배포를 가정하면:

| 프로세스 유형 | 개수 | 설명 |
|---|---|---|
| API 서버 | `A` (기본값 `DP`) | HTTP 요청 처리 및 입력 전처리 |
| 엔진 코어 | `DP` (기본값 1) | 스케줄러 및 KV 캐시 관리 |
| GPU 워커 | `N` (= `DP x TP`) | GPU당 1개, 모델 forward 실행 |
| DP 코디네이터 | `DP > 1`이면 1, 아니면 0 | DP rank 간 로드 밸런싱 |
| **총합** | **`A + DP + N` (+ `DP > 1`이면 1 추가)** | |

예를 들어 4 GPU 단일 노드 배포(`vllm serve -tp=4`)는 다음과 같습니다.

- API 서버 1개 + 엔진 코어 1개 + GPU 워커 4개 = **총 6개 프로세스**

<figure markdown="1">
![V1 Process Architecture - TP=4](../assets/design/arch_overview/v1_process_architecture_tp4.png)
</figure>

8 GPU 데이터 병렬 배포(`vllm serve -tp=2 -dp=4`)는 다음과 같습니다.

- API 서버 4개 + 엔진 코어 4개 + GPU 워커 8개 + DP 코디네이터 1개 = **총 17개 프로세스**

<figure markdown="1">
![V1 Process Architecture - TP=2, DP=4](../assets/design/arch_overview/v1_process_architecture_tp2_dp4.png)
</figure>

CPU 리소스 산정 권장 사항은
[CPU Resources for GPU Deployments](../configuration/optimization.md#cpu-resources-for-gpu-deployments)
문서를 참고하세요.

## LLM 엔진

`LLMEngine`와 `AsyncLLMEngine` 클래스는 vLLM 시스템 동작의 중심이며,
모델 추론과 비동기 요청 처리를 담당합니다.

![LLMEngine Diagram](../assets/design/arch_overview/llm_engine.excalidraw.png)

### LLMEngine

`LLMEngine` 클래스는 vLLM 엔진의 핵심 구성요소입니다.
클라이언트 요청을 받아 모델 출력으로 변환하는 역할을 하며,
입력 처리, 모델 실행(여러 호스트/GPU에 분산될 수 있음), 스케줄링,
출력 처리를 포함합니다.

- **입력 처리(Input Processing)**: 지정된 토크나이저를 사용해 입력 텍스트를 토큰화합니다.
- **스케줄링(Scheduling)**: 각 step에서 처리할 요청을 선택합니다.
- **모델 실행(Model Execution)**: 다중 GPU 분산 실행을 포함해 언어 모델 실행을 관리합니다.
- **출력 처리(Output Processing)**: 모델이 생성한 출력의 토큰 ID를 사람이 읽을 수 있는 텍스트로 디코딩합니다.

`LLMEngine` 코드는 [vllm/engine/llm_engine.py](../../vllm/engine/llm_engine.py)에 있습니다.

### AsyncLLMEngine

`AsyncLLMEngine` 클래스는 `LLMEngine`의 비동기 래퍼입니다.
`asyncio`를 사용해 들어오는 요청을 지속적으로 처리하는 백그라운드 루프를 만들며,
온라인 서빙 시나리오에 맞게 설계되어 다중 동시 요청 처리와 결과 스트리밍을 지원합니다.

OpenAI 호환 API 서버는 `AsyncLLMEngine`을 사용합니다.
더 단순한 예시로 [vllm/entrypoints/api_server.py](../../vllm/entrypoints/api_server.py)에
데모 API 서버도 제공됩니다.

`AsyncLLMEngine` 코드는 [vllm/engine/async_llm_engine.py](../../vllm/engine/async_llm_engine.py)에 있습니다.

## Worker

워커는 모델 추론을 실행하는 프로세스입니다.
vLLM은 GPU 같은 가속기 장치 하나를 프로세스 하나가 제어하는 일반적인 방식을 따릅니다.
예를 들어 텐서 병렬 2, 파이프라인 병렬 2를 사용하면 총 워커는 4개가 됩니다.
워커는 `rank`와 `local_rank`로 식별됩니다.
`rank`는 전역 오케스트레이션에 사용되고,
`local_rank`는 주로 가속기 할당과 파일 시스템/공유 메모리 같은 로컬 리소스 접근에 사용됩니다.

## 모델 러너

각 워커는 모델 로드와 실행을 담당하는 모델 러너 객체를 하나씩 가집니다.
입력 텐서 준비, cudagraph 캡처 등 모델 실행 로직의 상당 부분이 여기에 있습니다.

## 모델

각 모델 러너 객체는 실제 `torch.nn.Module` 인스턴스인 모델 객체를 하나 가집니다.
최종적으로 어떤 클래스가 선택되는지는
[huggingface_integration](huggingface_integration.md) 문서를 참고하세요.

## 클래스 계층

아래 그림은 vLLM의 클래스 계층을 보여줍니다.

![Class Hierarchy](../assets/design/hierarchy.png)

이 클래스 계층에는 몇 가지 중요한 설계 선택이 반영되어 있습니다.

1\. **확장성(Extensibility)**: 계층의 모든 클래스는 필요한 정보를 담은
설정 객체를 받습니다.
[VllmConfig](https://github.com/vllm-project/vllm/blob/d1c6799b8870e513bf4f2305cbf6cda9fc3d773b/vllm/config.py#L2036)
클래스가 중심 설정 객체로 전달됩니다.
계층이 깊기 때문에 각 클래스는 자신이 관심 있는 설정을 읽어야 합니다.
모든 설정을 하나의 객체로 캡슐화하면 설정 객체를 쉽게 전달하고 필요한 값을 꺼내 쓸 수 있습니다.
예를 들어(LLM 추론 분야가 매우 빠르게 변하기 때문에 흔한 일입니다)
모델 러너에만 영향을 주는 새 기능을 추가한다고 가정해 보겠습니다.
이 경우 `VllmConfig`에 새 옵션을 추가하면,
전체 설정 객체가 전달되므로 엔진/워커/모델 클래스 생성자를 바꾸지 않고도
모델 러너에서 바로 해당 옵션을 사용할 수 있습니다.

2\. **일관성(Uniformity)**: 모델 러너는 모델을 생성/초기화할 때 일관된 인터페이스가 필요합니다.
vLLM은 50개가 넘는 인기 오픈소스 모델 유형을 지원하며,
각 모델은 고유한 초기화 로직을 가집니다.
모델마다 생성자 시그니처가 다르면 모델 러너는 복잡하고 오류 가능성이 큰
검사 로직 없이는 어떤 방식으로 생성자를 호출해야 하는지 알 수 없습니다.
모델 클래스 생성자를 통일하면 모델 러너가 모델 타입 세부사항을 몰라도
쉽게 모델을 만들고 초기화할 수 있습니다.
이는 모델 조합에도 유용합니다.
비전-언어 모델은 비전 모델과 언어 모델을 함께 구성하는 경우가 많은데,
생성자 형식을 통일하면 두 모델을 쉽게 만들고 조합할 수 있습니다.

!!! note
    이 변경을 지원하기 위해 모든 vLLM 모델 시그니처가 다음과 같이 업데이트되었습니다.

    ```python
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
    ```

    잘못된 인자가 실수로 전달되는 것을 막기 위해 생성자는 keyword-only로 바뀌었습니다.
    따라서 이전 방식의 설정이 전달되면 생성자가 오류를 발생시킵니다.
    vLLM 내부 모델은 이미 모두 이 변경을 반영했습니다.
    트리 밖(out-of-tree)에서 등록한 모델은 개발자가 직접 업데이트해야 하며,
    예를 들어 아래처럼 shim 코드를 추가해 이전 생성자 시그니처를 새 시그니처에 맞출 수 있습니다.

    ??? code

        ```python
        class MyOldModel(nn.Module):
            def __init__(
                self,
                config,
                cache_config: Optional[CacheConfig] = None,
                quant_config: Optional[QuantizationConfig] = None,
                lora_config: Optional[LoRAConfig] = None,
                prefix: str = "",
            ) -> None:
                ...

        from vllm.config import VllmConfig
        class MyNewModel(MyOldModel):
            def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
                config = vllm_config.model_config.hf_config
                cache_config = vllm_config.cache_config
                quant_config = vllm_config.quant_config
                lora_config = vllm_config.lora_config
                super().__init__(config, cache_config, quant_config, lora_config, prefix)

        from packaging import version
        if version.parse(__version__) >= version.parse("0.6.4"):
            MyModel = MyNewModel
        else:
            MyModel = MyOldModel
        ```

    이렇게 하면 모델은 vLLM의 구버전과 신버전 모두에서 동작할 수 있습니다.

3\. **초기화 시점의 샤딩 및 양자화(Sharding and Quantization at Initialization)**:
특정 기능은 모델 가중치 변경이 필요합니다.
예를 들어 텐서 병렬은 가중치 샤딩이 필요하고,
양자화는 가중치 양자화가 필요합니다.
이 기능을 구현하는 방법은 크게 두 가지입니다.
하나는 모델 초기화 후 가중치를 바꾸는 방법이고,
다른 하나는 모델 초기화 중에 가중치를 바꾸는 방법입니다.
vLLM은 후자를 선택했습니다.
전자는 대규모 모델에 확장성이 좋지 않습니다.
예를 들어 405B 모델(약 810GB 가중치)을
H100 80GB GPU 16개에서 실행한다고 가정해 보겠습니다.
이상적으로는 각 GPU가 50GB만 로드해야 합니다.
초기화 후 가중치를 바꾸는 방식이면,
각 GPU가 먼저 전체 810GB를 로드한 뒤 샤딩해야 하므로 메모리 오버헤드가 매우 커집니다.
반면 초기화 중 샤딩하면 각 레이어는 필요한 샤드만 생성하므로
메모리 오버헤드가 훨씬 작아집니다.
양자화도 같은 원리가 적용됩니다.
또한 모델 생성자에 `prefix` 인자를 추가해,
모델이 prefix에 따라 다른 방식으로 초기화되도록 했습니다.
이는 모델의 서로 다른 부분을 다르게 양자화하는 비균일 양자화에서 특히 유용합니다.
`prefix`는 보통 최상위 모델에서는 빈 문자열이고,
서브모델에서는 `"vision"` 또는 `"language"` 같은 문자열입니다.
일반적으로 체크포인트 파일의 state dict에서 해당 모듈 이름과 일치합니다.

이 설계의 단점 중 하나는 vLLM 개별 컴포넌트 유닛 테스트가 어려워질 수 있다는 점입니다.
모든 컴포넌트가 완전한 설정 객체로 초기화되어야 하기 때문입니다.
이를 해결하기 위해 모든 필드를 `None`으로 둔 기본 설정 객체를 만드는
기본 초기화 함수를 제공합니다.
테스트 대상 컴포넌트가 설정 객체의 일부 필드만 필요하다면,
기본 설정 객체를 만든 뒤 필요한 필드만 설정해 격리 테스트를 수행할 수 있습니다.
또한 vLLM 테스트 중 다수는 전체 시스템을 검증하는 end-to-end 테스트이므로,
이 점이 큰 문제는 아닙니다.

요약하면 완전한 설정 객체 `VllmConfig`는
모든 vLLM 클래스가 공유하는 엔진 레벨의 전역 상태로 볼 수 있습니다.
