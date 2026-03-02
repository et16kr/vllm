# CUDA Graphs

이 문서는 기존 [torch.compile 통합](torch_compile.md)을 넘어 vLLM v1에 도입된 새로운 CUDA Graphs 모드를 소개합니다. 요약하면 다음을 수행했습니다.

1. 유연한 `cudagraph_mode` 설정 추가
2. full CUDA Graphs 지원을 컴파일과 직교(orthogonal)하도록 분리
3. 배치별로 원하는 런타임 모드와 CUDA Graph를 자동 선택하는 중앙 제어기 CUDA Graphs dispatcher 도입

이 문서에서는 다음을 다룹니다.

* 동기(Motivation)
* CUDA Graphs 모드
* 상세 설계
* CUDA Graphs 모드별 사용 예시

!!! note
    이 문서에서 pure decode(`max_query_len=1`) 또는 speculative decode(`max_query_len =1+num_spec_tokens`) 배치를 **uniform decode** 배치라고 부르며, 그 반대(즉 prefill 또는 mixed prefill-decode 배치)는 **non-uniform** 배치라고 부릅니다.

!!! note
    아래 내용은 대부분 <https://github.com/vllm-project/vllm/pull/20059>의 마지막 커밋을 기반으로 합니다.

## 동기(Motivation)

초기 piecewise 컴파일은 cudagraph 미지원 연산(주로 attention)을 제외하고 piecewise cudagraph 캡처를 가능하게 하려는 목적에서 시작되었습니다. 덕분에 모든 attention 백엔드 호환성을 유지하면서도 cudagraph 가속 이점을 일부 얻을 수 있었습니다. 이후 attention이 cudagraph를 지원하는 경우 지연 시간을 더 줄이기 위해, piecewise 컴파일을 하지 않는 "full cudagraphs" 지원도 추가했습니다. 그러나 컴파일과 cudagraph 캡처가 강하게 결합되면서 유연성이 낮은 all-or-nothing 경험이 되었습니다. 많은 attention 백엔드가 통합된 "full" CUDA Graphs 캡처에 아직 준비되지 않았고(예: 현재는 FlashAttention 3만 지원), 또는 pure decode 배치에서만 CUDA Graphs를 지원했습니다(예: Flashinfer, FlashMLA, Mamba 등). 그 결과 성능/호환성 트레이드오프가 혼란스러워지고, CUDA Graphs 지원이 일관되지 않으며, 코드 구조 복잡도도 증가했습니다.

이에 따라 다음 기능을 가진 더 세밀한 CUDA Graphs 해법이 필요해졌습니다.

* prefill/mixed 배치와 (uniform-)decode 배치를 명시적으로 구분해 별도 캡처
* CUDAGraph 캡처 로직을 컴파일 로직에서 최대한 분리(기능 직교성), 즉:
    * 동일한 컴파일 그래프를 사용해 piecewise와 full cudagraph를 모두 캡처
    * 컴파일 없이 full cudagraph 캡처 지원
* 배치 구성에 따라 런타임에서 full/piecewise cudagraph 간 디스패치
* CUDAGraph 동작을 중앙 제어해 코드 복잡도를 낮추고 확장성 확보

이 기능들은 시작 시간/성능 트레이드오프와 기능 지원 조합을 더 유연하게 선택할 수 있게 해줍니다.

## `CudagraphModes`

[CUDAGraphMode][vllm.config.compilation.CUDAGraphMode]는 `CompilationConfig.cudagraph_mode`에서 조정하는 단일 노브입니다.

* `NONE` - CUDA Graphs를 끕니다. 디버깅에 적합합니다.
* `PIECEWISE` - 단일 모드 전략(과거 기본값). 가장 유연합니다. attention 등 CUDA Graphs 비호환 연산은 eager로 남기고, 나머지는 CUDA Graphs로 실행합니다. piecewise 컴파일이 필요합니다.
* `FULL` - 단일 모드 전략. non-uniform 배치에 대해서만 full CUDA Graphs를 캡처하고, uniform-decode 배치는 동일 batch_size의 non-uniform 배치 CUDA Graph를 재사용합니다(호환되기 때문). 작은 모델 또는 짧은 프롬프트 워크로드에 유리할 수 있습니다.
* `FULL_DECODE_ONLY` - uniform decode에만 full CUDA Graph 사용, prefill/mixed 등에는 cudagraph 미사용. P/D 구성에서 prefill보다 decode가 중요한 decode 인스턴스에 적합하며, 이 경우 `PIECEWISE` CUDA Graphs에 필요한 메모리를 절약할 수 있습니다.
* `FULL_AND_PIECEWISE` - (기본 모드) uniform decode는 full CUDA Graph, 나머지는 piecewise CUDA Graph를 사용. 일반적으로 가장 높은 성능(특히 작은 모델/MoE 저지연)에 유리하지만 메모리 사용량이 가장 크고 캡처 시간도 가장 깁니다.

기본값: v1 + piecewise 컴파일 가능 시 성능 향상을 위해 기본값은 `FULL_AND_PIECEWISE`입니다(풀링 모델은 여전히 `PIECEWISE`). 그 외(예: piecewise 컴파일 불가)에는 기본값이 `NONE`입니다.

`NONE`, `PIECEWISE`, `FULL`은 단일 모드 설정으로 각각 과거 eager 실행, piecewise CUDA Graphs, full CUDA Graphs 구현과 단순 동등합니다. 반면 `FULL_DECODE_ONLY`, `FULL_AND_PIECEWISE`는 새로 추가된 이중 모드 설정이며, 런타임 배치에 따라 구체 런타임 모드를 동적으로 전환하는 디스패치가 필요합니다.

!!! note
    여기서 단일 모드 `NONE`, `PIECEWISE`, `FULL`은 CUDA Graphs 디스패치의 런타임 모드로 취급합니다. 이중 모드를 사용하면 dispatcher는 배치 구성에 따라 항상 구성원 모드 중 하나(적절한 그래프가 없으면 `NONE` 포함)로 디스패치합니다.

cascade attention은 cudagraph 호환은 아니지만, 이제 가능한 모든 cudagraph 모드 설정과는 호환됩니다. 배치가 cascade attention을 사용하면, 가능한 경우 항상 `PIECEWISE` 모드로 디스패치되고(없으면 `NONE`) 실행됩니다.

!!! note
    모든 CUDA Graph 모드가 모든 attention 백엔드와 호환되는 것은 아닙니다. 지원 가능한 가장 가까운 모드로 자동 "다운그레이드"합니다. 예를 들어 백엔드가 pure decode/uniform 배치에서만 CUDA Graphs를 지원하면, piecewise 컴파일이 켜져 있을 때 `FULL`은 `FULL_AND_PIECEWISE`로, 아니면 `FULL_DECODE_ONLY`로 변환됩니다.

## 상세 설계

### 개요

새 CUDA Graphs 로직은 piecewise 컴파일 위에 구축되며, 이중 CUDA Graphs 런타임 모드 전환을 지원합니다. 시스템 핵심 컴포넌트는 다음과 같습니다.

* [CUDAGraphWrapper][vllm.compilation.cuda_graph.CUDAGraphWrapper]: 감싼 callable에 대해 CUDAGraph 캡처/재생을 처리하는 래퍼
* [CudagraphDispatcher][vllm.v1.cudagraph_dispatcher.CudagraphDispatcher]: CUDA Graphs의 단일 진실 공급원(single source of truth)으로 동작하며 디스패치를 담당하는 중앙 제어기
* [CUDAGraphMode][vllm.config.compilation.CUDAGraphMode]: 지원/런타임 모드를 나타내는 enum
* [BatchDescriptor][vllm.forward_context.BatchDescriptor]: 디스패치에 쓰이는 런타임 배치의 고유 표현

아래 그림은 inductor 컴파일과 함께 CUDA Graphs를 사용하는 과거/현재 설계를 비교합니다. 과거에는 CUDA Graphs 로직과 컴파일 로직이 vLLM `PiecewiseBackend`에 강결합되어 있었고, CUDA Graphs가 사실상 `batch_size` 기준으로 암묵적으로 디스패치되었습니다. 현재는 CUDA Graphs 로직이 `CUDAGraphWrapper` 클래스로 분리되어 full/piecewise 기능을 모두 담당하고, 디스패치는 **런타임 모드**와 **BatchDescriptor**를 **디스패치 키**로 사용해 `CudagraphDispatcher`에서 **명시적**으로 수행됩니다.

**Before:**

![previous_design](../assets/design/cuda_graphs/previous_design.png)

**After:**

![new_design](../assets/design/cuda_graphs/current_design.png)

### `BatchDescriptor`

[BatchDescriptor][vllm.forward_context.BatchDescriptor]는 `ForwardContext` 내부에서 CUDA Graphs 런타임 모드와 함께 동작하며, 런타임 디스패치 키의 핵심 구조입니다. 원형은 다음과 같습니다.

```python
class BatchDescriptor(NamedTuple):
    num_tokens: int
    num_reqs: int
    uniform: bool = False
    has_lora: bool = False
```

여기서 `num_tokens`는 패딩된 토큰 길이일 수 있고, `uniform`은 모든 요청이 동일한 query 길이를 가지는지 나타냅니다. 많은 attention 백엔드는 배치가 uniform할 때만 full cudagraph를 지원합니다. pure decode 배치는 uniform이지만 query 길이가 1이 아닐 수도 있습니다(즉 `num_tokens == num_reqs`가 아님). 이는 spec-decode 검증 단계에서 "decode" 배치의 query 길이가 `1+num_spec_tokens`가 되는 경우입니다.

이 구조의 목표는 CUDA Graph 항목에 대응되는 최소 정보로 (패딩된) 배치를 유일하게 식별하는 것입니다.

!!! note
    향후 더 일반적인 상황을 위해 `BatchDescriptor` 원형이 확장될 수 있습니다. 예를 들어 `uniform_query_len` 같은 항목을 추가해 여러 uniform decode 길이 설정을 지원하거나(<https://github.com/vllm-project/vllm/pull/23679>), 또는 입력이 토큰 길이 중심이 아닌 모델(예: 일부 멀티모달 입력)의 CUDA Graphs 지원을 위해 다른 변경이 필요할 수 있습니다.

### `CudagraphDispatcher`

[CudagraphDispatcher][vllm.v1.cudagraph_dispatcher.CudagraphDispatcher]는 `FULL` 런타임 모드용 유효 디스패치 키 집합과 `PIECEWISE` 런타임 모드용 집합, 총 두 집합을 유지하고, 모델 forward 실행 전에 올바른 런타임 모드와 디스패치 키를 선택합니다. 초기 키(패딩 입력에 대한 대략적 batch_descriptor)를 받아 최종 런타임 모드와 최종 batch_descriptor를 반환한 뒤, forward context를 통해 해당 결정을 `CUDAGraphWrapper` 인스턴스들에 전달합니다. `CudagraphDispatcher`가 사용 가능한 CUDA Graph 키의 유일한 진실 공급원이며, wrapper 인스턴스는 forward context의 결정을 신뢰하기만 하면 됩니다. 이로써 wrapper 코드가 단순해지고 로직이 dispatcher에 중앙화됩니다.

디스패치 키는 dispatcher's `initialize_cudagraph_keys` 메서드에서 초기화되며, 이는 가능한 attention 백엔드 초기화가 완료된 뒤 gpu_model_runner가 호출합니다. 향후 여기에서 다양한 CUDA Graph 조합을 "준비"하도록 더 고도화할 수 있습니다. 현재는 compilation config의 `cudagraph_mode`의 `decode_mode`/`mixed_mode` 유효 조합과 `cudagraph_capture_sizes`를 기준으로 가능한 키를 추가합니다.

디스패치 코드는 다음과 같습니다.

```python
batch_descriptor=BatchDescriptor(num_tokens=num_input_tokens, uniform_decode=...)
runtime_mode, batch_descriptor = cudagraphdispatcher.dispatch(batch_descriptor)
# execution
with set_forward_context(
    ..., 
    cudagraph_runtime_mode=runtime_mode, 
    batch_descriptor=batch_descriptor,
):
     output = self.model(...)
```

`dispatch()` 내부에서는 적절한 CUDA Graphs 런타임 모드와 기존 디스패치 키를 찾아 반환합니다. 우선순위는 기본적으로 `FULL` > `PIECEWISE` > `None`입니다. 해당 키가 없으면 `NONE` 모드(eager 실행)를 반환합니다. 구현은 [여기](https://github.com/vllm-project/vllm/blob/main/vllm/v1/cudagraph_dispatcher.py#L91)에서 볼 수 있습니다.

모델 executor의 런타임 워크플로를 단순화하면 다음 그림과 같습니다.
![executor_runtime](../assets/design/cuda_graphs/executor_runtime.png)

### `CUDAGraphWrapper`

[CUDAGraphWrapper][vllm.compilation.cuda_graph.CUDAGraphWrapper] 인스턴스는 runnable을 감싸고 CUDA Graphs 기능이 추가된 runnable처럼 동작합니다. 각 wrapper 인스턴스는 `runtime_mode` 하나에 바인딩되며(`PIECEWISE` 또는 `FULL`), 캡처/재생 및 passthrough(직접 호출)를 담당합니다. 런타임에서 각 wrapper는 다음을 수행합니다.

1. 전역 forward context에서 runtime_mode와 batch_descriptor(디스패치 키)를 확인
2. runtime_mode가 `NONE`이거나 wrapper의 모드와 맞지 않으면 runnable을 직접 호출
3. 그 외(wrapper 모드와 runtime_mode가 일치)에는 CUDA Graph 캡처(키가 없으면 새 엔트리 생성 후 캐시) 또는 재생(키가 있으면 캐시 사용) 수행

위 단계는 wrapper가 forward context(dispatcher가 제어)의 내용을 그대로 신뢰한다는 가정에 기반합니다. 이 방식은 로직 단순화/중앙화로 복잡도와 상태 불일치 위험을 줄이고, `FULL`/`PIECEWISE` 두 모드에서 동일 wrapper 클래스를 재사용할 수 있게 합니다. 구현은 [여기](https://github.com/vllm-project/vllm/blob/f751e50b7a2aae3110d83ed0d88202fc91b3e78a/vllm/compilation/cuda_graph.py#L106)를 참고하세요.

#### 중첩 래퍼(Nested Wrapper) 설계

full CUDA Graphs와 piecewise CUDA Graphs를 공존 가능하게 만드는 핵심 메커니즘은 중첩 CUDA Graphs wrapper 설계입니다. 단일 piecewise FX 그래프를 사용하는 piecewise 컴파일 위에서 동작합니다. full CUDA Graph 기능을 위해 모델 전체 바깥을 `FULL` 모드 wrapper로 감싸고, 동시에 각 piecewise backend는 컴파일 내부에서 `PIECEWISE` 모드 wrapper로 감쌉니다.

아래 플로우 차트가 동작 방식을 보여줍니다.
![wrapper_flow](../assets/design/cuda_graphs/wrapper_flow.png)

따라서 `FULL` 런타임 모드에서는 piecewise wrapper가 활성화되지 않으므로 full CUDA Graph 캡처/재생이 안전합니다. `PIECEWISE` 모드도 유사하게 `FULL` 모드 wrapper와 충돌하지 않습니다. `NONE` 모드에서는 `FULL`/`PIECEWISE` wrapper 모두 비활성화되어 eager 실행으로 그대로 내려갑니다.

### Full CUDA Graph 캡처와 워밍업

CUDA Graph 캡처는 runner가 비-`NONE` 런타임 모드로 `_dummy_run`을 통해 처음 모델 forward를 호출할 때 발생합니다. full CUDA Graph 캡처에서는 attention metadata를 적절히 설정해 attention 백엔드가 원하는 커널 루틴을 실행하도록 하여, prefill/mixed 배치와 uniform_decode 배치를 명시적으로 각각 캡처합니다. 두 케이스를 구분하는 가장 중요한 속성은 attn_metadata의 `max_query_len`입니다(대부분의 attention 백엔드에서 해당). uniform_decode의 경우 원하는 `uniform_query_len`로 설정하고, non-uniform_decode 배치의 경우 `num_tokens`로 설정합니다.

CUDA Graph wrapper는 더 이상 warm-up 로직을 관리하지 않습니다. warm-up 과정은 이제 GPU model runner가 직접 제어하며, warm-up eager 실행을 위해 `NONE` 런타임 모드를 사용합니다. full CUDA Graph를 워밍업할 때는 warmup `dummy_run` 호출에서 attention이 명시적으로 실행되도록 하는 것도 중요합니다.

## Attention 백엔드의 CUDA Graphs 호환성

attention 백엔드의 CUDA Graphs 호환성을 나타내기 위해 [AttentionCGSupport][vllm.v1.attention.backend.AttentionCGSupport] enum을 도입했습니다. 이 enum은 attention 백엔드의 CUDA Graphs 지원 능력을 나타내며, 값은 능력 순서(`ALWAYS` > `UNIFORM_BATCH` > `UNIFORM_SINGLE_TOKEN_DECODE` > `NEVER`)로 정렬됩니다.

```python
class AttentionCGSupport(enum.Enum):
    """ Constants for the CUDA Graphs support of the attention backend
    Here we do not consider the cascade attention, as currently
    it is never CUDA Graphs supported."""

    ALWAYS = 3
    """CUDA Graphs always supported; supports mixed-prefill-decode"""
    UNIFORM_BATCH = 2
    """CUDA Graphs supported for batches the only contain query lengths that are
    the same, this can be used for spec-decode 
        i.e. "decodes" are 1 + num_speculative_tokens"""
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    """CUDA Graphs supported for batches the only contain query_len==1 decodes"""
    NEVER = 0
    """NO CUDA Graphs support"""
```

하이브리드 attention 백엔드(예: mamba mixer 모델)가 있는 경우, 모든 백엔드 중 최소 capability를 모델의 최종 capability로 사용하며, 필요하면 비호환 CUDA Graphs 모드를 가장 적합한 모드로 다운그레이드합니다. 예를 들어 최소 capability가 `UNIFORM_BATCH`면 `FULL`을 `FULL_AND_PIECEWISE`로, `NEVER`면(-O3 컴파일 모드 기준) `PIECEWISE`로 내립니다. 전체 fallback 정책은 [이 코드][vllm.v1.worker.gpu_model_runner.GPUModelRunner._check_and_update_cudagraph_mode]를 참고하세요.

작성 시점 기준 full CUDA Graphs를 지원하는 백엔드는 다음과 같습니다.

| Attention Backend | cudagraph_support | Comments |
|:---|:---|:---|
| FlashAttention v2 | `UNIFORM_BATCH` | 실제로는 `ALWAYS`이나 성능 이유로 `FULL_AND_PIECEWISE`로 폴백하는 우회 적용 |
| FlashAttention v3 | `ALWAYS` | 두 배치 유형 모두에 대한 통합 루틴을 가져 `FULL` 모드가 유리 |
| Triton Attention | `ALWAYS` | prefill/mixed와 pure decode 배치에 서로 다른 커널이 있어 `FULL_AND_PIECEWISE` 선호 |
| AITER FlashAttention | `UNIFORM_BATCH`| |
| FlashInfer | `UNIFORM_SINGLE_TOKEN_DECODE` | Blackwell에서 TRTLLM attention 사용 시 `UNIFORM_BATCH`로 설정 예정 |
| FlashMLA | `UNIFORM_BATCH` | |
| FlashInferMLA | `UNIFORM_BATCH` | |
| FlashInferMLASparse | `UNIFORM_BATCH` | |
| AITER MLA | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| CUTLASS MLA | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| Mamba attention| `UNIFORM_SINGLE_TOKEN_DECODE` | |

표에 없는 백엔드는 모두 `NEVER`로 선언됩니다.

## 사용 가이드

현재 CLI에서는 compilation_config의 `cudagraph_mode`에 대문자 문자열을 직접 사용합니다: `--compilation-config '{"cudagraph_mode": "..."}'`. 여기서 `...`에는 `NONE`, `PIECEWISE`, `FULL`, `FULL_DECODE_ONLY`, `FULL_AND_PIECEWISE` 중 하나를 넣습니다. `PIECEWISE` 관련 모드는 모두 piecewise 컴파일이 필요하고, `FULL` 관련 모드는 attention 백엔드의 CUDA Graphs 지원이 필요합니다. 예:

```bash
vllm serve --model meta-llama/Llama-3.1-8B-Instruct --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
```

### Python 예시

```python
import os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")

import vllm
from vllm.config import CUDAGraphMode

compilation_config = {"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}
model = vllm.LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",
    compilation_config=compilation_config,
)
sampling_params = vllm.SamplingParams(
    temperature=0,  # greedy decoding
    max_tokens=1024,
)
outputs = model.generate(
    ["My name is John and"],
    sampling_params=sampling_params,
)
```

### Piecewise 컴파일과 full graph 커스텀 패스(attention fusion, sequence parallelism)

아쉽게도 일부 커스텀 compile pass는 효과를 내려면 전체 그래프를 봐야 하므로 piecewise 컴파일과 호환되지 않습니다. 여기에는 `AttnFusionPass`, `SequenceParallelismPass`가 포함됩니다. 단기 해법으로 attention fusion이 활성화되면 piecewise 컴파일을 자동 비활성화(`splitting_ops=[]`)합니다. 이때 CUDA Graph 모드는 백엔드 지원에 따라 `FULL` 또는 `FULL_DECODE_ONLY`를 사용합니다. 다만 이 방식도 다른 최적화와의 비호환 및 성능 트레이드오프 혼란을 초래합니다.

장기적으로는 Dynamo 직후가 아니라 Inductor에서 그래프를 분할할 수 있는 기능을 추가했습니다. `CompilationConfig.use_inductor_graph_partition=True`로 활성화할 수 있지만, 현재는 실험적이며 `torch>=2.9`에서만 사용 가능합니다. 이 방식은 전체 그래프를 컴파일해야 하고 piecewise 컴파일 산출물을 재사용할 수 없어 컴파일 시간이 늘어납니다. vLLM이 2.9를 지원하면 piecewise cudagraph 캡처 가속 효과도 있어 이를 기본 접근으로 전환할 계획입니다.

## 성능 관련

예시는 아래 링크를 참고하세요.

* [20059#issuecomment-3160858458](https://github.com/vllm-project/vllm/pull/20059#issuecomment-3160858458)
* [20059#issuecomment-3188735226](https://github.com/vllm-project/vllm/pull/20059#issuecomment-3188735226)
* [20059#issuecomment-3219888738](https://github.com/vllm-project/vllm/pull/20059#issuecomment-3219888738)
