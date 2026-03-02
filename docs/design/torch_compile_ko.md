# `torch.compile` 통합

vLLM의 V1 아키텍처에서는 `torch.compile`이 기본 활성화되어 있으며, 프레임워크의 핵심 요소입니다. 이 문서는 `torch.compile` 사용 방식을 이해할 수 있도록 간단한 워크스루 예시를 제공합니다.

예시 전체에서 일반적인 Llama 모델을 실행하고, 세부 동작을 모두 보기 위해 debug 레벨 로깅을 켭니다. 사용할 명령은 `VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Llama-3.2-1B`입니다.

!!! note
    `torch.compile` 통합의 최신 진행 상황과 추가 정보는 이 [블로그 글](https://blog.vllm.ai/2025/08/20/torch-compile.html)을 참고하세요.

## 컴파일 캐시

아주 자세한 로그를 보면 다음과 같은 메시지를 확인할 수 있습니다.

```console
INFO 03-07 03:06:55 [backends.py:409] Using cache directory: ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0 for vLLM's torch.compile
```

vLLM은 가능한 모든 요소를 고려해 컴파일 산출물(artifact)을 저장할 디렉터리를 결정합니다. 따라서 배포 환경에서 `~/.cache/vllm/torch_compile_cache` 전체를 그대로 복사해 컴파일 시간을 크게 줄이고, vLLM 인스턴스 시작 시간을 단축할 수 있습니다.

고려 요소는 다음과 같습니다.

- 관련 설정 전체(각 설정의 `compute_hash` 함수, [config 폴더](../../vllm/config) 참고)
- PyTorch 설정([compiler_interface.py](../../vllm/compilation/compiler_interface.py)의 `compute_hash` 함수 참고)
- 모델의 `forward` 함수와 `forward`에서 호출되는 관련 함수(아래 설명)

이 요소들을 함께 고려하기 때문에, 일반적으로 캐시 사용의 안전성이 보장되고 예상치 못한 동작을 유발하지 않습니다. 그래서 캐시는 기본 활성화입니다. 컴파일 과정을 디버깅하거나 캐시가 문제 원인이라고 의심된다면 환경변수 `VLLM_DISABLE_COMPILE_CACHE=1`로 캐시를 비활성화할 수 있습니다.

vLLM의 `torch.compile` 통합에서 중요한 특징은, 요청을 받기 전에 컴파일을 모두 끝내도록 보장한다는 점입니다. 어떤 요청도 추가 컴파일을 트리거하지 않습니다. 그렇지 않으면 엔진이 해당 요청에서 블로킹되고 응답 시간에 예기치 않은 스파이크가 생기기 때문입니다.

기본적으로 캐시는 컴파일 산출물을 바이너리로 저장합니다. 디버깅 목적으로 생성된 코드를 직접 보고 싶다면 compilation config에서 `compile_cache_save_format=unpacked`를 설정하거나, 이 값을 생략하고 환경변수 `VLLM_COMPILE_CACHE_SAVE_FORMAT=unpacked`를 설정하면 됩니다.

## 동적 shape와 vLLM guard dropping

`torch.compile`은 필요하면 동적 shape에 대한 guard를 적극적으로 추가하도록 설계되어 있습니다.
이는 많은 guard를 제거(drop)하려는 vLLM의 `torch.compile` 접근과 긴장 관계를 가집니다. 일부 guard는 실제로 중요할 수 있기 때문입니다.

`torch.compile`은 동적 shape를 두 종류로 구분합니다: `backed`, `unbacked`.
`backed` 동적 shape에는 guard가 걸리며, guard가 추가되지 않는다는 보장은 없습니다. 사용자 코드, dynamo, inductor, autograd 모두 guard를 추가할 수 있습니다. 또한 0/1 특수화(0/1 specialization)와 관련해, 해당 범위 분기를 실제로 만나지 않아도 backed 심볼은 0, 1, 혹은 >=2로 무조건 특수화됩니다.

반대로 `unbacked` 동적 shape는 guard가 걸리지 않으며 0/1 특수화도 일어나지 않도록 보장됩니다. 다만 값이 필요한 분기를 만났는데 명시적 unbacked 처리 로직이 없으면 data dependent error(DDE)가 발생할 수 있습니다. 프레임워크는 DDE를 던지기보다 일반 경로를 선택하도록 수렴하고 있습니다. `unbacked`의 단점은 성능 버그 또는 일반 경로 선택으로 인한 최적화 기회 손실, 그리고 고정된 비예시 입력 기반 힌트 사용(곧 override_hint API로 개선 예정)입니다. 일반 경로 선택의 예로 contiguous 여부를 심볼릭하게 증명할 수 없을 때 `contiguous()`/`reshape()`에서 입력이 비연속이라고 가정해 clone이 추가될 수 있습니다.

`backed_size_oblivious`는 unbacked 명시 처리가 있는 위치에서 backed 심볼을 unbacked처럼 취급하도록 하는 플래그입니다. 이 모드에서는 프레임워크 코드에서 0/1 특수화가 대부분 회피되고 기본 0/1 특수화도 발생하지 않습니다. 다만 사용자 코드나 커스텀 패스 등으로 인해 `torch.compile`이 여전히 guard를 추가할 가능성은 있습니다. `backed_size_oblivious`는 PyTorch compile에서 실험적 기능이며 향후 deprecated될 수 있습니다. 그럼에도 `backed`보다 안전한 선택이고, `unbacked` 대비 성능 저하 가능성도 더 낮습니다.

### 동적 shape 설정

`DynamicShapesConfig`의 `type` 필드로 동적 shape 동작을 제어할 수 있습니다.
선택지는 `BACKED`(기본값), `UNBACKED`, `BACKED_SIZE_OBLIVIOUS`입니다.

#### 오프라인 추론 예시(LLM 클래스 사용)

오프라인 추론에서 `LLM` 클래스를 사용할 때는 `compilation_config` 파라미터로 동적 shape를 설정할 수 있습니다.

```python
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, DynamicShapesConfig, DynamicShapesType

# Example: Using backed_size_oblivious (experimental, safer than backed)
llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    compilation_config=CompilationConfig(
        dynamic_shapes_config=DynamicShapesConfig(
            type=DynamicShapesType.BACKED_SIZE_OBLIVIOUS
        )
    )
)

# Example: Using unbacked (strongest guarantee against guards)
llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    compilation_config=CompilationConfig(
        dynamic_shapes_config=DynamicShapesConfig(
            type=DynamicShapesType.UNBACKED
        )
    )
)

# Generate outputs
prompts = ["Hello, my name is", "The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)
```

#### 온라인 서빙 예시(`vllm serve` 사용)

온라인 서빙에서 `vllm serve`를 사용할 때는 `--compilation-config` 플래그로 동적 shape를 설정할 수 있습니다.

```bash
# Example: Using unbacked
vllm serve meta-llama/Llama-3.2-1B \
  --compilation-config '{"dynamic_shapes_config": {"type": "unbacked"}}'


# Alternative: Using dot notation (simpler for single values)
vllm serve meta-llama/Llama-3.2-1B -cc.dynamic_shapes_config.type=unbacked
```

#### 어떤 모드를 선택할까

- **BACKED**(기본값): 최대 성능을 위해 guard의 잠재적으로 안전하지 않은 제거를 감수할 수 있을 때 사용합니다. guard가 비정상적으로 추가된 뒤 무시될 수 있습니다.

- **UNBACKED**: guard에 대한 가장 강한 회피 보장이 필요할 때 사용합니다.
  가장 보수적인 옵션이며 일부 최적화 기회를 놓칠 수 있습니다.

- **BACKED_SIZE_OBLIVIOUS**: guard 회피와 성능의 균형이 필요할 때 사용합니다.
  실험적 모드이며 BACKED보다 안전하지만 UNBACKED만큼 보수적이지는 않습니다.

## Python 코드 컴파일

아주 자세한 로그에서 다음과 같은 내용을 볼 수 있습니다.

??? console "Logs"

      ```text
      DEBUG 03-07 03:06:52 [decorators.py:203] Start compiling function <code object forward at 0x7f08acf40c90, file "xxx/vllm/model_executor/models/llama.py", line 339>

      DEBUG 03-07 03:06:54 [backends.py:370] Traced files (to be considered for compilation cache):
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/_dynamo/polyfills/builtins.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/nn/modules/container.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/nn/modules/module.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/attention/layer.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/distributed/communication_op.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/distributed/parallel_state.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/custom_op.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/activation.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/layernorm.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/linear.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/rotary_embedding.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/vocab_parallel_embedding.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/models/llama.py

      DEBUG 03-07 03:07:07 [backends.py:462] Computation graph saved to ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py
      DEBUG 03-07 03:07:07 [wrapper.py:105] Dynamo transformed code saved to ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/transformed_code.py
      ```

이 로그는 Python 코드 컴파일, 즉 Dynamo의 그래프 캡처 과정에 대한 것입니다. `xxx/vllm/model_executor/models/llama.py:339`의 함수(컴파일 대상 모델의 `forward`)를 trace합니다. `forward` 수행 중 Dynamo가 인라인하는 다른 함수들도 함께 trace되며, 로그처럼 `xxx/torch/nn/modules/module.py`의 PyTorch 함수(`nn.Module`의 속성 접근이 함수 호출을 유발), vLLM의 통신/attention/activation 함수 등이 포함됩니다. trace된 모든 파일은 캐시 디렉터리 결정 시 고려됩니다. 따라서 위 파일들에 코드 변경이 있으면 컴파일 캐시 미스가 발생하고 재컴파일됩니다.

Dynamo 컴파일 결과는 새 함수로 생성되어 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/transformed_code.py`에 저장됩니다. 보통 이 함수는 모듈에서 텐서를 꺼낸 뒤 trace된 computation graph로 전달합니다. computation graph는 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py`에 저장됩니다.

## 연산 그래프 처리

computation graph는 모든 텐서에 대한 shape annotation을 포함합니다. 입력은 input ids, position ids, 모델의 가중치와 버퍼이고, 출력은 최종 hidden state입니다. lm head projection과 sampling 연산은 그래프에 포함되지 않습니다.

그래프 입력 중 대부분은 정적 shape입니다. 모델 가중치/버퍼는 모델 수명 동안 변하지 않기 때문입니다. 오직 input ids와 position ids만 심볼릭 shape(배치마다 변할 수 있는 shape)을 가집니다. 다만 이 둘은 동일한 심볼릭 shape를 공유합니다. 즉 computation graph에서 실제로 변하는 크기는 배치 크기(현재 forward pass에서 처리하는 토큰 수)뿐입니다.

attention 연산은 KV cache와 복잡한 shape 상호작용이 필요해 복잡합니다. 다행히 attention 출력 shape는 attention query 입력 shape와 동일합니다. 따라서 전체 attention 연산을 PyTorch custom op `torch.ops.vllm.unified_attention_with_output`으로 감싸서 Dynamo가 내부 연산을 들여다보지 않게 합니다. 이렇게 하면 attention이 복잡하더라도 Dynamo 관점에서 모델 computation graph를 full-graph로 캡처할 수 있습니다.

computation graph는 `splitting_ops`(보통 attention 연산)를 기준으로 더 잘게 분할됩니다. 그래서 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py` 파일에는 분할된 그래프 조각(submodule)이 다수 보입니다.

- attention 연산 자체가 하나의 submodule
- 한 attention 연산부터 다음 attention 연산 직전까지의 계산 구간이 하나의 submodule

각 submodule은 인덱스로 식별되며 개별 처리됩니다.

## 연산 그래프 컴파일

아주 자세한 로그에서는 다음도 볼 수 있습니다.

```console
DEBUG 03-07 03:52:37 [backends.py:134] store the 0-th graph for shape None from inductor via handle ('fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py')
DEBUG 03-07 03:52:39 [backends.py:134] store the 1-th graph for shape None from inductor via handle ('f7fmlodmf3h3by5iiu2c4zarwoxbg4eytwr3ujdd2jphl4pospfd', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/ly/clyfzxldfsj7ehaluis2mca2omqka4r7mgcedlf6xfjh645nw6k2.py')
...
DEBUG 03-07 03:52:45 [backends.py:134] store the 15-th graph for shape None from inductor via handle ('f7fmlodmf3h3by5iiu2c4zarwoxbg4eytwr3ujdd2jphl4pospfd', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/ly/clyfzxldfsj7ehaluis2mca2omqka4r7mgcedlf6xfjh645nw6k2.py')
DEBUG 03-07 03:52:45 [backends.py:134] store the 16-th graph for shape None from inductor via handle ('fvj3ccoi7m34f3dnr4itmu55mmun44l5xymwhrjlwisylsk7q6jy', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/tf/ctfftkglj7b4lcttq5cymx6cew372uoauupqn6ldsvpiucavqcjc.py')
```

이는 첫 번째 그래프 조각(shape `None`, 심볼릭 shape)이 Inductor로 컴파일되었고, key가 `fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw`이며, 컴파일된 커널이 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py`에 저장됨을 의미합니다. 해당 파일을 열어 Inductor가 최종적으로 실행하는 코드를 확인할 수 있습니다.

한 가지 더 보면, 1번째 그래프와 15번째 그래프는 key가 같고, 0번째와 16번째는 다릅니다. 이는 예상된 결과입니다. attention op 기준으로 그래프를 분할하면 고유 서브그래프는 3개가 되기 때문입니다.

- attention 이전의 첫 레이어
- attention과 attention 사이의 중간 레이어들
- attention 이후의 마지막 레이어

이미 캐시 디렉터리가 있다면(예: 같은 코드를 두 번째 실행), 다음 로그를 보게 됩니다.

```console
DEBUG 03-07 04:00:45 [backends.py:86] Directly load the 0-th graph for shape None from inductor via handle ('fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py')
```

이 경우 Inductor 컴파일은 완전히 생략되고, 이전 실행에서 만든 산출물을 디스크에서 바로 읽어옵니다.

위 예시는 일반 shape(심볼릭 shape)에 대한 Inductor 컴파일입니다. 특정 shape에 대해 컴파일하는 것도 가능합니다. 예:

```bash
vllm serve meta-llama/Llama-3.2-1B \
  --compilation_config '{"compile_sizes": [1, 2, 4, 8]}'
```

이렇게 하면 배치 크기 `1, 2, 4, 8` 각각에 대해 전용 커널을 컴파일합니다. 이 경우 computation graph의 shape가 모두 정적이고 알려진 상태이므로 최대 성능을 위한 auto-tuning을 켭니다. 첫 실행은 느릴 수 있지만, 다음 실행부터는 튜닝을 우회하고 튜닝된 커널을 바로 사용합니다.

shape가 모두 알려지면 `torch.compile`은 여러 설정을 비교해 더 좋은 커널 설정을 찾을 수 있습니다. 예를 들어 아래 로그를 볼 수 있습니다.

??? console "Logs"

    ```
    AUTOTUNE mm(8x2048, 2048x3072)
      triton_mm_4 0.0130 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=2
      triton_mm_8 0.0134 ms 97.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=4
      triton_mm_12 0.0148 ms 87.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=4, num_warps=4
      mm 0.0160 ms 81.6%
      triton_mm_16 0.0165 ms 78.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=8
      triton_mm_3 0.0199 ms 65.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=2
      triton_mm_1 0.0203 ms 64.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=2, num_warps=2
      triton_mm_7 0.0203 ms 64.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=3, num_warps=4
      triton_mm_2 0.0208 ms 62.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=4
      triton_mm_11 0.0215 ms 60.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 2.0428 seconds and 7.5727 seconds precompiling
    ```

이 로그는 `8x2048x3072` shape의 행렬곱에 대해 `torch.compile`이 다양한 Triton 템플릿 설정을 시도했고, 기본 코드(cublas 호출)보다 훨씬 빠른 설정을 찾았음을 의미합니다.

다만 auto-tuning은 시간이 많이 걸립니다(모델 크기/배치 크기에 따라 수초~수분). 결과를 캐시할 수는 있지만 사용자 친화성을 위해 기본값은 비활성화되어 있습니다. 최대 성능이 필요하면 특정 shape 컴파일과 함께 시도하는 것을 권장합니다.

## CUDAGraph 캡처

vLLM V1 아키텍처는 piecewise 컴파일과 정렬되는 piecewise CUDAGraph를 사용합니다. 앞서 설명했듯 전체 computation graph를 분할하고, attention 연산 사이 구간(첫 attention 이전 그래프, 마지막 attention 이후 그래프 포함)만 CUDAGraph로 캡처합니다. 이는 일반적인 관찰에 기반합니다. attention 사이 계산은 보통 토큰 단위라 CUDAGraph 적용이 쉽고, attention 자체는 CUDAGraph 호환성이 까다롭습니다. 따라서 attention은 eager로 실행하고 나머지는 CUDAGraph로 실행해 attention 유연성을 유지합니다.

piecewise CUDAGraph는 세밀한 메모리 관리도 제공합니다. 목표는 attention 커널만 CUDAGraph에서 제외하고, 나머지 모듈과 메모리 할당 연산은 CUDAGraph에 포함하는 것입니다. 그래서 V1 attention은 attention의 출력 텐서를 attention 입력으로 받는 구조를 사용합니다.

CUDAGraph는 컴파일러 백엔드가 캡처/관리하고, 해당 배치 크기의 그래프가 있으면 재생(replay)합니다. 모델 호출자(model runner)는 입력 버퍼 관리만 올바르게 하면 되고, 중간 버퍼는 컴파일러 백엔드가 자동으로 관리합니다.

기본적으로 vLLM은 CUDAGraph 캡처 대상 크기 집합을 자동 결정합니다. `cudagraph_capture_sizes` 설정으로 직접 지정할 수도 있습니다.

```bash
vllm serve meta-llama/Llama-3.2-1B \
  --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8]}'
```

이 경우 지정한 크기에 대해서만 CUDAGraph를 캡처합니다. 캡처 대상을 세밀하게 제어할 때 유용합니다.

### 전체 CUDAGraph 캡처

attention 백엔드가 CUDAGraph 호환이면 attention까지 포함한 full CUDAGraph 캡처가 가능합니다. 이는 작은 모델이나 MOE의 decode 속도처럼 일부 경우 성능을 향상시킬 수 있습니다. 자세한 내용은 [CUDA Graphs](cuda_graphs.md)를 참고하세요.
