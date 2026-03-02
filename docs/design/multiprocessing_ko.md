# Python 멀티프로세싱

## 디버깅

알려진 이슈와 해결 방법은
[문제 해결 가이드](../usage/troubleshooting.md#python-multiprocessing)를 참고하세요.

## 소개

!!! important
    이 문서의 소스 코드 참조는 2024년 12월 작성 시점의 코드 상태를 기준으로 합니다.

vLLM에서 Python 멀티프로세싱 사용이 복잡한 이유는 다음과 같습니다.

- vLLM이 라이브러리 형태로 사용되며, vLLM을 사용하는 쪽의 코드를 vLLM이 제어할 수 없음
- 멀티프로세싱 시작 방식과 vLLM 의존성 사이의 비호환성이 환경마다 다름

이 문서는 vLLM이 이러한 문제를 어떻게 다루는지 설명합니다.

## 멀티프로세싱 방식

[Python multiprocessing methods](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)에는 다음이 있습니다.

- `spawn` - 새 Python 프로세스를 생성합니다. Windows와 macOS 기본값입니다.

- `fork` - `os.fork()`를 사용해 Python 인터프리터를 포크합니다. Python 3.14 미만의 Linux 기본값입니다.

- `forkserver` - 요청 시 새 프로세스를 포크해주는 서버 프로세스를 먼저 띄웁니다. Python 3.14 이상의 Linux 기본값입니다.

### 트레이드오프

`fork`는 가장 빠르지만 스레드를 사용하는 의존성과는 호환되지 않습니다. macOS에서 `fork`를 사용하면 프로세스가 크래시할 수 있습니다.

`spawn`은 의존성과의 호환성이 더 좋지만, vLLM을 라이브러리로 쓸 때 문제가 될 수 있습니다. 호출 코드에 `__main__` 가드(`if __name__ == "__main__":`)가 없으면, vLLM이 새 프로세스를 만들 때 코드가 의도치 않게 다시 실행됩니다. 이로 인해 무한 재귀 등 문제가 발생할 수 있습니다.

`forkserver`는 서버 프로세스를 띄우고 필요할 때 프로세스를 포크합니다. 하지만 vLLM을 라이브러리로 사용할 때는 `spawn`과 동일한 문제가 있습니다. 서버 프로세스 자체가 `spawn`으로 생성되므로 `__main__` 가드가 없는 코드가 다시 실행됩니다.

`spawn`과 `forkserver` 모두 `fork`처럼 전역 상태를 상속하는 방식에 의존하면 안 됩니다.

## 의존성과의 호환성

여러 vLLM 의존성은 `spawn` 사용을 선호하거나 사실상 요구합니다.

- <https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing>
- <https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors>
- <https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html?highlight=multiprocessing#torch-multiprocessing-for-dataloaders>

정확히 말하면, 이런 의존성을 초기화한 뒤 `fork`를 사용하면 알려진 문제가 발생합니다.

## 현재 상태(v0)

환경변수 `VLLM_WORKER_MULTIPROC_METHOD`로 vLLM이 사용할 시작 방식을 제어할 수 있습니다. 현재 기본값은 `fork`입니다.

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/envs.py#L339-L342>

`vllm` 명령으로 실행되어 프로세스 제어권이 우리에게 있는 경우에는, 호환성이 가장 넓은 `spawn`을 사용합니다.

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/scripts.py#L123-L140>

`multiproc_xpu_executor`는 `spawn` 사용을 강제합니다.

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/executor/multiproc_xpu_executor.py#L14-L18>

그 외에도 `spawn`을 하드코딩한 위치가 몇 군데 있습니다.

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/distributed/device_communicators/all_reduce_utils.py#L135>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/entrypoints/openai/api_server.py#L184>

관련 PR:

- <https://github.com/vllm-project/vllm/pull/8823>

## v1의 이전 상태

v1 엔진 코어에서 멀티프로세싱 사용 여부를 제어하는 환경변수 `VLLM_ENABLE_V1_MULTIPROCESSING`가 있었고, 기본값은 비활성이었습니다.

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/envs.py#L452-L454>

이 옵션이 켜지면 v1 `LLMEngine`이 엔진 코어 실행을 위해 새 프로세스를 생성했습니다.

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/llm_engine.py#L93-L95>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/llm_engine.py#L70-L77>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/core_client.py#L44-L45>

앞서 언급한 이유들(의존성과의 호환성, 라이브러리 사용 코드와의 충돌) 때문에 기본값은 비활성이었습니다.

### v1에서의 변경

Python `multiprocessing`만으로 모든 환경에서 완벽히 동작하는 쉬운 해법은 없습니다. 첫 단계로 v1이 호환성을 최대화하도록 "최선의 선택(best effort)"을 하게 만들 수 있습니다.

- 기본값은 `fork`.
- 메인 프로세스를 우리가 제어함이 확실한 경우(`vllm` 명령 실행)는 `spawn` 사용.
- `cuda`가 이미 초기화되어 있음을 감지하면 `spawn`을 강제하고 경고 출력.
  `fork`가 깨지는 것이 확실하므로 가능한 최선의 대응입니다.

이 시나리오에서 여전히 깨지는 것으로 알려진 경우는, vLLM을 라이브러리로 쓰는 코드가 vLLM 호출 전에 `cuda`를 초기화한 경우입니다. 이때 경고 메시지에는 `__main__` 가드를 추가하거나 멀티프로세싱을 비활성화하라는 안내가 포함됩니다.

이 실패 케이스가 발생하면 사용자는 상황 설명 메시지 2개를 보게 됩니다. 첫 번째는 vLLM 로그입니다.

```console
WARNING 12-11 14:50:37 multiproc_worker_utils.py:281] CUDA was previously
    initialized. We must use the `spawn` multiprocessing start method. Setting
    VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See
    https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing
    for more information.
```

두 번째는 Python이 직접 발생시키는 예외입니다.

```console
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.

        To fix this issue, refer to the "Safe importing of main module"
        section in https://docs.python.org/3/library/multiprocessing.html
```

## 검토했던 대안

### `__main__` 가드 존재 여부 탐지

vLLM을 라이브러리로 사용하는 코드에 `__main__` 가드가 있는지 감지할 수 있다면 더 나은 동작을 할 수 있다는 의견이 있었습니다. 같은 질문을 다룬 라이브러리 작성자의 [stackoverflow 글](https://stackoverflow.com/questions/77220442/multiprocessing-pool-in-a-python-class-without-name-main-guard)도 있습니다.

현재 프로세스가 원래 `__main__` 프로세스인지, 이후 생성된 `spawn` 프로세스인지는 감지할 수 있습니다. 하지만 코드에 `__main__` 가드가 있는지를 신뢰성 있게 감지하는 것은 쉽지 않아 보입니다.

이 옵션은 현실성이 낮아 폐기했습니다.

### `forkserver` 사용

처음에는 `forkserver`가 좋은 해결책처럼 보입니다.
하지만 동작 방식상 vLLM을 라이브러리로 사용할 때 `spawn`과 같은 문제를 가집니다.

### 항상 `spawn` 강제

정리를 단순화하려면 항상 `spawn`을 강제하고, vLLM을 라이브러리로 사용할 때는 `__main__` 가드가 필수라고 문서화할 수 있습니다. 하지만 이는 기존 코드를 깨뜨릴 수 있고, `LLM` 클래스를 가능한 쉽게 사용하게 하려는 목표에 반합니다.

사용자에게 이 복잡성을 떠넘기기보다, vLLM 쪽에서 최대한 동작하도록 복잡성을 유지하기로 했습니다.

## 향후 작업

앞으로는 이러한 문제를 우회할 수 있는 다른 워커 관리 접근법을 검토할 수 있습니다.

1. `forkserver`와 비슷한 방식을 직접 구현하되, 프로세스 매니저를 자체 서브프로세스와 커스텀 엔트리포인트로 띄우는 방법(`vllm-manager` 프로세스 시작).

2. 요구사항에 더 잘 맞는 다른 라이브러리 탐색. 예:

- <https://github.com/joblib/loky>
