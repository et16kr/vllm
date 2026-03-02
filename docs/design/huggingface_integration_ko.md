# Hugging Face와의 통합

이 문서는 vLLM이 Hugging Face 라이브러리와 어떻게 통합되는지 설명합니다. `vllm serve`를 실행했을 때 내부에서 어떤 일이 일어나는지 단계별로 살펴봅니다.

예를 들어 `vllm serve Qwen/Qwen2-7B`를 실행해 인기 있는 Qwen 모델을 서빙한다고 가정해보겠습니다.

1. `model` 인자는 `Qwen/Qwen2-7B`입니다. vLLM은 해당 모델이 존재하는지 대응되는 설정 파일 `config.json`을 확인해 판단합니다. 구현은 이 [코드 스니펫](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L162-L182)을 참고하세요. 이 과정에서:
    - `model` 인자가 기존 로컬 경로를 가리키면, vLLM은 해당 경로에서 설정 파일을 직접 로드합니다.
    - `model` 인자가 사용자명/모델명 형태의 Hugging Face 모델 ID이면, vLLM은 먼저 Hugging Face 로컬 캐시에서 설정 파일을 찾습니다. 이때 `model` 인자를 모델명으로, `--revision` 인자를 리비전으로 사용합니다. Hugging Face 캐시 동작은 [공식 문서](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome)를 참고하세요.
    - `model` 인자가 Hugging Face 모델 ID이지만 캐시에 없으면, vLLM은 Hugging Face 모델 허브에서 설정 파일을 다운로드합니다. 구현은 [이 함수](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L91)를 참고하세요. 입력 인자로는 모델명(`model`), 리비전(`--revision`), 그리고 허브 접근 토큰(`HF_TOKEN` 환경변수)이 사용됩니다. 이 예시에서는 [config.json](https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json)을 다운로드합니다.

2. 모델 존재를 확인한 뒤, vLLM은 설정 파일을 로드하고 dictionary로 변환합니다. 구현은 이 [코드 스니펫](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L185-L186)을 참고하세요.

3. 다음으로 vLLM은 설정 dictionary의 `model_type` 필드를 [검사](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L189)해 사용할 설정 객체를 [생성](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L190-L216)합니다. vLLM이 직접 지원하는 `model_type` 값 목록은 [여기](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L48)에 있습니다. 목록에 없는 `model_type`이면, vLLM은 `model`, `--revision`, `--trust_remote_code`를 인자로 [AutoConfig.from_pretrained](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained)를 사용해 설정 클래스를 로드합니다. 이때 다음을 참고해야 합니다:
    - Hugging Face도 자체적으로 설정 클래스 결정 로직을 가집니다. 다시 `model_type` 필드를 사용해 transformers 라이브러리에서 클래스 이름을 찾습니다. 지원 모델 목록은 [여기](https://github.com/huggingface/transformers/tree/main/src/transformers/models)에서 확인할 수 있습니다. `model_type`을 찾지 못하면, Hugging Face는 config JSON의 `auto_map` 필드를 사용해 클래스 이름을 결정합니다. 구체적으로는 `auto_map` 아래의 `AutoConfig` 필드입니다. 예시는 [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V2.5/blob/main/config.json)를 참고하세요.
    - `auto_map` 아래 `AutoConfig` 필드는 모델 저장소의 모듈 경로를 가리킵니다. Hugging Face는 해당 모듈을 import하고 `from_pretrained` 메서드를 사용해 설정 클래스를 로드합니다. 이 과정은 일반적으로 임의 코드 실행 가능성을 동반하므로 `--trust_remote_code`를 켠 경우에만 수행됩니다.

4. 이후 vLLM은 설정 객체에 과거 호환 패치를 적용합니다. 대부분 RoPE 설정과 관련되어 있으며, 구현은 [여기](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/transformers_utils/config.py#L244)를 참고하세요.

5. 마지막으로 vLLM은 초기화할 모델 클래스에 도달합니다. vLLM은 설정 객체의 `architectures` 필드를 사용해 어떤 모델 클래스를 초기화할지 결정합니다. 아키텍처 이름과 모델 클래스의 매핑은 [registry](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/model_executor/models/registry.py#L80)에서 관리합니다. 아키텍처 이름이 registry에 없으면 해당 모델 아키텍처는 vLLM에서 지원되지 않는다는 뜻입니다. `Qwen/Qwen2-7B`의 `architectures`는 `["Qwen2ForCausalLM"]`이며, 이는 [vLLM 코드](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/model_executor/models/qwen2.py#L364)의 `Qwen2ForCausalLM` 클래스에 대응됩니다. 이 클래스는 다양한 설정값에 따라 초기화됩니다.

이 외에도 vLLM이 Hugging Face에 의존하는 요소가 2가지 더 있습니다.

1. **토크나이저(Tokenizer)**: vLLM은 입력 텍스트 토크나이징에 Hugging Face 토크나이저를 사용합니다. 토크나이저는 `model`을 모델명으로, `--revision`을 리비전으로 사용해 [AutoTokenizer.from_pretrained](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained)로 로드됩니다. `vllm serve`에서 `--tokenizer` 인자를 지정하면 다른 모델의 토크나이저를 사용할 수도 있습니다. 관련 인자로 `--tokenizer-revision`, `--tokenizer-mode`가 있습니다. 의미는 Hugging Face 문서를 참고하세요. 이 로직은 [get_tokenizer](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/transformers_utils/tokenizer.py#L87) 함수에서 확인할 수 있습니다. 또한 vLLM은 토크나이저를 얻은 뒤 비용이 큰 일부 속성을 [vllm.tokenizers.hf.get_cached_tokenizer][]에 캐시합니다.

2. **모델 가중치(Model weight)**: vLLM은 `model`을 모델명으로, `--revision`을 리비전으로 사용해 Hugging Face 모델 허브에서 모델 가중치를 다운로드합니다. vLLM은 허브에서 어떤 파일을 내려받을지 제어하기 위해 `--load-format` 인자를 제공합니다. 기본 동작은 safetensors 형식을 우선 시도하고, safetensors가 없으면 PyTorch bin 형식으로 폴백합니다. `--load-format dummy`를 지정하면 가중치 다운로드를 건너뛸 수 있습니다.
    - 분산 추론 로딩 효율과 임의 코드 실행 안전성 측면에서 safetensors 형식 사용을 권장합니다. 자세한 내용은 [공식 문서](https://huggingface.co/docs/safetensors/en/index)를 참고하세요. 이 로직은 [여기](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/model_executor/model_loader/loader.py#L385)에서 확인할 수 있습니다.

이로써 vLLM과 Hugging Face 통합 과정이 완료됩니다.

요약하면, vLLM은 Hugging Face 모델 허브 또는 로컬 디렉터리에서 `config.json`, 토크나이저, 모델 가중치를 읽어옵니다. 그리고 설정 클래스는 vLLM 자체 구현, Hugging Face transformers, 혹은 모델 저장소에서 로드한 클래스를 사용합니다.
