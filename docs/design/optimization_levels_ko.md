<!-- markdownlint-disable -->

# 최적화 레벨

## 개요

vLLM은 이제 최적화 레벨(`-O0`, `-O1`, `-O2`, `-O3`)을 지원합니다. 최적화 레벨은 사용자가 시작 시간(startup time)과 성능 사이를 직관적으로 절충할 수 있게 해줍니다. 레벨이 높을수록 성능은 좋아지지만 시작 시간은 길어집니다. 각 최적화 레벨에는 기본 설정(default)이 연결되어 있어 사용자가 별도 튜닝 없이도 원하는 성능을 얻을 수 있습니다. 중요한 점은, 최적화 레벨이 설정하는 값은 어디까지나 기본값이며, 사용자가 명시적으로 지정한 설정은 덮어쓰지 않는다는 것입니다.

## 레벨 요약 및 사용 예시
```bash
# CLI usage
python -m vllm.entrypoints.api_server --model RedHatAI/Llama-3.2-1B-FP8 -O0

# Python API usage
from vllm.entrypoints.llm import LLM

llm = LLM(
    model="RedHatAI/Llama-3.2-1B-FP8",
    optimization_level=0
)
```

#### `-O1`: 빠른 최적화
- **시작 시간**: 중간 수준
- **성능**: Inductor 컴파일, `CUDAGraphMode.PIECEWISE`
- **사용 사례**: 대부분의 개발 시나리오에서 균형 잡힌 선택

```bash
# CLI usage
python -m vllm.entrypoints.api_server --model RedHatAI/Llama-3.2-1B-FP8 -O1

# Python API usage
from vllm.entrypoints.llm import LLM

llm = LLM(
    model="RedHatAI/Llama-3.2-1B-FP8",
    optimization_level=1
)
```

#### `-O2`: 전체 최적화(기본값)
- **시작 시간**: 더 긴 시작 시간
- **성능**: `-O1` + `CUDAGraphMode.FULL_AND_PIECEWISE`
- **사용 사례**: 성능이 중요한 프로덕션 워크로드. 기본 사용 시나리오이며, 이전 기본 동작과도 매우 유사합니다. 주요 차이는 noop 및 fusion 플래그가 활성화된다는 점입니다.

```bash
# CLI usage (default, so optional)
python -m vllm.entrypoints.api_server --model RedHatAI/Llama-3.2-1B-FP8 -O2

# Python API usage
from vllm.entrypoints.llm import LLM

llm = LLM(
    model="RedHatAI/Llama-3.2-1B-FP8",
    optimization_level=2  # This is the default
)
```

#### `-O3`: 완전 최적화
아직 개발 중입니다. 향후 릴리스에서 API 변경을 방지하기 위한 인프라가 추가된 상태이며,
현재 동작은 `O2`와 동일합니다.

## 문제 해결

### 자주 발생하는 이슈

1. **시작 시간이 너무 긴 경우**: 더 빠른 시작을 위해 `-O0` 또는 `-O1` 사용
2. **컴파일 오류**: 추가 디버깅 정보를 위해 `debug_dump_path` 사용
3. **성능 이슈**: 프로덕션에서는 `-O2` 사용 여부 확인
