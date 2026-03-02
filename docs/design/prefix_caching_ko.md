# 자동 Prefix 캐싱

prefix 캐싱 KV-cache 블록은 중복 프롬프트 계산을 피하기 위한 LLM 추론 최적화로 널리 사용됩니다. 핵심 아이디어는 단순합니다. 처리된 요청의 KV-cache 블록을 캐시해 두고, 이전 요청과 동일한 prefix를 가진 새 요청이 들어오면 해당 블록을 재사용하는 것입니다. prefix 캐싱은 거의 공짜에 가까운 최적화이면서 모델 출력은 바꾸지 않기 때문에, 많은 공개 엔드포인트(OpenAI, Anthropic 등)와 대부분의 오픈소스 LLM 추론 프레임워크(SGLang 등)에서 널리 사용됩니다.

prefix 캐싱 구현 방법은 다양하지만, vLLM은 해시 기반 접근을 선택했습니다. 구체적으로 각 KV-cache 블록은 블록 내부 토큰과, 해당 블록 이전 prefix 토큰을 함께 사용해 해시합니다.

```text
                    Block 1                  Block 2                  Block 3
         [A gentle breeze stirred] [the leaves as children] [laughed in the distance]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
Block 3: |<------------------ prefix -------------------->| |<--- block tokens ---->|
```

위 예시에서 첫 번째 블록의 KV cache는 "A gentle breeze stirred" 토큰으로 고유 식별할 수 있습니다. 세 번째 블록은 블록 토큰 "laughed in the distance"와 prefix 토큰 "A gentle breeze stirred the leaves as children"를 함께 써서 고유 식별할 수 있습니다. 따라서 블록 해시는 `hash(tuple[components])` 형태로 만들 수 있으며, `components`는 다음과 같습니다.

* 부모 해시 값(Parent hash value): 부모 해시 블록의 해시 값
* 블록 토큰(Block tokens): 해당 블록의 토큰 튜플. 정확한 토큰을 포함하는 이유는 잠재적 해시 충돌 가능성을 낮추기 위해서입니다.
* 추가 해시(Extra hashes): 블록을 유일하게 만들기 위해 필요한 다른 값들. 예: LoRA ID, 멀티모달 입력 해시(아래 예시 참고), 멀티테넌트 환경에서 캐시 격리를 위한 cache salt

!!! note "Note 1"
    전체 블록(full block)만 캐싱합니다.

!!! note "Note 2"
    이전 버전에서는 해시 키가 충돌 없이 고유하다고 보장되지 않았습니다. v0.11부터 기본 해싱 알고리즘은 `sha256`이며, 이로써 충돌 위험을 해결했습니다.

    `vllm serve`에서는 `--prefix-caching-hash-algo`로 해싱 알고리즘을 제어할 수 있습니다.
    - `sha256` (기본값): 직렬화에 Python `pickle` 사용. Python/vLLM 버전이 다르면 해시 재현성이 보장되지 않을 수 있습니다.
    - `sha256_cbor`: 직렬화에 `cbor2` 사용. 재현 가능하고 언어 간 호환되는 해시를 제공합니다. 환경 간 결정론적 캐싱이 필요할 때 권장됩니다.
    - `xxhash`: Pickle 직렬화 + xxHash(128-bit)를 사용해 더 빠른 비암호학적 해싱을 수행합니다. 선택적 `xxhash` 패키지가 필요합니다. 중요: 암호학적으로 안전하지 않은 해싱 알고리즘을 사용하면 이론적으로 해시 충돌 위험이 증가하며, 멀티테넌트 환경에서 정의되지 않은 동작 또는 민감 정보 노출로 이어질 수 있습니다. 충돌 가능성 자체는 여전히 매우 낮지만, 성능 이점과 보안 리스크 허용 수준을 비교한 뒤 활성화해야 합니다.
    - `xxhash_cbor`: canonical CBOR 직렬화와 xxHash를 결합해 재현 가능한 해시를 생성합니다. 선택적 `xxhash` 패키지가 필요합니다.

**멀티모달 입력 해싱 예시**
이 예시에서는 멀티모달 입력(예: 이미지)에서 prefix 캐싱이 어떻게 동작하는지 보여줍니다. 다음과 같은 메시지가 들어온다고 가정해 보겠습니다.

```text
messages = [
    {"role": "user",
     "content": [
         {"type": "text",
          "text": "What's in this image?"
         },
         {"type": "image_url",
          "image_url": {"url": image_url},
         },
    ]},
]
```

이는 다음 프롬프트로 변환됩니다.

```text
Prompt:
    <s>[INST]What's in this image?\n[IMG][/INST]

Tokenized prompt:
    [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, 10, 4]

Prompt with placeholders (<P>):
    [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, <P>, <P>, ..., <P>, 4]
```

보듯이 토크나이징 이후 `[IMG]`는 placeholder 토큰 시퀀스로 바뀌며, prefill 단계에서 이 placeholder는 이미지 임베딩으로 치환됩니다. 이 경우 prefix 캐싱의 과제는 placeholder만으로는 서로 다른 이미지를 구분하기 어렵다는 점입니다. 이를 해결하기 위해 프런트엔드 이미지 프로세서가 생성한 이미지 해시를 함께 인코딩합니다. 예를 들어 위 프롬프트에서(블록 크기 16, placeholder 41개 가정) 블록 해시는 다음과 같습니다.

```text
Block 0
    Parent hash: None
    Token IDs: 1, 3, 7493, 1681, 1294, 1593, 3937, 9551, <p>, ..., <p>
    Extra hash: <image hash>
Block 1
    Parent hash: Block 0 hash
    Token IDs: <p>, ..., <p>
    Extra hash: <image hash>
Block 2
    Parent hash: Block 1 hash
    Token IDs: <p>, ..., <p>
    Extra hash: <image hash>
Block 3
    Parent hash: Block 2 hash
    Token IDs: <p>, ..., <p>, 4
    Extra hash: <image hash>
```

이 문서의 나머지에서는 먼저 vLLM v1에서 prefix 캐싱에 사용하는 데이터 구조를 소개하고, 이어서 주요 KV cache 연산(allocate, append, free, eviction)의 prefix 캐싱 워크플로를 설명합니다. 마지막으로 end-to-end 예시를 통해 전체 흐름을 보여줍니다.

**보안을 위한 캐시 격리(Cache Isolation)**
공유 환경에서 프라이버시를 강화하기 위해, vLLM은 요청 단위 salt를 통한 prefix 캐시 재사용 격리를 지원합니다. 요청에 `cache_salt`를 포함하면 이 값이 첫 번째 블록 해시에 주입되어, 같은 salt를 가진 요청끼리만 캐시 블록을 재사용할 수 있습니다. 이를 통해 공격자가 지연 시간 차이로 캐시된 내용을 추론하는 타이밍 기반 공격을 방지할 수 있습니다. 성능 저하 없이 보호를 제공합니다.

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Here is a document with details about the world series: ..."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ],
  "cache_salt": "your-cache-salt"
}
```

이 설정에서는 공통 salt를 명시적으로 공유한 사용자/요청 집합(trust group) 내에서만 캐시가 공유되며, 그 외 요청은 격리됩니다.

## 데이터 구조

vLLM v1의 prefix 캐싱은 KV cache manager에서 구현됩니다. 기본 빌딩 블록은 "Block" 데이터 클래스입니다(단순화).

```python
class KVCacheBlock:
    # The block ID (immutable)
    block_id: int
    # The block hash (will be assigned when the block is full,
    # and will be reset when the block is evicted).
    block_hash: BlockHash
    # The number of requests using this block now.
    ref_cnt: int

    # The pointers to form a doubly linked list for the free queue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
```

강조할 설계 포인트는 2가지입니다.

1. KV cache manager 초기화 시점에 모든 `KVCacheBlock`을 미리 할당해 블록 풀을 구성합니다. 이렇게 하면 Python 객체 생성 오버헤드를 피하고, 항상 전체 블록을 추적하기 쉽습니다.
2. `KVCacheBlock` 내부에 free queue용 이중 연결 리스트 포인터를 직접 두어 queue를 바로 구성합니다. 장점은 다음과 같습니다.
    1. 중간 원소를 tail로 옮기는 연산을 O(1)로 처리할 수 있습니다.
    2. 원소를 감싸는 별도 Python queue(`deque` 등)를 도입하지 않아도 됩니다.

그 결과 KV cache manager 초기화 시 다음 컴포넌트를 갖게 됩니다.

![Component Overview](../assets/design/prefix_caching/overview.png)

* 블록 풀(Block Pool): `KVCacheBlock` 리스트
* Free Block Queue: 조작을 위해 head/tail 블록 포인터만 저장
* Cache blocks: 해시 키 -> 블록 ID 매핑
* Request blocks: 요청 ID -> 할당된 블록 ID 매핑

## 연산

### 블록 할당(Block Allocation)

**신규 요청(New request):**
스케줄러가 신규 요청을 스케줄링할 때 KV cache 블록 할당 워크플로는 다음과 같습니다.

1. 스케줄러가 `kv_cache_manager.get_computed_blocks()`를 호출해 이미 계산된 블록 시퀀스를 가져옵니다. 요청 프롬프트 토큰을 해싱해 cache blocks에서 조회하는 방식입니다.
2. 스케줄러가 `kv_cache_manager.allocate_slots()`를 호출합니다. 내부 단계는 다음과 같습니다.
    1. 새로 필요한 블록 수를 계산하고, 충분한 블록이 없으면 반환합니다.
    2. 계산된 블록을 "touch"합니다. 계산된 블록의 reference count를 1 증가시키고, 그 블록을 다른 요청이 쓰고 있지 않았다면 free queue에서 제거합니다. 이렇게 해야 해당 계산 블록이 evict되지 않습니다. (다음 섹션 예시 참고)
    3. free queue head를 pop하여 새 블록을 할당합니다. head 블록이 캐시 블록이면 이때 해당 블록을 "evict"해, 이후 다른 요청이 재사용하지 못하게 합니다.
    4. 할당된 블록이 이미 토큰으로 가득 찬 상태라면, 즉시 cache blocks에 추가해 같은 배치의 다른 요청이 재사용할 수 있게 합니다.

**실행 중 요청(Running request):**
스케줄러가 실행 중 요청을 스케줄링할 때 KV cache 블록 할당 워크플로는 다음과 같습니다.

1. 스케줄러가 `kv_cache_manager.allocate_slots()`를 호출합니다. 내부 단계는 다음과 같습니다.
    1. 새로 필요한 블록 수를 계산하고, 충분한 블록이 없으면 반환합니다.
    2. free queue head를 pop하여 새 블록을 할당합니다. head 블록이 캐시 블록이면 이때 해당 블록을 "evict"해, 이후 다른 요청이 재사용하지 못하게 합니다.
    3. 기존 블록 슬롯과 신규 블록 슬롯에 token ID를 append합니다. 블록이 가득 차면 cache blocks에 추가해 캐시합니다.

**중복 블록(Duplicated blocks)**
블록 크기가 4이고, 프롬프트 ABCDEF + 디코딩 길이 3인 요청(Request 1)을 보낸다고 가정해 보겠습니다.

```text
Prompt: [A, B, C, D, E, F]
Output: [G, H, I]

Time 0:
  Tokens: [A, B, C, D, E, F, G]
  Block Table: [0 (ABCD), 1 (EFG)]
  Cache Blocks: 0
Time 1:
  Tokens: [A, B, C, D, E, F, G, H]
  Block Table: [0 (ABCD), 1 (EFGH)]
  Cache Blocks: 0, 1
Time 2:
  Tokens: [A, B, C, D, E, F, G, H, I]
  Block Table: [0 (ABCD), 1 (EFGH), 2 (I)]
  Cache Blocks: 0, 1
```

이제 블록 0, 1은 캐시된 상태입니다. 같은 요청(Request 2)을 greedy sampling으로 다시 보내면 Request 1과 동일한 출력이 생성됩니다.

```text
Prompt: [A, B, C, D, E, F]
Output: [G, H, I]

Time 0:
  Tokens: [A, B, C, D, E, F, G]
  Block Table: [0 (ABCD), 3 (EFG)]
  Cache Blocks: 0, 1
Time 1:
  Tokens: [A, B, C, D, E, F, G, H]
  Block Table: [0 (ABCD), 3 (EFGH)]
  Cache Blocks: 0, 1, 3
```

보면 블록 3은 새로 가득 찬 블록이라 캐시되지만, 실제로는 블록 1과 중복입니다. 즉 같은 블록을 두 번 캐시한 상태입니다. v0에서는 블록 3이 중복임을 감지하면 블록 3을 해제하고 Request 2가 블록 1을 쓰도록 바꿔 block table을 Time 1에서 `[0, 1]`로 만들었습니다. 하지만 vLLM v1의 block table은 append-only여서 `[0, 3]`을 `[0, 1]`로 변경할 수 없습니다. 따라서 E-H 해시 키에 대해 중복 블록이 생기며, 이 중복은 요청이 해제될 때 제거됩니다.

### 해제(Free)

요청이 끝나면 해당 블록을 쓰는 다른 요청이 없을 때(reference count = 0) 모든 블록을 해제합니다. 아래 예시에서는 request 1을 해제하면서 연결된 블록 2, 3, 4, 8을 해제합니다. 해제된 블록은 free queue tail에 *역순(reverse order)*으로 추가됩니다. 요청의 마지막 블록일수록 더 많은 토큰을 해싱하므로 다른 요청이 재사용할 가능성이 낮고, 따라서 먼저 evict되는 편이 유리하기 때문입니다.

![Free queue after a request us freed](../assets/design/prefix_caching/free.png)

### 축출(Eviction, LRU)

free queue의 head 블록(가장 오래 사용되지 않은 블록, LRU)이 캐시된 블록이라면, 다른 요청이 사용하지 못하도록 해당 블록을 evict해야 합니다. 구체적 단계는 다음과 같습니다.

1. free queue head에서 블록을 pop합니다. 이 블록이 evict 대상 LRU 블록입니다.
2. cache blocks에서 해당 블록 ID를 제거합니다.
3. 블록 해시를 제거합니다.

## 예시

이 예시에서는 블록 크기를 4(각 블록이 토큰 4개 캐시), KV-cache manager 총 블록 수를 10으로 가정합니다.

**Time 1: 캐시가 비어 있고 새 요청이 들어옵니다.** 블록 4개를 할당합니다. 그중 3개는 이미 가득 차 캐시되고, 네 번째 블록은 4개 중 3개 토큰만 찬 부분 채움 상태입니다.

![Example Time 1](../assets/design/prefix_caching/example-time-1.png)

**Time 2: Request 0이 block 3을 가득 채우고, 디코딩을 계속하기 위해 새 블록을 요청합니다.** block 3을 캐시하고 block 4를 할당합니다.

![Example Time 2](../assets/design/prefix_caching/example-time-3.png)

**Time 3: Request 1이 길이 14의 프롬프트로 들어오며, 앞 10개 토큰이 request 0과 동일합니다.** 3번째 블록은 4개 중 2개 토큰만 일치하므로, 앞 2개 블록(8토큰)만 캐시 히트합니다.

![Example Time 3](../assets/design/prefix_caching/example-time-4.png)

**Time 4: Request 0이 종료되어 해제됩니다.** 블록 2, 3, 4가 free queue에 역순으로 추가됩니다(단, 블록 2, 3은 여전히 캐시 상태). 블록 0, 1은 Request 1이 사용 중이므로 free queue에 추가되지 않습니다.

![Example Time 4](../assets/design/prefix_caching/example-time-5.png)

**Time 5: Request 1이 종료되어 해제됩니다.**

![Example Time 5](../assets/design/prefix_caching/example-time-6.png)

**Time 6: Request 2가 길이 29의 프롬프트로 들어오며, 앞 12개 토큰이 request 0과 동일합니다.** free queue 순서가 `7 - 8 - 9 - 4 - 3 - 2 - 6 - 5 - 1 - 0`이었더라도, 캐시 히트 블록(0, 1, 2)은 할당 전에 touch되어 queue에서 제거되므로 free queue는 `7 - 8 - 9 - 4 - 3 - 6 - 5`가 됩니다. 따라서 실제 할당 블록은 0(캐시), 1(캐시), 2(캐시), 7, 8, 9, 4, 3(evicted)입니다.

![Example Time 6](../assets/design/prefix_caching/example-time-7.png)
