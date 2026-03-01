# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """스케줄링 정책 열거형."""

    FCFS = "fcfs"
    PRIORITY = "priority"


class RequestQueue(ABC):
    """요청 큐 인터페이스."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """정책에 따라 요청을 큐에 추가한다."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """정책에 따라 큐에서 요청 하나를 꺼낸다."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """큐에서 제거하지 않고 다음 요청을 조회한다."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """요청을 큐 앞쪽에 삽입한다."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """다른 큐의 요청들을 현재 큐 앞쪽에 삽입한다."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """특정 요청을 큐에서 제거한다."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """여러 요청을 큐에서 제거한다."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """큐가 비어 있지 않은지 확인한다."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """큐 길이를 반환한다."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """정책 순서대로 큐를 순회한다."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """deque 기반 선입선출(FCFS) 요청 큐."""

    def add_request(self, request: Request) -> None:
        """FCFS 정책으로 요청을 추가한다."""
        self.append(request)

    def pop_request(self) -> Request:
        """FCFS 정책으로 큐 앞 요청을 꺼낸다."""
        return self.popleft()

    def peek_request(self) -> Request:
        """FCFS 큐의 다음 요청을 조회한다."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """요청을 큐 앞쪽에 넣는다."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """다른 큐의 요청들을 현재 큐 앞쪽에 넣는다.

        참고:
            `deque.extendleft`를 사용하므로 입력 순서의 역순으로 삽입된다.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """특정 요청을 제거한다."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """여러 요청을 제거한다."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque는 in-place 필터링을 지원하지 않으므로 재구성한다.
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """큐가 비어 있지 않은지 반환한다."""
        return len(self) > 0

    def __len__(self) -> int:
        """큐 길이를 반환한다."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """FCFS 순서로 큐를 순회한다."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """힙 기반 우선순위 요청 큐.

    Request의 정렬 규칙을 따르며,
    `priority`가 작을수록 먼저 처리된다.
    `priority`가 같으면 `arrival_time`이 빠른 요청이 먼저 처리된다.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """우선순위 정책으로 요청을 추가한다."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """우선순위가 가장 높은(값이 작은) 요청을 꺼낸다."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """다음 요청을 제거하지 않고 조회한다."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """우선순위 정책에 따라 요청을 추가한다.

        참고:
            우선순위 큐에는 물리적인 '앞쪽' 개념이 없다.
            삽입 후 `(priority, arrival_time)` 순서가 다시 적용된다.
        """
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """다른 큐의 요청들을 우선순위 정책에 따라 추가한다.

        참고:
            우선순위 큐에는 물리적인 '앞쪽' 개념이 없다.
            삽입 후 `(priority, arrival_time)` 순서가 다시 적용된다.
        """
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """특정 요청을 제거한다."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """여러 요청을 제거한다."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """큐가 비어 있지 않은지 반환한다."""
        return bool(self._heap)

    def __len__(self) -> int:
        """큐 길이를 반환한다."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """우선순위 순서대로 큐를 순회한다."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """스케줄링 정책에 맞는 요청 큐를 생성한다."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
