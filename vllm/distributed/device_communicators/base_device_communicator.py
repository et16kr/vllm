# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from weakref import WeakValueDictionary

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


class Cache:
    def __init__(self):
        self._cache: WeakValueDictionary = WeakValueDictionary()
        self._lock = threading.RLock()  # 스레드 안전성을 위한 재진입 가능 락

    def get_or_create(self, kwargs, func):
        # kwargs로부터 해시 가능한 키를 생성한다.
        key = tuple(sorted((k, v) for k, v in kwargs.items()))

        with self._lock:
            instance = self._cache.get(key)
            if instance is None:
                instance = func(**kwargs)
                self._cache[key] = instance
            return instance


class All2AllManagerBase:
    rank: int
    world_size: int

    def __init__(self, cpu_group, tcp_store_group=None):
        self.cpu_group = cpu_group
        self.tcp_store_group = tcp_store_group

        # 공통 속성을 계산한다.
        from vllm.distributed.parallel_state import (
            get_dp_group,
            get_tp_group,
            in_the_same_node_as,
        )

        # all2all은 dp/tp 그룹이 합쳐진 ep 그룹에서 동작한다.
        self.dp_group = get_dp_group()
        self.tp_group = get_tp_group()

        # 이 객체 생성 시점에는 self.ep_group이 아직 구성 중이므로
        # self.ep_group을 사용하지 않는다.
        self.dp_rank = self.dp_group.rank_in_group
        self.dp_world_size = self.dp_group.world_size
        self.rank = cpu_group.rank()
        self.world_size = cpu_group.size()

        # all2all 통신은 보통 intra-node/inter-node 구현이 분리되어 있다.
        if tcp_store_group is None:
            self.internode = not all(in_the_same_node_as(cpu_group, source_rank=0))
        else:
            self.internode = not all(
                in_the_same_node_as(tcp_store_group, source_rank=0)
            )

    def get_handle(self, kwargs):
        # kwargs를 바탕으로 all2all 통신 핸들을 가져온다.
        # 레이어별 설정은 서로 다를 수 있다.
        # 예: 한 레이어는 hidden size 1024, 다른 레이어는 2048.
        # 일반적으로 내부 구현은 핸들을 캐시하여 동일한 설정에 재사용한다.
        raise NotImplementedError

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        # 하위 클래스는 다음 중 하나를 구현해야 한다.
        # - extra_tensors 처리 구현
        # - extra_tensors 미지원 시 명확한 예외 발생
        raise NotImplementedError

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        # 하위 클래스는 다음 중 하나를 구현해야 한다.
        # - extra_tensors 처리 구현
        # - extra_tensors 미지원 시 명확한 예외 발생
        raise NotImplementedError

    def set_num_sms(self, num_sms: int):
        pass

    def max_sms_used(self) -> int | None:
        return None  # None means it could use the whole GPU

    def combine(self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False):
        raise NotImplementedError

    def destroy(self):
        pass


class DeviceCommunicatorBase:
    """
    디바이스별 communicator의 베이스 클래스.
    communicator 초기화 시 `cpu_group`을 사용할 수 있다.
    디바이스가 PyTorch와 통합되어 있어(PyTorch가 해당 통신 백엔드를
    인식할 수 있어) `device_group`을 제공할 수 있으면 함께 전달된다.
    """

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
    ):
        self.device = device or torch.device("cpu")
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.unique_name = unique_name

        # stateless process group인지 확인한다.
        from torch.distributed.distributed_c10d import _world

        is_stateless = _world.pg_map.get(cpu_group, None) is None

        if is_stateless:
            # stateless 그룹에서는 torch.distributed 메서드를 사용할 수 없다.
            self.rank = cpu_group.rank()
            self.world_size = cpu_group.size()
            assert global_ranks is not None
            assert global_world_size is not None
            self.ranks = global_ranks
            self.global_rank = self.ranks[self.rank]
            self.global_world_size = global_world_size
            self.rank_in_group = self.rank
        else:
            self.rank = dist.get_rank(cpu_group)
            self.world_size = dist.get_world_size(cpu_group)
            self.ranks = dist.get_process_group_ranks(cpu_group)
            self.global_rank = dist.get_rank()
            self.global_world_size = dist.get_world_size()
            self.rank_in_group = dist.get_group_rank(self.cpu_group, self.global_rank)

        use_ep = False
        all2all_backend = None
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        if config is not None:
            # data parallel(모든 data parallel rank가 함께 forward를 수행하는
            # 결합형 data parallel)을 사용하는 경우에는
            # expert parallel에서 쓰는 all2all manager를 초기화한다.
            use_ep = config.parallel_config.data_parallel_size > 1
            all2all_backend = config.parallel_config.all2all_backend

        self.is_ep_communicator = unique_name.split(":")[0] == "ep"
        self.use_all2all = self.is_ep_communicator and use_ep
        self.all2all_backend = all2all_backend
        self.all2all_manager: All2AllManagerBase | None = None

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(input_, group=self.device_group)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            # 음수 dim을 양수로 변환한다.
            dim += input_.dim()
        input_size = input_.size()
        # NOTE: 여기서는 concat 방식 all-gather를 사용해야 한다.
        # stack 방식 all-gather는 torch.compile과 호환성 이슈가 있다.
        # 참고: https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        # 출력 텐서를 할당한다.
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        # all-gather를 수행한다.
        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        # shape을 복원한다.
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        return output_tensor

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        raise NotImplementedError

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        # GPU를 1개만 사용하는 경우 연산을 생략한다.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )

        if dim < 0:
            # 음수 dim을 양수로 변환한다.
            dim += input_.dim()

        # Note: input_tensor를 contiguous로 만들지 않으면 결과가 틀릴 수 있다.
        # reduce_scatter_tensor 쪽 버그 가능성이 있다.
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output_tensor = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        # reduce-scatter 연산을 수행한다.
        torch.distributed.reduce_scatter_tensor(
            output_tensor, input_tensor, group=self.device_group
        )

        # 반환 전에 shape을 복원한다.
        return output_tensor.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        """
        NOTE: 모든 rank에서 입력 텐서가 동일한 디바이스에 있다고 가정한다.
        NOTE: `dst`는 목적지 rank의 local rank다.
        """
        world_size = self.world_size
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            # 음수 dim을 양수로 변환한다.
            dim += input_.dim()

        # 출력 텐서를 할당한다.
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None
        # gather를 수행한다.
        torch.distributed.gather(
            input_, gather_list, dst=self.ranks[dst], group=self.device_group
        )
        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """텐서를 목적지 rank로 블로킹 방식으로 전송한다."""
        """NOTE: `dst`는 목적지 rank의 local rank다."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """소스 rank에서 텐서를 수신한다."""
        """NOTE: `src`는 소스 rank의 local rank다."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """소스 rank의 텐서를 모든 rank로 브로드캐스트한다."""
        if self.world_size == 1:
            return tensor
        torch.distributed.broadcast(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self):
        pass

    def prepare_communication_buffer_for_model(self, model: torch.nn.Module) -> None:
        """
        모델의 통신 버퍼를 준비한다.
        """
        if not self.is_ep_communicator:
            return

        moe_modules = [
            module
            for module in model.modules()
            # TODO(bnell): isinstance를 쓰는 게 맞지만 현재는 어렵다.
            # quant_method.maybe_init_modular_kernel 존재 여부를 확인하는
            # 방식도 검토할 수 있다.
            if (
                module.__class__.__name__ == "FusedMoE"
                or module.__class__.__name__ == "SharedFusedMoE"
            )
        ]
        for module in moe_modules:
            module.maybe_init_modular_kernel()

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        hidden states와 router logits를 적절한 디바이스로 디스패치한다.
        베이스 클래스에서는 no-op이다.
        """
        if extra_tensors is not None:
            return hidden_states, router_logits, extra_tensors
        return hidden_states, router_logits

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        hidden states와 topk weights/ids를 적절한 디바이스로 디스패치한다.
        베이스 클래스에서는 no-op이다.
        """
        if extra_tensors is not None:
            return hidden_states, topk_weights, topk_ids, extra_tensors
        return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        적절한 디바이스에서 hidden states와 router logits를 결합한다.
        베이스 클래스에서는 no-op이다.
        """
        return hidden_states

    def batch_isend_irecv(self, p2p_ops: list):
        raise NotImplementedError
