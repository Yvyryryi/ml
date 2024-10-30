from lightorch.nn.criterions import LighTorchLoss, ELBO, Loss, MSELoss, BinaryCrossEntropy
from typing import Tuple, Optional
from torch import Tensor
import torch

class TV(LighTorchLoss):
    def __init__(self, factor: float = 1) -> None:
        super().__init__(
            labels=self.__class__.__name__,
            factors={self.__class__.__name__: factor},
        )

    def forward(self, **kwargs) -> Tuple[Tensor, Tensor]:
        out = torch.norm(kwargs['input'][:, :-1] - kwargs['input'][:, 1:], p=1).sum()
        return out, out * self.factors[self.__class__.__name__]

class SeismicVelocityLoss(LighTorchLoss):
    def __init__(self, omega: float, damping: float, factor: float = 1) -> None:
        super().__init__(
            labels=self.__class__.__name__,
            factors={self.__class__.__name__: factor},
        )
        self.omega = omega
        self.damping = damping

    def v_t(self, t: Tensor, v: Tensor) -> Tensor:
        return torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    def v_tt(self, t: Tensor, v: Tensor) -> Tensor:
        v_t = self.v_t(t, v)
        return torch.autograd.grad(v_t, t, grad_outputs=torch.ones_like(v_t), create_graph=True)[0]

    def pil(self, t: Tensor, v: Tensor) -> Tensor:
        v_t = self.v_t(t, v)
        v_tt = self.v_tt(t, v)
        pde_residual = v_tt + 2 * self.damping * v_t + self.omega**2 * v
        return torch.mean(pde_residual ** 2)

    def forward(self, **kwargs) -> Tensor:
        t: Tensor = kwargs['time']
        v: Tensor = kwargs['velocity']
        return self.pil(t, v)

class BinaryLoss(BinaryCrossEntropy):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = "mean", factor: float = 1) -> None:
        super().__init__(weight, size_average, reduce, reduction, factor)
    def forward(self, **kwargs) -> Tuple[Tensor, Tensor]:
        return super().forward(
            input = kwargs['binary_pred'],
            target = kwargs['binary_target'],
        )


def criterion(beta: float, *args) -> LighTorchLoss:
    elbo = ELBO(beta, MSELoss(factor = args[0]))
    tv = TV(args[1])
    binary = BinaryLoss(factor = args[2])
    return Loss(elbo, tv, binary)
