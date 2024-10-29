from lightorch.nn.criterions import LighTorchLoss, ELBO, Loss, MSELoss
import torch.nn.functional as f
from torch import Tensor
import torch

class TV(LighTorchLoss):
    def __init__(self, factor: float = 1) -> None:
        super().__init__(
            labels=self.__class__.__name__,
            factors={self.__class__.__name__: factor},
        )

    def forward(self, **kwargs) -> Tensor:
        return torch.norm(kwargs['reconstruction'][:, :-1] - kwargs['reconstruction'][:, 1:], p=1).sum()

class BinaryCrossEntropy(LighTorchLoss):
    def __init__(self, factor: float = 1) -> None:
        super().__init__(
            labels=self.__class__.__name__,
            factors={self.__class__.__name__: factor},
        )
    def forward(self, **kwargs) -> Tensor:
        return f.binary_cross_entropy(kwargs['binary_pred'], kwargs['binary_target'])

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

def criterion(beta: float, *args) -> LighTorchLoss:
    elbo = ELBO(beta, MSELoss(args[0]))
    physics = SeismicVelocityLoss(omega=2*torch.pi, damping=0.1, factor=args[1])
    binary = BinaryCrossEntropy(args[2])
    return Loss(
        elbo,
        # physics,
        binary
    )
