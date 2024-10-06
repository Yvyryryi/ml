from lightorch.nn.criterions import LighTorchLoss, ELBO, BinaryCrossEntropy, Loss, MSE
from torch import Tensor
import torch

class TV(LighTorchLoss):
    def __init__(self, factor: float = 1) -> None:
        super().__init__(
            labels=self.__class__.__name__,
            factors={self.__class__.__name__: factor},
        )

    def forward(self, kwargs) -> Tensor:
        return torch.norm(kwargs['input'][:, :-1] - kwargs['input'][:, 1:], p=1).sum()


class PhysicsInformedLoss(LighTorchLoss):
    def __init__(self, c: float, g: float, factor: float = 1) -> None:
        super().__init__(
            labels=self.__class__.__name__,
            factors={self.__class__.__name__: factor},
        )
        self.c = c
        self.g = g

    def u_tt(self, t: Tensor, z: Tensor) -> Tensor:
        u = self(t, z)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
        return u_tt

    def u_zz(self, t: Tensor, z: Tensor) -> Tensor:
        u = self(t, z)
        u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
        return u_zz

    def u_z(self, t: Tensor, z: Tensor) -> Tensor:
        u = self(t, z)
        u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        return u_z

    def pil(self, t: Tensor, z: Tensor) -> Tensor:
        u_tt_val = self.u_tt(t, z)
        u_zz_val = self.u_zz(t, z)
        u_z_val = self.u_z(t, z)
        pde_residual = u_tt_val - (self.c ** 2 * u_zz_val + self.g * u_z_val)
        return torch.mean(pde_residual ** 2)

    def forward(self, x: Tensor) -> Tensor:
        z: Tensor = ...
        return self.pil(x, z)


def criterion(beta: float, *args) -> LighTorchLoss:
    return Loss(
        ELBO(beta, MSE(args[0])),
        TV(args[1]),
        PhysicsInformedLoss(c=1, g=3.71, factor=args[2]),
        BinaryCrossEntropy(args[3])
    )

