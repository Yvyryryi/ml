from lightorch.nn.criterions import LighTorchLoss
from torch import Tensor
import torch

class criterion(LighTorchLoss):
    def __init__(self) -> None:
        super().__init__(
            labels = ,
            factors = ,
        )

    def u_tt(self, t: Tensor, z: Tensor):
        u = self(t, z)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
        return u_tt

    def u_zz(self, t: Tensor, z: Tensor):
        u = self(t, z)
        u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
        return u_zz

    def u_z(self, t: Tensor, z: Tensor):
        u = self(t, z)
        u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        return u_z

    def pil(self, t: Tensor, z: Tensor, c: float, g: float): ## different g for mars and lunar
        u_tt_val = self.u_tt(t, z)
        u_zz_val = self.u_zz(t, z)
        u_z_val = self.u_z(t, z)
        pde_residual = u_tt_val - (c**2 * u_zz_val + g * u_z_val)
        return torch.mean(pde_residual**2)

    def forward(self, signal: Tensor, classification: Tensor) -> Tensor:
        ## ELBO (signal)
        ## Total Variance (signal)
        ## PIL (signal)
        ## binary (classification)
        return


