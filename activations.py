from torch import nn, Tensor
import torch

class Cosine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        return torch.cos(x)

class Sine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        return torch.sen(x)
