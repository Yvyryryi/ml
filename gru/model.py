from typing import Sequence
from lightorch.nn.sequential.residual import GRU
from lightorch.training.supervised import Module
from torch import nn, Tensor
from torch.nn.modules import Sequential
from .loss import criterion


class SignalProcessingGRU(GRU):
    def __init__(self, input_size: int, hidden_size: int, gru_layers: int, res_layers: int, bias: bool = True, batch_first: bool = True, dropout: float = 0, bidirectional: bool = False, device: Union[Any, None] = None, dtype: Union[Any, None] = None) -> None:
        super().__init__(input_size, hidden_size, gru_layers, res_layers, bias, batch_first, dropout, bidirectional, device, dtype)

class Model(Module):
    def __init__(self, **hparams) -> None:
        super().__init__(**hparams)
        self.model = SignalProcessingGRU()
        self.criterion = criterion()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
