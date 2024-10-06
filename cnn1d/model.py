from lightorch.training.supervised import Module
from typing import Sequence

from torch import nn, Tensor
from ..loss import criterion

def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    activation: nn.Module,
    dropout: float
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm1d(out_channels),
        activation(),
        nn.Dropout(p = dropout)
    )


def trans_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    activation: nn.Module,
    dropout: float
) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm1d(out_channels),
        activation(),
        nn.Dropout(p = dropout)
    )

class SignalProcessing1DCNN(nn.Sequential):
    def __init__(
        self,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        paddings: Sequence[int],
        dropout: float,
        activation: nn.Module
    ) -> None:
        super().__init__(
            *[
                conv_block(in_channel, out_channel, kernel_size, stride, padding, activation, dropout) \
                for out_channel, in_channel, kernel_size, stride, padding in \
                zip(channels[1:], channels[:-1], kernel_sizes, strides, paddings)
            ]
        )

class Model(Module):
    def __init__(self, **hparams) -> None:
        super().__init__(**hparams)
        self.model = SignalProcessing1DCNN()
        self.criterion = criterion(hparams['beta'], *hparams['lambdas'])

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
