from lightorch.training.supervised import Module
from lightorch.nn.sequential.residual import LSTM
from torch import Tensor
from ..loss import criterion

class SignalProcessingLSTM(LSTM):
    def __init__(self, input_size: int, hidden_size: int, lstm_layers: int, res_layers: int, bias: bool = True, batch_first: bool = True, dropout: float = 0, bidirectional: bool = False, proj_size: int = 0, device: Union[Any, None] = None, dtype: Union[Any, None] = None) -> None:
        super().__init__(input_size, hidden_size, lstm_layers, res_layers, bias, batch_first, dropout, bidirectional, proj_size, device, dtype)

class Model(Module):
    def __init__(self, **hparams) -> None:
        super().__init__(**hparams)
        self.model = SignalProcessingLSTM(
            hparams['input_size'],
            hparams['hidden_size'],
            hparams['lstm_layers'],
            hparams['res_layers'],
            True,
            True,
            hparams['dropout'],
            hparams['bidirectional'],
        )
        self.criterion = criterion(hparams['beta'], *hparams['lambdas'])

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
