from lightorch.training.supervised import Module
from typing import Sequence, Dict, Any, Tuple
from torch import nn, Tensor
from loss import criterion
import torch

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
        nn.Dropout(p=dropout)
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
        nn.Dropout(p=dropout)
    )

class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        paddings: Sequence[int],
        dropout: float,
        activation: nn.Module,
        latent_dim: int
    ) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            *[conv_block(in_channel, out_channel, kernel_size, stride, padding, activation, dropout)
              for in_channel, out_channel, kernel_size, stride, padding in
              zip([input_channels] + list(channels[:-1]), channels, kernel_sizes, strides, paddings)]
        )
        self.fc_mu = nn.Linear(200, latent_dim)
        self.fc_logvar = nn.Linear(200, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.conv_layers(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        paddings: Sequence[int],
        dropout: float,
        activation: nn.Module,
        output_channels: int
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 200)
        self.conv_layers = nn.Sequential(
            *[trans_conv_block(in_channel, out_channel, kernel_size, stride, padding, activation, dropout)
              for in_channel, out_channel, kernel_size, stride, padding in
              zip(channels[:-1], channels[1:], kernel_sizes, strides, paddings)]
        )
        self.final_conv = nn.ConvTranspose1d(channels[-1], output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        x = self.fc(z)
        x = self.conv_layers(x)
        x = self.final_conv(x)
        return x.squeeze(1)

class BinaryClassifier(nn.LSTM):
    def __init__(self, input_size: int, layers: int, dropout: float) -> None:
        super().__init__(input_size = input_size, hidden_size = 1, num_layers = layers, dropout = dropout, batch_first = True)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.transpose(-2, -1))[0].squeeze(-1)

class VAE(nn.Module):
    def __init__(
        self,
        input_channels: int,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        paddings: Sequence[int],
        dropout: float,
        latent_dim: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(input_channels, encoder_channels, kernel_sizes, strides, paddings, dropout, nn.Tanh, latent_dim)
        self.decoder = Decoder(latent_dim, decoder_channels, kernel_sizes[::-1], strides[::-1], paddings[::-1], dropout, nn.Tanh, 1)

        self.binary_classifier = BinaryClassifier(input_channels, layers, dropout)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        var = torch.exp(0.5 * logvar)
        eps = torch.randn_like(var, device = 'cuda')
        return mu + eps * var

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        binary_pred = self.binary_classifier(x)
        recon_x = self.decoder(z)
        return recon_x, binary_pred, mu, logvar

class Model(Module):
    def __init__(self, **hparams: Dict[str, Any]) -> None:
        super().__init__(**hparams)
        self.vae = VAE(
            input_channels=hparams['input_channels'],
            encoder_channels=hparams['encoder_channels'],
            decoder_channels=hparams['decoder_channels'],
            kernel_sizes=hparams['kernel_sizes'],
            strides=hparams['strides'],
            paddings=hparams['paddings'],
            dropout=hparams['dropout'],
            latent_dim=hparams['latent_dim'],
            layers=hparams['layers'],
        )
        self.criterion = criterion(hparams['beta'], *hparams['lambdas'])

    def loss_forward(self, batch: Tensor, idx: int):
        _, input, binary_target = batch
        reconstruction, binary_pred, mu, logvar = self(input)
        return dict(
            input=reconstruction,
            target=input[:, 0, :],
            logvar=logvar,
            mu=mu,
            binary_pred=binary_pred,
            binary_target=binary_target,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.vae(x)
