import torch
import numpy as np
import torch.nn as nn


try:
    from torch.amp import autocast

    torch_amp_new = True
except:
    from torch.cuda.amp import autocast

    torch_amp_new = False

from torchaudio.transforms import AmplitudeToDB, MelSpectrogram


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        """
        Feature extraction module.

        Args:
            params (dict): Parameters for the spectrogram.
            aug_config (dict, optional): Configuration for data augmentation. Defaults to None.
            top_db (float, optional): Threshold for computing the amplitude to dB. Defaults to None.
            norm (str, optional): Normalization method. Defaults to "min_max".
        """
        super().__init__()

        self.audio2melspec = MelSpectrogram(
            n_fft=cfg.melspec.n_fft,
            hop_length=cfg.melspec.hop_length,
            win_length=cfg.melspec.win_length,
            n_mels=cfg.melspec.n_mels,
            sample_rate=cfg.audio.sample_rate,
            f_min=cfg.melspec.f_min,
            f_max=cfg.melspec.f_max,
            power=cfg.melspec.power,
        )
        self.amplitude_to_db = AmplitudeToDB(top_db=cfg.melspec.top_db)

        if cfg.melspec.norm == "mean_std":
            self.normalizer = MeanStdNorm()
        elif cfg.melspec.norm == "min_max":
            self.normalizer = MinMaxNorm()
        elif cfg.melspec.norm == "simple":
            self.normalizer = SimpleNorm()
        else:
            self.normalizer = nn.Identity()

    def forward(self, x):
        """
        Forward pass of the feature extractor.

        Args:
            x (torch.Tensor): Input audio data.

        Returns:
            torch.Tensor: Extracted features.
        """

        with (
            autocast("cuda", enabled=False)
            if torch_amp_new
            else autocast(enabled=False)
        ):
            melspec = self.audio2melspec(x.float())
            melspec = self.amplitude_to_db(melspec)
            melspec = self.normalizer(melspec)

        return melspec

class MinMaxNorm(nn.Module):
    def __init__(self, eps=1e-6):
        """
        Module for performing min-max normalization on input data.

        Args:
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps

    def forward(self, X):
        """
        Forward pass of the min-max normalization module.

        Args:
            X (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Normalized data.
        """
        min_ = torch.amax(X, dim=(1, 2), keepdim=True)
        max_ = torch.amin(X, dim=(1, 2), keepdim=True)
        return (X - min_) / (max_ - min_ + self.eps)

class SimpleNorm(nn.Module):
    def __init__(self):
        """
        Module for performing simple normalization on input data.
        """
        super().__init__()

    def forward(self, x):
        """
        Forward pass of the simple normalization module.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Normalized data.
        """
        return (x - 40) / 80

class MeanStdNorm(nn.Module):
    def __init__(self, eps=1e-6):
        """
        Module for performing mean and standard deviation normalization on input data.

        Args:
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps

    def forward(self, X):
        """
        Forward pass of the mean and standard deviation normalization module.

        Args:
            X (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Normalized data.
        """
        mean = X.mean((1, 2), keepdim=True)
        std = X.reshape(X.size(0), -1).std(1, keepdim=True).unsqueeze(-1)
        return (X - mean) / (std + self.eps)
