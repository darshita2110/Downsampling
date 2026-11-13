"""
BS-Net-Conv (Cai, Liu & Cai, "BS-Nets: An End-to-End Framework for Band Selection of Hyperspectral Image",
IEEE TGRS 58(3):1969-1984, 2020, arXiv:1904.08269), used here as the "UBS-CNN" method
(Unsupervised Band Selection with a CNN). PyTorch re-implementation.

No published method is defined under the exact name "UBS-CNN"; see documentation/UBSCNN.md for why
BS-Net-Conv was chosen. The layer list follows Table IV of the paper and the authors' code
(github.com/AngryCai/BS-Nets, BS_Net_Conv.py); where the paper is silent the code is followed:
input batch normalisation, global AVERAGE pooling, 'valid' 3x3 convolutions, batch norm after every
hidden convolution.

Input  : a patch x of shape (C, a, a) - C channels (here 8), a x a pixels (16 x 16), values in [0, 1]
                         ┌────────────────── Band Attention Module (BAM) ───────────────────┐
x -> BatchNorm -> x_n -> Conv3x3(64, valid)+BN+ReLU -> GlobalAvgPool -> FC 128 ReLU -> FC C Sigmoid -> w (C weights)
                 x_n ⊗ w  (band re-weighting, BRW: every channel multiplied by its weight)
                         ┌──────────────────── Reconstruction network (RecNet) ─────────────┐
z -> Conv3x3(128)+BN+ReLU -> Conv3x3(64)+BN+ReLU -> ConvT3x3(64)+BN+ReLU -> ConvT3x3(128)+BN+ReLU -> Conv1x1(C) Sigmoid -> x_hat
       16 -> 14                14 -> 12               12 -> 14                  14 -> 16                 (spatial size)

Loss (paper Eq. 10, code: MSE + L1 activity regulariser on w):
    L = mean((x - x_hat)^2) + lambda * sum over samples and channels of |w|      (lambda = 0.01)
The L1 term pushes most weights towards 0 so that only a few channels carry the reconstruction.

Selection (paper Eq. 11): average w over all training patches; the K channels with the largest
average weight are selected.
"""
import torch
from torch import nn


class BandAttention(nn.Module):
    """BAM: one weight in (0, 1) per channel, computed from the spatial-spectral content of the patch."""

    def __init__(self, n_bands, conv_filters=64, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(n_bands, conv_filters, 3, padding=0),
                                  nn.BatchNorm2d(conv_filters), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(conv_filters, hidden), nn.ReLU(),
                                nn.Linear(hidden, n_bands), nn.Sigmoid())

    def forward(self, x):
        h = self.conv(x).mean(dim=(2, 3))          # global average pooling -> (N, 64)
        return self.fc(h)                          # (N, C)


class ReconstructionNet(nn.Module):
    """RecNet: conv encoder + transposed-conv decoder restoring all C channels of the patch."""

    def __init__(self, n_bands):
        super().__init__()

        def block(layer, ch):
            return [layer, nn.BatchNorm2d(ch), nn.ReLU()]
        self.net = nn.Sequential(
            *block(nn.Conv2d(n_bands, 128, 3), 128),           # 16 -> 14
            *block(nn.Conv2d(128, 64, 3), 64),                  # 14 -> 12
            *block(nn.ConvTranspose2d(64, 64, 3), 64),          # 12 -> 14
            *block(nn.ConvTranspose2d(64, 128, 3), 128),        # 14 -> 16
            nn.Conv2d(128, n_bands, 1), nn.Sigmoid())           # 1x1 conv back to C channels in [0, 1]

    def forward(self, z):
        return self.net(z)


class BSNetConv(nn.Module):
    def __init__(self, n_bands):
        super().__init__()
        self.input_norm = nn.BatchNorm2d(n_bands)     # as in the authors' code
        self.bam = BandAttention(n_bands)
        self.recnet = ReconstructionNet(n_bands)

    def forward(self, x):
        xn = self.input_norm(x)
        w = self.bam(xn)                               # (N, C) band weights
        z = xn * w[:, :, None, None]                   # band re-weighting
        return self.recnet(z), w


def bsnet_loss(x, x_hat, w, l1_lambda):
    """Returns (total, mse, l1) for one batch. L1 is summed over samples and bands, as in the authors' code
    (TensorFlow l1_regularizer used as an activity regulariser sums |w| over the whole batch)."""
    mse = torch.mean((x - x_hat) ** 2)
    l1 = l1_lambda * w.abs().sum()
    return mse + l1, mse, l1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def band_weights(model, patches, batch_size=500):
    """Attention weights for every patch (evaluation mode, no gradient). patches: float tensor (N, C, a, a)."""
    model.eval()
    out = [model(patches[i:i + batch_size])[1] for i in range(0, len(patches), batch_size)]
    return torch.cat(out).numpy()


@torch.no_grad()
def reconstruction_mse(model, patches, batch_size=500):
    """Mean squared error of the network's own patch reconstruction (all channels re-weighted, not a subset)."""
    model.eval()
    se, n = 0.0, 0
    for i in range(0, len(patches), batch_size):
        x = patches[i:i + batch_size]
        se += float(((model(x)[0] - x) ** 2).sum())
        n += x.numel()
    return se / max(n, 1)
