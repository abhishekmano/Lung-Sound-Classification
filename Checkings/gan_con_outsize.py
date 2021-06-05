import argparse
from typing import NewType
import torch
import torchaudio
import joblib
import numpy
import torch.nn as nn
import torch.nn.functional as F


class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary
    If batch shuffle is enabled, only a single shuffle is applied to the entire
    batch, rather than each sample in the batch.
    """

    def __init__(self, shift_factor, batch_shuffle=False):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor
        self.batch_shuffle = batch_shuffle

    def forward(self, x):
        # Return x if phase shift is disabled
        if self.shift_factor == 0:
            return x

        if self.batch_shuffle:
            # Make sure to use PyTorcTrueh to generate number RNG state is all shared
            k = (
                int(torch.Tensor(1).random_(0, 2 * self.shift_factor + 1))
                - self.shift_factor
            )

            # Return if no phase shift
            if k == 0:
                return x

            # Slice feature dimension
            if k > 0:
                x_trunc = x[:, :, :-k]
                pad = (k, 0)
            else:
                x_trunc = x[:, :, -k:]
                pad = (0, -k)

            # Reflection padding
            x_shuffle = F.pad(x_trunc, pad, mode="reflect")

        else:
            # Generate shifts for each sample in the batch
            k_list = (
                torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1)
                - self.shift_factor
            )
            k_list = k_list.numpy().astype(int)

            # Combine sample indices into lists so that less shuffle operations
            # need to be performed
            k_map = {}
            for idx, k in enumerate(k_list):
                k = int(k)
                if k not in k_map:
                    k_map[k] = []
                k_map[k].append(idx)

            # Make a copy of x for our output
            x_shuffle = x.clone()

            # Apply shuffle to each sample
            for k, idxs in k_map.items():
                if k > 0:
                    x_shuffle[idxs] = F.pad(
                        x[idxs][..., :-k], (k, 0), mode="reflect")
                else:
                    x_shuffle[idxs] = F.pad(
                        x[idxs][..., -k:], (0, -k), mode="reflect")

        assert x_shuffle.shape == x.shape, "{}, {}".format(
            x_shuffle.shape, x.shape)
        return x_shuffle


class Discriminator(nn.Module):
    def __init__(self, num_dec_features, num_channels):
        super(Discriminator, self).__init__()

        #torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.cnv1 = nn.utils.spectral_norm(
            nn.Conv1d(num_channels, num_dec_features, 8, bias=False)
        )
        self.lkr1 = nn.LeakyReLU(0.2, inplace=True)
        self.fs1 = PhaseShuffle(2)
        self.cnv2 = nn.utils.spectral_norm(
            nn.Conv1d(num_dec_features, num_dec_features * 2, 8, bias=False)
        )
        self.lkr2 = nn.LeakyReLU(0.2, inplace=True)
        self.fs2 = PhaseShuffle(2)

        self.cnv3 = nn.utils.spectral_norm(
            nn.Conv1d(
                num_dec_features * 2, num_dec_features * 4, 8, bias=False
            )
        )
        self.lkr3 = nn.LeakyReLU(0.2, inplace=True)
        self.fs3 = PhaseShuffle(2)

        self.cnv4 = nn.utils.spectral_norm(
            nn.Conv1d(
                num_dec_features * 4, num_dec_features * 8, 8, bias=False
            )
        )
        self.lkr4 = nn.LeakyReLU(0.2, inplace=True)
        self.fs4 = PhaseShuffle(2)

        self.cnv5 = nn.utils.spectral_norm(
            nn.Conv1d(
                num_dec_features * 8, num_dec_features * 16, 8, bias=False
            )
        )
        self.lkr5 = nn.LeakyReLU(0.2, inplace=True)
        self.fs5 = PhaseShuffle(2)

        self.true_fake = nn.Linear(487904, 1)  # output(1,3088,158)
        self.classifier = nn.Linear(487904, 2)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = F.log_softmax()

    def forward(self, x, case):
        x = self.cnv1(x)
        x = self.lkr1(x)
        self.fs1
        print("conv1", x.size())

        x = self.cnv2(x)
        x = self.lkr2(x)
        self.fs2
        print("conv2", x.size())

        x = self.cnv3(x)
        x = self.lkr3(x)
        self.fs1
        print("conv3", x.size())

        x = self.cnv4(x)
        x = self.lkr4(x)
        self.fs1
        print("conv4", x.size())

        x = self.cnv5(x)
        self.fs1
        print("conv5", x.size())

        x = x.view(-1, 1, x.size(1) * x.size(2))

        print("permute", x.size())
        # x = x.unsqueeze(0)
        # x = x.permute(0, 2, 1)
        print("permute", x.size())
        #print("Reshaped hidden-size:",reshaped_hidden.size())
        if(case == 1):  # sigmoid
            linear = self.true_fake(x)
            print("fc1", linear.size())
            out = self.sigmoid(linear)
            print("out", out.size())
            return out
        else:  # softmax
            linear = self.classifier(x)
            print("fc1", linear.size())
            out = F.log_softmax(linear, dim=2)
            print("out", out.size())
            return out


model = Discriminator(193, 1)
x = torch.randn(2, 1, 193)

# Let's print it
res = model(x, 2)
print(res)
