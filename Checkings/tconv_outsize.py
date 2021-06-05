import argparse
from typing import NewType
import torch
import torchaudio
import joblib
import numpy
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, num_enc_features, num_channels, num_z):
        super(Generator, self).__init__()
        self.num_enc_features = num_enc_features
        self.fc1 = nn.Linear(num_z, 16 * 158 * num_enc_features)
        self.cnvt1 = nn.ConvTranspose1d(
            num_enc_features * 16,
            num_enc_features * 8,
            8,
            stride=1,
            padding=0,
            output_padding=0,
            bias=False,
        )
        self.r1 = nn.ReLU(True)
        self.cnvt2 = nn.ConvTranspose1d(
            num_enc_features * 8,
            num_enc_features * 4,
            8,
            stride=1,
            padding=0,
            output_padding=0,
            bias=False,
        )
        self.r2 = nn.ReLU(True)
        self.cnvt3 = nn.ConvTranspose1d(
            num_enc_features * 4,
            num_enc_features * 2,
            8,
            stride=1,
            padding=0,
            output_padding=0,
            bias=False,
        )
        self.r3 = nn.ReLU(True)
        self.cnvt4 = nn.ConvTranspose1d(
            num_enc_features * 2,
            num_enc_features,
            8,
            stride=1,
            padding=0,
            output_padding=0,
            bias=False,
        )
        self.r4 = nn.ReLU(True)
        self.cnvt5 = nn.ConvTranspose1d(
            num_enc_features,
            num_channels,
            8,
            stride=1,
            padding=0,
            output_padding=0,
            bias=False,
        )
        self.th = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        print("fc1", x.size())

        x = x.view(-1, self.num_enc_features * 16, 158)
        print("Re shape", x.size())

        x = self.cnvt1(x)
        x = self.r1(x)
        print("conv1", x.size())

        x = self.cnvt2(x)
        x = self.r2(x)
        print("conv2", x.size())

        x = self.cnvt3(x)
        x = self.r3(x)
        print("conv3", x.size())

        x = self.cnvt4(x)
        x = self.r4(x)
        print("conv4", x.size())

        x = self.cnvt5(x)
        x = self.th(x)
        print("conv5", x.size())

        return x  # Output: torch.Size([N, 1, 10])


model = Generator(193, 1, 500)
x = torch.randn(1, 500)

# Let's print it
model(x)
