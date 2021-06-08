import argparse
from typing import NewType
import torch
from torch.nn.modules import upsampling
import torchaudio
import joblib
import numpy
import torch.nn as nn
import torch.nn.functional as F


class Upsampling1D(torch.nn.Module):
    def __init__(self, scale_factor):
        super(Upsampling1D, self).__init__()
        self.upsampling2D = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x):
        x = torch.unsqueeze(x, 3)
        x = self.upsampling2D(x)
        x = x[:, :, :, 0]
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.a = nn.Conv1d(1, 64, 3)
        self.a1 = nn.ReLU(True)
        self.a2 = nn.MaxPool1d(2, 2)
        self.a3 = nn.Conv1d(64, 64, 3)
        self.a4 = nn.ReLU(True)
        self.a5 = nn.MaxPool1d(2, 2)
        self.a6 = nn.Conv1d(64, 32, 3)
        self.a7 = nn.ReLU(True)
        self.a8 = nn.MaxPool1d(2, 2)
        self.a9 = nn.Conv1d(32, 16, 3)
        self.a10 = nn.ReLU(True)
        self.a11 = nn.MaxPool1d(2, 2)
        self.a12 = nn.Conv1d(16, 8, 3)
        self.a13 = nn.ReLU(True)
        self.a14 = nn.MaxPool1d(2, 2)

        self.b = nn.Conv1d(2, 8, 3)
        self.b1 = nn.ReLU(True)
        self.b2 = Upsampling1D(2)
        self.b3 = nn.Conv1d(8, 16, 3)
        self.b4 = nn.ReLU(True)
        self.b5 = Upsampling1D(2)
        self.b6 = nn.Conv1d(16, 32, 3)
        self.b7 = nn.ReLU(True)
        self.b8 = Upsampling1D(2)
        self.b9 = nn.Conv1d(32, 64, 3)
        self.b10 = nn.ReLU(True)
        self.b11 = Upsampling1D(2)
        # nn.ZeroPad1d(),
        self.b13 = nn.Conv1d(64, 1, 4)

    def forward(self, x):
        x = self.a(x)
        print(x.size())
        x = self.a1(x)
        # print(x.size())
        x = self.a2(x)
        print(x.size())
        x = self.a3(x)
        print(x.size())
        x = self.a4(x)
        # print(x.size())
        x = self.a5(x)
        print(x.size())
        x = self.a6(x)
        print(x.size())
        x = self.a7(x)
        # print(x.size())
        x = self.a8(x)
        print(x.size())
        x = self.a9(x)
        print(x.size())
        x = self.a10(x)
        # print(x.size())
        x = self.a11(x)
        print(x.size())
        x = self.a12(x)
        print(x.size())
        x = self.a13(x)
        # print(x.size())
        x = self.a14(x)
        print(x.size())
        x = x.reshape(-1, 2, 16)
        print("reshaped", x.size())

        x = self.b(x)
        print(x.size())
        x = self.b1(x)
        # print(x.size())
        x = self.b2(x)
        print(x.size())
        x = self.b3(x)
        print(x.size())
        x = self.b4(x)
        # print(x.size())
        x = self.b5(x)
        print(x.size())
        x = self.b6(x)
        print(x.size())
        x = self.b7(x)
        # print(x.size())
        x = self.b8(x)
        print(x.size())
        x = self.b9(x)
        print(x.size())
        x = self.b10(x)
        # print(x.size())
        x = self.b11(x)
        print(x.size())
        x = self.b13(x)
        print(x.size())

        return x


model = Autoencoder()
x = torch.randn(16, 1, 193)

# Let's print it
model(x)
