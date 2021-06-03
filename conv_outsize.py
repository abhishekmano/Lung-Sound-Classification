import argparse
from typing import NewType
import torch
import torchaudio
import joblib
import numpy
import torch.nn as nn
import torch.nn.functional as F


class NetM3(nn.Module):
    def __init__(self):
        super(NetM3, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        # input should be 512x30 so this outputs a 512x1
        self.avgPool = nn.AvgPool1d(25)
        self.fc1 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        print("conv1", x.size())
        x = self.pool1(x)
        print("pool1", x.size())
        x = self.conv2(x)
        print("conv2", x.size())
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        print("pool2", x.size())
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        print("conv3", x.size())
        x = self.pool3(x)
        print("pool3", x.size())
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        print("conv4", x.size())
        x = self.pool4(x)
        print("pool4", x.size())
        x = self.avgPool(x)
        print("avg pool", x.size())
        x = x.permute(0, 2, 1)  # change the 512x1 to 1x512
        print("perm", x.size())
        x = self.fc1(x)
        print("linear", x.size())
        print("softmax", F.log_softmax(x, dim=2).size())
        return F.log_softmax(x, dim=2)  # Output: torch.Size([N, 1, 10])


model = NetM3()
x = torch.randn(1, 1, 26440)

# Let's print it
model(x)
