import argparse
import torch
import torchaudio
import joblib
import numpy
import torch.nn as nn
import torch.nn.functional as F


def loadmodel(filename):

    torch.backends.cudnn.benchmark = True

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
            self.avgPool = nn.AvgPool1d(120)
            self.fc1 = nn.Linear(512, 2)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(self.bn1(x))
            x = self.pool1(x)
            x = self.conv2(x)
            x = F.relu(self.bn2(x))
            x = self.pool2(x)
            x = self.conv3(x)
            x = F.relu(self.bn3(x))
            x = self.pool3(x)
            x = self.conv4(x)
            x = F.relu(self.bn4(x))
            x = self.pool4(x)
            x = self.avgPool(x)
            x = x.permute(0, 2, 1)  # change the 512x1 to 1x512
            x = self.fc1(x)
            return F.log_softmax(x, dim=2)  # Output: torch.Size([N, 1, 10])

    def predict_out():
        model_path = 'savedmodels/unsup2.pt'
        audio_cnn = NetM3()
        audio_cnn.load_state_dict(torch.load(
            model_path, map_location=torch.device('cpu')))
        # audio_cnn = torch.load('savedmodels/unsup_2class.pt',
        #                        map_location=torch.device('cpu'))
        path = "./static/uploads/" + filename
        audio_tensor, sample_rate = torchaudio.load(path, normalize=True)
        with torch.no_grad():
            audio_tensor = audio_tensor  # .to("cuda")
            print("size of tensor:", audio_tensor.size())
            print(type(audio_tensor))
            print("tensor:", audio_tensor.size())
            print(audio_tensor)
            audio_tensor = torch.unsqueeze(audio_tensor, 0)
            print("size of unsueezed tensor:", audio_tensor.size())

            output = audio_cnn(audio_tensor)
            print("Output tensor: ", output)
            print("output size: ", output.size())
            output = output.permute(1, 0, 2)[0]
            print("Output tensor", output)
            pred = output.max(1, keepdim=True)[1]
            pred = pred.squeeze(0).squeeze(0).item()
            print("predicted:", pred)
            return pred

    result = predict_out()
    return result
