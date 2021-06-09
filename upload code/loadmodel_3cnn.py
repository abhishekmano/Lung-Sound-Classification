import argparse
from re import X
import torch
from torch.serialization import load
import torchaudio
import joblib
import numpy as np

import librosa
import csv

import torch.nn as nn
import torch.nn.functional as F


def loadmodel_3cnn(filename):

    torch.backends.cudnn.benchmark = True

    def feature_extract(audio, sr):
        audio = audio.squeeze()
        y = audio.numpy()

        stft = librosa.stft(y)
        stft = np.abs(stft)

        mfcc = librosa.feature.mfcc(y, sr, n_mfcc=40)
        mfcc = np.mean(mfcc, axis=1)

        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        chroma = np.mean(chroma, axis=1)  # 12 features

        mel = librosa.feature.melspectrogram(y, sr)
        mel = np.mean(mel, axis=1)  # 128

        contrast = librosa.feature.spectral_contrast(y, sr)
        contrast = np.mean(contrast, axis=1)  # 7

        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz = np.mean(tonnetz, axis=1)  # 6

        features = np.hstack([mfcc, chroma, mel, contrast, tonnetz])
        features = features.reshape((1, 193))

        features = torch.from_numpy(features)

        features = features.float()

        return features

    class NetM3(nn.Module):
        def __init__(self):
            super(NetM3, self).__init__()
            self.conv1 = nn.Conv1d(1, 128, 3)
            self.bn1 = nn.BatchNorm1d(128)
            self.pool1 = nn.MaxPool1d(3)
            self.conv2 = nn.Conv1d(128, 128, 3)
            self.bn2 = nn.BatchNorm1d(128)
            self.pool2 = nn.MaxPool1d(3)
            self.conv3 = nn.Conv1d(128, 128, 3)
            self.bn3 = nn.BatchNorm1d(128)
            self.pool3 = nn.MaxPool1d(3)
            self.conv4 = nn.Conv1d(128, 128, 3)
            self.bn4 = nn.BatchNorm1d(128)
            self.pool4 = nn.MaxPool1d(3)
            # self.avgPool = nn.AvgPool1d(25)  # input should be 512x30 so this outputs a 512x1
            self.fc1 = nn.Linear(128, 3)

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
            #x = self.avgPool(x)
            x = x.permute(0, 2, 1)  # change the 512x1 to 1x512
            x = self.fc1(x)
            return F.log_softmax(x, dim=2)  # Output: torch.Size([N, 1, 2])

    def consistency(value):
        if(value < 0.65):
            return "Moderate"
        elif(value < 0.85):
            return "High"
        else:
            return "Very High"

    def predict_out():
        #model_path = 'savedmodels/cnn_3class.pt'
        model_path = 'savedmodels/best_model_cnn_3class.pt'
        audio_cnn = NetM3()
        audio_cnn.load_state_dict(torch.load(
            model_path, map_location=torch.device('cpu')))

        path = "./static/uploads/" + filename
        try:
            audio_tensor, sample_rate = torchaudio.load(
                path, normalize=True)  # [ 1 , 193 ]
        except RuntimeError:
            print("Couldnt open file")
            return "101", "Error", [0, 0, 0]
        feature = feature_extract(audio_tensor, sample_rate)
        with torch.no_grad():
            audio_tensor = feature  # [1 , 193]

            audio_tensor = torch.unsqueeze(audio_tensor, 0)  # [1,1,193]
            output = audio_cnn(audio_tensor)  # [1,1,2]
            output = output.permute(1, 0, 2)[0]  # [1,2]
            #print("model output:", output)

            softmax = nn.Softmax(dim=1)
            res = softmax(output)
            #print("Softmax", res)

            z = res.squeeze(0).numpy() * 100
            z = [round(elem, 2)for elem in z]  # percentage of each class
            print(z)

            pred = output.max(1, keepdim=True)[1]
            pred = pred.squeeze(0).squeeze(0).item()

            x_group = ["Crack", "Normal", "Wheeze"]
            x = x_group[pred]
            y = consistency(res[0][pred])
            #print("predicted:", pred)

            return x, y, z

    x, y, z = predict_out()
    print(x, y, z)
    return x, y, z


# for running only comment out when on web
# loadmodel_3cnn("crack_1.wav")
# loadmodel_3cnn("crack_2.wav")
# loadmodel_3cnn("crack_3.wav")
# loadmodel_3cnn("crack_4.wav")
# loadmodel_3cnn("crack_5.wav")
# loadmodel_3cnn("crack_6.wav")
# loadmodel_3cnn("crack_7.wav")
# loadmodel_3cnn("crack_8.wav")
# loadmodel_3cnn("crack_9.wav")
# loadmodel_3cnn("crack_10.wav")
