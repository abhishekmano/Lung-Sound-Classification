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


def loadmodel_3gan(filename):

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

    class Discriminator(nn.Module):
        def __init__(self, num_dec_features, num_channels):
            super(Discriminator, self).__init__()

            #torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            self.main = nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv1d(num_channels, 128, 3, bias=False)
                ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool1d(3),
                nn.utils.spectral_norm(
                    nn.Conv1d(128, 128, 3, bias=False)
                ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool1d(3),
                nn.utils.spectral_norm(
                    nn.Conv1d(
                        128, 128, 3, bias=False
                    )
                ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool1d(3),
                nn.utils.spectral_norm(
                    nn.Conv1d(
                        128, 128, 3, bias=False
                    )
                ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool1d(3),
            )
            self.true_fake = nn.Linear(128, 1)  # output(1,3088,158)
            self.classifier = nn.Linear(128, 3)
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=2)

        def forward(self, features, case):
            hidden = self.main(features)
            # print("Hidden-size:",hidden.size())
            reshaped_hidden = hidden.view(-1, 1, hidden.size(1)*hidden.size(2))
            #print("Reshaped hidden-size:",reshaped_hidden.size())
            if(case == 1):  # sigmoid
                linear = self.true_fake(reshaped_hidden)
                # print(linear.size())
                out = self.sigmoid(linear)
                # print(out.shape)
                return out
            else:  # softmax
                linear = self.classifier(reshaped_hidden)
                # print(linear.size())
                out = F.log_softmax(linear, dim=2)
                #out = self.softmax(linear)
                # print(out.shape)
                return out

    def consistency(value):
        if(value < 0.65):
            return "Moderate"
        elif(value < 0.85):
            return "High"
        else:
            return "Very High"

    def predict_out():
        model_path = 'savedmodels/GAN_3class_best_discriminator.pt'
        discriminator = Discriminator(193, 1)
        discriminator.load_state_dict(torch.load(
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
            output = discriminator(audio_tensor, 2)  # [1,1,2]
            output = output.permute(1, 0, 2)[0]  # [1,2]
            #print("model output:", output)

            softmax = nn.Softmax(dim=1)
            res = softmax(output)
            #print("Softmax", res)

            z = res.squeeze(0).numpy() * 100
            z = [round(elem, 2)for elem in z]  # percentage of each class
            # print(z)

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
# loadmodel_3gan("crack_1.wav")
# loadmodel_3gan("crack_2.wav")
# loadmodel_3gan("crack_3.wav")
# loadmodel_3gan("crack_4.wav")
# loadmodel_3gan("crack_5.wav")
# loadmodel_3gan("crack_6.wav")
# loadmodel_3gan("crack_7.wav")
# loadmodel_3gan("crack_8.wav")
# loadmodel_3gan("crack_9.wav")
# loadmodel_3gan("crack_10.wav")

# loadmodel_3gan("normal1.wav")
# loadmodel_3gan("normal2.wav")
# loadmodel_3gan("normal3.wav")
# loadmodel_3gan("normal4.wav")
# loadmodel_3gan("normal5.wav")
# loadmodel_3gan("normal6.wav")
# loadmodel_3gan("normal7.wav")
# loadmodel_3gan("normal8.wav")
# loadmodel_3gan("normal9.wav")
# loadmodel_3gan("normal10.wav")
