import argparse
from re import X
import torch
from torch.serialization import load
import torchaudio
import joblib
import numpy as np

import librosa
import csv
import sklearn
from sklearn import svm

import torch.nn as nn
import torch.nn.functional as F

import pickle


def loadmodel_svm(filename):

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

    class Upsampling1D(torch.nn.Module):
        def __init__(self, scale_factor):
            super(Upsampling1D, self).__init__()
            self.upsampling2D = nn.UpsamplingBilinear2d(
                scale_factor=scale_factor)

        def forward(self, x):
            x = torch.unsqueeze(x, 3)
            x = self.upsampling2D(x)
            x = x[:, :, :, 0]
            return x

    class Autoencoder(nn.Module):
        def __init__(self, z_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 64, 3),
                nn.ReLU(True),
                nn.MaxPool1d(2, 2),
                nn.Conv1d(64, 64, 3),
                nn.ReLU(True),
                nn.MaxPool1d(2, 2),
                nn.Conv1d(64, 32, 3),
                nn.ReLU(True),
                nn.MaxPool1d(2, 2),
                nn.Conv1d(32, 16, 3),
                nn.ReLU(True),
                nn.MaxPool1d(2, 2),
                nn.Conv1d(16, 8, 3),
                nn.ReLU(True),
                nn.MaxPool1d(2, 2)
            )

            self.decoder = nn.Sequential(
                nn.Conv1d(2, 8, 3),
                nn.ReLU(True),
                Upsampling1D(2),
                nn.Conv1d(8, 16, 3),
                nn.ReLU(True),
                Upsampling1D(2),
                nn.Conv1d(16, 32, 3),
                nn.ReLU(True),
                Upsampling1D(2),
                nn.Conv1d(32, 64, 3),
                nn.ReLU(True),
                Upsampling1D(2),
                # nn.ZeroPad1d(),
                nn.Conv1d(64, 1, 4)
            )

        def forward(self, x):
            z = self.encoder(x)
            # print(z.size())
            z = z.reshape(-1, 1, 32)
            y = z.reshape(-1, 2, 16)
            # print(y.size())
            xhat = self.decoder(y)
            return z, xhat

    def consistency(value):
        if(value < 0.65):
            return "Moderate"
        elif(value < 0.85):
            return "High"
        else:
            return "Very High"

    def predict_out():
        # model_path = 'savedmodels/cnn_2class.pt'               #initial model

        z_dim = 64
        encoder_path = 'savedmodels/autoencoder_state.pt'
        svm_path = 'savedmodels/best_test_svm_afterencoding.sav'
        encoder = Autoencoder(z_dim)

        # encoder = torch.load(encoder_path, map_location=torch.device('cpu'))
        encoder.load_state_dict(torch.load(
            encoder_path, map_location=torch.device('cpu')))
        clf = svm.OneClassSVM
        clf = pickle.load(open(svm_path, "rb"))
        #encoder.load_state_dict(torch.load(encoder_path, map_location=torch.device('cpu')))

        mse_loss = nn.MSELoss()

        path = "./static/uploads/" + filename

        print("Going to Process: ", filename)
        try:
            audio_tensor, sample_rate = torchaudio.load(
                path, normalize=True)  # [ 1 , 193 ]
        except RuntimeError:
            print("Couldnt open file")
            return "101", "Error", 0
        feature = feature_extract(audio_tensor, sample_rate)
        with torch.no_grad():
            audio_tensor = feature  # [1 , 193]

            audio_tensor = torch.unsqueeze(audio_tensor, 0)  # [1,1,193]

            z, xhat = encoder(audio_tensor)
            z = z.cpu().detach()
            z = z.squeeze(0).squeeze(0)
            # print(z.size())

            xhat = xhat.cpu().detach()
            loss = mse_loss(xhat, audio_tensor)
            loss = loss.cpu().detach()
            # print(loss)  # re constructed loss
            z = z.tolist()
            # print(z)
            loss = loss.tolist()
            # print(loss)
            full_feature = z + [loss]
            # print(len(full_feature))

            pred = clf.predict([full_feature])
            scores = clf.decision_function([full_feature]).ravel() * (-1)
            print(scores)
            print(pred)
            prediction = ''
            if(pred[0] == -1):
                prediction = 'Anomaly'
            else:
                prediction = 'Normal'

            return prediction, scores[0]

            # output = audio_cnn(audio_tensor)  # [1,1,2]
            # output = output.permute(1, 0, 2)[0]  # [1,2]
            # # print(output)

            # softmax = nn.Softmax(dim=1)
            # res = softmax(output)
            # # print(res)

            # z = res.squeeze(0).numpy()
            # z = [round(elem, 2)for elem in z]

            # pred = output.max(1, keepdim=True)[1]
            # pred = pred.squeeze(0).squeeze(0).item()

            # x_group = ["Abnormal", "Normal"]
            # x = x_group[pred]
            # y = consistency(res[0][pred])
            # z = int(res[0][pred].item()*100)
            #print("predicted:", pred)

            # return x, y, z

    x, y = predict_out()
    # x, y, z = predict_out()
    # print(x, y, z)
    return x, y


# print(sklearn.__version__)
# loadmodel_svm("crack_1.wav")
# for running only comment out when on web
# loadmodel("crack_1.wav")
# loadmodel("crack_2.wav")
# loadmodel("crack_3.wav")
# loadmodel("crack_4.wav")
# loadmodel("crack_5.wav")
# loadmodel("crack_6.wav")
# loadmodel("crack_7.wav")
# loadmodel("crack_8.wav")
# loadmodel("crack_9.wav")
# loadmodel("crack_10.wav")

# loadmodel("wheeze1.wav")
# loadmodel("wheeze2.wav")
# loadmodel("wheeze3.wav")
# loadmodel("wheeze4.wav")
# loadmodel("wheeze5.wav")
