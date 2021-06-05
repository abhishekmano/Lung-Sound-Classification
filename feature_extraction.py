import argparse
import torch
import torchaudio
import joblib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MFCC
import librosa
#from datasets.audio import get_audio_dataset
#from models.audiocnn import AudioCNN

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d",
    "--data-directory",
    type=str,
    required=False,
    help="Directory where subfolders of audio reside",
)

parser.add_argument("-e", "--num-epochs", type=int,
                    default=10, help="Number of epochs")
parser.add_argument("-b", "--batch-size", type=int,
                    default=50, help="Batch size")


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
        self.avgPool = nn.AvgPool1d(128)
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


def feature_extract(audio, sr):
    audio = audio.squeeze()
    y = audio.numpy()
    # print(y.dtype)

    stft = librosa.stft(y)
    stft = np.abs(stft)

    #print("stft", stft, stft.shape)

    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=40)
    #print("mfcc", mfcc, mfcc.shape)

    mfcc = np.mean(mfcc, axis=1)

    #print("mfcc", mfcc, mfcc.shape)

    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma = np.mean(chroma, axis=1)

    # print("chroma", chroma, chroma.shape)       #12 features

    mel = librosa.feature.melspectrogram(y, sr)
    mel = np.mean(mel, axis=1)

    # print("mel", mel, mel.shape)    #128

    contrast = librosa.feature.spectral_contrast(y, sr)
    contrast = np.mean(contrast, axis=1)

    # print("contrast", contrast, contrast.shape)    #7

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tonnetz = np.mean(tonnetz, axis=1)

    # print("tonnetz", tonnetz, tonnetz.shape)    #6

    features = np.hstack([mfcc, chroma, mel, contrast, tonnetz])
    # print(features.shape)
    features = features.reshape((1, 193))

    print("size of reshape", features.shape)

    features = torch.from_numpy(features)
    # print(features.size())
    # print(features.dtype)
    # print(features[0][0:10])
    features = features.float()
    # print(features.dtype)
    # print(features[0][0:10])

    return features


def predict_out():
    model_path = 'upload code/savedmodels/unsup2.pt'
    audio_cnn = NetM3()
    # audio_cnn.load_state_dict(torch.load(
    #     model_path, map_location=torch.device('cpu')))

    path = "upload code/static/uploads/normal1.wav"
    audio_tensor, sample_rate = torchaudio.load(path, normalize=True)
    print("sample rate: ", sample_rate)
    print(audio_tensor.dtype)
    feature = feature_extract(audio_tensor, sample_rate)
    with torch.no_grad():
        audio_tensor = feature  # .to("cuda")

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


if __name__ == "__main__":
    args = parser.parse_args()
    # main(**vars(args))
    out = predict_out()
