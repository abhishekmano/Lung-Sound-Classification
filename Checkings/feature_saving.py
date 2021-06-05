from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# import torchaudio
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Iterable
import librosa
import numpy as np
import random
from functools import partial
from torchaudio.transforms import MFCC

import torch
import torchaudio
from torchvision.datasets import DatasetFolder


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

    #print("size of reshape", features.shape)

    features = torch.from_numpy(features)
    # print(features.size())
    # print(features.dtype)
    # print(features[0][0:10])
    features = features.float()
    # print(features.dtype)
    # print(features[0][0:10])

    return features


def audio_loader(path, max_length_in_seconds, pad_and_truncate):
    audio_tensor, sample_rate = torchaudio.load(path, normalize=True)
    max_length = sample_rate * max_length_in_seconds
    audio_size = audio_tensor.size()

    if pad_and_truncate:
        if audio_size[1] < max_length:
            difference = max_length - audio_size[1]
            padding = torch.zeros(audio_size[0], difference)
            padded_audio = torch.cat([audio_tensor, padding], 1)
            return padded_audio

        if audio_size[1] > max_length:
            random_idx = random.randint(0, audio_size[1] - max_length)
            return audio_tensor.narrow(1, random_idx, max_length)
    feature = feature_extract(audio_tensor, sample_rate)
    # return torch.randn(1,193)
    return feature
    # return audio_tensor


def get_audio_dataset(datafolder, max_length_in_seconds=6, pad_and_truncate=False):
    loader_func = partial(
        audio_loader,
        max_length_in_seconds=max_length_in_seconds,
        pad_and_truncate=pad_and_truncate,
    )
    dataset = DatasetFolder(datafolder, loader_func, ".wav")
    return dataset


class LungSoundDataset(Dataset):
    def __init__(
        self, file_path: Path,
    ):
        """[summary]
        Args:
            file_path (Path): Path to Tuple
        """

        self.data = torch.load(file_path)

    def __getitem__(self, index):
        # format the file path and load the file
        sound = self.data[index]

        return sound[0], sound[1]

    def __len__(self):
        return len(self.data)


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
        self.fc1 = nn.Linear(128, 2)

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
        return F.log_softmax(x, dim=2)  # Output: torch.Size([N, 1, 10])


def train(model, epoch, train_loader, device, optimizer, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        print(data, target)
        print(data.size(), target.size())
        data, target = data.to(device), target.to(device)
        output = model(data)
        break
        # original output dimensions are batchSizex1x10
        output = output.permute(1, 0, 2)
        # the loss functions expects a batchSizex10 input
        loss = F.nll_loss(output[0], target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx *
                    len(data), len(train_loader.dataset), 100.0 *
                    batch_idx / len(train_loader), loss
                )
            )
    print("Train complete")


def test(model, test_loader, device):
    print("Reached test")
    model.eval()
    print("In test")
    test_loss = 0
    correct = 0
    loop = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if(loop % 50 == 0):
                print(loop)
            loop += 1
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.permute(1, 0, 2)[0]
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )
    return test_loss, accuracy


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NetM3()
    if torch.cuda.device_count() > 1:
        print("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    #path = os.path.join(model_dir, "model.pth")
    path = input("Filename")
    path = "savedmodels/" + path + ".pt"
    torch.save(model.state_dict(), path)


def save_feature(dataset, filename, print_freq=100):
    train_list = []
    for i in range(len(dataset)):
        if(i % print_freq == 0):
            print(i, "/", len(dataset))
        train_list.append(dataset[i])
    print(len(train_list))
    train_list = tuple(train_list)
    torch.save(train_list, filename+".pt")
    print("*******Model Saved*******")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int,
                        default=16, help="train batch size")
    parser.add_argument(
        "--test-batch-size", type=int, default=32, help="test batch size",
    )
    parser.add_argument("--epochs", type=int, default=1,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Learning rate step gamma")
    parser.add_argument("--weight-decay", type=float,
                        default=0.0001, help="Optimizer regularization")
    parser.add_argument("--stepsize", type=int,
                        default=50, help="Step LR size")
    parser.add_argument("--model", type=str, default="m3")
    parser.add_argument("--num-workers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--cv", type=int, default=0,
                        help="0: No cross validation 1: with cross validation")

    # Container environment
    parser.add_argument("--model-dir", type=str,
                        default=os.getenv("SM_MODEL_DIR", "./"))
    if os.getenv("SM_HOSTS") is not None:
        # parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
        parser.add_argument("--data-dir", type=str,
                            default=os.environ["SM_CHANNEL_TRAINING"])
        # parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # On SageMaker
    if os.getenv("SM_HOSTS") is not None:
        print("Running on sagemaker")
        datapath = Path(args.data_dir)
        csvpath = datapath / "UrbanSound8K.csv"
        print("datapath", datapath)
        print("csvpath", csvpath)
    # Local
    else:
        print("Running on local")
        full_filepath = Path(__file__).resolve()
        #parent_path = full_filepath.parent.parent.parent.parent
        #parent_path = full_filepath.parent.parent.parent.parent.parent / "MyDrive"
        # above path is for mano
        parent_path = "/media/abhishek/01D465FA6B9CE220/Btech/Lung Sound Data Set/Lung_Data_3Class_Full"
        print(parent_path)

        train_path = parent_path + "/Train"
        valid_path = parent_path + "/Valid"
        test_path = parent_path + "/Test"
        # csvpath = datapath / "UrbanSound8K.csv"

    kwargs = {"num_workers": args.num_workers,
              "pin_memory": True} if torch.cuda.is_available() else {}
    print(kwargs)

    # 10 fold cross validation
    all_scores = []
    if args.cv == 1:
        print("Cross validation: Enable")
        cv = 10
    else:
        cv = 1

    for i in range(1, cv + 1):
        # folders = list(range(1, 11))
        # test_folder = [i]
        # train_folder = set(folders) - set([i])
        print(f"***** Processing fold({i}) *****")

        return 0
        full_set = get_audio_dataset(
            parent_path, max_length_in_seconds=6, pad_and_truncate=True)
        save_feature(full_set, "full_set_3class.pt")
        # train_set = get_audio_dataset(
        #    train_path, max_length_in_seconds=6, pad_and_truncate=True)
        # valid_set = get_audio_dataset(
        #     valid_path, max_length_in_seconds=6, pad_and_truncate=True)
        # test_set = get_audio_dataset(
        #     test_path, max_length_in_seconds=6, pad_and_truncate=True)
        # print(test_set[10])
        #save_feature(train_set, "train_set_3class", 50)
        #save_feature(valid_set, "valid_set_3class", 50)
        #save_feature(test_set, "test_set_3class", 50)
        print("********END********")

        # train_loader = torch.utils.data.DataLoader(
        #     train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        # valid_loader = torch.utils.data.DataLoader(
        #     valid_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        # test_loader = torch.utils.data.DataLoader(
        #     test_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        print("Loading model:", args.model)
        if args.model == "m3":
            model = NetM3()
        else:
            model = NetM3()

        if torch.cuda.device_count() > 1:
            print("There are {} gpus".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

        model.to(device)

        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.stepsize, gamma=args.gamma)

        log_interval = args.log_interval

        for epoch in range(1, args.epochs + 1):
            print("Learning rate:", scheduler.get_last_lr()[0])
            train(model, epoch, train_loader, device, optimizer, log_interval)
            return 0
            loss, accuracy = test(model, test_loader, device)
            loss2, accuracy2 = test(model, test_loader2, device)
            scheduler.step()

        print(f"Accuracy for fold ({i}): {accuracy}")
        all_scores.append(accuracy)

    print(f"Final score: {sum(all_scores)/len(all_scores):.2f}%")

    # Save Model
    save_model(model, args.model_dir)


if __name__ == "__main__":
    out = main()
