# python code to plot and check wav
from os import stat_result
import librosa
import numpy as np
import scipy
import scipy.io.wavfile
import matplotlib.pyplot as plt
import librosa.display
import glob
import os
from pydub import AudioSegment
dir = os.getcwd()
os.chdir(dir)

for file in glob.glob("*.wav"):
    y, sr = librosa.load(file, sr=None)

    print(sr)
    print(y.shape)
    plt.figure(figsize=(14, 5))
    plt.title("Original")
    librosa.display.waveplot(y, sr=sr)    # Code to display the original
    plt.show()
