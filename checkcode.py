# python code to plot and check 2 wav
from os import stat_result
import librosa
import numpy as np
import scipy
import scipy.io.wavfile
import matplotlib.pyplot as plt
import librosa.display
y, sr = librosa.load('wave1', sr=22000)


plt.figure(figsize=(14, 5))
plt.title("Original")
plt.subplot(211)
librosa.display.waveplot(y, sr=sr)    # Code to display the original

y, sr = librosa.load('wav2', sr=None)
plt.subplot(212)
librosa.display.waveplot(y, sr=sr)

plt.show()
