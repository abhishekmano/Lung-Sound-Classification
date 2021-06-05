# python code to check whther audio signals are properly padded
from os import POSIX_FADV_DONTNEED
import librosa
import wave
import math
from numpy.lib.function_base import extract
import scipy.io.wavfile as wf
import numpy as np
import glob
import matplotlib.pyplot as plt
import librosa.display
import scipy
import scipy.io.wavfile


for file in glob.glob("*.wav"):
    X, sr = librosa.load(file, sr=22000)
    assert len(X) == 132000, file + " length should be 132000"
