import librosa
import wave
import math
from numpy.lib.function_base import extract
import scipy.io.wavfile as wf
import numpy as np
import glob


def read24bitwave(lp_wave):
    nFrames = lp_wave.getnframes()
    buf = lp_wave.readframes(nFrames)
    reshaped = np.frombuffer(buf, np.int8).reshape(nFrames, -1)
    short_output = np.empty((nFrames, 2), dtype=np.int8)
    short_output[:, :] = reshaped[:, -2:]
    short_output = short_output.view(np.int16)
    # return numpy array to save memory via array slicing
    return (lp_wave.getframerate(), np.divide(short_output, 32768).reshape(-1))


def bitrate_channels(lp_wave):
    bps = (lp_wave.getsampwidth() / lp_wave.getnchannels())  # bytes per sample
    return (bps, lp_wave.getnchannels())


def extract2FloatArr(lp_wave, str_filename):
    (bps, channels) = bitrate_channels(lp_wave)

    if bps in [1, 2, 4]:
        (rate, data) = wf.read(str_filename)
        divisor_dict = {1: 255, 2: 32768}
        if bps in [1, 2]:
            divisor = divisor_dict[bps]
            data = np.divide(data, float(divisor))  # clamp to [0.0,1.0]
        return (rate, data)

    elif bps == 3:
        # 24bpp wave
        return read24bitwave(lp_wave)

    else:
        raise Exception(
            'Unrecognized wave format: {} bytes per sample'.format(bps))


def resample(current_rate, data, target_rate):
    x_original = np.linspace(0, 100, len(data))
    x_resampled = np.linspace(0, 100, int(
        len(data) * (target_rate / current_rate)))
    resampled = np.interp(x_resampled, x_original, data)
    return (target_rate, resampled.astype(np.float32))

#wav, sr = librosa.load("101_1b1_Al_sc_Meditron.wav",sr = None)


def read_wav_file(str_filename, target_rate):
    #filename = "101_1b1_Pr_sc_Meditron.wav"
    wav = wave.open(str_filename, mode='r')
    (sample_rate, data) = extract2FloatArr(wav, str_filename)
    print(sample_rate)
    if (sample_rate != target_rate):
        (_, data) = resample(sample_rate, data, target_rate)

    wav.close()
    return (target_rate, data.astype(np.float32))


for file in glob.glob("*.wav"):
    sr, X = read_wav_file(file, 22000)
    print('sampling rate is:', sr)
    print('wav is :', X)
    print("length of wav is", len(X))
