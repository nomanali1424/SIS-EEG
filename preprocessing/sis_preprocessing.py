import numpy as np
import math
from scipy.signal import spectrogram
import functools

# -------------------------------
# Channel layout
# -------------------------------
SHUFFLED_CH = [
    128,32,25,21,127,17,126,14,8,2,1,125,
    48,43,38,33,26,22,15,9,10,3,
    123,122,44,39,34,28,27,23,18,16,
    121,114,120,119,24,19,11,4,
    124,117,116,115,110,109,108,113,
    40,35,29,20,12,5,
    118,111,93,104,103,98,
    41,36,30,7,13,6,
    112,105,102,101,100,107,
    49,56,45,46,47,42,37,31,
    97,96,95,99,57,50,57,58,55,
    106,80,87,79,86,92,91,
    52,53,54,61,62,72,78,77,85,84,90,94,
    63,64,59,60,66,67,71,76,82,83,89,88,
    68,65,69,70,73,74,75,81,
    129,129,129,129
]


# -------------------------------
# Sliding window helpers
# -------------------------------
def rolling_window(a, shape):
    s = (a.shape[0] - shape[0] + 1,
         a.shape[1] - shape[1] + 1) + shape
    strides = a.strides + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)


def window2(arr, shape=(2, 2)):
    r, c = shape
    out = np.full((arr.shape[0] + r, arr.shape[1] + c), np.nan)
    out[1:-1, 1:-1] = arr
    return rolling_window(out, shape)


def check_arr(arr):
    for i in range(2):
        for j in range(2):
            if np.isnan(arr[i][j]) or arr[i][j] == 129:
                return False
    return True


# -------------------------------
# SIS channel reshuffling
# -------------------------------
def apply_sis(eeg_data):
    y = np.array(SHUFFLED_CH).reshape(11, 12)
    stride_arr = window2(y, (2, 2))

    valid_blocks = []
    for i in range(stride_arr.shape[0]):
        for j in range(stride_arr.shape[1]):
            if check_arr(stride_arr[i][j]):
                valid_blocks.append(stride_arr[i][j])

    valid_blocks = np.array(valid_blocks)

    reshuffled = []
    for sample in eeg_data:
        b = []
        for block in valid_blocks:
            for ch in block.flatten():
                b.append(sample[int(ch) - 1])
        reshuffled.append(b)

    return np.array(reshuffled)


# -------------------------------
# Combine dimensions
# -------------------------------
def combine_dims(a, i=0, n=1):
    s = list(a.shape)
    combined = functools.reduce(lambda x, y: x*y, s[i:i+n+1])
    return np.reshape(a, s[:i] + [combined] + s[i+n+1:])


# -------------------------------
# Spectrogram generation
# -------------------------------
def generate_spectrograms(eeg_data, fs=250, nperseg=125, noverlap=62):
    size_dataset = len(eeg_data)
    new_size = size_dataset // 4

    f_size = math.ceil((nperseg + 1) / 2)
    t_size = int((eeg_data[0].size - noverlap) / (nperseg - noverlap))

    X = np.zeros((new_size, f_size, t_size, 4))

    for i in range(new_size):
        for j in range(4):
            signal_data = eeg_data[4*i + j]
            _, _, Sxx = spectrogram(
                signal_data, fs,
                nperseg=nperseg,
                noverlap=noverlap,
                mode='psd'
            )
            X[i, :, :, j] = Sxx

    return X / 255.0
