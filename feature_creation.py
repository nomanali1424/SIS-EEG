import numpy as np
import math
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import functools




def _combine_dims(a, i=0, n=1):
    s = list(a.shape)
    combined = np.prod(s[i:i+n+1])
    return np.reshape(a, s[:i] + [combined] + s[i+n+1:])


def _generate_spatial_windows(arr):

    windows = []

    for i in range(arr.shape[0] - 1):
        for j in range(arr.shape[1] - 1):
            block = arr[i:i+2, j:j+2]
            if not np.any(block == 129):
                windows.append(block)

    return windows







# ==============================
# Public API
# ==============================
def create_features(eeg_data, eeg_labels, feature_type="SIS"):

    feature_type = feature_type.upper()

    if feature_type == "WSIS":
        return _create_wsis_features(eeg_data, eeg_labels)

    elif feature_type == "SIS":
        return _create_sis_features(eeg_data, eeg_labels)

    else:
        raise ValueError("feature_type must be 'WSIS' or 'SIS'")



def _create_wsis_features(eeg_data, eeg_labels):

    # Flatten subject dimension
    eeg_data = _combine_dims(np.array(eeg_data), 0)

    # Expand labels per channel (128 channels assumed)
    expanded_labels = []
    for label in eeg_labels:
        for _ in range(128):
            expanded_labels.append(label)

    y = to_categorical(np.array(expanded_labels))

    fs = 250
    nperseg = 125
    noverlap = 62

    size_dataset = len(eeg_data)
    f_size = math.ceil((nperseg + 1) / 2)
    t_size = int((eeg_data[0].size - noverlap) / (nperseg - noverlap))

    X_full = np.zeros((size_dataset, f_size, t_size, 1))

    for i in range(size_dataset):
        X = eeg_data[i]
        _, _, Sxx = spectrogram(X, fs, nperseg=nperseg,
                                noverlap=noverlap, mode='psd')
        X_full[i, :, :, 0] = Sxx

    X_full /= 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42
    )

    input_shape = X_train.shape[1:]

    return X_train, X_test, y_train, y_test, input_shape



def _create_sis_features(eeg_data, eeg_labels):

    shuffled_ch = [128,32,25,21,127,17,126,14,8,2,1,
                   125,48,43,38,33,26,22,15,9,10,3,
                   123,122,44,39,34,28,27,23,18,16,
                   121,114,120,119,24,19,11,4,124,117,
                   116,115,110,109,108,113,40,35,29,
                   20,12,5,118,111,93,104,103,98,41,
                   36,30,7,13,6,112,105,102,101,100,
                   107,49,56,45,46,47,42,37,31,97,96,
                   95,99,57,50,57,58,55,106,80,87,79,
                   86,92,91,52,53,54,61,62,72,78,77,85,
                   84,90,94,63,64,59,60,66,67,71,76,82,
                   83,89,88,68,65,69,70,73,74,75,81,129,
                   129,129,129]
    twoD = np.array(shuffled_ch).reshape(11,12)


    eeg_data = np.array(eeg_data)

    # Create spatial windows
    stride_windows = _generate_spatial_windows(twoD)

    reshuffled = []

    for sample in eeg_data:
        temp_sample = []
        for window in stride_windows:
            for row in window:
                for ch in row:
                    temp_sample.append(sample[ch - 1])
        reshuffled.append(temp_sample)

    reshuffled = np.array(reshuffled)

    # Expand labels (106 patches)
    expanded_labels = []
    for label in eeg_labels:
        for _ in range(len(stride_windows)):
            expanded_labels.append(label)

    y = to_categorical(np.array(expanded_labels))

    reshuffled = _combine_dims(reshuffled, 0)

    fs = 250
    nperseg = 125
    noverlap = 62

    new_size = len(reshuffled) // 4
    f_size = math.ceil((nperseg + 1) / 2)
    t_size = int((reshuffled[0].size - noverlap) / (nperseg - noverlap))

    X_full = np.zeros((new_size, f_size, t_size, 4))

    for i in range(new_size):
        for j in range(4):
            idx = i * 4 + j
            _, _, Sxx = spectrogram(
                reshuffled[idx],
                fs,
                nperseg=nperseg,
                noverlap=noverlap,
                mode='psd'
            )
            X_full[i, :, :, j] = Sxx

    X_full /= 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42
    )

    input_shape = X_train.shape[1:]

    return X_train, X_test, y_train, y_test, input_shape
