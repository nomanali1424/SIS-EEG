import os
import scipy.io as sio
import numpy as np
import pandas as pd

# -------------------------------
# Label mapping
# -------------------------------
def label_mapping(arousal):
    if arousal <= 3:
        return 0
    elif arousal <= 6:
        return 1
    else:
        return 2


# -------------------------------
# EEG data loader
# -------------------------------
def load_eeg_data(
    eeg_path,
    rating_csv_path
):
    _, _, files = next(os.walk(eeg_path))

    rating = pd.read_excel(rating_csv_path)

    eeg_data = []
    eeg_labels = []

    for f in files:
        if f.endswith(".mat"):
            file_str = f[:-4]
            df = rating[rating["Subject"].str.contains(file_str)]

            if not df.empty:
                mat = sio.loadmat(os.path.join(eeg_path, f))
                data = mat.get('eegData')

                df = df[['Subject', 'valence', 'arousal']].drop_duplicates()
                arousal = df['arousal'].values[0]

                eeg_data.append(data)
                eeg_labels.append(label_mapping(arousal))

    eeg_data = np.array(eeg_data)
    eeg_labels = np.array(eeg_labels)

    return eeg_data, eeg_labels
