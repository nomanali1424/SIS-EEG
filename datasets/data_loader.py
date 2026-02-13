import os
import numpy as np
import pandas as pd
import scipy.io as sio
import pickle


def load_dataset(dataset_name, label_mapper):
    dataset_name = dataset_name.upper()

    if dataset_name == "DENS":
        return _load_dens(label_mapper)

    elif dataset_name == "DEAP":
        return _load_deap(label_mapper)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def _load_dens(label_mapper):

    base_path = "data/DENS/Emotional/"
    rating = pd.read_excel(
        "data/DENS/wholeFrequencyDependentDataWithVADLFR_ReFormattingWholeFrequencyVA.xlsx"
    )

    eeg_data = []
    eeg_labels = []

    _, _, files = next(os.walk(base_path))

    for f in files:
        if not f.endswith(".mat"):
            continue

        file_str = f[:-4]
        df = rating[rating["Subject"].str.contains(file_str)]

        if df.empty:
            continue

        df = df[['Subject', 'valence', 'arousal', 'Dominance']]
        df = df.drop_duplicates()

        valence = df["valence"].values[0]
        arousal = df["arousal"].values[0]
        dominance = df["Dominance"].values[0]

        mat = sio.loadmat(os.path.join(base_path, f))
        data = mat.get("eegData")

        # 🔥 Mode-aware mapping
        if label_mapper.mode == 'VAD':
            label = label_mapper(
                valence=valence,
                arousal=arousal,
                dominance=dominance
            )

        elif label_mapper.mode == 'A':
            label = label_mapper(arousal=arousal)

        elif label_mapper.mode == 'V':
            label = label_mapper(valence=valence)

        else:
            raise ValueError(f"Unsupported mode {label_mapper.mode} for DENS")

        eeg_data.append(data)
        eeg_labels.append(label)

    eeg_data = np.array(eeg_data)
    eeg_labels = np.array(eeg_labels)

    return eeg_data, eeg_labels



def _load_deap(label_mapper):

    subject_list = [f"{i:02d}" for i in range(1, 33)]

    data = []
    labels = []

    for sub in subject_list:

        with open(f"data/DEAP/s{sub}.dat", "rb") as file:
            subject = pickle.load(file, encoding="latin1")

            for trial in range(40):

                eeg_data = subject["data"][trial]
                trial_labels = subject["labels"][trial]

                valence = trial_labels[0]
                arousal = trial_labels[1]
                dominance = trial_labels[2]

                # 🔥 Mode-aware mapping
                if label_mapper.mode == 'VAD':
                    label = label_mapper(
                        valence=valence,
                        arousal=arousal,
                        dominance=dominance
                    )

                elif label_mapper.mode == 'A':
                    label = label_mapper(arousal=arousal)

                elif label_mapper.mode == 'V':
                    label = label_mapper(valence=valence)

                else:
                    raise ValueError(f"Unsupported mode {label_mapper.mode} for DEAP")

                # Use EEG channels only (1–32)
                for ch in range(1, 33):
                    data.append(eeg_data[ch])
                    labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    return data, labels
