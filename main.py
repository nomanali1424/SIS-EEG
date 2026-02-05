import os
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from datasets.eeg_loader import load_eeg_data
from preprocessing.sis_preprocessing import (
    apply_sis, combine_dims, generate_spectrograms
)
from models.cnn_lstm_model import build_model


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log("Script started")

# -------------------------------
# Paths
# -------------------------------
EEG_PATH = r"C:\Users\Noman\Desktop\Github\SIS-EEG-Development\data\DENS\Emotional"
RATING_CSV = r"C:\Users\Noman\Desktop\Github\SIS-EEG-Development\data\DENS\wholeFrequencyDependentDataWithVADLFR_ReFormattingWholeFrequencyVA.xlsx"

log("Paths set")

# -------------------------------
# Load data
# -------------------------------
log("Loading EEG data...")
eeg_data, eeg_labels = load_eeg_data(EEG_PATH, RATING_CSV)
log(f"EEG data loaded | data shape: {eeg_data.shape} | labels: {eeg_labels.shape}")

# -------------------------------
# Expand labels
# -------------------------------
log("Expanding labels...")
labels = []
for lab in eeg_labels:
    labels.extend([lab] * 106)
labels = to_categorical(labels)
log(f"Labels expanded | shape: {labels.shape}")

# -------------------------------
# SIS preprocessing
# -------------------------------
log("Applying SIS channel reshuffling...")
sis_data = apply_sis(eeg_data)
log(f"SIS output shape: {sis_data.shape}")

log("Combining dimensions...")
sis_data = combine_dims(sis_data, 0)
log(f"Combined SIS shape: {sis_data.shape}")

# -------------------------------
# Spectrograms (THIS IS SLOW)
# -------------------------------
log("Generating spectrograms (this may take several minutes)...")
X = generate_spectrograms(sis_data)
log(f"Spectrogram tensor shape: {X.shape}")

# -------------------------------
# Train / Test split
# -------------------------------
log("Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)
log(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

# -------------------------------
# Model
# -------------------------------
log("Building model...")
model = build_model(X_train.shape[1:])
model.summary(print_fn=log)

log("Compiling model...")
model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# Callbacks
# -------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        "DENS_SW_A_3.h5",
        save_best_only=True,
        monitor="val_accuracy"
    )
]

# -------------------------------
# Training
# -------------------------------
log("Starting training...")
history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=256,
    validation_data=(X_test, y_test),
    class_weight={0:2.77, 1:1.85, 2:0.48},
    callbacks=callbacks,
    verbose=1   # IMPORTANT
)

log("Training finished")

# -------------------------------
# Evaluation
# -------------------------------
log("Evaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

log(f"Final Test Loss: {test_loss:.4f}")
log(f"Final Test Accuracy: {test_acc:.4f}")

log("Script finished successfully")
