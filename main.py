import argparse
import os
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import ConfusionMatrixDisplay

from datasets.utils import LabelMapper
from datasets.data_loader import load_dataset
from feature_creation import create_features
from model import build_model

import csv
from datetime import datetime


# =====================================
# Utility Functions
# =====================================

def get_checkpoint_path(args):
    base_dir = "checkpoints"

    feature_dir = os.path.join(
        base_dir,
        args.dataset_name,
        args.feature_type
    )

    os.makedirs(feature_dir, exist_ok=True)

    filename = f"{args.dataset_name}_{args.feature_type}_{args.task}_{args.num_classes}.h5"

    return os.path.join(feature_dir, filename)


def get_results_dir(args):
    result_dir = os.path.join(
        "results",
        args.dataset_name,
        args.feature_type,
        args.task,
        str(args.num_classes)
    )

    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def save_training_plots(history, args):
    result_dir = get_results_dir(args)

    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "accuracy.png"))
    plt.close()

    # Loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "loss.png"))
    plt.close()


def save_confusion_matrix(y_test, y_pred, args):
    result_dir = get_results_dir(args)

    plt.figure()
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap='Blues',
        normalize='all',
        values_format='.01%'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
    plt.close()




def log_experiment(args, test_acc):

    log_path = os.path.join("results", "experiment_log.csv")
    os.makedirs("results", exist_ok=True)

    file_exists = os.path.isfile(log_path)

    with open(log_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header only once
        if not file_exists:
            writer.writerow([
                "Dataset",
                "FeatureType",
                "Task",
                "NumClasses",
                "TestAccuracy",
                "Timestamp"
            ])

        writer.writerow([
            args.dataset_name,
            args.feature_type,
            args.task,
            args.num_classes,
            round(float(test_acc), 5),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])



# =====================================
# Main Pipeline
# =====================================

def main(args):

    print("Running experiment with config:")
    print(args)

    # Label mapping
    label_mapper = LabelMapper(
        mode=args.task,
        num_classes=args.num_classes
    )

    # Load dataset
    data, labels = load_dataset(args.dataset_name, label_mapper)
    print("Loading data and labels, it will take some time...")

    X_train, X_test, y_train, y_test, input_shape = create_features(
        data,
        labels,
        feature_type=args.feature_type
    )

    # Build model
    model = build_model(input_shape, args.num_classes)

    # Compute class weights dynamically
    y_train_labels = np.argmax(y_train, axis=1)
    classes = np.unique(y_train_labels)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train_labels
    )

    class_weights = dict(zip(classes, weights))

    print("Class Weights:", class_weights)

    # Setup callbacks
    checkpoint_path = get_checkpoint_path(args)

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=30
    )

    mc = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True
    )

    # Compile
    model.compile(
        optimizer="Adam",
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    # 8️⃣ Train
    history = model.fit(
        X_train,
        y_train,
        epochs=300,
        batch_size=256,
        verbose=1,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=[es, mc]
    )

    # Load best model
    saved_model = load_model(checkpoint_path)

    #Evaluate
    _, test_acc = saved_model.evaluate(X_test, y_test, verbose=0)
    print("Test Accuracy:", test_acc)

    # Saving log
    log_experiment(args, test_acc)

    # Save plots
    save_training_plots(history, args)

    y_pred = saved_model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    save_confusion_matrix(y_test_labels, y_pred, args)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', choices=['DEAP', 'DENS'], required=True)
    parser.add_argument('--task', choices=['A', 'V', 'VAD'], required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--feature_type', choices=['SIS', 'WSIS'], required=True)

    args = parser.parse_args()

    main(args)
