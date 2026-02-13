import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model

from datasets.utils import LabelMapper
from datasets.data_loader import load_dataset
from feature_creation import create_features


# =====================================
# Path Utilities
# =====================================

def get_checkpoint_path(args):
    return os.path.join(
        "checkpoints",
        args.dataset_name,
        args.feature_type,
        f"{args.dataset_name}_{args.feature_type}_{args.task}_{args.num_classes}.h5"
    )


def get_results_dir(args):
    return os.path.join(
        "results",
        args.dataset_name,
        args.feature_type,
        args.task,
        str(args.num_classes)
    )


# =====================================
# Plot Saving Utilities
# =====================================

def save_feature_maps(feature_map, save_path, title, max_maps=8):
    num_maps = min(feature_map.shape[-1], max_maps)

    plt.figure(figsize=(15, 4))
    for i in range(num_maps):
        plt.subplot(1, num_maps, i+1)
        plt.imshow(feature_map[0, :, :, i], aspect='auto', cmap='viridis')
        plt.axis('off')
        plt.title(f'F{i}')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_gradcam_overlay(sample, heatmap, save_path, pred_class):
    plt.figure(figsize=(6, 4))
    plt.imshow(sample[0, :, :, 0], aspect='auto', cmap='gray')
    plt.imshow(heatmap, alpha=0.5, cmap='jet', aspect='auto')
    plt.title(f"Grad-CAM (pred={pred_class})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =====================================
# Grad-CAM
# =====================================

def grad_cam(model, img, class_idx, layer_name):
    grad_model = tf.keras.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = heatmap.numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    return heatmap


# =====================================
# Main
# =====================================

def main(args):

    print("Running interpretability for:", args)

    # Load model
    checkpoint_path = get_checkpoint_path(args)
    saved_model = load_model(checkpoint_path)

    # Reload dataset & features
    label_mapper = LabelMapper(args.task, args.num_classes)
    data, labels = load_dataset(args.dataset_name, label_mapper)

    X_train, X_test, y_train, y_test, _ = create_features(
        data,
        labels,
        feature_type=args.feature_type
    )

    # Select sample
    sample_idx = 0
    sample = X_test[sample_idx:sample_idx+1]

    # Feature Map Extraction
    layer_names = ['conv2d', 'conv2d_1']

    feature_model = tf.keras.Model(
        inputs=saved_model.input,
        outputs=[saved_model.get_layer(name).output for name in layer_names]
    )

    feature_maps = feature_model.predict(sample)

    result_dir = get_results_dir(args)

    save_feature_maps(
        feature_maps[0],
        os.path.join(result_dir, "feature_maps_layer1.png"),
        "Conv Layer 1 Feature Maps"
    )

    save_feature_maps(
        feature_maps[1],
        os.path.join(result_dir, "feature_maps_layer2.png"),
        "Conv Layer 2 Feature Maps"
    )

    # Grad-CAM
    pred_class = np.argmax(saved_model.predict(sample))

    heatmap = grad_cam(
        saved_model,
        sample,
        pred_class,
        layer_name='conv2d_1'
    )

    save_gradcam_overlay(
        sample,
        heatmap,
        os.path.join(result_dir, "gradcam.png"),
        pred_class
    )

    print("Interpretability results saved to:", result_dir)


# =====================================
# Entry Point
# =====================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--feature_type', required=True)

    args = parser.parse_args()

    main(args)
