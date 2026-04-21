"""
MangoSense – Multi-Modal Training Script
=========================================
Trains a disease classification model that fuses:
    • Image features  – extracted by a fine-tuned EfficientNetB0 backbone
    • Symptom features – encoded by a small Dense branch

Architecture overview::

    IMAGE (224×224×3) ──► EfficientNetB0 ──► GlobalAvgPool ──► image_features (1280)
                                                                        │
    SYMPTOMS (N binary) ──► Dense(64, relu) ──► symptom_features (64)  │
                                                        │               │
                                          Concatenate ◄────────────────┘
                                                │
                                    Dense(256, relu) ── Dropout(0.5)
                                                │
                                    Dense(128, relu) ── Dropout(0.5)
                                                │
                                    Dense(num_classes, softmax)

Training phases:
    1. Freeze backbone → train head only (warm-up)
    2. Unfreeze top layers of backbone → fine-tune end-to-end

Usage:
    python train_model.py                    # uses default config.yaml
    python train_model.py --config my.yaml   # custom config file
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_split(
    data_dir: Path, split: str, has_symptoms: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load pre-processed numpy arrays for *split* (train / val / test)."""
    images = np.load(data_dir / f"{split}_images.npy")
    labels = np.load(data_dir / f"{split}_labels.npy")
    symptoms = (
        np.load(data_dir / f"{split}_symptoms.npy") if has_symptoms else None
    )
    return images, labels, symptoms


def make_tf_dataset(
    images: np.ndarray,
    labels_onehot: np.ndarray,
    symptoms: np.ndarray | None,
    batch_size: int,
    shuffle: bool = False,
    augment_fn=None,
    prefetch: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    """
    Build a ``tf.data.Dataset`` from numpy arrays.

    When *symptoms* is provided the dataset yields
    ``((image, symptom), label)`` tuples; otherwise ``(image, label)``.
    """
    if symptoms is not None:
        ds = tf.data.Dataset.from_tensor_slices(
            ((images, symptoms), labels_onehot)
        )
    else:
        ds = tf.data.Dataset.from_tensor_slices((images, labels_onehot))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), seed=42)

    ds = ds.batch(batch_size)

    if augment_fn is not None:
        if symptoms is not None:
            ds = ds.map(
                lambda xy, y: ((augment_fn(xy[0]), xy[1]), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            ds = ds.map(
                lambda x, y: (augment_fn(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

    return ds.prefetch(prefetch)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def build_augmentation_layer(aug_cfg: dict) -> keras.Sequential:
    """Return a Keras Sequential model with data-augmentation layers."""
    h_flip = aug_cfg.get("horizontal_flip", True)
    v_flip = aug_cfg.get("vertical_flip", False)
    if h_flip and v_flip:
        flip_mode = "horizontal_and_vertical"
    elif h_flip:
        flip_mode = "horizontal"
    elif v_flip:
        flip_mode = "vertical"
    else:
        flip_mode = "horizontal"  # safe default

    aug = keras.Sequential(
        [
            layers.RandomFlip(flip_mode),
            layers.RandomRotation(aug_cfg.get("rotation_range", 30) / 360.0),
            layers.RandomZoom(aug_cfg.get("zoom_range", 0.2)),
            layers.RandomTranslation(
                height_factor=aug_cfg.get("height_shift_range", 0.15),
                width_factor=aug_cfg.get("width_shift_range", 0.15),
            ),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(
                factor=aug_cfg.get("brightness_range", [0.8, 1.2])[1] - 1.0
            ),
        ],
        name="augmentation",
    )
    return aug


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(
    num_classes: int,
    num_symptoms: int,
    model_cfg: dict,
) -> keras.Model:
    """
    Build the multi-modal model.

    Parameters
    ----------
    num_classes:  Number of disease classes.
    num_symptoms: Number of binary symptom features (0 if no symptom data).
    model_cfg:    Sub-dict from config.yaml under ``model``.
    """
    l2 = keras.regularizers.l2(model_cfg.get("l2_regularization", 1e-4))

    # ── Image branch ──────────────────────────────────────────────────────
    backbone = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=model_cfg.get("backbone_weights", "imagenet"),
    )
    backbone.trainable = False  # frozen during phase 1

    img_input = keras.Input(shape=(224, 224, 3), name="image_input")
    # EfficientNetB0 expects pixel values in [0, 255]; rescale from [0, 1]
    x = layers.Rescaling(255.0, name="rescale_to_255")(img_input)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="image_gap")(x)
    # x shape: (batch, 1280)

    # ── Symptom branch ────────────────────────────────────────────────────
    if num_symptoms > 0:
        sym_input = keras.Input(shape=(num_symptoms,), name="symptom_input")
        s = layers.Dense(
            model_cfg.get("symptom_embedding_dim", 64),
            activation="relu",
            kernel_regularizer=l2,
            name="symptom_embed",
        )(sym_input)
        fused = layers.Concatenate(name="fusion")([x, s])
        model_inputs = [img_input, sym_input]
    else:
        fused = x
        model_inputs = img_input

    # ── Fusion head ───────────────────────────────────────────────────────
    dropout_rate = model_cfg.get("dropout_rate", 0.5)
    for i, units in enumerate(model_cfg.get("fusion_dense_units", [256, 128])):
        fused = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=l2,
            name=f"fusion_dense_{i}",
        )(fused)
        fused = layers.Dropout(dropout_rate, name=f"fusion_drop_{i}")(fused)

    output = layers.Dense(num_classes, activation="softmax", name="predictions")(fused)

    return keras.Model(inputs=model_inputs, outputs=output, name="MangoSense_MultiModal")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def build_callbacks(
    train_cfg: dict, output_cfg: dict, fold: int | None = None
) -> list[keras.callbacks.Callback]:
    logs_dir = Path(output_cfg["logs_dir"])
    if fold is not None:
        logs_dir = logs_dir / f"fold_{fold}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    return [
        keras.callbacks.EarlyStopping(
            monitor=train_cfg.get("early_stopping_monitor", "val_loss"),
            patience=train_cfg.get("early_stopping_patience", 10),
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=train_cfg.get("reduce_lr_factor", 0.5),
            patience=train_cfg.get("reduce_lr_patience", 5),
            min_lr=train_cfg.get("reduce_lr_min_lr", 1e-6),
            verbose=1,
        ),
        keras.callbacks.TensorBoard(log_dir=str(logs_dir), histogram_freq=1),
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compile_model(model: keras.Model, lr: float, num_classes: int) -> None:
    # Always use categorical_crossentropy; the model always outputs num_classes
    # units with softmax (including the binary case with 2 units).
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_backbone(model: keras.Model, fine_tune_from: int) -> None:
    """Unfreeze backbone layers at and after *fine_tune_from* index."""
    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_from]:
        layer.trainable = False
    trainable_count = sum(1 for l in backbone.layers if l.trainable)
    print(
        f"[INFO] Fine-tuning: {trainable_count} / {len(backbone.layers)} "
        f"backbone layers unfrozen (from index {fine_tune_from})"
    )


def train_full(
    cfg: dict,
    train_images: np.ndarray,
    train_labels_oh: np.ndarray,
    train_symptoms: np.ndarray | None,
    val_images: np.ndarray,
    val_labels_oh: np.ndarray,
    val_symptoms: np.ndarray | None,
    num_classes: int,
    num_symptoms: int,
    augment_fn,
    fold: int | None = None,
) -> tuple[keras.Model, dict]:
    """
    Run two-phase training and return (best_model, combined_history).

    Datasets are built internally so each phase can use its own batch size.
    """
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    output_cfg = cfg["output"]

    model = build_model(num_classes, num_symptoms, model_cfg)
    if fold is None:
        model.summary(line_length=100)

    # Phase 1 – warm-up (backbone frozen)
    print("\n── Phase 1: Warm-up (backbone frozen) ──")
    train_ds_p1 = make_tf_dataset(
        train_images, train_labels_oh, train_symptoms,
        batch_size=train_cfg["phase1_batch_size"],
        shuffle=True,
        augment_fn=augment_fn,
    )
    val_ds_p1 = make_tf_dataset(
        val_images, val_labels_oh, val_symptoms,
        batch_size=train_cfg["phase1_batch_size"],
    )
    compile_model(model, train_cfg["phase1_learning_rate"], num_classes)
    callbacks = build_callbacks(train_cfg, output_cfg, fold)
    h1 = model.fit(
        train_ds_p1,
        validation_data=val_ds_p1,
        epochs=train_cfg["phase1_epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2 – fine-tuning
    print("\n── Phase 2: Fine-tuning (partial backbone unfrozen) ──")
    train_ds_p2 = make_tf_dataset(
        train_images, train_labels_oh, train_symptoms,
        batch_size=train_cfg["phase2_batch_size"],
        shuffle=True,
        augment_fn=augment_fn,
    )
    val_ds_p2 = make_tf_dataset(
        val_images, val_labels_oh, val_symptoms,
        batch_size=train_cfg["phase2_batch_size"],
    )
    unfreeze_backbone(model, model_cfg.get("fine_tune_from_layer", 100))
    compile_model(model, train_cfg["phase2_learning_rate"], num_classes)
    callbacks = build_callbacks(train_cfg, output_cfg, fold)
    h2 = model.fit(
        train_ds_p2,
        validation_data=val_ds_p2,
        epochs=train_cfg["phase2_epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    # Merge histories
    history = {}
    for key in h1.history:
        history[key] = h1.history[key] + h2.history[key]

    return model, history


# ---------------------------------------------------------------------------
# Evaluation & visualisation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: keras.Model,
    test_ds: tf.data.Dataset,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    test_symptoms: np.ndarray | None,
    class_names: list[str],
    plots_dir: Path,
    prefix: str = "",
) -> dict:
    """Compute metrics and generate visualisation plots."""
    print("\n[INFO] Evaluating on test set …")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"  Test loss    : {test_loss:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")

    # Predictions
    if test_symptoms is not None:
        y_pred_probs = model.predict([test_images, test_symptoms], verbose=0)
    else:
        y_pred_probs = model.predict(test_images, verbose=0)

    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_labels

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    print("\n" + classification_report(y_true, y_pred, target_names=class_names))

    num_classes = len(class_names)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(8, num_classes), max(6, num_classes - 1)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"{prefix}Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    cm_path = plots_dir / f"{prefix}confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved confusion matrix → {cm_path}")

    # ROC curves (one-vs-rest, multi-class)
    if num_classes > 1:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        if num_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

        plt.figure(figsize=(10, 7))
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_pred_probs[:, i])
            plt.plot(fpr, tpr, lw=1.5, label=f"{cls} (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlim([0, 1])
        plt.ylim([0, 1.02])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{prefix}ROC Curves (one-vs-rest)")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        roc_path = plots_dir / f"{prefix}roc_curves.png"
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"[INFO] Saved ROC curves → {roc_path}")

    return {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "classification_report": report,
    }


def plot_training_history(history: dict, plots_dir: Path, prefix: str = "") -> None:
    """Plot accuracy and loss curves from a training history dict."""
    epochs = range(1, len(history["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["accuracy"], label="Train accuracy")
    ax1.plot(epochs, history["val_accuracy"], label="Val accuracy")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(epochs, history["loss"], label="Train loss")
    ax2.plot(epochs, history["val_loss"], label="Val loss")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.suptitle(f"{prefix}Training History")
    plt.tight_layout()
    path = plots_dir / f"{prefix}training_history.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved training history plot → {path}")


# ---------------------------------------------------------------------------
# K-Fold cross-validation
# ---------------------------------------------------------------------------

def run_kfold(
    cfg: dict,
    all_images: np.ndarray,
    all_labels: np.ndarray,
    all_symptoms: np.ndarray | None,
    class_names: list[str],
    num_classes: int,
    num_symptoms: int,
    augment_fn,
) -> list[dict]:
    """Run stratified K-fold cross-validation and return per-fold metrics."""
    cv_cfg = cfg["cross_validation"]
    train_cfg = cfg["training"]
    output_cfg = cfg["output"]
    plots_dir = Path(output_cfg["plots_dir"])

    skf = StratifiedKFold(
        n_splits=cv_cfg["n_folds"],
        shuffle=cv_cfg.get("shuffle", True),
        random_state=cv_cfg.get("random_seed", 42),
    )
    labels_onehot = tf.keras.utils.to_categorical(all_labels, num_classes)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(all_images, all_labels), start=1
    ):
        print(f"\n{'='*60}")
        print(f"  K-FOLD  –  Fold {fold_idx}/{cv_cfg['n_folds']}")
        print(f"{'='*60}")

        fold_sym_train = all_symptoms[train_idx] if all_symptoms is not None else None
        fold_sym_val = all_symptoms[val_idx] if all_symptoms is not None else None

        _, history = train_full(
            cfg,
            all_images[train_idx],
            labels_onehot[train_idx],
            fold_sym_train,
            all_images[val_idx],
            labels_onehot[val_idx],
            fold_sym_val,
            num_classes,
            num_symptoms,
            augment_fn,
            fold=fold_idx,
        )

        plot_training_history(
            history, plots_dir, prefix=f"fold{fold_idx}_"
        )

        best_val_acc = max(history["val_accuracy"])
        print(f"[INFO] Fold {fold_idx} best val accuracy: {best_val_acc:.4f}")
        fold_results.append(
            {"fold": fold_idx, "best_val_accuracy": float(best_val_acc)}
        )

    mean_acc = np.mean([r["best_val_accuracy"] for r in fold_results])
    std_acc = np.std([r["best_val_accuracy"] for r in fold_results])
    print(f"\n[INFO] K-Fold results: {mean_acc:.4f} ± {std_acc:.4f}")
    fold_results.append({"mean_val_accuracy": float(mean_acc), "std_val_accuracy": float(std_acc)})
    return fold_results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    train_cfg = cfg["training"]
    output_cfg = cfg["output"]
    aug_cfg = cfg["augmentation"]

    # Optional mixed-precision training
    if cfg.get("hardware", {}).get("use_mixed_precision", False):
        mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision enabled (float16)")

    # ── Create output directories ─────────────────────────────────────────
    for key in ("models_dir", "history_dir", "plots_dir", "logs_dir"):
        Path(output_cfg[key]).mkdir(parents=True, exist_ok=True)
    plots_dir = Path(output_cfg["plots_dir"])

    # ── Load preprocessed data ────────────────────────────────────────────
    data_dir = Path(cfg["data"]["preprocessed_output_dir"])
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Preprocessed data directory not found: {data_dir}\n"
            "Run preprocessing.py first."
        )

    with open(data_dir / "class_names.json") as fh:
        class_names: list[str] = json.load(fh)
    num_classes = len(class_names)

    with open(data_dir / "symptom_columns.json") as fh:
        symptom_cols: list[str] = json.load(fh)
    num_symptoms = len(symptom_cols)
    has_symptoms = num_symptoms > 0

    print(f"[INFO] Classes ({num_classes}): {class_names}")
    print(f"[INFO] Symptom features: {num_symptoms}")

    # Load splits
    train_images, train_labels, train_symptoms = load_split(data_dir, "train", has_symptoms)
    val_images, val_labels, val_symptoms = load_split(data_dir, "val", has_symptoms)
    test_images, test_labels, test_symptoms = load_split(data_dir, "test", has_symptoms)

    # ── Augmentation ──────────────────────────────────────────────────────
    augment_fn = build_augmentation_layer(aug_cfg)

    # ── Build test tf.data dataset (for evaluate_model) ───────────────────
    test_labels_oh = tf.keras.utils.to_categorical(test_labels, num_classes)
    test_ds = make_tf_dataset(
        test_images, test_labels_oh, test_symptoms,
        batch_size=train_cfg["phase1_batch_size"],
    )

    # ── Optional K-Fold cross-validation ─────────────────────────────────
    cv_cfg = cfg["cross_validation"]
    if cv_cfg.get("enabled", False):
        all_images = np.concatenate([train_images, val_images], axis=0)
        all_labels = np.concatenate([train_labels, val_labels], axis=0)
        all_symptoms_cv = (
            np.concatenate([train_symptoms, val_symptoms], axis=0)
            if has_symptoms else None
        )
        fold_results = run_kfold(
            cfg, all_images, all_labels, all_symptoms_cv,
            class_names, num_classes, num_symptoms, augment_fn,
        )
        cv_results_path = Path(output_cfg["history_dir"]) / "kfold_results.json"
        with open(cv_results_path, "w") as fh:
            json.dump(fold_results, fh, indent=2)
        print(f"[INFO] K-Fold results saved → {cv_results_path}")

    # ── Final training on full train set ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL MODEL TRAINING")
    print("=" * 60)
    train_labels_oh = tf.keras.utils.to_categorical(train_labels, num_classes)
    val_labels_oh = tf.keras.utils.to_categorical(val_labels, num_classes)
    best_model, history = train_full(
        cfg,
        train_images, train_labels_oh, train_symptoms,
        val_images, val_labels_oh, val_symptoms,
        num_classes, num_symptoms, augment_fn,
    )

    # Save training history
    history_path = Path(output_cfg["history_dir"]) / "final_history.json"
    with open(history_path, "w") as fh:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.items()},
            fh,
            indent=2,
        )
    print(f"[INFO] Training history saved → {history_path}")

    # ── Visualisations ────────────────────────────────────────────────────
    plot_training_history(history, plots_dir, prefix="final_")

    metrics = evaluate_model(
        best_model,
        test_ds,
        test_images,
        test_labels,
        test_symptoms,
        class_names,
        plots_dir,
        prefix="final_",
    )

    # Save metrics
    metrics_path = Path(output_cfg["history_dir"]) / "final_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[INFO] Evaluation metrics saved → {metrics_path}")

    # ── Save model ────────────────────────────────────────────────────────
    model_path = Path(output_cfg["models_dir"]) / output_cfg["best_model_name"]
    best_model.save(str(model_path))
    print(f"\n[DONE] Best model saved → {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MangoSense multi-modal training")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()
    main(args.config)
