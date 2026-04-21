"""
MangoSense – Data Preprocessing Script
=======================================
Prepares image data (leaf + fruit) and symptom CSV data for training the
multi-modal disease classification model.

Outputs (saved to config 'preprocessed_output_dir'):
    - train_images.npy, val_images.npy, test_images.npy
    - train_labels.npy, val_labels.npy, test_labels.npy
    - train_symptoms.npy, val_symptoms.npy, test_symptoms.npy  (if CSV present)
    - class_names.json
    - label_encoder.pkl
    - preprocessing_stats.json  (pixel mean/std for sanity checks)

Usage:
    python preprocessing.py                    # uses default config.yaml
    python preprocessing.py --config my.yaml   # custom config file
"""

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def collect_image_paths(root_dir: str) -> tuple[list[str], list[str]]:
    """
    Walk *root_dir* and collect all image file paths together with their
    class label (directory name one level below *root_dir*).

    Expected layout::

        root_dir/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img1.jpg
    """
    image_paths: list[str] = []
    labels: list[str] = []
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root_dir}")

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in supported:
                image_paths.append(str(img_path))
                labels.append(class_dir.name)

    return image_paths, labels


def load_and_preprocess_image(
    path: str, target_size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Load a single image from *path*, resize to *target_size* and return a
    float32 array with pixel values in [0, 1].
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def load_images_batch(
    paths: list[str], target_size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Load and preprocess a list of image paths into a single NumPy array."""
    images = []
    for p in tqdm(paths, desc="Loading images", unit="img"):
        images.append(load_and_preprocess_image(p, target_size))
    return np.stack(images, axis=0)


def load_symptoms_csv(
    csv_path: str, image_paths: list[str]
) -> tuple[np.ndarray | None, list[str]]:
    """
    Load symptom features from a CSV file and align them with *image_paths*.

    Expected CSV columns:
        image_path   – relative or absolute path matching entries in *image_paths*
        symptom_*    – binary (0/1) or float columns for each symptom
        disease_label – class name (used for verification only; labels come from
                        the directory structure)

    Returns:
        symptoms  – float32 array shaped (N, num_symptoms) aligned with
                    *image_paths*, or ``None`` if CSV is absent.
        symptom_cols – list of symptom column names.
    """
    if not csv_path or not Path(csv_path).exists():
        print("[INFO] symptoms_data.csv not found – training without symptom features.")
        return None, []

    df = pd.read_csv(csv_path)
    symptom_cols = [c for c in df.columns if c not in ("image_path", "disease_label")]

    # Build a lookup: filename stem → symptom row
    df["_stem"] = df["image_path"].apply(lambda p: Path(p).stem)
    stem_to_row = dict(zip(df["_stem"], df[symptom_cols].values.astype(np.float32)))

    symptoms = []
    for p in image_paths:
        stem = Path(p).stem
        row = stem_to_row.get(stem)
        if row is None:
            # No symptom data available for this image – use zero vector
            row = np.zeros(len(symptom_cols), dtype=np.float32)
        symptoms.append(row)

    return np.stack(symptoms, axis=0), symptom_cols


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def stratified_split(
    image_paths: list[str],
    labels_encoded: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Perform two-step stratified splitting:
        all → train  +  temp (val + test)
        temp → val   +  test

    Returns a dict with keys ``train``, ``val``, ``test`` each mapping to an
    index array into *image_paths* / *labels_encoded*.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Split ratios must sum to 1.0"
    )

    indices = np.arange(len(image_paths))
    temp_ratio = val_ratio + test_ratio  # fraction for val + test combined

    idx_train, idx_temp, _, y_temp = train_test_split(
        indices,
        labels_encoded,
        test_size=temp_ratio,
        stratify=labels_encoded,
        random_state=random_seed,
    )
    val_fraction_of_temp = val_ratio / temp_ratio
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=1.0 - val_fraction_of_temp,
        stratify=y_temp,
        random_state=random_seed,
    )
    return {"train": idx_train, "val": idx_val, "test": idx_test}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_class_distribution(labels: list[str], title: str, save_path: str) -> None:
    """Bar chart of per-class sample counts."""
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(max(8, len(unique) * 1.2), 5))
    plt.bar(unique, counts, color="steelblue", edgecolor="black")
    plt.title(title)
    plt.xlabel("Disease class")
    plt.ylabel("Number of images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved class distribution plot → {save_path}")


def plot_sample_images(
    image_paths: list[str],
    labels: list[str],
    n_per_class: int = 3,
    target_size: tuple[int, int] = (224, 224),
    save_path: str = "sample_images.png",
) -> None:
    """Grid of sample images, *n_per_class* images per disease class."""
    class_names = sorted(set(labels))
    n_classes = len(class_names)
    fig, axes = plt.subplots(
        n_classes, n_per_class, figsize=(n_per_class * 3, n_classes * 3)
    )
    if n_classes == 1:
        axes = [axes]

    for row_idx, cls in enumerate(class_names):
        cls_paths = [p for p, l in zip(image_paths, labels) if l == cls][:n_per_class]
        for col_idx in range(n_per_class):
            ax = axes[row_idx][col_idx] if n_per_class > 1 else axes[row_idx]
            if col_idx < len(cls_paths):
                img = load_and_preprocess_image(cls_paths[col_idx], target_size)
                ax.imshow(img)
                if col_idx == 0:
                    ax.set_ylabel(cls, fontsize=9, rotation=45, ha="right")
            ax.axis("off")

    plt.suptitle("Sample images per class", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved sample images plot → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    img_cfg = cfg["image"]
    split_cfg = cfg["split"]

    target_size = (img_cfg["height"], img_cfg["width"])
    out_dir = Path(data_cfg["preprocessed_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = Path(cfg["output"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Collect image paths from leaf + fruit directories
    # ------------------------------------------------------------------
    all_paths: list[str] = []
    all_labels: list[str] = []

    for key in ("leaf_images_dir", "fruit_images_dir"):
        dir_path = data_cfg[key]
        if Path(dir_path).exists():
            paths, labels = collect_image_paths(dir_path)
            all_paths.extend(paths)
            all_labels.extend(labels)
            print(f"[INFO] {key}: {len(paths)} images, {len(set(labels))} classes")
        else:
            print(f"[WARN] Directory not found, skipping: {dir_path}")

    if not all_paths:
        raise RuntimeError(
            "No images found. Check 'leaf_images_dir' and 'fruit_images_dir' in config."
        )

    print(f"\n[INFO] Total images collected: {len(all_paths)}")
    print(f"[INFO] Classes: {sorted(set(all_labels))}")

    # ------------------------------------------------------------------
    # 2. Encode labels
    # ------------------------------------------------------------------
    le = LabelEncoder()
    labels_encoded = le.fit_transform(all_labels)
    class_names = list(le.classes_)

    # Save encoder and class names
    with open(out_dir / "label_encoder.pkl", "wb") as fh:
        pickle.dump(le, fh)
    with open(out_dir / "class_names.json", "w") as fh:
        json.dump(class_names, fh, indent=2)
    print(f"[INFO] Encoded {len(class_names)} classes: {class_names}")

    # ------------------------------------------------------------------
    # 3. Stratified split
    # ------------------------------------------------------------------
    idx_map = stratified_split(
        all_paths,
        labels_encoded,
        train_ratio=split_cfg["train"],
        val_ratio=split_cfg["val"],
        test_ratio=split_cfg["test"],
        random_seed=split_cfg["random_seed"],
    )
    for split_name, idxs in idx_map.items():
        print(f"[INFO] {split_name}: {len(idxs)} samples")

    # ------------------------------------------------------------------
    # 4. Load & preprocess symptom data
    # ------------------------------------------------------------------
    symptoms, symptom_cols = load_symptoms_csv(
        data_cfg.get("symptoms_csv", ""), all_paths
    )

    # ------------------------------------------------------------------
    # 5. Load images (per split) and save as .npy
    # ------------------------------------------------------------------
    stats: dict = {}
    for split_name, idxs in idx_map.items():
        print(f"\n[INFO] Processing {split_name} split …")
        split_paths = [all_paths[i] for i in idxs]
        split_labels = labels_encoded[idxs]

        imgs = load_images_batch(split_paths, target_size)

        np.save(out_dir / f"{split_name}_images.npy", imgs)
        np.save(out_dir / f"{split_name}_labels.npy", split_labels)

        if symptoms is not None:
            np.save(out_dir / f"{split_name}_symptoms.npy", symptoms[idxs])

        # Record per-split statistics
        stats[split_name] = {
            "n_samples": int(len(idxs)),
            "pixel_mean": float(imgs.mean()),
            "pixel_std": float(imgs.std()),
        }

    # Save column names so the training script knows symptom dimensionality
    with open(out_dir / "symptom_columns.json", "w") as fh:
        json.dump(symptom_cols, fh, indent=2)

    with open(out_dir / "preprocessing_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    print("\n[INFO] All splits saved to:", out_dir)

    # ------------------------------------------------------------------
    # 6. Visualisations
    # ------------------------------------------------------------------
    plot_class_distribution(
        all_labels,
        title="Class distribution (all data)",
        save_path=str(plots_dir / "class_distribution.png"),
    )

    plot_sample_images(
        all_paths,
        all_labels,
        n_per_class=3,
        target_size=target_size,
        save_path=str(plots_dir / "sample_images.png"),
    )

    print("\n[DONE] Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MangoSense data preprocessing")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()
    main(args.config)
