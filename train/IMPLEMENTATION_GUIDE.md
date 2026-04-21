# MangoSense – Custom CNN Implementation Guide

A comprehensive walkthrough for training the multi-modal mango disease
classification model used by the **MangoSense** mobile application.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Design](#2-architecture-design)
3. [Dataset Structure](#3-dataset-structure)
4. [Environment Setup](#4-environment-setup)
5. [Preprocessing Pipeline](#5-preprocessing-pipeline)
6. [Training Strategy](#6-training-strategy)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Integrating with the Mobile App](#8-integrating-with-the-mobile-app)
9. [Troubleshooting & Optimisation Tips](#9-troubleshooting--optimisation-tips)

---

## 1. Project Overview

| Item | Detail |
|------|--------|
| **Task** | Multi-class mango disease classification |
| **Visual data** | ~2 500 leaf images + ~500 fruit images (≈ 3 000 total) |
| **Supplementary data** | User-selected symptom checkboxes from the mobile app |
| **Backbone** | EfficientNetB0 (pretrained on ImageNet) |
| **Approach** | Multi-modal learning: image features ⊕ symptom features |
| **Target deployment** | Android / iOS via TensorFlow Lite |

### Why multi-modal learning?

With only ~3 000 images spread across multiple classes, a purely visual
classifier is likely to overfit.  When a user reports visible symptoms
(yellowing, black spots, wilting, etc.) through the mobile app, those
categorical signals give the model extra discriminative power that does not
require more labelled images.

---

## 2. Architecture Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Multi-Modal Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  IMAGE INPUT (224×224×3)      SYMPTOM INPUT (N binary features)     │
│        │                                  │                        │
│        ▼                                  ▼                        │
│  ┌───────────────────┐         ┌──────────────────────┐            │
│  │  EfficientNetB0   │         │  Dense(64, relu)     │            │
│  │  (pretrained,     │         │  + L2 regularisation │            │
│  │   ImageNet)       │         └──────────────────────┘            │
│  └───────────────────┘                    │                        │
│        │                                  │  (64 features)         │
│        ▼ GlobalAveragePooling2D           │                        │
│  (1 280 image features)                   │                        │
│        │                                  │                        │
│        └──────────────┬───────────────────┘                        │
│                       ▼                                            │
│              Concatenate (1 344 features)                          │
│                       │                                            │
│              Dense(256, relu) → Dropout(0.5)                       │
│                       │                                            │
│              Dense(128, relu) → Dropout(0.5)                       │
│                       │                                            │
│              Dense(num_classes, softmax)                           │
│                       │                                            │
│              Disease prediction                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Design rationale

| Component | Reason |
|-----------|--------|
| **EfficientNetB0** | Best accuracy-to-parameter ratio; exported to TFLite easily |
| **ImageNet pretraining** | Learned low-level feature detectors (edges, textures) transfer well to plant images |
| **GlobalAveragePooling2D** | Reduces spatial dimensions to a fixed vector; avoids fully-connected overfitting |
| **Symptom Dense branch** | Lightweight; adds discriminative signal without large parameter count |
| **Concatenation fusion** | Simple but effective; lets the head learn which modality to trust |
| **Dropout 0.5** | Primary regulariser for the small dataset |

---

## 3. Dataset Structure

Organise your raw data as shown below **before** running `preprocessing.py`:

```
data/
├── leaf_images/
│   ├── anthracnose/
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   ├── powdery_mildew/
│   │   └── …
│   └── healthy/
│       └── …
├── fruit_images/
│   ├── anthracnose/
│   │   └── …
│   └── healthy/
│       └── …
└── symptoms_data.csv
```

### `symptoms_data.csv` format

| Column | Type | Description |
|--------|------|-------------|
| `image_path` | string | Relative path or filename of the image |
| `symptom_yellowing` | 0 / 1 | Leaf / fruit shows yellowing |
| `symptom_black_spots` | 0 / 1 | Black or dark spots visible |
| `symptom_wilting` | 0 / 1 | Leaf shows wilting |
| `symptom_lesions` | 0 / 1 | Visible lesions or wounds |
| `symptom_powdery_coating` | 0 / 1 | White powdery coating |
| *(add more as needed)* | | |
| `disease_label` | string | Ground-truth class (same as folder name) |

> **Note:** If `symptoms_data.csv` is absent, the training scripts fall back
> to image-only mode automatically.

### Class balance tips

* Aim for **at least 100 images per class** for meaningful generalisation.
* If one class has far fewer samples, apply extra augmentation or consider
  using class weights (see [Troubleshooting](#9-troubleshooting--optimisation-tips)).
* Use the class distribution plot (`plots/class_distribution.png`) generated
  by `preprocessing.py` to spot imbalances early.

---

## 4. Environment Setup

### Prerequisites

* Python 3.10 or 3.11
* pip ≥ 23
* NVIDIA GPU with CUDA 11.8+ (optional but strongly recommended)

### Installation

```bash
# 1. Clone the repository (if you haven't already)
git clone https://github.com/meowbroken/MangoSense.git
cd MangoSense/train

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### GPU setup (optional)

Install the GPU-enabled TensorFlow build:

```bash
pip install tensorflow[and-cuda]   # TF ≥ 2.14 bundled CUDA
# or
pip install tensorflow-gpu>=2.13.0
```

Verify GPU is visible:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## 5. Preprocessing Pipeline

Run `preprocessing.py` **once** before any training runs:

```bash
cd MangoSense/train
python preprocessing.py --config config.yaml
```

### What the script does

```
collect_image_paths()
        │
        ▼
load_symptoms_csv()   ← aligns symptom rows with image paths
        │
        ▼
LabelEncoder.fit()    ← integer-encodes class names
        │
        ▼
stratified_split()    ← 70 / 15 / 15 with reproducible seed
        │
        ▼
load_images_batch()   ← resizes to 224×224, scales to [0, 1]
        │
        ▼
save .npy arrays + class_names.json + label_encoder.pkl
        │
        ▼
plot_class_distribution() + plot_sample_images()
```

### Output files

After a successful run you will find these files in
`data/preprocessed/`:

```
train_images.npy      val_images.npy      test_images.npy
train_labels.npy      val_labels.npy      test_labels.npy
train_symptoms.npy    val_symptoms.npy    test_symptoms.npy
class_names.json
label_encoder.pkl
symptom_columns.json
preprocessing_stats.json
```

And diagnostic plots in `plots/`:

```
class_distribution.png
sample_images.png
```

### Tuning preprocessing via `config.yaml`

```yaml
split:
  train: 0.70      # fraction for training
  val:   0.15      # fraction for validation
  test:  0.15      # fraction for held-out testing
  random_seed: 42  # change to get a different split

image:
  height: 224
  width:  224
```

---

## 6. Training Strategy

### Quick start

```bash
cd MangoSense/train
python train_model.py --config config.yaml
```

### Two-phase fine-tuning

| Phase | Backbone | LR | Purpose |
|-------|----------|----|---------|
| 1 – Warm-up | Frozen | 1 × 10⁻³ | Train the fusion head on fixed features |
| 2 – Fine-tune | Partially unfrozen (layers ≥ 100) | 1 × 10⁻⁵ | Adapt high-level backbone features to mango images |

Freezing the backbone during warm-up prevents the randomly initialised head
from destroying the pretrained weights before the head has learned something
useful.

### Data augmentation

Applied only to the **training set**:

| Transform | Value |
|-----------|-------|
| Random horizontal flip | enabled |
| Random rotation | ± 30° |
| Random zoom | ± 20 % |
| Random translation | ± 15 % (H and W) |
| Random contrast | ± 20 % |
| Random brightness | ± 20 % |

All augmentation parameters can be adjusted in `config.yaml` under
`augmentation:`.

### Early stopping & LR scheduling

* **EarlyStopping** – monitors `val_loss`, patience = 10 epochs, restores
  best weights automatically.
* **ReduceLROnPlateau** – halves LR when `val_loss` stagnates for 5 epochs;
  minimum LR = 1 × 10⁻⁶.

### Stratified K-Fold cross-validation

Enable in `config.yaml`:

```yaml
cross_validation:
  enabled: true
  n_folds: 5
```

K-Fold trains *k* separate models and reports mean ± std validation accuracy.
Use this to check how much variance exists between different data splits
before committing to a single final model.

> ⚠️  K-Fold multiplies training time by *k*.  Use `enabled: false` for
> quick iteration; enable it for your final thesis experiments.

### Saved artefacts

After training:

```
models/
    mangosense_multimodal_best.keras   ← best model weights
training_history/
    final_history.json
    final_metrics.json
    kfold_results.json                 ← only if K-Fold enabled
plots/
    final_training_history.png
    final_confusion_matrix.png
    final_roc_curves.png
    fold1_training_history.png … (K-Fold)
logs/
    <TensorBoard event files>
```

Launch TensorBoard to monitor training in real time:

```bash
tensorboard --logdir logs/
```

---

## 7. Evaluation Metrics

The script automatically computes and saves the following:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Fraction of correctly classified samples |
| **Precision (per class)** | TP / (TP + FP) |
| **Recall (per class)** | TP / (TP + FN) |
| **F1-score (per class)** | Harmonic mean of precision and recall |
| **Macro F1** | Unweighted average F1 across all classes |
| **Confusion Matrix** | Visual heatmap of predicted vs. true labels |
| **ROC / AUC** | One-vs-rest ROC curves with AUC for each class |

### Interpreting the confusion matrix

A diagonal entry (*i*, *i*) is the number of samples from class *i* that
were correctly predicted.  Off-diagonal entries reveal which classes the model
confuses most often.  Common confusions to watch for:

* `anthracnose` ↔ `black_spot` (similar dark lesion appearance)
* `healthy` ↔ `early_stage_disease` (subtle visual differences)

If a class is frequently confused, consider:

1. Collecting more samples for that class.
2. Adding a distinguishing symptom feature.
3. Applying targeted augmentation (colour jitter to emphasise lesion colours).

---

## 8. Integrating with the Mobile App

### Export to TensorFlow Lite

```python
import tensorflow as tf

model = tf.keras.models.load_model("models/mangosense_multimodal_best.keras")

# Convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # INT8 quantisation
tflite_model = converter.convert()

with open("models/mangosense.tflite", "wb") as f:
    f.write(tflite_model)
```

### Running inference in the app

The mobile app must supply **two inputs** to the TFLite model:

1. **Image tensor** – shape `[1, 224, 224, 3]`, dtype `float32`, values in
   `[0, 1]`.
2. **Symptom tensor** – shape `[1, N]`, dtype `float32`, binary values
   `0.0` or `1.0` for each symptom checkbox.

The output is a `[1, num_classes]` softmax probability array.  Take
`argmax` and map the index back to a class name using `class_names.json`.

---

## 9. Troubleshooting & Optimisation Tips

### Model is overfitting (train acc >> val acc)

| Fix | How |
|-----|-----|
| Increase dropout | `model.dropout_rate: 0.6` in config |
| Increase L2 | `model.l2_regularization: 0.001` |
| More augmentation | Increase rotation / zoom / translation ranges |
| Reduce model capacity | Decrease `fusion_dense_units` |
| Add more data | Use web-scraped or augmented images |

### Model is underfitting (both train and val acc are low)

| Fix | How |
|-----|-----|
| Train longer | Increase `phase1_epochs` / `phase2_epochs` |
| Increase LR | Try `phase1_learning_rate: 0.003` |
| Unfreeze more layers | Reduce `fine_tune_from_layer` (e.g. 80) |
| Reduce augmentation | Strong augmentation can be too noisy for small data |

### Class imbalance

If some classes have far fewer samples, add class weights:

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(enumerate(class_weights))

model.fit(
    train_ds,
    class_weight=class_weight_dict,
    …
)
```

### Out-of-memory errors

* Reduce `phase1_batch_size` to 8 or 16.
* Enable mixed precision: `hardware.use_mixed_precision: true`.
* Reduce image size to `160×160` (minor accuracy drop).

### Training is very slow (CPU only)

* Install a GPU build of TensorFlow.
* Use Google Colab (free T4 GPU) if no local GPU is available.
* Reduce the number of K-Fold splits temporarily.

### Symptom data missing for some images

The preprocessing script fills missing symptom rows with **zeros** (no
symptoms reported).  This is equivalent to a user clicking "none of the
above" and is a safe default.

---

*Last updated: April 2026 – MangoSense thesis project*
