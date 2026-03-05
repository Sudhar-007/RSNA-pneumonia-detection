# 🫁 RSNA Pneumonia Detection — YOLOv8 Object Detection

A full end-to-end deep learning pipeline that detects pneumonia in chest X-rays using YOLOv8. Built on the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) dataset — the model doesn't just classify whether pneumonia is present, it **localises exactly where** in the lung it appears by drawing bounding boxes around regions of opacity.

> Built on a Lenovo LOQ 2024 (Ryzen 7 7435HS, RTX 4060 8GB, 24GB RAM) — trained fully locally with CUDA.

---

## 📋 Table of Contents

- [What This Is](#what-this-is)
- [Dataset](#dataset)
- [Full Pipeline](#full-pipeline)
- [Environment Setup](#environment-setup)
- [Problems Encountered & How They Were Fixed](#problems-encountered--how-they-were-fixed)
- [Results](#results)
- [How To Run](#how-to-run)
- [Project Structure](#project-structure)
- [What I Learned](#what-i-learned)

---

## What This Is

Most pneumonia detection projects treat this as a **binary classification** problem — does this image have pneumonia or not? This project goes further by treating it as an **object detection** problem.

Instead of just predicting yes/no, the model:
- Scans the entire chest X-ray
- Draws bounding boxes around regions of lung opacity
- Assigns a confidence score to each detection

This is significantly harder than classification — the model must understand not just *that* pneumonia is present but *where* it is. That's far more clinically meaningful.

---

## Dataset

**Source:** [RSNA Pneumonia Detection Challenge — Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

| Property | Value |
|---|---|
| Total images | ~26,000 chest X-rays |
| Format | DICOM (.dcm) — raw medical scanner format |
| Labels | CSV with bounding box coordinates + Target column |
| Positive rate | ~22% (pneumonia present) |
| Train split | 21,349 images |
| Val split | 5,336 images |

The dataset has significant **class imbalance** — ~78% of images are healthy (Target=0), ~22% have pneumonia (Target=1). This is intentional — it reflects real-world clinical distribution.

---

## Full Pipeline

### Step 1 — DICOM → PNG Conversion

Medical imaging uses DICOM format — a complex file type that stores both image data and patient metadata. Standard ML frameworks can't read it directly.

Each `.dcm` file was converted to `.png` using `pydicom`:

```python
import pydicom
import numpy as np
from PIL import Image
import os

def convert_dicom_to_png(dicom_path, output_path):
    dcm = pydicom.dcmread(dicom_path)
    pixel_array = dcm.pixel_array
    # Normalize to 0-255
    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
    img = Image.fromarray(pixel_array.astype(np.uint8))
    img.save(output_path)
```

### Step 2 — CSV → YOLO Label Conversion

The RSNA CSV provided bounding boxes in pixel coordinates:
```
patientId, x, y, width, height, Target
abc123, 300, 240, 150, 180, 1
```

YOLO requires normalized coordinates in a specific format:
```
class_index x_center y_center width height
```
Where all values are between 0 and 1.

Conversion logic:
```python
import pandas as pd
import os

def convert_labels(csv_path, output_dir, img_size=1024):
    df = pd.read_csv(csv_path)
    # Only process rows with pneumonia (Target=1)
    df = df[df['Target'] == 1].dropna(subset=['x', 'y', 'width', 'height'])

    for patient_id, group in df.groupby('patientId'):
        label_path = os.path.join(output_dir, f"{patient_id}.txt")
        with open(label_path, 'w') as f:
            for _, row in group.iterrows():
                x_center = (row['x'] + row['width'] / 2) / img_size
                y_center = (row['y'] + row['height'] / 2) / img_size
                w = row['width'] / img_size
                h = row['height'] / img_size
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
```

Healthy images (Target=0) intentionally have no label file — YOLO treats them as background/negative samples automatically.

### Step 3 — Train/Val Split

Dataset was split 80/20:
- `images/train` — 21,349 images
- `images/val` — 5,336 images
- `labels/train` — 4,810 label files (pneumonia cases only)
- `labels/val` — 1,202 label files

### Step 4 — Dataset Config (rsna.yaml)

```yaml
path: E:\Model\rsna-pneumonia-detection-challenge\rsna_yolo_dataset
train: images/Train
val: images/val

nc: 1
names:
  0: pneumonia
```

### Step 5 — Training

```bash
yolo detect train data=rsna.yaml model=yolov8n.pt epochs=20 imgsz=512 batch=8 device=0
```

---

## Environment Setup

```bash
# Create virtual environment with Python 3.11
python -m venv venv
venv\Scripts\activate.bat  # Windows

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Ultralytics
pip install ultralytics

# Verify GPU is available
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:
```
True
NVIDIA GeForce RTX 4060 Laptop GPU
```

---

## Problems Encountered & How They Were Fixed

This section documents every failure hit during this project. These weren't tutorial mistakes — they were real engineering problems that took systematic diagnosis to identify.

---

### ❌ Problem 1 — Missing `nc` Field in rsna.yaml

**Symptom:** Model trained for a full hour on 21,349 images. All loss values (box_loss, dfl_loss) were exactly zero every epoch. Precision, recall, mAP50 all zero. Graphs completely flat.

**Why it happened:** The `rsna.yaml` config was missing the `nc: 1` field (number of classes). Without it, YOLOv8 had no valid class definition — every bounding box in every label file was silently rejected during training. The GPU ran real computation for an hour on images with effectively zero valid labels attached.

**What made it hard to spot:** YOLO gave no error. No warning. It just trained normally and produced garbage metrics. The model appeared to run successfully.

**Fix:** Added `nc: 1` to rsna.yaml. One line.

**Before (broken graphs):**

![Broken training graphs — all flat at zero](broken_result.png)

**After (working):**

![Working training graphs — losses decreasing, mAP climbing](results/results.png)

---

### ❌ Problem 2 — Folder Casing Mismatch (`Labels` vs `labels`)

**Symptom:** Even after fixing `nc`, training showed `0 images found` and `WARNING: Labels are missing or empty`.

**Why it happened:** The dataset folders were named `Images` and `Labels` (capital first letter). YOLO internally constructs the label path by replacing `images` → `labels` (lowercase) in the image path. On Linux this would be an instant failure. On Windows, the OS resolved the path anyway — but YOLO's cache scanner still flagged all images as unlabelled backgrounds because of the mismatch in its internal path logic.

**Fix:** Renamed folders to lowercase:
```cmd
ren Images images
ren Labels labels
```

---

### ❌ Problem 3 — Stale Cache Files

**Symptom:** After fixing the folder names, scanner was still reading old broken cache data from previous failed runs.

**Why it happened:** YOLO caches dataset scans in `.cache` files for speed. The cached data was from the broken runs — it remembered the labels as missing and kept using that.

**Fix:** Deleted stale cache files before retraining:
```cmd
del images\Train.cache
del images\val.cache
```

---

### ❌ Problem 4 — Virtual Environment Lost

**Symptom:** `ModuleNotFoundError: No module named 'torch'` when running yolo command.

**Why it happened:** The original working environment with CUDA PyTorch was on a separate machine that stopped functioning. Running `yolo` from the system Python (3.13) which had no torch installed.

**Fix:** Rebuilt the virtual environment from scratch on Python 3.11, reinstalled PyTorch with CUDA and Ultralytics. Full environment rebuild took under 15 minutes.

---

### ❌ Problem 5 — Corrupt Stray File in Training Folder

**Symptom:** Warning — `ignoring corrupt image: Images\Train\New Bitmap image.bmp`

**Why it happened:** A stray empty bitmap file had been accidentally created in the training images folder.

**Fix:**
```cmd
del "Images\Train\New Bitmap image.bmp"
```

---

## Results

### Final Model (yolov8n, 20 epochs, imgsz=512)

| Metric | Value |
|---|---|
| mAP50 | 0.344 |
| mAP50-95 | 0.143 |
| Precision | 0.407 |
| Recall | 0.362 |
| Training time | 1.749 hours |
| GPU | RTX 4060 Laptop (8GB) |

### Training Curves

![Training results](results/results.png)

All losses (box, cls, dfl) decrease consistently across 20 epochs on both train and val sets — no overfitting. mAP50 climbs from 0.05 at epoch 1 to 0.344 by epoch 20, with room to improve further with more epochs or a larger model variant.

### Context On The Numbers

mAP50 is measured at 50% IoU (Intersection over Union) threshold — meaning a detection only counts as correct if the predicted bounding box overlaps the ground truth box by at least 50%. This is significantly harder than classification accuracy.

A naive classifier that always predicts "no pneumonia" would score ~75% classification accuracy on this dataset due to class imbalance — but would score 0% mAP50 since it never produces any boxes. mAP50 cannot be gamed by class imbalance, making it a more honest metric for this task.

---

## How To Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/rsna-pneumonia-detection
cd rsna-pneumonia-detection
```

### 2. Download the dataset
Download from [Kaggle RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) and place in the project directory.

### 3. Set up environment
```bash
python -m venv venv
venv\Scripts\activate.bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics pydicom pandas pillow
```

### 4. Convert DICOM to PNG
```bash
python convert_dicom.py
```

### 5. Generate YOLO labels
```bash
python prepare_labels.py
```

### 6. Train
```bash
yolo detect train data=rsna.yaml model=yolov8n.pt epochs=20 imgsz=512 batch=8 device=0
```

### 7. Run inference on a single image
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/xray.png
```

---

## Project Structure

```
rsna-pneumonia-detection/
│
├── convert_dicom.py        # DICOM → PNG conversion
├── prepare_labels.py       # CSV → YOLO label conversion
├── rsna.yaml               # Dataset config
├── requirements.txt
├── README.md
│
├── rsna_yolo_dataset/
│   ├── images/
│   │   ├── Train/          # 21,349 PNG images
│   │   └── val/            # 5,336 PNG images
│   └── labels/
│       ├── Train/          # 4,810 label files
│       └── val/            # 1,202 label files
│
└── results/
    ├── results.png         # Training curves
    ├── broken_results.png  # Pre-fix broken training curves
    └── val_predictions.jpg # Sample predictions on validation set
```

---

## What I Learned

**Technical:**
- DICOM is the standard medical imaging format — not directly readable by standard image libraries, requires `pydicom` to extract pixel arrays
- YOLO label format requires normalized coordinates (0–1), not pixel values — a subtle but critical difference
- YOLO silently ignores invalid labels with no error — zero box_loss is the only symptom of a completely broken label pipeline
- Virtual environments are per-machine — losing a machine means rebuilding the environment from scratch
- Cache files from broken runs can persist and cause misleading behaviour on subsequent runs
- Class imbalance makes classification accuracy a dishonest metric — mAP50 is immune to this

**Engineering:**
- Systematic debugging — isolating each component (yaml, labels, paths, environment) individually rather than changing everything at once
- Reading training metrics to diagnose problems — flat losses mean no valid labels, not a model failure
- GPU memory management — yolov8n uses ~9% of an RTX 4060's VRAM, leaving significant headroom for larger variants

---

## Tech Stack

- Python 3.11
- PyTorch 2.5.1 + CUDA 12.1
- Ultralytics YOLOv8
- pydicom
- pandas
- PIL / Pillow
- NVIDIA RTX 4060 Laptop GPU

---

*Dataset: RSNA Pneumonia Detection Challenge — Radiological Society of North America*
