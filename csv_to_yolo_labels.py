import os
import pandas as pd
from tqdm import tqdm

CSV_PATH = r"E:\Model\rsna-pneumonia-detection-challenge\stage_2_train_labels.csv"
IMG_DIR = r"E:\Model\rsna-pneumonia-detection-challenge\rsna_yolo_dataset\images\train"
LABEL_DIR = r"E:\Model\rsna-pneumonia-detection-challenge\rsna_yolo_dataset\labels\train"

os.makedirs(LABEL_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# RSNA: Target=1 means pneumonia (bounding box exists)
df = df[df["Target"] == 1]

# Image size in RSNA is 1024x1024 (standard for this dataset)
IMG_W = 1024
IMG_H = 1024

# Group by patientId (one image may have multiple boxes)
grouped = df.groupby("patientId")

for patient_id, group in tqdm(grouped, desc="Creating YOLO label files"):
    label_path = os.path.join(LABEL_DIR, f"{patient_id}.txt")

    lines = []
    for _, row in group.iterrows():
        x = row["x"]
        y = row["y"]
        w = row["width"]
        h = row["height"]

        # Convert to YOLO format (normalized)
        x_center = (x + w / 2) / IMG_W
        y_center = (y + h / 2) / IMG_H
        w_norm = w / IMG_W
        h_norm = h / IMG_H

        # class 0 = pneumonia
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

print("Done! Label files created.")
