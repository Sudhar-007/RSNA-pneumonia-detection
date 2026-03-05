import os
import cv2
import pydicom
import numpy as np
from tqdm import tqdm

DICOM_DIR = r"E:\Model\rsna-pneumonia-detection-challenge\stage_2_train_images"
OUT_DIR = r"E:\Model\rsna-pneumonia-detection-challenge\rsna_yolo_dataset\images\train"

os.makedirs(OUT_DIR, exist_ok=True)

files = [f for f in os.listdir(DICOM_DIR) if f.lower().endswith(".dcm")]

print("Found DICOM files:", len(files))

for f in tqdm(files, desc="Converting DICOM -> PNG"):
    path = os.path.join(DICOM_DIR, f)

    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)

    # Normalize to 0–255
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)

    # Convert grayscale -> 3 channel
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    out_name = f.replace(".dcm", ".png").replace(".DCM", ".png")
    out_path = os.path.join(OUT_DIR, out_name)

    cv2.imwrite(out_path, img)

print("Done! Converted:", len(files), "images.")
