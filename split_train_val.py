import os
import random
import shutil
from tqdm import tqdm

random.seed(42)

BASE = r"E:\Model\rsna-pneumonia-detection-challenge\rsna_yolo_dataset"

IMG_TRAIN = os.path.join(BASE, "images", "train")
IMG_VAL = os.path.join(BASE, "images", "val")

LBL_TRAIN = os.path.join(BASE, "labels", "train")
LBL_VAL = os.path.join(BASE, "labels", "val")

os.makedirs(IMG_VAL, exist_ok=True)
os.makedirs(LBL_VAL, exist_ok=True)

# List all PNG images
imgs = [f for f in os.listdir(IMG_TRAIN) if f.endswith(".png")]

random.shuffle(imgs)

val_size = int(0.2 * len(imgs))
val_imgs = imgs[:val_size]

for img_file in tqdm(val_imgs, desc="Moving validation files"):
    img_id = img_file.replace(".png", "")

    src_img = os.path.join(IMG_TRAIN, img_file)
    dst_img = os.path.join(IMG_VAL, img_file)

    src_lbl = os.path.join(LBL_TRAIN, img_id + ".txt")
    dst_lbl = os.path.join(LBL_VAL, img_id + ".txt")

    shutil.move(src_img, dst_img)

    # Move label if it exists (only pneumonia images have labels)
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)

print("Done! Train/Val split completed.")
