"""
Convert Roboflow YOLO detection dataset to classification folder format.
Pools all splits, does 80/20 stratified re-split.

Input:  roboflow_model/ (YOLO detection: images/ + labels/)
Output: cls_dataset/train/{ClassName}/*.jpg + cls_dataset/val/{ClassName}/*.jpg
"""
import os
import yaml
import shutil
import random
from collections import defaultdict

random.seed(42)

SRC = "roboflow_model"
DST = "cls_dataset"

# Step 1: Read class names
with open(os.path.join(SRC, "data.yaml")) as f:
    data = yaml.safe_load(f)
names = data["names"]
print(f"Classes: {len(names)}")

# Step 2: Pool ALL images from all splits
all_images = []  # (image_path, label_path)
for split in ["train", "valid", "test"]:
    img_dir = os.path.join(SRC, split, "images")
    lbl_dir = os.path.join(SRC, split, "labels")
    if not os.path.isdir(img_dir):
        continue
    for fname in os.listdir(img_dir):
        if not fname.endswith((".jpg", ".jpeg", ".png")):
            continue
        lbl_name = os.path.splitext(fname)[0] + ".txt"
        lbl_path = os.path.join(lbl_dir, lbl_name)
        if os.path.exists(lbl_path):
            all_images.append((os.path.join(img_dir, fname), lbl_path))

print(f"Total images pooled: {len(all_images)}")

# Step 3: Group by class
class_images = defaultdict(list)
for img_path, lbl_path in all_images:
    with open(lbl_path) as f:
        line = f.readline().strip()
        if not line:
            continue
        cls_idx = int(line.split()[0])
        cls_name = names[cls_idx]
        class_images[cls_name].append(img_path)

print(f"Classes with images: {len(class_images)}")

# Step 4: Stratified 80/20 split
train_items = []  # (cls_name, img_path)
val_items = []

for cls_name, imgs in sorted(class_images.items()):
    random.shuffle(imgs)
    n = len(imgs)
    # At least 1 in val, rest in train
    n_val = max(1, int(n * 0.2))
    n_train = n - n_val
    for img in imgs[:n_train]:
        train_items.append((cls_name, img))
    for img in imgs[n_train:]:
        val_items.append((cls_name, img))

print(f"Train: {len(train_items)}, Val: {len(val_items)}")

# Step 5: Create folder structure and copy
if os.path.exists(DST):
    shutil.rmtree(DST)

for split, items in [("train", train_items), ("val", val_items)]:
    for cls_name, img_path in items:
        dst_dir = os.path.join(DST, split, cls_name)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(img_path, dst_dir)

# Step 6: Verify
print("\n-- Verification --")
for split in ["train", "val"]:
    split_dir = os.path.join(DST, split)
    classes = sorted(os.listdir(split_dir))
    total = sum(len(os.listdir(os.path.join(split_dir, c))) for c in classes)
    print(f"{split}: {total} images across {len(classes)} classes")

# Show demo deck cards
demo = ["Archers", "Arrows", "Fireball", "Giant", "Knight", "Mini_Pekka", "Minions", "Musketeer"]
print("\n-- Demo deck distribution --")
for card in demo:
    t = len(os.listdir(os.path.join(DST, "train", card))) if os.path.isdir(os.path.join(DST, "train", card)) else 0
    v = len(os.listdir(os.path.join(DST, "val", card))) if os.path.isdir(os.path.join(DST, "val", card)) else 0
    print(f"  {card}: train={t}, val={v}")

print("\nDone. Dataset ready at:", os.path.abspath(DST))
