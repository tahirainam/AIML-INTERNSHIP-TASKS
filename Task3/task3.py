# ==========================================================================
# Task 3 – Satellite Image Analysis for Deforestation Monitoring
# ==========================================================================
# What this script does
# ---------------------
# 1.  Reads the Planet Amazon satellite dataset
# 2.  Creates a binary label  ─  "forest" (1) vs. "deforested / other" (0)
# 3.  Fine-tunes a pretrained ResNet-18 CNN to classify each 256×256 image tile
# 4.  Runs “change detection” on two images of the *same* location
#     (predicts forest-probability on T1 & T2, highlights losses)
# 5.  Saves CSV predictions + change-maps for visual inspection
#
# Heavy training on the full dataset can take hours.  For a demo in minutes
# the code samples 5 000 pictures (≈10 %); adjust SAMPLE_FRAC for your GPU/CPU.
# ==========================================================================

# --------------------------------------------------------------------------
# 0.  Install (first time only):
# --------------------------------------------------------------------------
# pip install pandas numpy torch torchvision pillow tqdm matplotlib scikit-learn

# --------------------------------------------------------------------------
# 1.  Imports & config
# --------------------------------------------------------------------------
import os, random, glob, csv, math, shutil, time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import matplotlib.pyplot as plt

# ---------- adjustable paths -------------------------------------------------
DATA_ROOT   = Path(r"E:\datasets\planet_amazon")  # change to your dataset folder
TRAIN_CSV   = DATA_ROOT / "train_v2.csv"          # original CSV with multi-labels
TRAIN_IMG   = DATA_ROOT / "train-jpg"             # folder of .jpg tiles
MODEL_OUT   = "forest_classifier_resnet18.pth"
SAMPLE_FRAC = 0.10            # fraction of rows to keep (1.0 → full dataset)
BATCH_SIZE  = 32
NUM_EPOCHS  = 3               # small for demo; increase for real training
LR          = 3e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------
# 2.  Build binary labels (“forest” vs “non-forest”)
#     Original Planet labels include: primary, agriculture, clear, slash_burn…
#     We treat any tile containing 'primary' OR 'water' as FOREST (1)
# --------------------------------------------------------------------------
df = pd.read_csv(TRAIN_CSV)
if SAMPLE_FRAC < 1.0:
    df = df.sample(frac=SAMPLE_FRAC, random_state=42).reset_index(drop=True)

def label_to_binary(tags: str) -> int:
    tags_set = set(tags.split())
    return int(bool({"primary", "water"} & tags_set))  # 1 any forest / water

df["forest"] = df["tags"].apply(label_to_binary)

print(f"[INFO] Dataset size (sample): {len(df):,} images")
print(df["forest"].value_counts())

# --------------------------------------------------------------------------
# 3.  PyTorch custom Dataset
# --------------------------------------------------------------------------
class PlanetDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['image_name']}.jpg"
        img = Image.open(img_path).convert("RGB")
        y   = torch.tensor(row["forest"], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)
        return img, y

# Image augmentations & normalization matching ImageNet
train_tfms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225])
])

val_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225])
])

# Train / validation split (80 / 20)
val_frac = 0.2
val_size = int(len(df)*val_frac)
train_df = df.iloc[val_size:].reset_index(drop=True)
val_df   = df.iloc[:val_size].reset_index(drop=True)

train_ds = PlanetDataset(train_df, TRAIN_IMG, transform=train_tfms)
val_ds   = PlanetDataset(val_df,   TRAIN_IMG, transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# --------------------------------------------------------------------------
# 4.  Model – ResNet-18, last layer replaced by 1-neuron sigmoid
# --------------------------------------------------------------------------
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1),
    nn.Sigmoid()
)
model.to(DEVICE)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --------------------------------------------------------------------------
# 5.  Training loop
# --------------------------------------------------------------------------
def evaluate(loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).flatten()
            preds_label = (preds > 0.5).float()
            correct += (preds_label == y).sum().item()
            total   += y.size(0)
    model.train()
    return correct / total

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        preds = model(x_batch).flatten()
        loss  = criterion(preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y_batch.size(0)

    train_acc = evaluate(train_loader)
    val_acc   = evaluate(val_loader)
    print(f"Epoch {epoch+1}: Loss={running_loss/len(train_ds):.4f}, "
          f"Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")

# Save weights
torch.save(model.state_dict(), MODEL_OUT)
print(f"[INFO] Model saved to {MODEL_OUT}")

# --------------------------------------------------------------------------
# 6.  Inference helper - Predict forest probability for one image
# --------------------------------------------------------------------------
infer_tfms = val_tfms
def predict_proba(img_path: Path) -> float:
    img = Image.open(img_path).convert("RGB")
    x   = infer_tfms(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        proba = model(x).item()
    return proba  # probability that tile is *forest*

# --------------------------------------------------------------------------
# 7.  Simple Change Detection
#     User supplies two images of the SAME location at two dates.
#     We predict forest-probability at T1 & T2 and flag forest→non-forest.
# --------------------------------------------------------------------------
def change_detection(img_t1: Path, img_t2: Path, out_png: Path):
    p1, p2 = predict_proba(img_t1), predict_proba(img_t2)
    print(f"[INFO] Forest prob T1={p1:.2f}, T2={p2:.2f}")

    # Heuristic: significant drop (>0.5) => deforestation alert
    if p1 - p2 > 0.50:
        print("⚠️  Possible deforestation detected!")
    else:
        print("No major forest loss detected.")

    # Create side-by-side comparison for visual report
    img1 = Image.open(img_t1).convert("RGB")
    img2 = Image.open(img_t2).convert("RGB")
    concat = Image.new("RGB", (img1.width*2, img1.height))
    concat.paste(img1, (0,0))
    concat.paste(img2, (img1.width,0))
    concat.save(out_png)
    print(f"[INFO] Saved comparison image → {out_png}")

# Example demo (update with real pair of tiles of same coordinates)
# change_detection(Path("tile_123_2020.jpg"),
#                  Path("tile_123_2024.jpg"),
#                  Path("deforestation_check.png"))

# --------------------------------------------------------------------------
# 8.  Bulk prediction CSV (optional)
# --------------------------------------------------------------------------
def bulk_predict(df_subset, outfile="tile_forest_probs.csv"):
    model.eval()
    records = []
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset),
                       desc="Bulk predicting"):
        img_path = TRAIN_IMG / f"{row['image_name']}.jpg"
        proba = predict_proba(img_path)
        records.append({"image_name": row['image_name'],
                        "forest_proba": proba})
    pd.DataFrame(records).to_csv(outfile, index=False)
    print(f"[INFO] Saved {outfile}")

# bulk_predict(df.iloc[:1000])  # small sample
