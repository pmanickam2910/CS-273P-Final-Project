"""
STEP 2 — Imports, config, and dataset loading
Run: exec(open("step2_data.py").read())
"""

import os, glob, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import timm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── Reproducibility ───────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Read dataset path ─────────────────────────────────────────────────
with open("dataset_path.txt") as f:
    path = f.read().strip()

DATA_ROOT = Path(path)
metadata_matches = list(DATA_ROOT.rglob("HAM10000_metadata.csv"))
assert len(metadata_matches) > 0, "metadata CSV not found"
METADATA = metadata_matches[0]

all_jpg_paths = list(DATA_ROOT.rglob("*.jpg"))
print(f"Total .jpg images found: {len(all_jpg_paths)}")
IMAGE_LOOKUP = {p.stem: str(p) for p in all_jpg_paths}

df = pd.read_csv(METADATA)
print(f"Total rows: {len(df)} | Unique lesions: {df['lesion_id'].nunique()}")

# ── Class config ──────────────────────────────────────────────────────
CLASS_NAMES  = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

print("\nClass Distribution:")
print(df["dx"].value_counts())

# ── Map paths ─────────────────────────────────────────────────────────
df["path"]  = df["image_id"].map(IMAGE_LOOKUP)
df = df.dropna(subset=["path"]).reset_index(drop=True)
df["label"] = df["dx"].map(CLASS_TO_IDX)
print(f"Final dataset size: {len(df)}")

# ── Patient-level split (70 / 15 / 15) ───────────────────────────────
unique_lesions = df["lesion_id"].unique()
train_lesions, temp_lesions = train_test_split(unique_lesions, test_size=0.30, random_state=SEED)
val_lesions,   test_lesions = train_test_split(temp_lesions,   test_size=0.50, random_state=SEED)

train_df = df[df["lesion_id"].isin(train_lesions)].reset_index(drop=True)
val_df   = df[df["lesion_id"].isin(val_lesions)].reset_index(drop=True)
test_df  = df[df["lesion_id"].isin(test_lesions)].reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"No lesion overlap: {len(set(train_lesions) & set(test_lesions)) == 0}")

# ── Transforms ────────────────────────────────────────────────────────
IMG_SIZE = 224

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Dataset ───────────────────────────────────────────────────────────
class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df        = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label

# ── Weighted sampler ──────────────────────────────────────────────────
def make_weighted_sampler(dataframe):
    class_counts   = dataframe["label"].value_counts().sort_index().values
    class_weights  = 1.0 / class_counts
    sample_weights = class_weights[dataframe["label"].values]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(dataframe),
        replacement=True,
    )

# ── DataLoaders ───────────────────────────────────────────────────────
BATCH_SIZE = 32

train_loader = DataLoader(SkinLesionDataset(train_df, train_transforms),
                          batch_size=BATCH_SIZE, sampler=make_weighted_sampler(train_df),
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(SkinLesionDataset(val_df, val_transforms),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(SkinLesionDataset(test_df, val_transforms),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")
print("Step 2 complete.")
