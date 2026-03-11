# ── 1. Install packages ───────────────────────────────────────────────
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "timm", "grad-cam", "-q"])

# ── 2. Imports ────────────────────────────────────────────────────────
import os, random, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm

from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── 3. Config ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── 4. Load dataset ───────────────────────────────────────────────────
DATA_ROOT = Path("/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000")
all_jpg_paths = list(DATA_ROOT.rglob("*.jpg"))
print(f"Total images found: {len(all_jpg_paths)}")
IMAGE_LOOKUP = {p.stem: str(p) for p in all_jpg_paths}

df = pd.read_csv(DATA_ROOT / "HAM10000_metadata.csv")
print(f"Total rows: {len(df)} | Unique lesions: {df['lesion_id'].nunique()}")

CLASS_NAMES  = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

print("\nClass Distribution:")
print(df["dx"].value_counts())

df["path"]  = df["image_id"].map(IMAGE_LOOKUP)
df = df.dropna(subset=["path"]).reset_index(drop=True)
df["label"] = df["dx"].map(CLASS_TO_IDX)
print(f"Final dataset size: {len(df)}")

# ── 5. Patient-level split (70/15/15) ─────────────────────────────────
unique_lesions = df["lesion_id"].unique()
train_lesions, temp_lesions = train_test_split(unique_lesions, test_size=0.30, random_state=SEED)
val_lesions,   test_lesions = train_test_split(temp_lesions,   test_size=0.50, random_state=SEED)

train_df = df[df["lesion_id"].isin(train_lesions)].reset_index(drop=True)
val_df   = df[df["lesion_id"].isin(val_lesions)].reset_index(drop=True)
test_df  = df[df["lesion_id"].isin(test_lesions)].reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"No lesion overlap: {len(set(train_lesions) & set(test_lesions)) == 0}")

# ── 6. Transforms ─────────────────────────────────────────────────────
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

# ── 7. Dataset & DataLoaders ──────────────────────────────────────────
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

def make_weighted_sampler(dataframe):
    class_counts   = dataframe["label"].value_counts().sort_index().values
    class_weights  = 1.0 / class_counts
    sample_weights = class_weights[dataframe["label"].values]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(dataframe),
        replacement=True,
    )

BATCH_SIZE   = 32
train_loader = DataLoader(SkinLesionDataset(train_df, train_transforms),
                          batch_size=BATCH_SIZE, sampler=make_weighted_sampler(train_df),
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(SkinLesionDataset(val_df,   val_transforms),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(SkinLesionDataset(test_df,  val_transforms),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

# ── 8. Models ─────────────────────────────────────────────────────────
NUM_CLASSES = 7

def build_efficientnet(num_classes=NUM_CLASSES):
    model = timm.create_model("efficientnet_b2", pretrained=True)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model

def build_vit(num_classes=NUM_CLASSES):
    model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
    embed_dim = model.embed_dim
    model.head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(embed_dim, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model

def build_efficientnet_frozen(num_classes=NUM_CLASSES):
    model = timm.create_model("efficientnet_b2", pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model

# sanity check
dummy = torch.randn(2, 3, 224, 224)
eff   = build_efficientnet()
vit   = build_vit()
print("EfficientNet-B2 output:", eff(dummy).shape)
print("ViT-Small output:      ", vit(dummy).shape)
eff_params = sum(p.numel() for p in eff.parameters()) / 1e6
vit_params = sum(p.numel() for p in vit.parameters()) / 1e6
print(f"EfficientNet-B2 params: {eff_params:.1f}M")
print(f"ViT-Small params:       {vit_params:.1f}M")

# ── 9. Training infrastructure ────────────────────────────────────────
_USE_AMP = torch.cuda.is_available()

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_labels = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(smooth_labels * log_probs).sum(dim=-1)
        return loss.mean()

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if _USE_AMP:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if _USE_AMP:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss   = criterion(logits, labels)
        else:
            logits = model(images)
            loss   = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        probs  = F.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    bal_acc  = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, bal_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)

def train_model(model, model_name, num_epochs=20, lr=5e-5):
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    model     = model.to(DEVICE)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    scaler    = torch.cuda.amp.GradScaler() if _USE_AMP else None
    history          = {"train_loss": [], "train_acc": [], "val_loss": [], "val_bal_acc": []}
    best_val_acc     = 0.0
    best_weights     = None
    patience         = 5
    patience_counter = 0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, DEVICE)
        val_loss, val_bal_acc, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_bal_acc"].append(val_bal_acc)
        if val_bal_acc > best_val_acc:
            best_val_acc     = val_bal_acc
            best_weights     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Bal-Acc: {val_bal_acc:.4f} | "
                  f"Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} — no improvement for {patience} epochs")
            break
    print(f"\nBest Val Balanced Accuracy: {best_val_acc:.4f}")
    model.load_state_dict(best_weights)
    torch.save(best_weights, f"{model_name}_best.pth")
    return model, history

# ── 10. Train both models ─────────────────────────────────────────────
efficientnet_model = build_efficientnet()
efficientnet_model, eff_history = train_model(efficientnet_model, "EfficientNet-B2", num_epochs=20)

vit_model = build_vit()
vit_model, vit_history = train_model(vit_model, "ViT-Small", num_epochs=20)

# ── 11. Training curves ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
eff_epochs = range(1, len(eff_history["train_loss"]) + 1)
vit_epochs = range(1, len(vit_history["train_loss"]) + 1)
axes[0].plot(eff_epochs, eff_history["val_bal_acc"], "b-o", markersize=3, label="EfficientNet-B2")
axes[0].plot(vit_epochs, vit_history["val_bal_acc"], "r-s", markersize=3, label="ViT-Small")
axes[0].set_title("Validation Balanced Accuracy", fontsize=13)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Balanced Accuracy")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(eff_epochs, eff_history["train_loss"], "b-o", markersize=3, label="EfficientNet-B2")
axes[1].plot(vit_epochs, vit_history["train_loss"], "r-s", markersize=3, label="ViT-Small")
axes[1].set_title("Training Loss", fontsize=13)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.suptitle("CNN vs ViT — Training Dynamics", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved training_curves.png")

# ── 12. Ablation study ────────────────────────────────────────────────
frozen_model = build_efficientnet_frozen()
frozen_model, _ = train_model(frozen_model, "EfficientNet-B2-Frozen", num_epochs=10)

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
_, frozen_bal_acc, _, _, _ = evaluate(frozen_model,       val_loader, criterion, DEVICE)
_, eff_bal_acc,    _, _, _ = evaluate(efficientnet_model, val_loader, criterion, DEVICE)
_, vit_bal_acc,    _, _, _ = evaluate(vit_model,          val_loader, criterion, DEVICE)

ablation_df = pd.DataFrame({
    "Model Variant": ["EfficientNet-B2 (Frozen)", "EfficientNet-B2 (Fine-Tuned)", "ViT-Small (Fine-Tuned)"],
    "Val Balanced Acc": [f"{frozen_bal_acc:.3f}", f"{eff_bal_acc:.3f}", f"{vit_bal_acc:.3f}"],
    "Delta vs Frozen":  ["—", f"+{eff_bal_acc-frozen_bal_acc:.3f}", f"+{vit_bal_acc-frozen_bal_acc:.3f}"],
})
print("\n── Ablation Study ──────────────────────────────────────")
print(ablation_df.to_string(index=False))

# ── 13. Test evaluation ───────────────────────────────────────────────
_, eff_test_acc, eff_preds, eff_labels, eff_probs = evaluate(efficientnet_model, test_loader, criterion, DEVICE)
_, vit_test_acc, vit_preds, vit_labels, vit_probs = evaluate(vit_model,          test_loader, criterion, DEVICE)

eff_probs = np.clip(eff_probs.astype(np.float64), 0, 1)
vit_probs = np.clip(vit_probs.astype(np.float64), 0, 1)
eff_probs = eff_probs / eff_probs.sum(axis=1, keepdims=True)
vit_probs = vit_probs / vit_probs.sum(axis=1, keepdims=True)

eff_auc = roc_auc_score(eff_labels, eff_probs, multi_class="ovr", average="macro")
vit_auc = roc_auc_score(vit_labels, vit_probs, multi_class="ovr", average="macro")

print("\n── Test Set Results ────────────────────────────────────")
print(f"EfficientNet-B2  | Balanced Acc: {eff_test_acc:.4f} | AUC-ROC: {eff_auc:.4f}")
print(f"ViT-Small/16     | Balanced Acc: {vit_test_acc:.4f} | AUC-ROC: {vit_auc:.4f}")

print("\n── EfficientNet-B2 Per-Class Report ────────────────────")
print(classification_report(eff_labels, eff_preds, target_names=CLASS_NAMES, digits=3))
print("\n── ViT-Small Per-Class Report ──────────────────────────")
print(classification_report(vit_labels, vit_preds, target_names=CLASS_NAMES, digits=3))

# ── 14. Confusion matrices ────────────────────────────────────────────
def plot_confusion_matrix(labels, preds, model_name):
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f"{model_name} — Normalized Confusion Matrix", fontsize=13)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_confusion_matrix(eff_labels, eff_preds, "EfficientNet-B2")
plot_confusion_matrix(vit_labels, vit_preds, "ViT-Small")

# ── 15. Final summary ─────────────────────────────────────────────────
summary = pd.DataFrame({
    "Model":             ["EfficientNet-B2 (CNN)", "ViT-Small/16 (Transformer)"],
    "Test Balanced Acc": [f"{eff_test_acc:.4f}", f"{vit_test_acc:.4f}"],
    "AUC-ROC (Macro)":   [f"{eff_auc:.4f}",     f"{vit_auc:.4f}"],
    "Params (M)":        [f"{eff_params:.1f}",   f"{vit_params:.1f}"],
})
print("\n══════════════════════════════════════════════════════")
print("FINAL RESULTS SUMMARY")
print("══════════════════════════════════════════════════════")
print(summary.to_string(index=False))
winner = "EfficientNet-B2" if eff_test_acc > vit_test_acc else "ViT-Small"
print(f"\n→ {winner} wins by {abs(eff_test_acc - vit_test_acc):.4f} balanced accuracy points")

# ── 16. GradCAM ───────────────────────────────────────────────────────
eff_cam = GradCAM(model=efficientnet_model, target_layers=[efficientnet_model.blocks[-1]])

def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)

vit_cam = GradCAM(model=vit_model, target_layers=[vit_model.blocks[-1].norm1],
                  reshape_transform=vit_reshape_transform)

samples = []
for cls in CLASS_NAMES:
    cls_df = test_df[test_df["dx"] == cls]
    if len(cls_df) > 0:
        row = cls_df.sample(1, random_state=SEED).iloc[0]
        samples.append((row["path"], cls, int(row["label"])))

n = len(samples)
fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
for i, (path, cls, label) in enumerate(samples):
    img_pil = Image.open(path).convert("RGB").resize((224, 224))
    img_np  = np.array(img_pil) / 255.0
    img_t   = val_transforms(img_pil).unsqueeze(0).to(DEVICE)
    target  = [ClassifierOutputTarget(label)]
    eff_overlay = show_cam_on_image(img_np.astype(np.float32), eff_cam(input_tensor=img_t, targets=target)[0], use_rgb=True)
    vit_overlay = show_cam_on_image(img_np.astype(np.float32), vit_cam(input_tensor=img_t, targets=target)[0], use_rgb=True)
    axes[i, 0].imshow(img_pil);     axes[i, 0].set_title(f"Original\n{cls}"); axes[i, 0].axis("off")
    axes[i, 1].imshow(eff_overlay); axes[i, 1].set_title("EfficientNet GradCAM"); axes[i, 1].axis("off")
    axes[i, 2].imshow(vit_overlay); axes[i, 2].set_title("ViT GradCAM");          axes[i, 2].axis("off")

plt.suptitle("GradCAM: Where CNN vs ViT Looks", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("gradcam_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nAll done!")
