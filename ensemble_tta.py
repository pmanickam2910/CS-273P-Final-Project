# ══════════════════════════════════════════════════════════════════════
# ENSEMBLE + TEST-TIME AUGMENTATION (TTA)
# Add this as a new cell AFTER all existing code has run
# ══════════════════════════════════════════════════════════════════════

import torch.nn.functional as F

# ── TTA Transforms ────────────────────────────────────────────────────
# We show each image in 5 different versions and average predictions
tta_transforms = [
    # Original
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    # Horizontal flip
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    # Vertical flip
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    # Rotate 90
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(90, 90)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    # Rotate 270
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(270, 270)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
]

# ── TTA Prediction function ───────────────────────────────────────────
@torch.no_grad()
def predict_with_tta(model, dataset_df, tta_transforms, device):
    """Run model on each image with all TTA versions and average probabilities."""
    model.eval()
    all_probs  = []
    all_labels = []

    for idx in range(len(dataset_df)):
        row   = dataset_df.iloc[idx]
        label = int(row["label"])
        img   = Image.open(row["path"]).convert("RGB")

        # Get predictions for each TTA version
        aug_probs = []
        for tfm in tta_transforms:
            img_t  = tfm(img).unsqueeze(0).to(device)
            logits = model(img_t)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            aug_probs.append(probs)

        # Average across all augmentations
        avg_prob = np.mean(aug_probs, axis=0)
        all_probs.append(avg_prob)
        all_labels.append(label)

    return np.vstack(all_probs), np.array(all_labels)

# ── Run TTA on both models ────────────────────────────────────────────
print("Running TTA for EfficientNet-B2 (this takes a few minutes)...")
eff_tta_probs, eff_tta_labels = predict_with_tta(efficientnet_model, test_df, tta_transforms, DEVICE)

print("Running TTA for ViT-Small (this takes a few minutes)...")
vit_tta_probs, vit_tta_labels = predict_with_tta(vit_model, test_df, tta_transforms, DEVICE)

# ── Ensemble: average EfficientNet + ViT probabilities ───────────────
ensemble_probs  = (eff_tta_probs + vit_tta_probs) / 2
ensemble_preds  = ensemble_probs.argmax(axis=1)
ensemble_labels = eff_tta_labels  # same for both

# ── Compute all metrics ───────────────────────────────────────────────
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# TTA-only results
eff_tta_preds = eff_tta_probs.argmax(axis=1)
vit_tta_preds = vit_tta_probs.argmax(axis=1)

eff_tta_acc = balanced_accuracy_score(eff_tta_labels, eff_tta_preds)
vit_tta_acc = balanced_accuracy_score(vit_tta_labels, vit_tta_preds)
ens_acc     = balanced_accuracy_score(ensemble_labels, ensemble_preds)

# Clip and normalize probs before AUC
def safe_probs(p):
    p = np.clip(p.astype(np.float64), 0, 1)
    return p / p.sum(axis=1, keepdims=True)

eff_tta_auc = roc_auc_score(eff_tta_labels, safe_probs(eff_tta_probs), multi_class="ovr", average="macro")
vit_tta_auc = roc_auc_score(vit_tta_labels, safe_probs(vit_tta_probs), multi_class="ovr", average="macro")
ens_auc     = roc_auc_score(ensemble_labels, safe_probs(ensemble_probs), multi_class="ovr", average="macro")

# ── Results comparison table ──────────────────────────────────────────
print("\n══════════════════════════════════════════════════════════════")
print("RESULTS: Baseline vs TTA vs Ensemble+TTA")
print("══════════════════════════════════════════════════════════════")
print(f"{'Method':<35} {'Bal Acc':>10} {'AUC-ROC':>10}")
print("─" * 57)
print(f"{'EfficientNet-B2 (baseline)':<35} {eff_test_acc:>10.4f} {eff_auc:>10.4f}")
print(f"{'ViT-Small (baseline)':<35} {vit_test_acc:>10.4f} {vit_auc:>10.4f}")
print("─" * 57)
print(f"{'EfficientNet-B2 + TTA':<35} {eff_tta_acc:>10.4f} {eff_tta_auc:>10.4f}")
print(f"{'ViT-Small + TTA':<35} {vit_tta_acc:>10.4f} {vit_tta_auc:>10.4f}")
print("─" * 57)
print(f"{'Ensemble (EfficientNet + ViT) + TTA':<35} {ens_acc:>10.4f} {ens_auc:>10.4f}")
print("══════════════════════════════════════════════════════════════")

best_acc = max(eff_test_acc, vit_test_acc, eff_tta_acc, vit_tta_acc, ens_acc)
if ens_acc == best_acc:
    print(f"\n→ Ensemble + TTA wins! Improvement over best single model: +{ens_acc - max(eff_test_acc, vit_test_acc):.4f}")
else:
    print(f"\n→ Best method: {best_acc:.4f}")

# ── Plot comparison bar chart ─────────────────────────────────────────
methods = [
    "EfficientNet\n(baseline)",
    "ViT\n(baseline)",
    "EfficientNet\n+TTA",
    "ViT\n+TTA",
    "Ensemble\n+TTA",
]
accs    = [eff_test_acc, vit_test_acc, eff_tta_acc, vit_tta_acc, ens_acc]
colors  = ["#4C72B0", "#DD8452", "#4C72B0", "#DD8452", "#2ca02c"]
alphas  = [0.5, 0.5, 0.9, 0.9, 1.0]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(methods, accs, color=colors)
for bar, alpha in zip(bars, alphas):
    bar.set_alpha(alpha)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{acc:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_title("Balanced Accuracy: Baseline vs TTA vs Ensemble", fontsize=13, fontweight="bold")
ax.set_ylabel("Balanced Accuracy")
ax.set_ylim(0.6, min(1.0, max(accs) + 0.05))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("ensemble_tta_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved ensemble_tta_comparison.png")
