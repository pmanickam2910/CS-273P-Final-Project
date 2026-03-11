"""
STEP 7 — Test evaluation, confusion matrices, results summary
Run: exec(open("step7_evaluate.py").read())
Requires step5 to have been run (models in memory).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from step4_training import evaluate, LabelSmoothingCrossEntropy

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

# ── Test evaluation ───────────────────────────────────────────────────
_, eff_test_acc, eff_preds, eff_labels, eff_probs = evaluate(efficientnet_model, test_loader, criterion, DEVICE)
_, vit_test_acc, vit_preds, vit_labels, vit_probs = evaluate(vit_model,          test_loader, criterion, DEVICE)

# Fix FP16 precision + clip negatives then renormalize
eff_probs = np.clip(eff_probs.astype(np.float64), 0, 1)
vit_probs = np.clip(vit_probs.astype(np.float64), 0, 1)
eff_probs = eff_probs / eff_probs.sum(axis=1, keepdims=True)
vit_probs = vit_probs / vit_probs.sum(axis=1, keepdims=True)

eff_auc = roc_auc_score(eff_labels, eff_probs, multi_class="ovr", average="macro")
vit_auc = roc_auc_score(vit_labels, vit_probs, multi_class="ovr", average="macro")

print("\n── Test Set Results ────────────────────────────────────")
print(f"EfficientNet-B2  | Balanced Acc: {eff_test_acc:.4f} | AUC-ROC: {eff_auc:.4f}")
print(f"ViT-Small/16     | Balanced Acc: {vit_test_acc:.4f} | AUC-ROC: {vit_auc:.4f}")

# ── Per-class reports ─────────────────────────────────────────────────
print("\n── EfficientNet-B2 Per-Class Report ────────────────────")
print(classification_report(eff_labels, eff_preds, target_names=CLASS_NAMES, digits=3))

print("\n── ViT-Small Per-Class Report ──────────────────────────")
print(classification_report(vit_labels, vit_preds, target_names=CLASS_NAMES, digits=3))

# ── Confusion matrices ────────────────────────────────────────────────
def plot_confusion_matrix(labels, preds, model_name):
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f"{model_name} — Normalized Confusion Matrix", fontsize=13)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    fname = f"confusion_matrix_{model_name.replace('/', '-')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {fname}")

plot_confusion_matrix(eff_labels, eff_preds, "EfficientNet-B2")
plot_confusion_matrix(vit_labels, vit_preds, "ViT-Small")

# ── Final summary ─────────────────────────────────────────────────────
eff_params = sum(p.numel() for p in efficientnet_model.parameters()) / 1e6
vit_params = sum(p.numel() for p in vit_model.parameters()) / 1e6

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
print("══════════════════════════════════════════════════════")

winner = "EfficientNet-B2" if eff_test_acc > vit_test_acc else "ViT-Small"
margin = abs(eff_test_acc - vit_test_acc)
print(f"\n→ {winner} wins by {margin:.4f} balanced accuracy points")

summary.to_csv("results_summary.csv", index=False)
print("Saved results_summary.csv")
print("\nStep 7 complete.")
