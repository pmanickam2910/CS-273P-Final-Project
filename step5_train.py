"""
STEP 5 — Train EfficientNet-B2 (20 epochs) and ViT-Small (20 epochs)
Run: exec(open("step5_train.py").read())
"""

import matplotlib.pyplot as plt

from step3_models import build_efficientnet, build_vit
from step4_training import train_model

NUM_EPOCHS = 20

# ── Train EfficientNet-B2 ─────────────────────────────────────────────
efficientnet_model = build_efficientnet()
efficientnet_model, eff_history = train_model(
    efficientnet_model, "EfficientNet-B2",
    train_loader, val_loader, DEVICE,
    num_epochs=NUM_EPOCHS,
)

# ── Train ViT-Small ───────────────────────────────────────────────────
vit_model = build_vit()
vit_model, vit_history = train_model(
    vit_model, "ViT-Small",
    train_loader, val_loader, DEVICE,
    num_epochs=NUM_EPOCHS,
)

# ── Plot training curves ──────────────────────────────────────────────
def plot_training_curves(eff_hist, vit_hist):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    eff_epochs = range(1, len(eff_hist["train_loss"]) + 1)
    vit_epochs = range(1, len(vit_hist["train_loss"]) + 1)

    axes[0].plot(eff_epochs, eff_hist["val_bal_acc"], "b-o", markersize=3, label="EfficientNet-B2")
    axes[0].plot(vit_epochs, vit_hist["val_bal_acc"], "r-s", markersize=3, label="ViT-Small")
    axes[0].set_title("Validation Balanced Accuracy", fontsize=13)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Balanced Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(eff_epochs, eff_hist["train_loss"], "b-o", markersize=3, label="EfficientNet-B2")
    axes[1].plot(vit_epochs, vit_hist["train_loss"], "r-s", markersize=3, label="ViT-Small")
    axes[1].set_title("Training Loss", fontsize=13)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("CNN vs ViT — Training Dynamics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved training_curves.png")

plot_training_curves(eff_history, vit_history)
print("\nStep 5 complete.")
