"""
STEP 6 — Ablation study
Run: exec(open("step6_ablation.py").read())
Requires step5 to have been run (models in memory).
"""

import pandas as pd
from step3_models import build_efficientnet_frozen
from step4_training import train_model, evaluate, LabelSmoothingCrossEntropy

# ── Train frozen backbone ─────────────────────────────────────────────
frozen_model = build_efficientnet_frozen()
frozen_model, _ = train_model(
    frozen_model, "EfficientNet-B2-Frozen",
    train_loader, val_loader, DEVICE,
    num_epochs=10,
)

# ── Evaluate all on val set ───────────────────────────────────────────
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

_, frozen_bal_acc, _, _, _ = evaluate(frozen_model,       val_loader, criterion, DEVICE)
_, eff_bal_acc,    _, _, _ = evaluate(efficientnet_model, val_loader, criterion, DEVICE)
_, vit_bal_acc,    _, _, _ = evaluate(vit_model,          val_loader, criterion, DEVICE)

ablation_df = pd.DataFrame({
    "Model Variant": [
        "EfficientNet-B2 (Frozen Backbone)",
        "EfficientNet-B2 (Full Fine-Tune)",
        "ViT-Small/16 (Full Fine-Tune)",
    ],
    "Val Balanced Acc": [f"{frozen_bal_acc:.3f}", f"{eff_bal_acc:.3f}", f"{vit_bal_acc:.3f}"],
    "Delta vs Frozen":  ["—", f"+{eff_bal_acc-frozen_bal_acc:.3f}", f"+{vit_bal_acc-frozen_bal_acc:.3f}"],
})

print("\n── Ablation Study ──────────────────────────────────────")
print(ablation_df.to_string(index=False))
print("\nStep 6 complete.")
