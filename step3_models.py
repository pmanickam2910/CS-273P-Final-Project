"""
STEP 3 — Model definitions
Run: exec(open("step3_models.py").read())
"""

import torch
import torch.nn as nn
import timm

NUM_CLASSES = 7

# ── EfficientNet-B2 ───────────────────────────────────────────────────
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

# ── ViT-Small/16 ──────────────────────────────────────────────────────
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

# ── EfficientNet-B2 Frozen (ablation) ────────────────────────────────
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

# ── Sanity check ──────────────────────────────────────────────────────
dummy = torch.randn(2, 3, 224, 224)
eff   = build_efficientnet()
vit   = build_vit()
print("EfficientNet-B2 output:", eff(dummy).shape)
print("ViT-Small output:      ", vit(dummy).shape)

eff_params = sum(p.numel() for p in eff.parameters()) / 1e6
vit_params = sum(p.numel() for p in vit.parameters()) / 1e6
print(f"EfficientNet-B2 params: {eff_params:.1f}M")
print(f"ViT-Small params:       {vit_params:.1f}M")
print("Step 3 complete.")
