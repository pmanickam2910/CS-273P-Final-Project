"""
STEP 8 — GradCAM visualizations
Run: exec(open("step8_gradcam.py").read())
Requires step5 to have been run (models in memory).
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── GradCAM setup ─────────────────────────────────────────────────────
eff_target_layer = [efficientnet_model.blocks[-1]]
vit_target_layer = [vit_model.blocks[-1].norm1]

def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)

eff_cam = GradCAM(model=efficientnet_model, target_layers=eff_target_layer)
vit_cam = GradCAM(model=vit_model, target_layers=vit_target_layer,
                  reshape_transform=vit_reshape_transform)

# ── Sample one image per class ────────────────────────────────────────
def get_sample_images(df, class_names, n_per_class=1):
    samples = []
    for cls in class_names:
        cls_df = df[df["dx"] == cls]
        if len(cls_df) > 0:
            row = cls_df.sample(n_per_class, random_state=SEED).iloc[0]
            samples.append((row["path"], cls, int(row["label"])))
    return samples

samples = get_sample_images(test_df, CLASS_NAMES)

# ── Visualize ─────────────────────────────────────────────────────────
n = len(samples)
fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

for i, (path, cls, label) in enumerate(samples):
    img_pil = Image.open(path).convert("RGB").resize((224, 224))
    img_np  = np.array(img_pil) / 255.0
    img_t   = val_transforms(img_pil).unsqueeze(0).to(DEVICE)
    target  = [ClassifierOutputTarget(label)]

    eff_grayscale = eff_cam(input_tensor=img_t, targets=target)
    vit_grayscale = vit_cam(input_tensor=img_t, targets=target)

    eff_overlay = show_cam_on_image(img_np.astype(np.float32), eff_grayscale[0], use_rgb=True)
    vit_overlay = show_cam_on_image(img_np.astype(np.float32), vit_grayscale[0], use_rgb=True)

    axes[i, 0].imshow(img_pil);    axes[i, 0].set_title(f"Original\nClass: {cls}", fontsize=10); axes[i, 0].axis("off")
    axes[i, 1].imshow(eff_overlay); axes[i, 1].set_title("EfficientNet-B2\nGradCAM", fontsize=10); axes[i, 1].axis("off")
    axes[i, 2].imshow(vit_overlay); axes[i, 2].set_title("ViT-Small\nGradCAM", fontsize=10);      axes[i, 2].axis("off")

plt.suptitle("GradCAM: Where CNN vs ViT Looks", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("gradcam_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved gradcam_comparison.png")
print("\nStep 8 complete. All done!")
