import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import matplotlib
import timm
matplotlib.use("Agg")

# ── Config ────────────────────────────────────────────────────────────
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_DESCRIPTIONS = {
    "akiec": "Actinic Keratosis / Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma ⚠️",
    "nv":    "Melanocytic Nevi (Mole)",
    "vasc":  "Vascular Lesion",
}
HIGH_RISK = {"mel", "bcc", "akiec"}
UNCERTAINTY_THRESHOLD = 0.5
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model definitions ─────────────────────────────────────────────────
def build_efficientnet(num_classes=7):
    model = timm.create_model("efficientnet_b2", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model

def build_vit(num_classes=7):
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
    embed_dim = model.embed_dim
    model.head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(embed_dim, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model

# ── Load models ───────────────────────────────────────────────────────
print("Loading models...")
efficientnet_model = build_efficientnet()
efficientnet_model.load_state_dict(
    torch.load("EfficientNet-B2_best.pth", map_location=DEVICE)
)
efficientnet_model = efficientnet_model.to(DEVICE).eval()

vit_model = build_vit()
vit_model.load_state_dict(
    torch.load("ViT-Small_best.pth", map_location=DEVICE)
)
vit_model = vit_model.to(DEVICE).eval()
print("Models loaded!")

# ── Transforms ────────────────────────────────────────────────────────
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tta_transforms_list = [
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(p=1.0),
                        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomVerticalFlip(p=1.0),
                        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomRotation((90,90)),
                        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomRotation((270,270)),
                        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
]

# ── GradCAM setup ─────────────────────────────────────────────────────
def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)

eff_cam = GradCAM(model=efficientnet_model, target_layers=[efficientnet_model.blocks[-1]])
vit_cam = GradCAM(model=vit_model, target_layers=[vit_model.blocks[-1].norm1],
                  reshape_transform=vit_reshape_transform)

# ── Prediction function ───────────────────────────────────────────────
def predict(image):
    if image is None:
        return "Please upload an image.", None

    img_pil = Image.fromarray(image).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_np  = np.array(img_pil) / 255.0

    def get_tta_probs(model):
        probs_list = []
        for tfm in tta_transforms_list:
            img_t = tfm(img_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(img_t)
            probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.mean(probs_list, axis=0)[0]

    eff_probs      = get_tta_probs(efficientnet_model)
    vit_probs      = get_tta_probs(vit_model)
    ensemble_probs = (eff_probs + vit_probs) / 2
    pred_idx       = ensemble_probs.argmax()
    pred_class     = CLASS_NAMES[pred_idx]
    confidence     = ensemble_probs[pred_idx]

    # Result text
    result  = f"## 🔬 Prediction: {CLASS_DESCRIPTIONS[pred_class]}\n\n"
    result += f"**Confidence:** {confidence:.1%}\n\n"
    result += "### Top 3 Predictions:\n"
    top3_idx = ensemble_probs.argsort()[::-1][:3]
    for i, idx in enumerate(top3_idx):
        bar = "█" * int(ensemble_probs[idx] * 20)
        result += f"{i+1}. **{CLASS_NAMES[idx]}** — {ensemble_probs[idx]:.1%} {bar}\n"
    result += "\n---\n"

    if confidence < UNCERTAINTY_THRESHOLD:
        result += "### ⚠️ Low Confidence — Please See a Doctor\n"
        result += "The model is uncertain about this image. Please consult a dermatologist.\n"
    elif pred_class in HIGH_RISK:
        result += "### 🚨 High-Risk Lesion Detected\n"
        result += "**Please consult a dermatologist as soon as possible.**\n"
    else:
        result += "### ✅ Low-Risk Lesion\n"
        result += "This appears to be a low-risk lesion. Consult a doctor if concerned.\n"

    result += "\n\n*⚠️ For educational purposes only — not a substitute for medical advice.*"

    # GradCAM
    img_t  = val_transforms(img_pil).unsqueeze(0).to(DEVICE)
    target = [ClassifierOutputTarget(int(pred_idx))]

    eff_grayscale = eff_cam(input_tensor=img_t, targets=target)
    vit_grayscale = vit_cam(input_tensor=img_t, targets=target)
    eff_overlay   = show_cam_on_image(img_np.astype(np.float32), eff_grayscale[0], use_rgb=True)
    vit_overlay   = show_cam_on_image(img_np.astype(np.float32), vit_grayscale[0], use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_pil);     axes[0].set_title("Original Image");       axes[0].axis("off")
    axes[1].imshow(eff_overlay); axes[1].set_title("EfficientNet GradCAM"); axes[1].axis("off")
    axes[2].imshow(vit_overlay); axes[2].set_title("ViT GradCAM");          axes[2].axis("off")
    plt.suptitle(f"Predicted: {pred_class} ({confidence:.1%} confidence)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    return result, fig

# ── Gradio Interface ──────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Skin Lesion Image"),
    outputs=[
        gr.Markdown(label="Diagnosis"),
        gr.Plot(label="GradCAM Visualization"),
    ],
    title="🔬 Skin Lesion Classifier — CNN vs ViT Ensemble",
    description="""
Upload a dermoscopic skin lesion image. The app uses an **Ensemble of EfficientNet-B2 (CNN) + ViT-Small (Transformer)**
with **Test-Time Augmentation** to classify into 7 categories and visualize where each model focused.

**Classes:** akiec · bcc · bkl · df · mel · nv · vasc

⚠️ *For educational purposes only — not a substitute for professional medical advice.*
    """,
    theme=gr.themes.Soft(),
)

demo.launch()
