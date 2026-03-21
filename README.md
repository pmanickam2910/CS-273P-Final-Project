# 🔬 Skin Lesion Classification: CNN vs Vision Transformer

**CS273P Final Project — UC Irvine**

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/mish1830/skin-lesion-classifier)
[![Demo Video](https://img.shields.io/badge/📹%20Demo-Google%20Drive-red)](https://drive.google.com/file/d/17QJzswjxweugf8wZNdH5KmZfB7i1r9rJ/view?usp=sharing)
[![Dataset](https://img.shields.io/badge/📦%20Dataset-HAM10000-green)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

---

## 🌟 Live Demo

**👉 Try the app here: [https://huggingface.co/spaces/mish1830/skin-lesion-classifier](https://huggingface.co/spaces/mish1830/skin-lesion-classifier)**

Upload any dermoscopic skin lesion image and get:
- Prediction from our CNN + ViT Ensemble with confidence score
- Top 3 class probabilities
- GradCAM visualizations showing where each model focused
- Clinical risk assessment (high-risk vs low-risk)

**📹 [Watch the full demo video](https://drive.google.com/file/d/17QJzswjxweugf8wZNdH5KmZfB7i1r9rJ/view?usp=sharing)**

---

## 📋 Project Overview

This project compares two fundamentally different deep learning architectures for skin lesion classification on the HAM10000 dermoscopic dataset:

- **EfficientNet-B2** — a Convolutional Neural Network (CNN) that learns local spatial features through convolutional filters
- **ViT-Small/16** — a Vision Transformer that uses self-attention to capture global relationships across image patches

Our contributions beyond a standard benchmark comparison:
- **Patient-level data splitting** — prevents data leakage, simulating deployment to genuinely new patients
- **Cohen's Kappa evaluation** — chance-corrected agreement benchmarked against inter-dermatologist agreement
- **Per-class threshold optimization** — F1-optimal decision thresholds derived from precision-recall curves for all 7 classes
- **GradCAM interpretability** — mechanistic explanation linking ViT's attention to the ABCDE clinical criteria
- **TTA + Ensemble** — Test-Time Augmentation and model ensembling (`ensemble_tta.py`)
- **Deployed web application** — publicly accessible clinical tool on Hugging Face Spaces

---

## 📊 Dataset: HAM10000

| Property       | Value                                          |
|----------------|------------------------------------------------|
| Full name      | Human Against Machine with 10000 training images |
| Total images   | 10,015 dermoscopic images                      |
| Classes        | 7 skin lesion types                            |
| Source         | Kaggle / ISIC Archive                          |

### Class Distribution

| Class    | Description              | Count | %     | Clinical Risk |
|----------|--------------------------|-------|-------|---------------|
| `nv`     | Melanocytic Nevi (Mole)  | 6,705 | 66.9% | Low           |
| `mel`    | Melanoma ⚠️              | 1,113 | 11.1% | **High**      |
| `bkl`    | Benign Keratosis         | 1,099 | 11.0% | Low           |
| `bcc`    | Basal Cell Carcinoma     | 514   | 5.1%  | **High**      |
| `akiec`  | Actinic Keratosis        | 327   | 3.3%  | **High**      |
| `df`     | Dermatofibroma           | 115   | 1.1%  | Low           |
| `vasc`   | Vascular Lesion          | 142   | 1.4%  | Low           |

> ⚠️ **Severe class imbalance**: `nv` makes up 66.9% of all images. A classifier always predicting `nv` achieves 66.9% accuracy but **0% melanoma recall**. This is why we use **Balanced Accuracy** as our primary metric.

### Data Splits (Patient-Level)

We split on `lesion_id`, not image ID, to prevent the same lesion appearing in both train and test sets. Overlap absence was verified programmatically.

| Split      | Images | %     |
|------------|--------|-------|
| Train      | 6,987  | 69.8% |
| Validation | 1,512  | 15.1% |
| Test       | 1,516  | 15.1% |

---

## 🏗️ Architecture

### EfficientNet-B2 (CNN)

```
Pretrained EfficientNet-B2 (ImageNet) — ~8.1M parameters
    └── Custom Classifier Head:
        ├── Dropout(0.5)
        ├── Linear(1408 → 256)
        ├── ReLU
        ├── Dropout(0.3)
        └── Linear(256 → 7)
```

### ViT-Small/16 (Transformer)

```
Pretrained ViT-Small/16 (ImageNet-21k) — ~21.8M parameters
196 non-overlapping 16×16 patches → 384-d tokens → 12 Transformer blocks (8 heads)
    └── Custom Classification Head:
        ├── Dropout(0.5)
        ├── Linear(384 → 256)
        ├── ReLU
        ├── Dropout(0.3)
        └── Linear(256 → 7)
```

Both models use an **identical two-layer head** — ensuring any performance difference reflects backbone architecture rather than head design.

### Key Design Choices

| Choice                        | Reason                                              |
|-------------------------------|-----------------------------------------------------|
| Two-layer classification head | More expressive than single linear layer            |
| Label Smoothing CE (ε=0.1)    | Calibrates probabilities; reduces overconfidence    |
| Weight decay = 1e-2           | Strong L2 regularization to reduce overfitting      |
| Early stopping (patience=5)   | Stops training before val accuracy degrades         |
| Balanced accuracy metric      | Prevents model from ignoring minority classes       |
| WeightedRandomSampler         | Minority classes sampled at equal frequency         |

---

## 🔧 Training Configuration

| Hyperparameter          | Value                                  |
|-------------------------|----------------------------------------|
| Image size              | 224 × 224                              |
| Batch size              | 32                                     |
| Learning rate           | 5e-5                                   |
| Optimizer               | AdamW                                  |
| Weight decay            | 1e-2                                   |
| Loss function           | Label Smoothing Cross-Entropy (ε=0.1)  |
| Max epochs              | 20                                     |
| Early stopping patience | 5                                      |
| Scheduler               | Cosine Annealing Warm Restarts (T₀=10) |
| Gradient clipping       | max norm = 1.0                         |
| Mixed precision         | FP16 (CUDA AMP)                        |
| Random seed             | 42                                     |

### Data Augmentation (Training Only)

| Augmentation   | Parameters                             | Reason                                   |
|----------------|----------------------------------------|------------------------------------------|
| Random H/V Flip| p = 0.5                                | Lesions have no canonical orientation    |
| Random Rotation| ±45°                                   | Dermoscopes can be held at any angle     |
| ColorJitter    | brightness/contrast/saturation/hue=0.4 | Accounts for inter-device variability    |
| RandomAffine   | translate, scale, shear                | Perspective variation                    |
| RandomErasing  | p = 0.2                                | Prevents relying on single salient patch |
| Normalize      | ImageNet mean/std                      | Consistent with pretrained weights       |

---

## 📈 Results

### Ablation Study

| Model Variant                     | Val Balanced Acc | Δ vs Frozen |
|-----------------------------------|------------------|-------------|
| EfficientNet-B2 (Frozen Backbone) | 0.480            | —           |
| EfficientNet-B2 (Full Fine-Tune)  | 0.733            | +0.252      |
| ViT-Small/16 (Full Fine-Tune)     | 0.761            | +0.280      |

**Fine-tuning is non-negotiable.** Frozen ImageNet features score barely above chance (14.3% random baseline).

### Baseline Model Comparison

| Model            | Val Bal-Acc | Test Bal-Acc | AUC-ROC  | Cohen's κ | Stopped     |
|------------------|-------------|--------------|----------|-----------|-------------|
| EfficientNet-B2  | 0.7345      | 0.6803       | 0.9413   | 0.5584    | Epoch 17/20 |
| **ViT-Small/16** | **0.7534**  | **0.7271**   | **0.9469** | **0.5974** | Epoch 15/20 |
| ViT Improvement  | +0.019      | +0.047       | +0.006   | +0.039    | 2 ep faster |

**ViT-Small wins on every metric.** The test-set gap (+0.043) exceeds the validation gap (+0.019) — the hallmark of superior generalization to genuinely unseen patients.

### Per-Class Recall (Test Set)

| Class  | EfficientNet-B2 | ViT-Small | Δ         | Risk     |
|--------|-----------------|-----------|-----------|----------|
| akiec  | 0.57            | 0.63      | +0.06     | **High** |
| bcc    | 0.70            | 0.77      | +0.07     | **High** |
| bkl    | 0.60            | 0.67      | +0.07     | Low      |
| df     | 0.70            | 0.61      | -0.09     | Low      |
| mel    | 0.58            | 0.75      | **+0.17** | **High** |
| nv     | 0.81            | 0.79      | -0.02     | Low      |
| vasc   | 0.81            | 0.88      | +0.07     | Low      |

> ⚠️ **Melanoma recall (+0.17)** — in a 1,000-patient screening, ViT correctly identifies **170 additional melanoma cases** for biopsy referral.

### Training Curves

![Training Curves](results/training_curves-2.png)

---

## 📐 Cohen's Kappa Analysis

Cohen's Kappa measures agreement beyond chance: **κ = (p_o − p_e) / (1 − p_e)**. It penalizes majority-class exploitation in a way balanced accuracy cannot.

| Model / Benchmark               | κ          | Interpretation      |
|---------------------------------|------------|---------------------|
| Random classifier               | ~0.000     | No agreement        |
| EfficientNet-B2                 | 0.5584     | Moderate            |
| **ViT-Small/16**                | **0.5974** | Moderate            |
| Inter-dermatologist (published) | 0.50–0.65  | Moderate-Good       |

**ViT-Small's κ = 0.5974 falls within the range of published inter-dermatologist agreement**, meaning our model approaches human-level consistency on this task. Both models fall in the Moderate range — appropriately reflecting that neither is ready for autonomous clinical decision-making.

---

## 🎯 Per-Class Threshold Analysis

A uniform 50% threshold is suboptimal under class imbalance. We computed precision-recall curves for each class and selected the **F1-maximizing threshold**.

| Class  | Opt. Threshold | Precision | Recall | F1    | Risk     |
|--------|----------------|-----------|--------|-------|----------|
| akiec  | 0.755          | 0.711     | 0.529  | 0.607 | **High** |
| bcc    | 0.303          | 0.726     | 0.792  | 0.758 | **High** |
| bkl    | 0.452          | 0.695     | 0.665  | 0.680 | Low      |
| df     | 0.192          | 0.750     | 0.783  | 0.766 | Low      |
| mel    | 0.732          | 0.590     | 0.497  | 0.539 | **High** |
| nv     | 0.157          | 0.921     | 0.925  | 0.923 | Low      |
| vasc   | 0.794          | 1.000     | 0.875  | 0.933 | Low      |

**Key findings:**
- `nv` threshold 0.157 — model is highly confident about moles (most training data)
- `bcc` threshold 0.303 — flag at low confidence, clinically appropriate for high-risk class
- `mel` F1-optimal threshold yields only 49.7% recall — a conservative clinical deployment should lower this threshold to prioritize recall over precision
- `vasc` achieves **perfect precision (1.000)** at threshold 0.794

---

## 🚀 TTA + Ensemble Results

`ensemble_tta.py` applies **Test-Time Augmentation** (5 orientations: original, H-flip, V-flip, 90°, 270°) and averages predictions from both models:

| Method                | Balanced Acc | AUC-ROC    | Best For                  |
|-----------------------|--------------|------------|---------------------------|
| EfficientNet-B2       | 0.6803       | 0.9413     | Baseline                  |
| ViT-Small/16          | 0.7271       | 0.9469     | Baseline                  |
| EfficientNet-B2 + TTA | 0.6871       | 0.9493     | —                         |
| ViT-Small/16 + TTA    | **0.7374**   | 0.9468     | Automated triage          |
| **Ensemble + TTA**    | 0.7233       | **0.9565** | Physician risk scoring    |

> **Key tradeoff:** ViT+TTA achieves best balanced accuracy. Ensemble+TTA achieves best AUC-ROC — most reliable for clinical probability ranking.

![Ensemble TTA Comparison](results/ensemble_tta_comparison.png)

---

## 🧠 GradCAM Interpretability

| Model        | Attention Pattern                       | Clinical Implication                              |
|--------------|-----------------------------------------|---------------------------------------------------|
| EfficientNet | Concentrated **single-blob** activation | Local receptive fields — one patch at a time      |
| ViT          | **Distributed multi-focal** activation  | Global self-attention — mirrors ABCDE reasoning   |

ViT's distributed attention naturally aligns with the **ABCDE diagnostic criteria**: Asymmetry requires comparing opposite halves, Border irregularity requires scanning the full perimeter, Color variation requires attending to multiple pigmentation zones. This is the mechanistic explanation for ViT's +0.11 melanoma recall gain.

![GradCAM Comparison](results/gradcam_comparison.png)

---

## 🌐 Web Application

**Live at: [https://huggingface.co/spaces/mish1830/skin-lesion-classifier](https://huggingface.co/spaces/mish1830/skin-lesion-classifier)**

| Feature                   | Description                                             |
|---------------------------|---------------------------------------------------------|
| Ensemble inference        | Runs both EfficientNet + ViT with 5-view TTA            |
| Top-3 predictions         | Shows confidence bars for top 3 classes                 |
| 🚨 High-risk alert        | Triggered for melanoma, BCC, actinic keratosis          |
| ⚠️ Low-confidence warning  | Shown when top probability < 50% — "See a doctor"      |
| ✅ Low-risk confirmation   | Shown for confidently predicted benign lesions          |
| GradCAM side-by-side      | EfficientNet and ViT heatmaps shown simultaneously      |

---

## 📁 Repository Structure

```
CS-273P-Final-Project/
├── step1_install.py          # Install packages, download dataset via kagglehub
├── step2_data.py             # Data loading, augmentation, patient-level splits
├── step3_models.py           # EfficientNet-B2 and ViT-Small model definitions
├── step4_training.py         # Training loop, early stopping, label smoothing loss
├── step5_train.py            # Train both models, save best checkpoints
├── step6_ablation.py         # Frozen vs fine-tuned ablation study
├── step7_evaluate.py         # Test evaluation, confusion matrices, Kappa, thresholds
├── step8_gradcam.py          # GradCAM visualizations for all 7 classes
├── ensemble_tta.py           # TTA + Ensemble evaluation
├── kaggle_complete.py        # Single-file version for Kaggle GPU notebook
├── app.py                    # Gradio web application (Hugging Face Spaces)
├── requirements.txt          # Python dependencies
├── EfficientNet-B2_best.pth  # Trained EfficientNet-B2 weights
├── ViT-Small_best.pth        # Trained ViT-Small weights
└── results/
    ├── training_curves-2.png
    ├── confusion_matrix_EfficientNet-B2.png
    ├── confusion_matrix_ViT-Small.png
    ├── gradcam_comparison.png
    ├── ensemble_tta_comparison.png
    ├── Melanoma.png
    └── Melanocytic Nevi.png
```

---

## 🚀 How to Run

### Option 1: Kaggle (Recommended — Free GPU)

1. Go to [Kaggle.com](https://kaggle.com) and create a notebook
2. Add the HAM10000 dataset: `kmader/skin-cancer-mnist-ham10000`
3. Enable GPU accelerator (T4 x2)
4. Upload `kaggle_complete.py` and run all cells
5. Training takes ~45 minutes on T4 GPU

### Option 2: Google Colab

1. Open the demo notebook in Colab
2. Enable GPU runtime (Runtime → Change runtime type → T4 GPU)
3. Run all cells — kagglehub handles dataset download automatically

### Option 3: Local (Mac/Linux)

```bash
git clone https://github.com/pmanickam2910/CS-273P-Final-Project.git
cd CS-273P-Final-Project
pip install -r requirements.txt

python step1_install.py   # Download dataset
python step2_data.py      # Prepare data splits
python step5_train.py     # Train both models (~45 min on GPU)
python step6_ablation.py  # Ablation study
python step7_evaluate.py  # Test evaluation + Kappa + thresholds
python step8_gradcam.py   # GradCAM visualizations
python ensemble_tta.py    # TTA + Ensemble evaluation
```

### Option 4: Run the Web App Locally

```bash
pip install gradio timm grad-cam torch torchvision
python app.py
# Open http://localhost:7860
```

---

## 📦 Dependencies

```
torch torchvision timm gradio grad-cam
Pillow numpy matplotlib scikit-learn pandas kagglehub
```

```bash
pip install -r requirements.txt
```

---

## 📝 Code Attribution

| Component                 | Source                                        |
|---------------------------|-----------------------------------------------|
| EfficientNet-B2 weights   | `timm` library (pretrained on ImageNet)       |
| ViT-Small/16 weights      | `timm` library (pretrained on ImageNet-21k)   |
| GradCAM implementation    | `pytorch-grad-cam` library                    |
| Training loop             | Written from scratch                          |
| Patient-level splitting   | Written from scratch                          |
| Label smoothing loss      | Written from scratch                          |
| Weighted random sampler   | PyTorch built-in, configured from scratch     |
| Cohen's Kappa             | scikit-learn `cohen_kappa_score`              |
| Threshold analysis        | Written from scratch using sklearn PR curves  |
| TTA + Ensemble pipeline   | Written from scratch                          |
| Web application           | Written from scratch using Gradio             |

---

## 🔑 Key Contributions

1. **Patient-level data splitting** — prevents data leakage; same patient's lesions never appear in both train and test
2. **Clinically motivated evaluation** — balanced accuracy, AUC-ROC, and Cohen's Kappa instead of standard accuracy
3. **Per-class threshold optimization** — F1-optimal thresholds from precision-recall curves for all 7 classes
4. **Interpretability analysis** — GradCAM comparison linking ViT's distributed attention to ABCDE criteria
5. **TTA + Ensemble** — ensemble achieves best AUC-ROC (0.9565), ViT+TTA achieves best balanced accuracy (0.7374)
6. **End-to-end deployment** — publicly accessible web app with uncertainty-aware clinical alerts

---

## 👥 Team

| Name            | Email            | Primary Responsibility                               |
|-----------------|------------------|------------------------------------------------------|
| Mishika Ahuja   | ahujamp@uci.edu  | Web app deployment, GradCAM UI, visualizations       |
| Pravin Kumar    | pmanicka@uci.edu | Model architecture, training infrastructure, ablation |
| Venkat Swaroop  | vveerla@uci.edu  | Data pipeline, Kappa, threshold analysis, TTA/ensemble |

**Course:** CS273P — Machine Learning in Healthcare
**Institution:** UC Irvine
**Year:** 2026

---

## 🏥 Clinical Threshold Justification

The web application uses two separate threshold concepts:

**1. System-level uncertainty threshold (50%):** When ensemble confidence falls below 50%, no single class is predicted with meaningful confidence for a 7-class problem. The model defers to a clinician.

**2. Per-class F1-optimal thresholds:** Derived from precision-recall curves (see table above). These represent the best operating point for each class independently and guide context-specific deployment tuning.

### Why 50% May Be Too Conservative for High-Risk Classes

| Class | F1-Optimal Threshold | Recall at F1-Opt | Clinical Recommendation |
|-------|---------------------|------------------|-------------------------|
| mel   | 0.732               | 49.7%            | Lower to ~0.3–0.4 to prioritize recall |
| bcc   | 0.303               | 79.2%            | Current threshold appropriate |
| akiec | 0.755               | 52.9%            | Lower to ~0.4 for screening |

In high-risk screening contexts, practitioners should lower mel and akiec thresholds below F1-optimal values to prioritize sensitivity over specificity.

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist for skin lesion evaluation.

---

## 📚 References

- Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*.
- Dosovitskiy, A., et al. (2021). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017*.
- Esteva, A., et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542(7639), 115–118.
- Codella, N., et al. (2019). Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by ISIC. *arXiv:1902.03368*.
- Szegedy, C., et al. (2016). Rethinking the Inception Architecture for Computer Vision (Label Smoothing). *CVPR 2016*.
- Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*.
