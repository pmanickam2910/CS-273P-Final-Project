"""
STEP 1 — Install dependencies & download dataset
Run: python step1_install.py
"""

import subprocess
import sys

packages = ["timm", "torchmetrics", "grad-cam", "kagglehub"]
print("Installing packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages + ["-q"])
print("All packages installed.\n")

import kagglehub
print("Downloading HAM10000 dataset via kagglehub...")
print("(You may be prompted for your Kaggle username + API token on first run)\n")
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
print(f"\nDataset downloaded to: {path}")

# Save the path so other steps can read it
with open("dataset_path.txt", "w") as f:
    f.write(path)
print("Path saved to dataset_path.txt")
