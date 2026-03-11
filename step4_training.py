"""
STEP 4 — Training infrastructure
Imported by other steps — not run directly.
Fixes: weight_decay=1e-2, early stopping patience=5, works on CPU and GPU.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

_USE_AMP = torch.cuda.is_available()


# ── Label Smoothing Loss ───────────────────────────────────────────────
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


# ── Single epoch training ─────────────────────────────────────────────
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


# ── Evaluation ────────────────────────────────────────────────────────
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


# ── Full training loop ────────────────────────────────────────────────
def train_model(model, model_name, train_loader, val_loader,
                device, num_epochs=20, lr=5e-5):
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    model     = model.to(device)
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
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_bal_acc, _, _, _ = evaluate(
            model, val_loader, criterion, device)
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
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Bal-Acc: {val_bal_acc:.4f} | "
                  f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} — no improvement for {patience} epochs")
            break

    print(f"\nBest Val Balanced Accuracy: {best_val_acc:.4f}")
    model.load_state_dict(best_weights)
    torch.save(best_weights, f"{model_name}_best.pth")
    print(f"Saved {model_name}_best.pth")
    return model, history
