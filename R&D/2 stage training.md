Nice — I’ll give you a complete, ready-to-run **two-phase pipeline** (PyTorch) that:

* creates a **diverse 100-class subset** (configurable) with **50% images per class**,
* runs **Phase 1** pretraining on that subset (AMP + OneCycleLR + Mixup),
* runs **Phase 2** fine-tuning on the full ImageNet (unfreeze strategy + CosineAnnealingLR),
* handles **BatchNorm recalibration**, **saving/loading checkpoints**, and **large-batch simulation (grad accumulation)** for your 16 GB RTX 5060 Ti,
* includes sensible hyperparameters/comments so you can tweak.

Drop this into a single file (e.g. `two_phase_imagenet.py`) and run. I kept things explicit and commented.

---

> ⚠️ Assumes: PyTorch + torchvision installed, `timm` optional (for Mixup). Adjust paths to your ImageNet layout: `data_root/train`, `data_root/val`.

```python
"""
two_phase_imagenet.py

Two-phase training pipeline:
- Phase 1: Pretrain on subset (N_classes_subset, frac_per_class)
- Phase 2: Fine-tune on full ImageNet (all classes, all images)
"""

import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from tqdm import tqdm

# Optional: timm Mixup if available (recommended)
try:
    from timm.data import Mixup
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

# ---------------------
# Config (tweak here)
# ---------------------
DATA_ROOT = "/path/to/imagenet"           # must contain "train" and "val"
OUT_DIR = "./checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_SUBSET_CLASSES = 100                 # stage1: number of classes to pick
SUBSET_FRAC = 0.5                        # stage1: fraction of images per chosen class
RANDOM_SEED = 42

IMG_SIZE = 224
PHASE1_EPOCHS = 25
PHASE1_BATCH = 256                        # batch per device
PHASE1_MIXUP = True

PHASE2_EPOCHS = 25
PHASE2_BATCH = 1024                       # effective global batch (use accumulation if needed)
PHASE2_FREEZE_UPTO = "layer2"             # freeze conv1,bn1,layer1,layer2
PHASE2_BASE_LR = 1e-4
PHASE2_HEAD_LR = 5e-4

NUM_WORKERS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For grad accumulation to simulate large batch (if GPU memory insufficient)
# effective_batch = PHASE2_BATCH, actual_batch = per-device batch passed to DataLoader
PHASE2_EFFECTIVE_BATCH = PHASE2_BATCH
PHASE2_ACTUAL_BATCH = 256                # set to a value that fits your GPU memory
ACCUM_STEPS = max(1, PHASE2_EFFECTIVE_BATCH // PHASE2_ACTUAL_BATCH)

# ---------------------
# Utility functions
# ---------------------
def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collate_fn_default(batch):
    return tuple(zip(*batch))

# Create subset indices: select K classes, and take fraction f of images per chosen class
def build_subset_indices(train_dir, num_classes=100, frac_per_class=0.5, seed=RANDOM_SEED):
    """
    train_dir: path to ImageNet train folder (class subfolders)
    returns: list of dataset indices to keep (relative to ImageFolder ordering)
    """
    # Build mapping from class -> list of indices
    dataset = datasets.ImageFolder(train_dir)
    class_to_indices = defaultdict(list)
    for idx, (_, class_idx) in enumerate(dataset.samples):
        class_to_indices[class_idx].append(idx)

    # choose classes uniformly at random (or choose first N for reproducibility)
    rng = random.Random(seed)
    all_classes = sorted(class_to_indices.keys())
    chosen_classes = rng.sample(all_classes, num_classes)

    selected_indices = []
    for cls in chosen_classes:
        indices = class_to_indices[cls]
        rng.shuffle(indices)
        k = max(1, int(len(indices) * frac_per_class))
        selected_indices.extend(indices[:k])

    selected_indices.sort()
    return selected_indices

# BN recalibration: run a few forward passes in train mode to update running stats
@torch.no_grad()
def recalibrate_bn(loader, model, device, num_iters=200):
    model.train()
    it = iter(loader)
    for i in range(num_iters):
        try:
            imgs, _ = next(it)
        except StopIteration:
            it = iter(loader)
            imgs, _ = next(it)
        imgs = imgs.to(device, non_blocking=True)
        model(imgs)  # forward only

# Save/Load helpers
def save_checkpoint(path, model, optimizer, epoch, scaler=None, extra=None):
    sd = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    }
    if scaler is not None:
        sd["scaler_state"] = scaler.state_dict()
    if extra:
        sd.update(extra)
    torch.save(sd, path)

def load_checkpoint(path, model, optimizer=None, scaler=None, map_location="cpu"):
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck["model_state"], strict=False)
    if optimizer and "optimizer_state" in ck:
        optimizer.load_state_dict(ck["optimizer_state"])
    if scaler and "scaler_state" in ck:
        scaler.load_state_dict(ck["scaler_state"])
    return ck

# ---------------------
# Transforms & Dataloaders
# ---------------------
def make_transforms(img_size=IMG_SIZE, train=True):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if train:
        t = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        t = transforms.Compose([
            transforms.Resize(int(img_size*1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return t

def build_dataloaders(data_root, subset_indices=None, batch_size=128, train=True, drop_last=True, num_workers=NUM_WORKERS):
    split = "train" if train else "val"
    ds = datasets.ImageFolder(os.path.join(data_root, split), transform=make_transforms(IMG_SIZE, train=train))
    if subset_indices is not None and train:
        ds = Subset(ds, subset_indices)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train and subset_indices is None,  # if using subset, you still want shuffle True
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last
    )
    return loader

# ---------------------
# Mixup helper (timm if available, else simple)
# ---------------------
class SimpleMixup:
    def __init__(self, alpha=0.2, prob=1.0, num_classes=1000):
        self.alpha = alpha
        self.prob = prob
        self.num_classes = num_classes

    def __call__(self, x, y):
        if self.alpha <= 0 or random.random() > self.prob:
            y_onehot = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float().to(x.device)
            return x, y_onehot
        lam = np.random.beta(self.alpha, self.alpha)
        bs = x.size(0)
        perm = torch.randperm(bs, device=x.device)
        x2 = x[perm]
        y2 = y[perm]
        mixed_x = lam * x + (1 - lam) * x2
        y_onehot = lam * torch.nn.functional.one_hot(y, num_classes=self.num_classes).float().to(x.device) + \
                   (1 - lam) * torch.nn.functional.one_hot(y2, num_classes=self.num_classes).float().to(x.device)
        return mixed_x, y_onehot

def get_mixup_fn(num_classes=1000, alpha=0.2, prob=1.0):
    if _HAS_TIMM:
        return Mixup(mixup_alpha=alpha, cutmix_alpha=1.0, prob=prob, switch_prob=0.5, label_smoothing=0.1, num_classes=num_classes)
    else:
        return SimpleMixup(alpha=alpha, prob=prob, num_classes=num_classes)

# Soft target loss (works with one-hot labels)
def soft_cross_entropy(outputs, targets):
    # outputs: logits; targets: one-hot distribution
    log_probs = torch.log_softmax(outputs, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()

# ---------------------
# Training & Validation loops
# ---------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler, mixup_fn=None, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Train", leave=False)
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, (imgs, labels) in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup (if timm Mixup returns mixed labels as soft or single param)
        if mixup_fn is not None:
            mixed = mixup_fn(imgs, labels)
            # timm Mixup sometimes returns (images, targets) where targets are soft labels
            if isinstance(mixed, tuple) and len(mixed) == 2:
                imgs_m, targets = mixed
                imgs = imgs_m
                # choose loss accordingly
                use_soft = True
            else:
                imgs, targets = mixed, None
                use_soft = False
        else:
            targets = None
            use_soft = False

        device_type = "cuda" if device.type == "cuda" else "cpu"
        with autocast(device_type=device_type, enabled=(device_type=="cuda")):
            outputs = model(imgs)
            if mixup_fn is not None:
                # If timm provided soft labels (tensor NxC), use soft loss
                if targets is not None and targets.dtype.is_floating_point:
                    loss = soft_cross_entropy(outputs, targets)
                else:
                    # fallback (shouldn't reach here)
                    loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)

            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # scheduler step per batch if OneCycle (call after optimizer.step)
            if scheduler is not None:
                scheduler.step()

        running_loss += float(loss.detach().cpu().item()) * imgs.size(0) * accumulation_steps  # de-scaled
        pred_labels = outputs.argmax(dim=1)
        correct += int((pred_labels == labels).sum().item())
        total += labels.size(0)

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(dataloader):
            pbar.set_postfix({"loss": f"{running_loss/total:.4f}", "acc": f"{100.*correct/total:.2f}%"})

    return running_loss / total, correct / total, scaler

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Val", leave=False)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    for batch_idx, (imgs, labels) in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device_type=device_type, enabled=(device_type=="cuda")):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        total_loss += float(loss.detach().cpu().item()) * imgs.size(0)
        pred_labels = outputs.argmax(dim=1)
        correct += int((pred_labels == labels).sum().item())
        total += labels.size(0)
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(dataloader):
            pbar.set_postfix({"val_loss": f"{total_loss/total:.4f}", "val_acc": f"{100.*correct/total:.2f}%"})
    return total_loss/total, correct/total

# ---------------------
# Model creation and freeze utilities
# ---------------------
def create_resnet50(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def freeze_upto(model, upto_name="layer2"):
    freeze_names = ["conv1", "bn1", "layer1", "layer2"]
    for name, child in model.named_children():
        if name in freeze_names:
            for p in child.parameters():
                p.requires_grad = False
        else:
            for p in child.parameters():
                p.requires_grad = True

# ---------------------
# Main orchestration
# ---------------------
def main():
    seed_everything()

    train_dir = os.path.join(DATA_ROOT, "train")
    val_dir = os.path.join(DATA_ROOT, "val")
    assert os.path.isdir(train_dir) and os.path.isdir(val_dir), "Invalid DATA_ROOT path."

    # Build subset indices for phase1
    print("Building subset indices for Phase 1 ...")
    subset_indices = build_subset_indices(train_dir, num_classes=NUM_SUBSET_CLASSES, frac_per_class=SUBSET_FRAC, seed=RANDOM_SEED)
    print(f"Phase1 subset size: {len(subset_indices)} samples across {NUM_SUBSET_CLASSES} classes.")

    # Dataloaders
    train_loader_phase1 = build_dataloaders(DATA_ROOT, subset_indices=subset_indices, batch_size=PHASE1_BATCH, train=True, drop_last=True)
    val_loader = build_dataloaders(DATA_ROOT, subset_indices=None, batch_size=256, train=False, drop_last=False)

    # Model
    num_classes_full = len(datasets.ImageFolder(train_dir).classes)
    model = create_resnet50(num_classes=num_classes_full)
    model = model.to(DEVICE).to(memory_format=torch.channels_last)

    # Criterion, optimizer, scheduler for Phase 1
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.1 * (PHASE1_BATCH / 256), momentum=0.9, weight_decay=1e-4, nesterov=True)

    # OneCycleLR: total_steps = epochs * steps_per_epoch
    steps_per_epoch = len(train_loader_phase1)
    total_steps = PHASE1_EPOCHS * steps_per_epoch
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.1 * (PHASE1_BATCH / 256),
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    scaler = GradScaler()
    mixup_fn = get_mixup_fn(num_classes=num_classes_full, alpha=0.2, prob=1.0) if PHASE1_MIXUP else None

    # Phase 1 training
    best_val_acc = 0.0
    for epoch in range(1, PHASE1_EPOCHS + 1):
        print(f"\n=== Phase 1 Epoch {epoch}/{PHASE1_EPOCHS} ===")
        train_loss, train_acc, scaler = train_one_epoch(model, train_loader_phase1, optimizer, scheduler, criterion, DEVICE, scaler, mixup_fn=mixup_fn)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        print(f"Phase1 Epoch {epoch} TrainLoss {train_loss:.4f} TrainAcc {100*train_acc:.2f}% | ValLoss {val_loss:.4f} ValAcc {100*val_acc:.2f}%")
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0 or val_acc > best_val_acc:
            ck_path = os.path.join(OUT_DIR, f"phase1_epoch{epoch}.pth")
            save_checkpoint(ck_path, model, optimizer, epoch, scaler=scaler)
            print(f"Saved {ck_path}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(os.path.join(OUT_DIR, "best_phase1.pth"), model, optimizer, epoch, scaler=scaler)

    # Save final phase1 weights
    save_checkpoint(os.path.join(OUT_DIR, "final_phase1.pth"), model, optimizer, PHASE1_EPOCHS, scaler=scaler)
    print("Phase 1 complete. Weights saved.")

    # -------------------------
    # Phase 2: Fine-tune on full dataset
    # -------------------------
    print("\n=== Phase 2: Fine-tune on full ImageNet ===")
    # Reload the best phase1 weights (optional)
    best_ck = os.path.join(OUT_DIR, "best_phase1.pth")
    if os.path.exists(best_ck):
        load_checkpoint(best_ck, model, optimizer=None, scaler=None, map_location=DEVICE)
        print(f"Loaded {best_ck} into model for finetuning.")

    # Freeze early layers
    freeze_upto(model, upto_name=PHASE2_FREEZE_UPTO)

    # Data loader for phase2 (full train)
    # If you need large effective batch, we use accumulation with a smaller per-step batch that fits GPU
    train_loader_phase2 = build_dataloaders(DATA_ROOT, subset_indices=None, batch_size=PHASE2_ACTUAL_BATCH, train=True, drop_last=True)
    val_loader = build_dataloaders(DATA_ROOT, subset_indices=None, batch_size=256, train=False, drop_last=False)

    # Optimizer with different LR for head
    param_groups = [
        {"params": model.layer3.parameters(), "lr": PHASE2_BASE_LR},
        {"params": model.layer4.parameters(), "lr": PHASE2_BASE_LR},
        {"params": model.fc.parameters(), "lr": PHASE2_HEAD_LR},
    ]
    optimizer2 = optim.AdamW(param_groups, weight_decay=1e-4)

    # Scheduler (Cosine)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=PHASE2_EPOCHS, eta_min=1e-6)

    scaler2 = GradScaler()
    # Mixup optional for FT, often disabled when fine-tuning on same dataset
    mixup_fn2 = None

    # Recalibrate BN running stats (important if many BN layers were frozen / dataset changed)
    print("Recalibrating BN stats with a few forward passes (no grads)...")
    recal_loader = build_dataloaders(DATA_ROOT, subset_indices=None, batch_size=256, train=True, drop_last=True)
    recalibrate_bn(recal_loader, model, DEVICE, num_iters=200)
    print("BN recalibration done.")

    best_val_acc_phase2 = 0.0
    for epoch in range(1, PHASE2_EPOCHS + 1):
        print(f"\n=== Phase 2 Epoch {epoch}/{PHASE2_EPOCHS} (accum_steps={ACCUM_STEPS}) ===")
        train_loss, train_acc, scaler2 = train_one_epoch(model, train_loader_phase2, optimizer2, scheduler2, criterion, DEVICE, scaler2, mixup_fn=mixup_fn2, accumulation_steps=ACCUM_STEPS)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        print(f"Phase2 Epoch {epoch} TrainLoss {train_loss:.4f} TrainAcc {100*train_acc:.2f}% | ValLoss {val_loss:.4f} ValAcc {100*val_acc:.2f}%")
        # Save best
        if val_acc > best_val_acc_phase2:
            best_val_acc_phase2 = val_acc
            save_checkpoint(os.path.join(OUT_DIR, "best_phase2.pth"), model, optimizer2, epoch, scaler2)

        # step scheduler per epoch (CosineAnnealingLR step once per epoch)
        scheduler2.step()

    save_checkpoint(os.path.join(OUT_DIR, "final_phase2.pth"), model, optimizer2, PHASE2_EPOCHS, scaler2)
    print("Phase 2 complete. Final weights saved.")

if __name__ == "__main__":
    main()
```

---

## Short explanation of important parts & best-practices used

* **Subset selection (`build_subset_indices`)**: picks `NUM_SUBSET_CLASSES` classes and `SUBSET_FRAC` fraction per class to form the reduced training set.
* **Mixup**: `timm` Mixup used if available; fallback simple mixup implemented.
* **OneCycleLR** in Phase 1**: good for short, aggressive pretraining (fast warmup + cosine anneal).
* **Gradient Accumulation** (Phase 2): simulates `batch_size=1024` by accumulating gradients across `ACCUM_STEPS`. Use when GPU memory can't hold big batches.
* **Freeze early layers**: `freeze_upto` sets `requires_grad=False` for conv1, bn1, layer1, layer2. Adapt if you want more/less freezing.
* **BatchNorm recalibration**: after loading weights / before fine-tuning, run several forward passes (no grads) to update running mean/var.
* **Channels-last memory format**: `.to(memory_format=torch.channels_last)` to speed convs on modern GPUs.
* **Saving**: checkpoints saved regularly; `best_phase1.pth` and `best_phase2.pth`.
* **AMP (autocast + GradScaler)** used everywhere.

---

## Hyperparameter notes & tuning suggestions

* If Phase1 underfits: increase epochs to 35–50 or increase max_lr slightly (use LR finder).
* If Phase2 shows degradation: unfreeze `layer2`/`layer1` gradually or reduce base LR by factor 2–10.
* For reproducibility: set `seed_everything`.
* If you want exact large-batch optimization for Phase2 (linear scaling rule), scale LR with effective batch size: `lr *= effective_batch / 256` — but if switching to AdamW, prefer fixed small LR (1e-4) and warmup.

---

If you want, I can:

* produce a **smaller example** to run quickly (e.g., CIFAR-style) so you can validate pipeline logic before ImageNet runs, or
* adapt the script to use **DDP** for multi-GPU,
* add **tensorboard/wandb logging** integration.

Which would you like next?
