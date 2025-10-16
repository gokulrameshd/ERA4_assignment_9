"""
train.py
--------------
End-to-end ResNet-34 training pipeline with:
✅ LR Finder (AMP + Auto reset + CSV export)
✅ OneCycleLR Scheduler
✅ Dynamic tqdm bars (train + val)
✅ ETA + GPU memory monitor
✅ Dynamic live plots (Acc/Loss/LR/Momentum)
✅ Best & Last weights saving
✅ Smart LR auto-scaling from LR Finder
Optimized: TF32, torch.compile, modern torch.amp usage
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_loader import get_dataloaders
from model import create_model
from cyclic_scheduler import create_onecycle_scheduler
from lr_finder import LRFinder

# Modern AMP imports
from torch.amp import autocast, GradScaler

# ==============================================================
# ⚙️ CONFIG
# ==============================================================
DATA_DIR = "data"
BATCH_SIZE = 512
IMG_SIZE = 224
NUM_EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_BEST = "best_weights.pth"
SAVE_LAST = "last_weights.pth"
LOG_FILE = "training_log.txt"
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

# Performance flags
torch.backends.cudnn.benchmark = True
# enable TF32 on Ampere/Ada GPUs for faster matmuls (negligible accuracy change)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# prefer Tensor Cores where possible
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
# ==============================================================


def format_mem(bytes_val):
    """Convert bytes to readable MB/GB format."""
    if bytes_val < 1024**2:
        return f"{bytes_val/1024:.1f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val/1024**2:.1f} MB"
    return f"{bytes_val/1024**3:.2f} GB"


# ==============================================================
# ⚡ Optimized Training & Validation Loops (AMP + non-blocking)
# ==============================================================
def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler=None, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    start_time = time.time()

    # If scaler is not passed, create one tied to device
    if scaler is None:
        scaler = GradScaler(device="cuda" if device == "cuda" else "cpu")

    device_type = "cuda" if device == "cuda" else "cpu"
    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="\033[92m🟢 Training\033[0m",
        leave=False,
        ncols=120
    )

    for batch_idx, (inputs, labels) in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward (autocast for GPU only)
        with autocast(device_type=device_type, dtype=torch.float16 if device_type == "cuda" else torch.float32):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Backward + optimizer step (with GradScaler)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # scheduler.step per batch (OneCycleLR expects per-step)
        if scheduler:
            scheduler.step()

        total_loss += float(loss.detach().cpu().item()) * inputs.size(0)
        _, preds = outputs.max(1)
        correct += int(preds.eq(labels).sum().item())
        total += labels.size(0)

        # Update tqdm less frequently to reduce overhead
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            avg_loss = total_loss / total
            acc = 100.0 * correct / total
            mem_alloc = (
                format_mem(torch.cuda.memory_allocated(device))
                if torch.cuda.is_available()
                else "N/A"
            )

            elapsed = time.time() - start_time
            iters_done = batch_idx + 1
            iters_left = len(dataloader) - iters_done
            eta = iters_left * (elapsed / max(1, iters_done))
            pbar.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "Acc": f"{acc:.2f}%",
                "Mem": mem_alloc,
                "ETA": f"{eta/60:.1f}m"
            })

    return total_loss / total, correct / total, scaler


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    start_time = time.time()

    device_type = "cuda" if device == "cuda" else "cpu"
    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="\033[94m🔵 Validating\033[0m",
        leave=False,
        ncols=120
    )

    with torch.no_grad():
        for batch_idx, (inputs, labels) in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # use autocast for faster inference on GPU
            with autocast(device_type=device_type, dtype=torch.float16 if device_type == "cuda" else torch.float32):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            total_loss += float(loss.detach().cpu().item()) * inputs.size(0)
            _, preds = outputs.max(1)
            correct += int(preds.eq(labels).sum().item())
            total += labels.size(0)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                avg_loss = total_loss / total
                acc = 100.0 * correct / total
                mem_alloc = (
                    format_mem(torch.cuda.memory_allocated(device))
                    if torch.cuda.is_available()
                    else "N/A"
                )

                elapsed = time.time() - start_time
                iters_done = batch_idx + 1
                iters_left = len(dataloader) - iters_done
                eta = iters_left * (elapsed / max(1, iters_done))
                pbar.set_postfix({
                    "Loss": f"{avg_loss:.4f}",
                    "Acc": f"{acc:.2f}%",
                    "Mem": mem_alloc,
                    "ETA": f"{eta/60:.1f}m"
                })

    return total_loss / total, correct / total


# ==============================================================
# 🚀 MAIN TRAIN FUNCTION
# ==============================================================
def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print(f"\n🚀 Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} on {DEVICE}")

    # -----------------------------------------------------------
    # 📦 Data
    train_loader, val_loader, num_classes = get_dataloaders(DATA_DIR, BATCH_SIZE, IMG_SIZE)

    # -----------------------------------------------------------
    # 🧠 Model setup
    model = create_model(num_classes=num_classes, pretrained=False).to(DEVICE)

    # Try to compile the model (PyTorch 2.x). Safe to ignore failures.
    try:
        model = torch.compile(model)
        print("⚡ model compiled with torch.compile()")
    except Exception:
        pass

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    # -----------------------------------------------------------
    # 🔍 LR Finder
    print("\n🔍 Running Learning Rate Finder...")
    # Use a small temp optimizer copy and model copy to run LR finder without altering original optimizer
    # We use the model and optimizer provided (lr_finder will cache & reset)
    lr_finder = LRFinder(model, optimizer, criterion, device=DEVICE, cache_dir=PLOTS_DIR)
    lr_finder.range_test(train_loader, start_lr=1e-6, end_lr=10, num_iter=100)

    # Plot and get LR suggestions (plot returns suggested_lr, safe_lr)
    best_lr, safe_lr = lr_finder.plot(
        save_path=os.path.join(PLOTS_DIR, "lr_finder_plot.png"),
        save_csv=True,
        suggest=True,
        annotate=True
    )

    # Fallback if LR finder fails
    if best_lr is None or not math.isfinite(best_lr) or best_lr <= 0:
        print("LR Finder returned invalid value. Falling back to 1e-3.")
        best_lr, safe_lr = 1e-3, 1e-3 * 0.3

    print(f"\nRaw Suggested LR: {best_lr:.6f}")
    print(f"Safe Max LR for OneCycleLR: {safe_lr:.6f}")

    # Clamp LR to safe bounds
    lr_floor, lr_ceiling = 1e-6, 0.1
    use_lr = float(max(lr_floor, min(safe_lr, lr_ceiling)))
    print(f"Final Selected LR → {use_lr:.6f}")

    # -----------------------------------------------------------
    # ♻️ Reset model + optimizer cleanly (recreate to clear any LR-finder state)
    print("Resetting model and optimizer after LR Finder...")
    model = create_model(num_classes=num_classes, pretrained=False).to(DEVICE)
    try:
        model = torch.compile(model)
    except Exception:
        pass
    optimizer = optim.SGD(model.parameters(), lr=use_lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------------
    # 🌀 OneCycleLR Scheduler (per step)
    scheduler = create_onecycle_scheduler(
        optimizer=optimizer,
        max_lr=use_lr,
        train_loader_len=len(train_loader),
        epochs=NUM_EPOCHS,
    )

    # ============================================================
    # Prepare a single GradScaler to pass through epochs (keeps state)
    scaler = GradScaler(device="cuda" if DEVICE == "cuda" else "cpu")

    # -----------------------------------------------------------
    # 📊 Training Loop
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": [], "mom": []}

    with open(LOG_FILE, "w") as log:
        log.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,Learning_Rate,Momentum\n")

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        train_loss, train_acc, scaler = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scheduler, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        current_lr = scheduler.get_last_lr()[0] if scheduler else use_lr
        current_mom = optimizer.param_groups[0].get("momentum", None)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["mom"].append(current_mom)

        epoch_time = time.time() - epoch_start

        print(
            f"[Epoch {epoch+1:03}/{NUM_EPOCHS}] | ⏱️ {epoch_time/60:.2f}m | "
            f"LR: {current_lr:.6f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%"
        )

        # ---- SAVE MODELS ----
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_BEST)
            print(f"New Best Accuracy: {best_acc*100:.2f}% (saved as {SAVE_BEST})\033[0m")

        torch.save(model.state_dict(), SAVE_LAST)

        # ---- LOG ----
        with open(LOG_FILE, "a") as log:
            log.write(
                f"{epoch+1},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{current_lr:.6f},{current_mom:.4f}\n"
            )

        # ---- DYNAMIC PLOTS ----
        epochs_so_far = range(1, epoch + 2)

        def save_plot(x, y_dict, title, ylabel, filename):
            plt.figure(figsize=(8, 5))
            for label, y in y_dict.items():
                plt.plot(x, y, marker='o', label=label)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, filename))
            plt.close()

        save_plot(epochs_so_far, {"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]}, "Accuracy", "Accuracy", "accuracy_live.png")
        save_plot(epochs_so_far, {"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]}, "Loss", "Loss", "loss_live.png")
        save_plot(epochs_so_far, {"Learning Rate": history["lr"]}, "Learning Rate", "LR", "lr_live.png")
        save_plot(epochs_so_far, {"Momentum": history["mom"]}, "Momentum", "Momentum", "momentum_live.png")

    # -----------------------------------------------------------
    print(f"\n🏁 Training Complete — Best Val Acc: {best_acc*100:.2f}%")
    print(f"✅ Best model: {SAVE_BEST}")
    print(f"✅ Last model: {SAVE_LAST}")
    print(f"🖼️ Live plots in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
