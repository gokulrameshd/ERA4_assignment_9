from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary
import os
import numpy as np
import time
from torch.cuda.amp import  GradScaler #autocast, old version 1.11.0
from data_loader import get_mixup_fn
from torch import autocast #new version 2.5.0
from gpu_stats import get_gpu_usage
# Let's visualize some of the images
import copy

# ========================
# CutMix / Mixup Utils
# ========================
def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)

    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
    y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def mixup_data(x, y, alpha=1.0):
    """Apply Mixup augmentation."""
    if alpha <= 0:
        return x, y, y, 1
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation."""
    if alpha <= 0:
        return x, y, y, 1
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    y_a, y_b = y, y[index]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion):

    correct = 0
    processed = 0
    train_loss = 0
    model.train()
    pbar = tqdm(train_loader)

    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)#F.nll_loss
        # train_losses.append(loss)
        
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm

        # pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += GetCorrectPredCount(y_pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc = 100*correct/processed
    train_loss = train_loss/len(train_loader)
    return train_loss, train_acc

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            y_pred = model(data)

            # accumulate loss (sum)
            test_loss += criterion(y_pred, target).item()
            correct += GetCorrectPredCount(y_pred, target)

    # average loss per batch
    test_loss /= len(test_loader)
    test_acc = 100. * correct / len(test_loader.dataset)

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc:.2f}%)\n")

    return test_loss, test_acc


def train_one_epoch(model, device, train_loader, optimizer, criterion, scheduler, scaler, epoch,total_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ✅ Step scheduler only once per batch, inside loop
        scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.*correct/total:.2f}%",
            "lr": f"{scheduler.get_last_lr()[0]:.5f}"
        })

    # ❌ DO NOT call scheduler.step() again here
    return total_loss / total, 100. * correct / total

# ========================
# Training & Evaluation
# ========================
def train_one_epoch_scaler_cutmix_mixup(model, device, train_loader, optimizer, criterion, scheduler, scaler,epoch, epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Randomly choose augmentation: CutMix or Mixup (or none)
        r = np.random.rand()
        if r < 0.25:
            imgs, y_a, y_b, lam = cutmix_data(imgs, targets, alpha=1.0)
        elif r < 0.5:
            imgs, y_a, y_b, lam = mixup_data(imgs, targets, alpha=1.0)
        else:
            y_a, y_b, lam = targets, targets, 1.0

        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += (lam * preds.eq(y_a).sum().item() +
                    (1 - lam) * preds.eq(y_b).sum().item())
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.*correct/total:.2f}%",
            "lr": f"{scheduler.get_last_lr()[0]:.5f}"
        })

    
    return total_loss / total, 100. * correct / total

@torch.no_grad()
def evaluate(model, device, test_loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for imgs, targets in test_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()
    return total_loss / total, 100. * correct / total


# ========================
# Training Loop
# ========================
def training_loop_with_scaler(model, device, train_loader, test_loader, optimizer, criterion, scheduler, epochs,CHECKPOINT_DIR="./checkpoints"):
    best_acc = 0.0
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    Learning_Rates = []
    best_accuracies = []
    scaler = torch.cuda.amp.GradScaler()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion, scheduler, scaler,epoch,epochs)
        val_loss, val_acc = evaluate(model, device, test_loader, criterion)
        train_losses.append(train_loss)
        test_losses.append(val_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(val_acc)
        Learning_Rates.append(scheduler.get_last_lr()[0])

        print(f"\nEpoch {epoch+1:03d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.5f}")

        # Save best checkpoint if the accuracy is better
        if val_acc > best_acc:
            best_acc = val_acc
            best_accuracies.append(best_acc)
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_acc': best_acc
            }, os.path.join(CHECKPOINT_DIR, "best.pth"))
            print(f"✅ Saved new best model ({best_acc:.2f}%)")

    print(f"Training complete. Best accuracy: {best_acc:.2f}%")
    return train_losses, test_losses, train_accuracies, test_accuracies, Learning_Rates, best_accuracies 

def training_loop_with_scaler_cutmix_mixup(model, device, train_loader, test_loader, optimizer, criterion, scheduler, epochs,CHECKPOINT_DIR="./checkpoints"):
    best_acc = 0.0
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    Learning_Rates = []
    best_accuracies = []
    scaler = torch.cuda.amp.GradScaler()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_scaler_cutmix_mixup(model, device, train_loader, optimizer, criterion, scheduler, scaler,epoch,epochs)
        val_loss, val_acc = evaluate(model, device, test_loader, criterion)
        train_losses.append(train_loss)
        test_losses.append(val_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(val_acc)
        Learning_Rates.append(scheduler.get_last_lr()[0])

        print(f"\nEpoch {epoch+1:03d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.5f}")

        # Save best checkpoint if the accuracy is better
        if val_acc > best_acc:
            best_acc = val_acc
            best_accuracies.append(best_acc)
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_acc': best_acc
            }, os.path.join(CHECKPOINT_DIR, "best.pth"))
            print(f"✅ Saved new best model ({best_acc:.2f}%)")

    print(f"Training complete. Best accuracy: {best_acc:.2f}%")
    return train_losses, test_losses, train_accuracies, test_accuracies, Learning_Rates, best_accuracies 


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
def train_one_epoch_imagenet(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    scheduler=None,
    scaler=None,
    mixup_fn=None,
    enable_last_channel= False,
    ema= None,
    num_classes=1000,
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    start_time = time.time()

        # Safer device detection
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    # AMP scaler (initialize if not provided)
    if scaler is None:
        scaler = GradScaler(device=device_type)

    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="\033[92m🟢 Training\033[0m",
        leave=False,
        ncols=120
    )

    for batch_idx, (inputs, labels) in pbar:
        if enable_last_channel:
            inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        else:
            inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # AMP autocast context — prefer bfloat16 if supported
        # with autocast(device_type=device_type, dtype=torch.float16 if device_type == "cuda" else torch.float32):
        with autocast(
            device_type=device_type,
            dtype=(
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16),
                    ):
            if mixup_fn is not None:
                mixed = mixup_fn(inputs, labels)
                if isinstance(mixed, tuple) and len(mixed) == 4:
                    # Custom SimpleMixup interface
                    inputs, y_a, y_b, lam = mixed
                    outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                else:
                    # timm-style Mixup interface
                    inputs, targets = mixed
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

        
         # Backward + optimizer step Backward pass (scaled)
        scaler.scale(loss).backward()
        # ✅ UNscale before clipping
        scaler.unscale_(optimizer)
        # ✅ Apply gradient clipping (L2 norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Optimizer + scaler steps
        scaler.step(optimizer)
        scaler.update()

        # Handle scheduler per-iteration (OneCycleLR etc.)
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            if scheduler._step_count < scheduler.total_steps:
                scheduler.step()
            # else:
            #     scheduler.step(epoch)
        # EMA update
        if ema is not None:
            with torch.no_grad():
                for ema_p, model_p in zip(ema.ema.parameters(), model.parameters()):
                    if model_p.dtype.is_floating_point:
                        ema_p.mul_(ema.decay).add_(model_p, alpha=1 - ema.decay)

        # Track metrics
        # total_loss += float(loss.detach().cpu().item()) * inputs.size(0)
        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += int(preds.eq(labels).sum().item())
        total += labels.size(0)

        # Update tqdm less frequently to reduce overhead
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(dataloader):
            avg_loss = total_loss / total
            acc = 100.0 * correct / total
            mem_alloc = (
                # format_mem(torch.cuda.memory_allocated(device))
                f"{torch.cuda.memory_allocated(device) / 1e9:.2f} GB"
                if torch.cuda.is_available()
                else "N/A"
            )

            elapsed = time.time() - start_time
            iters_done = batch_idx + 1
            iters_left = len(dataloader) - iters_done
            # eta = iters_left * (elapsed / max(1, iters_done))
            eta = (len(dataloader) - iters_done) * (elapsed / max(1, iters_done))
            pbar.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "Acc": f"{acc:.2f}%",
                "Mem": {mem_alloc},
                "ETA": f"{eta/60:.1f}m"
            })
        # stats = get_gpu_usage(device=device)
    # Epoch-end scheduler (for StepLR, CosineLR, etc.)
    if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        scheduler.step()

    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "scaler": scaler,
        "time": (time.time() - start_time)/60,
    }

@torch.no_grad()
def validate_imagenet(model, dataloader, criterion, device,num_classes=1000):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    start_time = time.time()

    # Safer device detection
    device_type = "cuda" if "cuda" in str(device) else "cpu"


    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="\033[94m🔵 Validating\033[0m",
        leave=False,
        ncols=120
    )

    # with torch.no_grad():
    for batch_idx, (inputs, labels) in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # use autocast for faster inference on GPU
        # with autocast(device_type=device_type, dtype=torch.float16 if device_type == "cuda" else torch.float32):
        with autocast(device_type=device_type,
                    dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),):

            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # total_loss += float(loss.detach().cpu().item()) * inputs.size(0)
        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += int(preds.eq(labels).sum().item())
        total += labels.size(0)

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            avg_loss = total_loss / total
            acc = 100.0 * correct / total
            mem_alloc = (
                # format_mem(torch.cuda.memory_allocated(device))
                f"{torch.cuda.memory_allocated(device) / 1e9:.2f} GB"
                if torch.cuda.is_available()
                else "N/A"
            )

            elapsed = time.time() - start_time
            # iters_done = batch_idx + 1
            # iters_left = len(dataloader) - iters_done
            # eta = iters_left * (elapsed / max(1, iters_done))
            eta = (len(dataloader) - (batch_idx + 1)) * (
                elapsed / max(1, (batch_idx + 1))
            )
            pbar.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "Acc": f"{acc:.2f}%",
                "Mem": {mem_alloc},
                "ETA": f"{eta/60:.1f}m"
            })

    return {
    "loss": total_loss / total,
    "acc": correct / total,
    "time": (time.time() - start_time)/60,
    }


def train_test_with_scheduler(model, device, train_loader, test_loader, optimizer, criterion, scheduler, epochs,save_path="./checkpoints/best_model.pth"):
    best_acc = 0
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    Learning_Rates = []
    best_accuracies = []
    for epoch in range(epochs):
        print("EPOCH:", epoch)
        train_loss, train_acc = train(model, device, train_loader, optimizer,criterion)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        scheduler.step(test_loss)
        print("Learning Rate:", scheduler.get_last_lr())
        train_losses.append(train_loss) 
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        Learning_Rates.append(scheduler.get_last_lr())
        if test_acc > best_acc:
            best_acc = test_acc
            best_accuracies.append(best_acc)
            torch.save(model.state_dict(), save_path)
            print(f"Saved model with accuracy: {best_acc:.2f}%")
    return train_losses, test_losses, train_accuracies, test_accuracies, Learning_Rates,best_accuracies

def plot_losses(train_losses, test_losses, train_accuracies, test_accuracies):
    t = [t_items.item() for t_items in train_losses]
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(t)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_accuracies)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_accuracies)
    axs[1, 1].set_title("Test Accuracy")

    plt.show()

def save_plot(x, y_dict, title, xlabel, ylabel, filename,plots_dir):
    plt.figure(figsize=(8, 5))
    for label, y in y_dict.items():
        # plt.plot(x, y, marker='o', label=label)
        min_len = min(len(x), len(y))
        plt.plot(x[:min_len], y[:min_len], marker='o', label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()

def get_model_summary(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return summary(model, input_size=(1, 28, 28))


def save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_acc, history, path="checkpoint.pth"):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if scaler else None,
        "best_acc": best_acc,
        "history": history,  # 👈 Save your plots' history too
    }
    torch.save(state, path)
    print(f"✅ Checkpoint saved at {path} (epoch {epoch})")

def load_best_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")  # or DEVICE if you prefer
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    print(f"✅ Loaded best model weights from {checkpoint_path}")
    return model


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cuda"):
    if not os.path.exists(path):
        print(f"⚠️ No checkpoint found at {path}")
        return 0, 0.0  # start fresh

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"✅ Model weights loaded from {path}")

    if optimizer and "optimizer_state" in checkpoint and checkpoint["optimizer_state"]:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("✅ Optimizer state restored")

    if scheduler and "scheduler_state" in checkpoint and checkpoint["scheduler_state"]:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        print("✅ Scheduler state restored")

    if scaler and "scaler_state" in checkpoint and checkpoint["scaler_state"]:
        scaler.load_state_dict(checkpoint["scaler_state"])
        print("✅ GradScaler state restored")

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_acc = checkpoint.get("best_acc", 0.0)
    history = checkpoint.get("history", {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": [], "mom": [],
                                        "train_time": [], "val_time": [], "time_lapsed": [], "total_time_epoch": []})
    print(f"🔁 Resuming from epoch {start_epoch} (best acc: {best_acc:.2f}%)")
    return start_epoch, best_acc, history


class ModelEMA:
    """Maintains an exponential moving average (EMA) of model weights."""
    def __init__(self, model, decay=0.999, model_fn=None):
        self.decay = decay
        device = next(model.parameters()).device

        # Clone the model for EMA
        if model_fn is not None:
            self.ema = model_fn().to(device)
        else:
            # fallback: deep copy the model
            self.ema = copy.deepcopy(model).to(device)

        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k], alpha=1 - self.decay)




def _set_trainable_layers(model, mode, target_layer=None):
    """Enable/disable layer parameters progressively based on mode."""
    # First, freeze everything
    for p in model.parameters():
        p.requires_grad = False

    if mode == "unfreeze":
        if target_layer == "layer3":
            for n, p in model.layer3.named_parameters(): p.requires_grad = True
            for n, p in model.layer4.named_parameters(): p.requires_grad = True
            for n, p in model.fc.named_parameters(): p.requires_grad = True
        elif target_layer == "layer2":
            for n, p in model.layer2.named_parameters(): p.requires_grad = True
            for n, p in model.layer3.named_parameters(): p.requires_grad = True
            for n, p in model.layer4.named_parameters(): p.requires_grad = True
            for n, p in model.fc.named_parameters(): p.requires_grad = True
        elif target_layer == "all":
            for p in model.parameters(): p.requires_grad = True

    elif mode == "freeze":
        # start with all trainable
        for p in model.parameters(): p.requires_grad = True
        if target_layer == "layer4":
            for n, p in model.layer1.named_parameters(): p.requires_grad = False
            for n, p in model.layer2.named_parameters(): p.requires_grad = False
            for n, p in model.layer3.named_parameters(): p.requires_grad = False
            print("Freezing layer1, layer2 and layer3 ")
        if target_layer == "layer3":
            for n, p in model.layer1.named_parameters(): p.requires_grad = False
            for n, p in model.layer2.named_parameters(): p.requires_grad = False
            print("Freezing layer1 and layer2 ")
        if target_layer == "layer2":
            for n, p in model.layer1.named_parameters(): p.requires_grad = False
            print("Freezing layer1")

def recreate_optimizer(model, base_lr, weight_decay=1e-4, momentum=0.9):
    # Only include trainable params
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    print(f"✅ Recreated optimizer with {len(params)} trainable parameter tensors")
    return optimizer

def forward_with_freeze(model, x):
    with torch.no_grad():
        x = model.layer1(x)
        x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    x = model.fc(x)
    return x

def disable_grad_for_frozen_layers(model):
    for name, module in model.named_children():
        if all(not p.requires_grad for p in module.parameters()):
            module.forward = torch.no_grad()(module.forward)
            
def set_trainable_layers(model, mode, target_layer):
    freeze_map = {
        "layer1": [model.layer1],
        "layer2": [model.layer1, model.layer2],
        "layer3": [model.layer1, model.layer2, model.layer3],
        "layer4": [model.layer1, model.layer2, model.layer3, model.layer4],
    }
    if mode == "freeze" and target_layer in freeze_map:
        for layer_number,block in enumerate(freeze_map[target_layer]):
            print(f"Freezing layer {layer_number+1}")
            for p in block.parameters():
                p.requires_grad = False
    elif mode == "unfreeze" and target_layer in freeze_map:
        for layer_number,block in enumerate(freeze_map[target_layer]):
            print(f"Unfreezing layer {layer_number+1}")
            for p in block.parameters():
                p.requires_grad = True

def build_optimizer(model, lr, weight_decay=1e-4, momentum=0.9):
    # only include params that require grad
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

def save_multiple_plots(epochs_so_far, history, PLOTS_DIR):
    save_plot(epochs_so_far, {"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]}, "Accuracy", "Epoch", "Accuracy", "accuracy_live.png",PLOTS_DIR)
    save_plot(epochs_so_far, {"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]}, "Loss", "Epoch", "Loss", "loss_live.png",PLOTS_DIR)
    save_plot(epochs_so_far, {"Learning Rate": history["lr"]}, "Learning Rate", "Epoch", "LR", "lr_live.png",PLOTS_DIR)
    save_plot(epochs_so_far, {"Momentum": history["mom"]}, "Momentum", "Epoch", "Momentum", "momentum_live.png",PLOTS_DIR)
    #plot train time vs accuracy
    save_plot(history["time_lapsed"], {"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]}, "Accuracy", "Time(m)", "Accuracy", "accuracy_time.png",PLOTS_DIR)
    save_plot(history["time_lapsed"], {"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]}, "Loss", "Time(m)", "Loss", "loss_time.png",PLOTS_DIR)

def save_weights(model, optimizer, scheduler, scaler, best_acc, best_weights,history, PLOTS_DIR, SAVE_BEST, SAVE_LAST, TXT_LOG_FILE, epoch, val_acc):
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), SAVE_BEST)
        print(f"New Best Accuracy: {best_acc*100:.2f}% (saved as {SAVE_BEST})\033[0m")
        with open(TXT_LOG_FILE, "a") as log:
            log.write(f"New Best Accuracy: {best_acc*100:.2f}% (saved as {SAVE_BEST})\033[0m \n")
        best_weights = model.state_dict()  # carry forward
    save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_acc, history,
                    path=SAVE_LAST)

    return best_acc,best_weights

def train_validate_save_weights_history_plots(model, train_loader, val_loader, optimizer, criterion, scheduler,
                                    scaler, mixup_fn,  ema, num_classes, PLOTS_DIR, 
                                    SAVE_BEST, SAVE_LAST, TXT_LOG_FILE, epoch, best_acc, best_weights,history, use_lr,
                                     CSV_LOG_FILE, NUM_EPOCHS, enable_last_channel, device):

    start_time = time.time()
    # ---------------------------------------
        # 🏋️ Train
        # ---------------------------------------
    train_results = train_one_epoch_imagenet(model, train_loader, optimizer, criterion, device,
                                            scheduler, scaler, mixup_fn=mixup_fn, 
                                            enable_last_channel = enable_last_channel,
                                            ema=ema,num_classes=num_classes)
    train_loss = train_results["loss"]
    train_acc = train_results["acc"]
    scaler = train_results["scaler"]
    train_time = train_results["time"]
    # ---------------------------------------
    # ✅ Validate
    # ---------------------------------------
    val_model = ema.ema if ema is not None else model
    val_results = validate_imagenet(val_model, val_loader, criterion, device,  num_classes=num_classes)
    val_loss = val_results["loss"]
    val_acc = val_results["acc"]
    val_time = val_results["time"]
    best_acc ,best_weights  = save_weights(model, optimizer, scheduler, scaler, best_acc,best_weights, history, PLOTS_DIR, SAVE_BEST, SAVE_LAST, TXT_LOG_FILE, epoch, val_acc)
    # ---------------------------------------
    # Log stats
    # ---------------------------------------
    current_lr = scheduler.get_last_lr()[0] if scheduler else use_lr
    current_mom = optimizer.param_groups[0].get("momentum", None)
    total_time_epoch = (time.time() - start_time)/60
    try:
        time_lapsed = history["time_lapsed"][-1] + total_time_epoch
    except :
        time_lapsed = total_time_epoch
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)
    history["mom"].append(current_mom)
    history["time_lapsed"].append(time_lapsed)
    history["train_time"].append(train_time)
    history["val_time"].append(val_time)
    history["total_time_epoch"].append(total_time_epoch)

    print(
        f"[Epoch {epoch+1:03}/{NUM_EPOCHS}] | ⏱️ {train_time:.2f}m | "
        f"LR: {current_lr:.6f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%"
    )

    # ---- LOG ----
    with open(TXT_LOG_FILE, "a") as log:
        log.write(
            # f"{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{current_lr:.6f},{current_mom:.4f}\n"
            f"[Epoch {epoch+1:03}/{NUM_EPOCHS}] | ⏱️ {train_time:.2f}m | "
            f"LR: {current_lr:.6f} | Train Acc: {train_acc*100:.2f}% | Train Loss: {train_loss:.4f} | Train Time: {train_time:.2f}m  | Val Acc: {val_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Time: {val_time:.2f}m | "
            f"Momentum: {current_mom:.4f} \n"
        )
    with open(CSV_LOG_FILE, "a") as log:
        log.write(f"{epoch+1:03},{train_loss:.4f},{train_acc:.4f},{train_time:.2f},{val_loss:.4f},{val_acc:.4f},{val_time:.2f},{current_lr:.6f},{current_mom:.4f}\n")

    # ---- DYNAMIC PLOTS ----
    epochs_so_far = range(1, epoch + 2)

    save_multiple_plots(epochs_so_far, history, PLOTS_DIR)

    return scaler, history, best_acc,best_weights

