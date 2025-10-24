Excellent — let’s make your **training fully resumable** 👇
We’ll implement a *clean resume system* that restores:

* ✅ Model weights
* ✅ Optimizer state
* ✅ Scheduler state
* ✅ AMP GradScaler state
* ✅ Epoch counter

---

## 🧩 1. Modify your training loop

You’ll add a few lines for **saving** and **resuming** checkpoints.

Here’s a solid template that integrates perfectly with your existing setup (OneCycleLR + AMP + MixUp etc.):

```python
import os
import torch

def save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_acc, path="checkpoint.pth"):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if scaler else None,
        "best_acc": best_acc
    }
    torch.save(state, path)
    print(f"✅ Checkpoint saved at {path} (epoch {epoch})")


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
    print(f"🔁 Resuming from epoch {start_epoch} (best acc: {best_acc:.2f}%)")
    return start_epoch, best_acc
```

---

## 🧠 2. Integrate into your main training script

```python
scaler = torch.cuda.amp.GradScaler()
best_acc = 0.0

# 🔁 Try to resume
start_epoch, best_acc = load_checkpoint(
    path="imagenet_resnet50.pth",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    device=device
)

for epoch in range(start_epoch, num_epochs):
    train_loss, train_acc, scaler = train_one_epoch_imagenet(
        model, train_loader, optimizer, criterion, device,
        scheduler=scheduler, scaler=scaler, use_mixup_fn=True
    )

    val_loss, val_acc = validate_imagenet(model, val_loader, criterion, device)

    # Save if best
    if val_acc > best_acc:
        best_acc = val_acc
        save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_acc,
                        path="imagenet_resnet50_best.pth")

    # Also save last
    save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_acc,
                    path="imagenet_resnet50_last.pth")
```

---

## ⚙️ 3. Important Notes

| Case                                      | Action                                                 |
| ----------------------------------------- | ------------------------------------------------------ |
| Changed learning rate / scheduler mid-run | Resume only weights; don’t load optimizer/scheduler    |
| Using `DistributedDataParallel`           | Call `torch.load(..., map_location="cuda:local_rank")` |
| Running on CPU temporarily                | Use `map_location="cpu"`                               |

---

## 💡 Optional: Auto-resume logic

You can automatically resume if a file exists:

```python
if os.path.exists("imagenet_resnet50_last.pth"):
    start_epoch, best_acc = load_checkpoint("imagenet_resnet50_last.pth", model, optimizer, scheduler, scaler, device)
else:
    start_epoch, best_acc = 0, 0.0
```

---

## ✅ TL;DR

Put this pattern together:

1. `save_checkpoint()` at the end of every epoch (or best model only)
2. `load_checkpoint()` before training starts
3. Include `scaler` and `scheduler` states for full AMP & LR recovery

---

Would you like me to extend this with **automatic resume-from-last** logic (so you can just run the same script and it continues automatically if a checkpoint exists)?
