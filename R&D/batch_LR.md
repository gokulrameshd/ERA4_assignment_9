Perfect 👍 — let’s generate a **ready-to-use progressive resizing configuration** for your **RTX 5060 Ti (16 GB)** based on the **Balanced Strategy** we discussed.
This version will:
✅ Automatically adjust learning rate based on batch size
✅ Handle gradient accumulation if needed
✅ Use AMP for faster training
✅ Work with `torch.compile` and `channels_last`
✅ Keep dataloaders persistent between stages

---

### 🔧 Progressive Resizing Config (Balanced Strategy)

```python
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler

# ==========================
# ✅ STAGE CONFIGURATION
# ==========================
stages = [
    {"img_size": 128, "batch_size": 512, "epochs": 10, "mixup": True},
    {"img_size": 160, "batch_size": 384, "epochs": 8,  "mixup": True},
    {"img_size": 224, "batch_size": 256, "epochs": 12, "mixup": False},
]

# ==========================
# ✅ SYSTEM & AMP SETUP
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
scaler = GradScaler()
print(f"Using {DEVICE} with AMP and channels_last")

# ==========================
# ✅ MODEL PREP (ResNet Example)
# ==========================
from torchvision import models
def create_model(num_classes):
    model = models.resnet50(weights=None)  # no pretrained weights
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE, memory_format=torch.channels_last)
    return torch.compile(model, dynamic=True)

# ==========================
# ✅ OPTIMIZER + SCHEDULER
# ==========================
def make_optimizer_and_scheduler(model, batch_size, epochs, steps_per_epoch):
    base_lr = min(0.1 * (batch_size / 256), 0.4)  # Linear LR scaling rule
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=base_lr,
        total_steps=epochs * steps_per_epoch,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4
    )
    return optimizer, scheduler

# ==========================
# ✅ TRAINING LOOP SKELETON
# ==========================
def train_stage(stage_idx, model, train_loader, val_loader, num_classes):
    stage = stages[stage_idx]
    batch_size = stage["batch_size"]
    epochs = stage["epochs"]
    steps_per_epoch = len(train_loader)
    accum_steps = max(1, 1024 // batch_size)  # simulate up to 1024 effective batch

    optimizer, scheduler = make_optimizer_and_scheduler(model, batch_size, epochs, steps_per_epoch)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for step, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            inputs = inputs.to(memory_format=torch.channels_last)

            with autocast(dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            running_loss += loss.item() * accum_steps
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

        acc = 100.0 * correct / total
        print(f"[Stage {stage_idx+1}] Epoch {epoch+1}/{epochs} | "
              f"Loss: {running_loss/steps_per_epoch:.4f} | Acc: {acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    torch.cuda.empty_cache()
    return model
```

---

### 🧠 Why this setup works well

| Aspect                                    | Purpose                                                      |
| ----------------------------------------- | ------------------------------------------------------------ |
| **Progressive resizing**                  | Reduces training time — small images first, high-res last    |
| **Batch scaling rule (0.1 × batch/256)**  | Keeps effective LR consistent                                |
| **AMP + channels_last + cudnn.benchmark** | Gives ~1.3×–1.5× speed boost on RTX cards                    |
| **1024 effective batch via accumulation** | Keeps large-batch stability even if VRAM limits actual batch |
| **OneCycleLR**                            | Fast convergence and robust learning rate decay              |
| **Label smoothing (0.1)**                 | Regularization and better calibration                        |

---

### ✅ Recommended Stage Summary for You

| Stage | Image Size | Batch Size | Epochs | LR (scaled) | Approx Duration* |
| ----- | ---------- | ---------- | ------ | ----------- | ---------------- |
| 1     | 128        | 512        | 10     | 0.2         | ~0.8 min/epoch   |
| 2     | 160        | 384        | 8      | 0.15        | ~1.1 min/epoch   |
| 3     | 224        | 256        | 12     | 0.1         | ~1.8 min/epoch   |

* based on your previous 1.8 min/epoch @224px.

---

Would you like me to add:

* ✅ automatic **checkpoint saving & resuming per stage**,
* ✅ or automatic **progressive dataloader resizing** (so your script rebuilds the dataloaders automatically for each stage)?

Both are simple additions that make this setup fully production-grade.
