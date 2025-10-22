Perfect 👌
Let’s now **add CutMix / MixUp directly into your DataLoader pipeline**, so you don’t have to toggle `use_mixup_fn` manually inside your training loop.

This will give you a **cleaner training step**, where augmentation logic (CutMix / MixUp) is encapsulated in the data pipeline — not buried inside your training loop.

---

### ✅ Recommended design pattern

We'll use `timm.data.Mixup` — a high-quality and efficient implementation that supports both **MixUp** and **CutMix** automatically.

---

### 🔧 Step 1 — Import and create a reusable augmentation mix

```python
from timm.data import Mixup
```

Then, add this **utility function** in your setup section:

```python
def get_mixup_cutmix_fn(mixup_alpha=0.2, cutmix_alpha=1.0, num_classes=1000):
    return Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=1.0,          # always apply; can adjust <1.0 if needed
        switch_prob=0.5,   # 50% chance to switch between mixup and cutmix
        mode='batch',      # can be 'batch', 'pair', or 'elem'
        label_smoothing=0.1,
        num_classes=num_classes,
    )
```

---

### 🔧 Step 2 — Apply this to your training DataLoader

We modify the `make_loader()` inside your existing code:

```python
def make_loader(img_size, batch_size, is_train=True, use_mix=False):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip() if is_train else transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    dataset = train_dataset if is_train else val_dataset
    dataset.transform = transform
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # --- 🧩 Wrap with CutMix / MixUp if enabled ---
    if is_train and use_mix:
        mixup_fn = get_mixup_cutmix_fn(num_classes=NUM_CLASSES)
        loader.mixup_fn = mixup_fn
    else:
        loader.mixup_fn = None

    return loader
```

---

### 🔧 Step 3 — Modify your training loop slightly

Replace your existing train call inside the epoch loop with:

```python
train_loss, train_acc, scaler = train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    DEVICE,
    scheduler,
    scaler,
    use_mixup_fn=False,  # ← disable this argument now
    num_classes=NUM_CLASSES,
    mixup_fn=getattr(train_loader, "mixup_fn", None)
)
```

---

### 🔧 Step 4 — Modify your `train_one_epoch()` to use the loader’s mixup directly

Here’s a clean pattern (replace the first few lines inside your loop):

```python
def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler, scaler, num_classes, mixup_fn=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        # Apply CutMix/MixUp if available
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets.argmax(dim=1) if targets.ndim > 1 else targets).sum().item()
        total += inputs.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc, scaler
```

---

### ✅ Benefits of this new approach

| Feature            | Old (manual toggle)        | New (DataLoader-integrated) |
| ------------------ | -------------------------- | --------------------------- |
| CutMix/MixUp logic | Inside training loop       | Attached to DataLoader      |
| Readability        | Mixed with training logic  | Clean separation            |
| Flexibility        | One augmentation per stage | Easily switch per stage     |
| Performance        | Same or slightly better    | Persistent + prefetching    |

---

### 🔄 Example stage config usage

```python
stages = [
    {"img_size": 128, "batch_size": 1024, "epochs": 5, "use_mix": True},
    {"img_size": 224, "batch_size": 256, "epochs": 5, "use_mix": False},
]

train_loaders = {
    s["img_size"]: make_loader(s["img_size"], s["batch_size"], True, s["use_mix"]) for s in stages
}
```

---

Would you like me to extend this so that **it dynamically adjusts MixUp/CutMix probability and alpha values per stage** (e.g., strong in early small-image stages, weaker later)?
That’s a common and very effective trick in progressive training.
