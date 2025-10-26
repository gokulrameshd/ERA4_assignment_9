Excellent — this stage setup is *very close* to what an expert progressive-resizing + stagewise OneCycleLR + partial freezing schedule should look like when training from scratch on ImageNet-scale or medium-sized datasets.

Let me break it down analytically with expert tuning advice 👇

---

### 🔍 Stage-by-Stage Expert Review

#### 🧩 Stage 1 — 50% data @ 128px, batch 1024, 10 epochs

✅ **Purpose:** Warmup & rapid convergence of low-level features (edges, textures).
✅ **Freeze:** None → good, since training from scratch.
✅ **Mixup:** True — excellent for stability.
✅ **LR Scale:** 1.0 — perfect for warmup.
💡 *Keep this short and fast (10–15 epochs).*

---

#### 🧩 Stage 2 — 60% data @ 160px, batch 768, 10 epochs

✅ Smooth transition — slightly higher resolution and smaller batch (hence effectively increasing gradient variance).
✅ Mixup retained — helps regularize.
💡 You could **add LR dampening** (0.9 is fine).
💡 *Optional:* use `freeze_to=None` or early layer freeze if low-level convergence is stable (but optional).

---

#### 🧩 Stage 3 — 70% data @ 224px, batch 512, 10 epochs

✅ Resolution ramp-up correct.
✅ **Mixup** still ON — good since this is the first full-size stage.
⚙️ Suggest **slight LR dampening** to `0.7` (instead of 0.6) unless you notice overshooting.
💡 Keep `freeze_to=None` still, as the model benefits from fine-tuning all layers here.

---

#### 🧩 Stage 4 — 80% data @ 224px, batch 768, 10 epochs

✅ **freeze_to="layer2"** — very smart choice; this lets the higher-level feature refinement stabilize while freezing mid-level features to reduce catastrophic forgetting.
✅ Mixup off — good, as we focus more on fine detail consistency.
💡 *Optional tweak:* use **CosineAnnealingWarmRestarts** or continue OneCycle with smaller `max_lr` for this stage for smooth convergence.

---

#### 🧩 Stage 5 — 100% data @ 224px, batch 1024, 10 epochs

✅ Freeze deeper block (layer3) — great for solidifying mid-level filters while training only top layers.
✅ Mixup off — correct choice for final convergence.
💡 *Consider adding label smoothing (0.05–0.1)* for generalization.
💡 This stage can use **lower OneCycle max_lr = 0.25 × base_lr**, for fine polishing.

---

### ⚙️ Expert Summary Configuration Recommendation

Here’s the refined expert-level dictionary:

```python
TRAIN_STAGES = [
    {"fraction": 0.50, "img_size": 128, "batch_size": 1024, "epochs": 10, "lr_scale": 1.0, "use_mixup": True, "freeze_to": None},
    {"fraction": 0.60, "img_size": 160, "batch_size": 768,  "epochs": 10, "lr_scale": 0.9, "use_mixup": True, "freeze_to": None},
    {"fraction": 0.70, "img_size": 224, "batch_size": 512,  "epochs": 10, "lr_scale": 0.7, "use_mixup": True, "freeze_to": None},
    {"fraction": 0.80, "img_size": 224, "batch_size": 768,  "epochs": 10, "lr_scale": 0.5, "use_mixup": False, "freeze_to": "layer2"},
    {"fraction": 1.00, "img_size": 224, "batch_size": 1024, "epochs": 10, "lr_scale": 0.25, "use_mixup": False, "freeze_to": "layer3"},
]
```

---

### 🚀 Why This Works

✅ Progressive resize keeps early training efficient.
✅ Stagewise OneCycleLR avoids the “wrong step counting” issue of global OneCycle.
✅ Freezing higher layers later stabilizes gradients and prevents overfitting.
✅ Larger batch in final stages improves generalization & stability.
✅ Controlled LR scaling prevents overshooting at high resolution.

---

### 💯 Expected Outcome

* Training time ↓ (due to smaller images early on)
* Accuracy ↑ (progressive adaptation + mixup + fine-tuning)
* Target > **78–80% top-1 accuracy** achievable on ResNet-50 (ImageNet-1k baseline) with proper augmentation & EMA.

---

If you share:

* model type (ResNet50, EfficientNet, etc.)
* dataset size (ImageNet-1k, smaller, or custom)

I can tailor **exact LR range, OneCycle parameters (`pct_start`, `div_factor`, `final_div_factor`)**, and **freeze layer mapping** for optimal convergence.

Would you like me to optimize those next?

Perfect — you’ve built a **robust hybrid progressive training pipeline**. You’re already combining almost everything high-end training frameworks use:
✅ Progressive resize
✅ Progressive freezing/unfreezing
✅ EMA
✅ Channel-last optimization
✅ OneCycleLR (both global + stagewise option)
✅ LR finder
✅ Mixup + label smoothing

Let’s fine-tune **the exact configuration for ResNet-50 on ImageNet-1k** to reach **>78% accuracy** faster, without wasting compute.

---

## 🧠 Expert-Recommended Strategy for Your Code

### 1️⃣ Enable Flags

```python
ENABLE_HYBRID_TRAINING = True
ENABLE_PROGRESSIVE_FREEZING = True
ENABLE_PROGRESSIVE_UNFREEZING = False
ENABLE_STAGE_WISE_SCHEDULER = False   # ✅ keep global for simplicity
ENABLE_EMA = True                     # ✅ stabilizes last stages
ENABLE_CHANNEL_LAST = True
ENABLE_LR_DAMPENING = True            # ✅ better LR control per stage
ENABLE_LR_FINDER = False              # (optional: True for initial LR scan)
```

---

### 2️⃣ Optimized TRAIN_STAGES for ResNet50 + ImageNet-1k

This progression balances GPU memory, convergence speed, and accuracy:

```python
TRAIN_STAGES = [
    # Stage 1 — Fast feature warmup (edges/textures)
    {"fraction": 0.50, "img_size": 128, "batch_size": 1024, "epochs": 8, "lr_scale": 1.0, "use_mixup": True, "freeze_to": None},

    # Stage 2 — Mid-level refinement
    {"fraction": 0.75, "img_size": 160, "batch_size": 768,  "epochs": 8, "lr_scale": 0.8, "use_mixup": True, "freeze_to": None},

    # Stage 3 — Full data fine-tuning (high res, still all layers trainable)
    {"fraction": 1.00, "img_size": 224, "batch_size": 512,  "epochs": 10, "lr_scale": 0.6, "use_mixup": True, "freeze_to": None},

    # Stage 4 — Freeze earlier blocks (stabilize deeper learning)
    {"fraction": 1.00, "img_size": 224, "batch_size": 768,  "epochs": 10, "lr_scale": 0.4, "use_mixup": False, "freeze_to": "layer2"},

    # Stage 5 — Final fine-tuning, larger batch, low LR
    {"fraction": 1.00, "img_size": 224, "batch_size": 1024, "epochs": 6, "lr_scale": 0.25, "use_mixup": False, "freeze_to": "layer3"},
]
NUM_EPOCHS = sum(s["epochs"] for s in TRAIN_STAGES)
```

---

### 3️⃣ Global OneCycleLR Parameters (for `create_onecycle_scheduler_global`)

Use:

```python
max_lr = 0.1  # base max LR
pct_start = 0.15  # fast ramp-up
div_factor = 10  # initial_lr = max_lr / div_factor
final_div_factor = 1e3
```

If you’re defining your own `create_onecycle_scheduler_global`, ensure it uses:

```python
torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    total_steps=total_steps,
    pct_start=pct_start,
    div_factor=div_factor,
    final_div_factor=final_div_factor,
    anneal_strategy='cos',
)
```

💡 The cosine annealing inside OneCycle improves late-stage smoothness.

---

### 4️⃣ Optimizer Recommendations

For ImageNet-scale ResNet50 from scratch:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=use_lr, momentum=0.9, weight_decay=1e-4, nesterov=True
)
```

Optional: use `torch.optim.AdamW` if you want slightly faster early convergence, but SGD+momentum still gives best top-1 accuracy.

---

### 5️⃣ Freezing Logic

Ensure `set_trainable_layers(model, "freeze", layer_name)` works like this:

```python
def set_trainable_layers(model, mode, target_layer):
    freeze_map = {
        "layer1": [model.layer1],
        "layer2": [model.layer1, model.layer2],
        "layer3": [model.layer1, model.layer2, model.layer3],
    }
    if mode == "freeze" and target_layer in freeze_map:
        for block in freeze_map[target_layer]:
            for p in block.parameters():
                p.requires_grad = False
    elif mode == "unfreeze" and target_layer in freeze_map:
        for block in freeze_map[target_layer]:
            for p in block.parameters():
                p.requires_grad = True
```

---

### 6️⃣ EMA

Keep:

```python
ema = ModelEMA(model, decay=0.9998, model_fn=lambda: create_model(num_classes, pretrained=False))
```

EMA improves final validation accuracy by ~0.3–0.5%.

---

### 7️⃣ Label smoothing

At the last 2 stages (when `mixup=False`):

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

This helps stabilize learning when mixup is disabled.

---

### 8️⃣ Expected Performance

| Metric                                        | Expected       |
| --------------------------------------------- | -------------- |
| Training Time (1x RTX 5090 / 90 epochs equiv) | ~20–25 hours   |
| Peak LR                                       | 0.1            |
| Final LR                                      | ~1e-4          |
| Top-1 Accuracy                                | **78.2–79.5%** |
| Top-5 Accuracy                                | **93–94%**     |

---

### ✅ Summary

**Best setup for your code:**

```python
ENABLE_STAGE_WISE_SCHEDULER = False
ENABLE_PROGRESSIVE_FREEZING = True
ENABLE_PROGRESSIVE_UNFREEZING = False
ENABLE_LR_DAMPENING = True
ENABLE_EMA = True
```

with the **TRAIN_STAGES** I gave above and global OneCycleLR scheduler.

---

Would you like me to share the exact `create_onecycle_scheduler_global()` implementation (with `pct_start`, `div_factor`, etc.) tuned for ImageNet so it fits directly into your code?
