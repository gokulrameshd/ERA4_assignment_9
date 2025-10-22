Nice — below is a practical, hardware-aware progressive-resizing plan tuned for your **RTX 5060 Ti 16 GB**. I give three sensible recipes (Fast / Balanced / Thorough) you can pick from depending on how much wall-clock time you want to spend vs final accuracy. For each stage I list **image size, per-GPU batch size, epochs**, plus recommended **LR / optimizer / scheduler / extras** and notes about grad accumulation and dataloader settings.

---

## Quick rules-of-thumb (applies to all recipes)

* ResNets accept variable input sizes if `avgpool = AdaptiveAvgPool2d((1,1))`. Good.
* Use **AMP** (autocast + GradScaler). Always.
* Use `channels_last` memory format (`model.to(memory_format=torch.channels_last)` and `inputs = inputs.to(..., memory_format=torch.channels_last)` if possible).
* Prefer `persistent_workers=True`, `pin_memory=True`, and `num_workers=4–8` (start with 4, increase if CPU can handle).
* If using `timm.Mixup`, ensure `drop_last=True` and batch size is even.
* If you want an *effective* larger batch than GPU memory allows, use **gradient accumulation**:
  `accum_steps = effective_batch // per_gpu_batch` (must be integer).
* OneCycleLR expects per-step `scheduler.step()` after `optimizer.step()`.

---

# Option A — FAST (get decent results quickly)

Goal: get to a good model fast (~few hours). Less final accuracy but cheap.

Stages:

1. **Stage 1 (coarse)**

   * img_size = **112**
   * per-GPU batch = **512** (if OOM → 256)
   * epochs = **10**
2. **Stage 2 (standard)**

   * img_size = **224**
   * per-GPU batch = **256**
   * epochs = **8**

Hyperparams:

* Optimizer: **SGD**(lr scaled by batch: `lr = 0.1 * (batch/256)`), momentum 0.9, weight_decay 1e-4
  (or **AdamW** lr = 5e-4 for faster convergence if you prefer)
* Scheduler: **OneCycleLR** per stage (total_steps = epochs*steps_per_epoch), max_lr = chosen lr
* Mixup: α=0.2 in Stage 1 only (optional)
* Label smoothing: 0.1
* Expected: fast training, moderate final accuracy.

When to use: quick experiments, hyperparameter sweeps.

---

# Option B — BALANCED (recommended for RTX 5060 Ti)

Goal: good accuracy / decent training time. This is my recommended default.

Stages:

1. **Stage 1 (coarse)**

   * img_size = **128**
   * per-GPU batch = **512 → if OOM set 256**
   * epochs = **10**
2. **Stage 2 (intermediate)**

   * img_size = **160**
   * per-GPU batch = **384 → if OOM set 256**
   * epochs = **8**
3. **Stage 3 (full)**

   * img_size = **224**
   * per-GPU batch = **256**
   * epochs = **12**

Hyperparams:

* Stage-wise optimizer: **SGD** with momentum 0.9, wd 1e-4. Use linear scaling: `base_lr = 0.1 * (batch/256)`. Example: batch=256 → lr=0.1; batch=512 → lr=0.2 (but be conservative; clamp to 0.1–0.4).
* Scheduler: OneCycleLR per stage (pct_start = 0.3) or short Cosine for longer phases.
* Mixup/CutMix: α=0.2 / 1.0 in Stages 1–2; reduce/disable in final stage if you want crisp eval.
* Label smoothing 0.1 throughout.
* Grad accumulation: not required unless you choose a larger effective batch.
* Expected: good balance — should reach high single-run accuracy in reasonable time.

Why recommended: starts with cheap resolution to learn general filters, progressively adds detail; avoids long compilation/reload overhead by caching dataloaders per size and compiling once per stage.

---

# Option C — THOROUGH (max accuracy, more time)

Goal: squeeze best performance; longer training.

Stages:

1. **Stage 1**: 160px, batch 384, epochs 15
2. **Stage 2**: 224px, batch 256, epochs 30
3. **Stage 3 (optional refine)**: 320px, batch 64–128, epochs 10

Hyperparams:

* Optimizer: start with **SGD** + OneCycle/long Cosine; optionally switch to **AdamW** for stage 1 then SGD for final polish.
* Use **strong augmentations** (RandAugment, Mixup/CutMix early), label smoothing 0.1.
* Consider EMA (weight averaging) for final eval.
* Use gradient accumulation if you want to simulate 1024 batch at 224.

When to use: final production model or when you want top-tier final accuracy.

---

## Practical recommended per-stage config (Balanced = default)

Copy-paste friendly:

```py
stages = [
    {"img_size": 128, "batch_size": 512, "epochs": 10, "mixup": True},
    {"img_size": 160, "batch_size": 384, "epochs": 8,  "mixup": True},
    {"img_size": 224, "batch_size": 256, "epochs": 12, "mixup": False},
]
```

Scheduler & LR example for SGD:

* per-stage `max_lr = 0.1 * (batch_size / 256)`
* Example: batch=512 → max_lr=0.2 (clamp to ≤0.4). If unstable, halve `max_lr`.

Use:

```py
optimizer = SGD(model.parameters(), lr=max_lr, momentum=0.9, weight_decay=1e-4)
scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=epochs*steps_per_epoch, pct_start=0.3)
```

If you want AdamW:

* lr ≈ **5e-4 – 1e-3** for head; lower for backbone when fine-tuning.

---

## Dataloader / system settings (critical)

* `num_workers = 4` (increase to 6 only if CPU and disk can keep up)
* `pin_memory=True`
* `persistent_workers=True` (and cache dataloader per size)
* `drop_last=True` for training (especially if mixup requires even batch), `drop_last=False` for validation
* Plotting: save every 5 epochs to avoid slowdowns

---

## Grad accumulation example (simulate larger effective batch)

If you want effective_batch = 1024 but per-GPU batch = 256:

```py
effective_batch = 1024
per_step = 256
accum_steps = effective_batch // per_step  # = 4
# in training loop: divide loss by accum_steps and call optimizer.step() only every accum_steps
```

---

## How to choose among the three

* If you want **fast experiments** → Option A
* If you want **best tradeoff** → Option B (my recommendation)
* If you chase **top accuracy** and have time → Option C

---

## Extra tips to speed things (so per-epoch time is low)

* **Cache dataloaders per size** — recreate only when `img_size` changes.
* **Compile once** with `torch.compile(model, dynamic=True)` (or compile per stage if warnings occur).
* **Use channels_last** and `torch.backends.cudnn.benchmark = True`.
* **Plot infrequently** (every 5–10 epochs).
* Keep `num_workers` tuned to CPU/Disk I/O.

---

If you want, I can:

* produce a **one-line config** you can drop into your script for the Balanced recipe, or
* modify your progressive script to use the Balanced stages and automatically compute `accum_steps` and LR scaling.

Which do you want me to generate next?
