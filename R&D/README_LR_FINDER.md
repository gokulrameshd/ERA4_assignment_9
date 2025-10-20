# 📘 Optimized Learning Rate Finder (`lr_finder.py`)

## 🔍 Overview

The **Learning Rate Finder (LRFinder)** is a specialized diagnostic tool designed to automatically determine the most effective learning rate (LR) for training deep learning models.  
Instead of guessing, the LR Finder runs a brief training session where the learning rate increases gradually from a very low value to a high one, observing how the loss changes at each step.

The resulting **Learning Rate vs. Loss curve** helps pinpoint the optimal LR range — the sweet spot where learning is fast, stable, and efficient.

---

## 🧩 Approach and Workflow

The core idea is based on the **LR Range Test** (Smith, 2017).  
It performs a short run with exponentially or linearly increasing learning rates, and records how loss evolves.

| Step | Stage | Description |
|------|--------|-------------|
| 1️⃣ | Initialize | Saves initial model & optimizer states |
| 2️⃣ | Range Test | Gradually increases LR for `num_iter` iterations |
| 3️⃣ | Loss Smoothing | Applies Savitzky–Golay or EMA to reduce noise |
| 4️⃣ | Early Stop | Automatically halts if loss diverges too fast |
| 5️⃣ | Analyze + Plot | Finds the steepest loss slope (best LR) |
| 6️⃣ | Reset | Restores original model/optimizer weights |

---

## ⚙️ Key Features and Enhancements

| Feature | Description | Benefit |
|----------|--------------|----------|
| 🧮 **Mixed Precision** | Uses `torch.amp.autocast` & `GradScaler` | 2× faster LR tests on GPU, lower VRAM usage |
| 🧠 **Savitzky–Golay Smoothing** | Uses polynomial filtering (if SciPy available) | Smooths loss curve, avoids spurious LR jumps |
| 💾 **CSV Logging** | Saves LR/Loss data to `lr_finder_log.csv` | Enables reproducibility & analysis |
| 🧯 **Adaptive Early Stopping** | Detects explosive loss and halts automatically | Saves time and GPU cycles |
| 📈 **Gradient of log(Loss)** | Computes gradient on log-smoothed loss | Finds true steepest descent point |
| 🎨 **Color Coded Plot Zones** | Visual zones: Yellow=Low, Green=Optimal, Red=High | Easier interpretation |
| ♻️ **Auto Reset** | Restores model & optimizer after test | Prevents corrupting main training |
| 🧩 **Optional Dependencies** | Gracefully falls back if `pandas` or `scipy` missing | Portable to minimal environments |

---

## 🧮 Constructor and Parameters

```python
class LRFinder:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device=None,
        memory_cache=True,
        cache_dir=None,
        scaler=None
    )
```

| Parameter | Description | Default |
|------------|--------------|----------|
| `model` | Model to test | Required |
| `optimizer` | Optimizer instance (SGD/Adam etc.) | Required |
| `criterion` | Loss function | Required |
| `device` | `"cuda"` or `"cpu"` | Inferred from model |
| `memory_cache` | Cache model & optimizer for reset | True |
| `cache_dir` | Optional path to store logs | None |
| `scaler` | Custom GradScaler for AMP | Auto-created |

---

## ⚡ `range_test()` — Core LR Sweep

```python
lr_finder.range_test(
    train_loader,
    start_lr=1e-7,
    end_lr=1,
    num_iter=100,
    step_mode="exp",
    smooth_f=0.05,
    diverge_th=5.0,
    use_amp=True,
    adaptive_stop=True
)
```

| Parameter | Purpose | Example | Notes |
|------------|----------|----------|-------|
| `train_loader` | DataLoader providing mini-batches | — | Required |
| `start_lr` | Starting LR | 1e-7 | Should be stable |
| `end_lr` | Final LR | 1 | Should cause divergence |
| `num_iter` | Iterations for LR test | 100 | Higher → smoother |
| `step_mode` | LR increase pattern (`exp` / `linear`) | `"exp"` | `"exp"` is preferred |
| `smooth_f` | EMA smoothing factor | 0.05 | Reduces noise |
| `diverge_th` | Divergence threshold | 5.0 | Stops when unstable |
| `adaptive_stop` | Detects sudden spikes | True | Saves time |

---

## 📊 `plot()` — Visualization and Suggested LR

```python
best_lr = lr_finder.plot(
    save_path="plots/lr_finder_plot.png",
    suggest=True,
    save_csv=True,
    annotate=True,
    auto_reset=True
)
```

### Plot Details:
- **X-axis**: Learning Rate (log scale)
- **Y-axis**: Smoothed Loss
- **Red Dashed Line**: Suggested LR (steepest slope)
- **Color Zones**:
  - 🟡 **Too Low LR** → model learns too slowly
  - 🟢 **Optimal Zone** → loss drops rapidly & stable
  - 🔴 **Too High LR** → loss diverges / unstable

---

## 🧠 `get_suggested_lr()` — Direct LR Retrieval

If you want to skip plotting and just get the suggested LR:

```python
best_lr = lr_finder.get_suggested_lr()
```

Returns: Best LR found based on the minimum gradient of log-smoothed loss.

---

## 📈 Example Workflow

```python
from lr_finder import LRFinder

# 1. Initialize
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")

# 2. Run range test
lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=1, num_iter=100)

# 3. Plot & Get best LR
best_lr = lr_finder.plot(save_path="plots/lr_finder_plot.png")

# 4. Reset before training
lr_finder.reset()

print(f"Suggested LR: {best_lr:.6f}")
```

---

## 🧩 Outputs

| File | Description |
|------|--------------|
| `lr_finder_plot.png` | Learning Rate vs. Smoothed Loss curve |
| `lr_finder_log.csv` | Saved LR–Loss pairs |
| `best_lr.txt` | Stores the best LR for later reuse |
| (Optional) Model restored to initial weights | Prevents corruption for next training |

---

## 🔍 Example Terminal Log

```
Finding LR: 100%|███████████████████████████████████████| 100/100 [00:11<00:00, 9.18it/s]
✅ LR range test complete! Use `.plot()` to visualize.
✅ Suggested Learning Rate: 7.74E-03
📈 LR Finder plot saved to plots/lr_finder_plot.png
🧾 Saved lr history to lr_finder_log.csv
♻️ Model and optimizer state restored after LR test.
```

---

## 🧭 Interpreting the Curve

| Region | Description | Interpretation |
|--------|--------------|----------------|
| 🟡 **Too Low** | Loss almost flat | Model learns too slowly |
| 🟢 **Optimal** | Loss drops steeply | Best LR range |
| 🔴 **Too High** | Loss rises sharply | LR too large — unstable gradients |

The **optimal LR** is typically **just before** the loss starts to rise again.

---

## 🚀 Impact on Training

| Aspect | Before LR Finder | After LR Finder |
|---------|------------------|-----------------|
| LR Choice | Manual guess | Data-driven optimal |
| Convergence | Slow or oscillating | 2–3× faster |
| Stability | Risk of divergence | Controlled and stable |
| Accuracy | Inconsistent | Early, smooth improvement |

---

## ⚠️ Common Issues & Fixes

| Issue | Cause | Solution |
|--------|--------|----------|
| Loss diverges immediately | End LR too high | Lower `end_lr` (e.g., 1 → 0.1) |
| Flat loss curve | Dataset too small | Increase `num_iter` |
| No suggested LR | Too few samples | Increase training batches |
| CUDA OOM | Batch size too large | Reduce `batch_size` temporarily |
| No Savitzky filter | SciPy missing | Auto-falls back to moving average |

---

## 💡 Best Practices

✅ Run LR Finder **before** every major training change (e.g., new model or dataset).  
✅ Use **smaller batch sizes** for finer LR resolution.  
✅ Apply the **suggested LR as `max_lr`** in `OneCycleLR`.  
✅ Store the plot for **future reference and comparison**.

---

## 📜 References

- Leslie N. Smith, *"Cyclical Learning Rates for Training Neural Networks"*, 2017.  
- FastAI’s `lr_find()` implementation (inspiration).  
- PyTorch LR Finder concept.

---

## 🎯 Summary

The **Optimized LR Finder** automates learning rate selection using numerical stability, gradient-based detection, and loss smoothing.  
It’s **fast**, **AMP-optimized**, and **reproducible**, making it ideal for modern training pipelines.

✅ Result: **Faster convergence, fewer experiments, and better models.**

---
