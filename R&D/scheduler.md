Excellent — that’s a key choice.
For **training ResNet-50 from scratch on ImageNet-1K**, the **learning-rate scheduler** you pick will strongly affect convergence speed and final accuracy.

Let’s go through what works best **for modern setups (2020–2025)** and why 👇

---

## 🔝 **Best Overall: Cosine Annealing with Warmup**

### ✅ **Recommended**

```python
import torch

warmup_epochs = 5
def cosine_lr_scheduler(optimizer, num_epochs, warmup_epochs=5):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + torch.cos(torch.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### 💡 Why it’s ideal

* **Smooth decay**: avoids abrupt drops like StepLR.
* **Strong early learning** but **fine-tuning at end**, improving final accuracy.
* **Warmup** stabilizes gradients early when BatchNorm stats are noisy.
* Used by: **timm**, **DeiT**, **ConvNeXt**, **EfficientNet**, etc.
* Works beautifully with **SGD**, **AdamW**, or **Lion** optimizers.
* Boosts Top-1 by ~+0.3–0.5% over step decay.

---

## ⚙️ Typical hyperparameters

| Parameter     | Typical value           | Notes                 |
| ------------- | ----------------------- | --------------------- |
| Initial LR    | 0.1 for SGD (batch 256) | scale ∝ batch size    |
| Warmup epochs | 5–10                    | Linear increase       |
| Max epochs    | 100–300                 | Depends on compute    |
| Min LR        | 1e-6                    | Don’t let it hit zero |

---

## ⚖️ **Alternatives (if you want variations)**

### 1️⃣ **One-Cycle LR** (`torch.optim.lr_scheduler.OneCycleLR`)

* Great if you use **AdamW**.
* Rapid rise → slow decay schedule.
* Works well for shorter runs (≤100 epochs).
* May need fine-tuned max_lr to avoid instability.

### 2️⃣ **StepLR (Legacy)**

* Used in original ResNet papers.
* Decays LR by 0.1 every 30 epochs.
* **Outdated** now — abrupt jumps can hurt convergence stability.

### 3️⃣ **Cosine Annealing with Restarts (SGDR)**

* Periodically resets LR to high value (T₀, T_mult).
* Can slightly improve generalization but adds complexity.
* Rarely used for full ImageNet runs now.

---

## 🚀 **Summary Recommendation**

| Use case                      | Scheduler                      | Why                                        |
| ----------------------------- | ------------------------------ | ------------------------------------------ |
| Standard ImageNet-1K training | **CosineAnnealingLR + Warmup** | Best balance of speed, stability, accuracy |
| Short or fine-tuning runs     | **OneCycleLR**                 | Faster convergence                         |
| Legacy baseline reproduction  | **StepLR (30, 0.1)**           | Matches old ResNet papers                  |

---

✅ **Verdict:**

> For your **ResNet-50 ImageNet-1K training from scratch**, use
> **Cosine Annealing with 5–10 epoch warmup** and **final LR floor ≈ 1e-6**.

