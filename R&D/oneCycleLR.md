Nice — good choice. OneCycleLR can be excellent when you pick `max_lr` well. Below I’ll give a compact, practical workflow you can run immediately, plus safe defaults and example code you can drop into your training loop.

# TL;DR (quick rules)

* **Best method:** Run a **LR range test (Smith’s LR finder)** and pick `max_lr` at (or just after) the LR where training loss is lowest / still decreasing before it blows up.
* **Quick default (if you can’t run finder):**

  * **SGD (momentum=0.9)** at reference batch 256 → `max_lr ≈ 0.1` (scale linearly with batch size: `max_lr *= batch_size / 256`).
  * **AdamW** at reference batch 256 → `max_lr ≈ 1e-3` (scale similarly).
* **OneCycle hyperparams I recommend:** `pct_start=0.3`, `anneal_strategy='cos'`, `div_factor=25`, `final_div_factor=1e4`.

---

# Why run an LR range test?

The LR finder runs the model for a short time while exponentially increasing LR and records training loss. The correct `max_lr` is where loss is still decreasing (often near the minimum loss point). This is the most reliable way to choose `max_lr` for your specific model, data, batch size and augmentations.

---

# Step-by-step LR Finder + OneCycle setup (PyTorch)

1. Do a short LR range test (e.g., 1000 iterations or 1–2 epochs on a fraction of data).
2. Plot `loss` vs `lr` (log scale). Find the LR where loss starts decreasing steeply and stays low. Avoid the LR where loss rises sharply.
3. Pick `max_lr` ≈ the LR at **minimum** loss or slightly (×1–3) above the min-but-still-stable region. A conservative rule: `max_lr = lr_at_min * 3` (choose lower multiplier if unstable).
4. Set OneCycleLR with that `max_lr`. Use `pct_start=0.3`, `anneal_strategy='cos'`.

---

# Example LR-finder code (compact, runnable)

```python
# LR range test (minimal implementation)
import torch
import math
from collections import deque
import matplotlib.pyplot as plt

def lr_range_test(model, optimizer, loss_fn, dataloader, device,
                  start_lr=1e-7, end_lr=10, num_iter=1000, smooth_beta=0.98):
    model.train()
    lr_mult = (end_lr / start_lr) ** (1.0 / num_iter)
    lr = start_lr
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    avg_loss = 0.0
    best_loss = 1e9
    losses = []
    lrs = []
    it = 0
    data_iter = iter(dataloader)
    while it < num_iter:
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, targets = next(data_iter)

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss_value = float(loss.item())

        # smooth loss
        avg_loss = smooth_beta * avg_loss + (1 - smooth_beta) * loss_value
        debias_loss = avg_loss / (1 - smooth_beta ** (it + 1))

        # record
        losses.append(debias_loss)
        lrs.append(lr)

        # update best
        if debias_loss < best_loss:
            best_loss = debias_loss

        # step
        loss.backward()
        optimizer.step()

        # update lr
        lr *= lr_mult
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        it += 1

    return lrs, losses

# Usage (pseudo):
# model, optimizer, loss_fn, train_loader, device already defined
# lrs, losses = lr_range_test(model, optimizer, loss_fn, train_loader, device, num_iter=1000)
# plt.semilogx(lrs, losses); plt.xlabel('lr'); plt.ylabel('loss')
```

**How to pick `max_lr` from the plot**

* Find LR where loss **stops decreasing** and starts rising or becomes noisy.
* A good `max_lr` is slightly BELOW that rising point, often near the minimum loss LR.
* If you want an automatic heuristic: pick LR at the **min loss** and set `max_lr = lr_min * 3` (or `*1.5` if you saw instability).

---

# Example OneCycleLR config (put into training loop)

```python
# Typical training config
epochs = 90
steps_per_epoch = len(train_loader)
total_steps = epochs * steps_per_epoch

# Suppose we've picked max_lr from the LR finder
max_lr = picked_max_lr  # scalar or a list for param groups

optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=max_lr,
    total_steps=total_steps,
    pct_start=0.3,
    anneal_strategy='cos',
    cycle_momentum=True,      # if using momentum optimizer
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=25.0,          # initial_lr = max_lr / div_factor
    final_div_factor=1e4      # final lr = max_lr / final_div_factor
)

# In your training loop:
for epoch in range(epochs):
    for batch in train_loader:
        # forward, backward, optimizer.step()
        scheduler.step()
```

Notes:

* `div_factor=25` makes initial lr `max_lr/25` (so model warms up from low lr to `max_lr`).
* `final_div_factor=1e4` causes the LR to end very small (helpful for fine-tuning at the end).
* If you use **AdamW**, consider `cycle_momentum=False` and adjust `div_factor` down (e.g., 10–20).

---

# Concrete numeric examples

Assume reference `batch_ref = 256`.

**If you use SGD (momentum 0.9)**:

* Reference `max_lr_ref = 0.1` for batch 256.
* For your batch size `B`: `max_lr = 0.1 * (B / 256)`.

  * e.g., if `B=64`: `max_lr = 0.1 * (64/256) = 0.025`.
* After LR-finder you might end up with `picked_max_lr` near that or 2–4× higher — trust the LR finder plot.

**If you use AdamW**:

* Reference `max_lr_ref = 1e-3` for batch 256.
* Scale: `max_lr = 1e-3 * (B / 256)`.

  * e.g., `B=64` → `max_lr = 2.5e-4`.
* AdamW often prefers ~1e-3 to 3e-4 ranges; validate with LR finder.

---

# Practical tips & gotchas

* **Use the same augmentation/settings** during LR-finder that you’ll use in training (Mixup, CutMix, large augmentations change loss curves).
* Keep the LR-finder run short (1000–2000 iterations) and on **training** data. Don’t use validation.
* If your loss is noisy, smooth it (as in the code). Choose the LR at minimum of smoothed loss.
* If using large batch sizes and distributed training, scale LR linearly with batch size but validate with LR finder.
* OneCycleLR is sensitive to a *bad* `max_lr` (divergence early). Start conservative if unsure.

---

## Final suggestion for your RTX 5060 Ti

You likely have limited batch size (maybe 64 or smaller). I recommend:

1. Run the short LR-finder above for 1000 iterations with your actual `train_loader` and augmentations.
2. Inspect plot: choose `max_lr` near the smoothed-min LR (or `*1.5–3` if stable).
3. Use OneCycleLR with `pct_start=0.3`, `div_factor=25`, `final_div_factor=1e4`, `anneal_strategy='cos'`.

If you want, tell me:

* which optimizer (SGD or AdamW),
* your batch size,
* approx `epochs` and whether you will use Mixup/CutMix,

and I’ll give a concrete numeric `max_lr` recommendation and the exact `OneCycleLR` call with numbers filled in.
