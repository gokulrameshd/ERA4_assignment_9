Excellent observation — and this is **a very common misunderstanding** when moving from fixed-size training to **progressive resizing**.
Even though you’re using **the same final image size (224)** and **same batch size (256)** at that point, the **epoch time increases (1.79 m → 2.8 m)** because of *how progressive resizing affects the training pipeline and resource utilization*.

Let’s break this down carefully 👇

---

## 🧩 Why Progressive Resizing Epochs Take Longer

Even when the **image size and batch size** look the same, the following hidden factors make progressive resizing slower:

---

### **1. DataLoader Rebuilding + Transform Overhead**

In progressive resizing, you **rebuild your `DataLoader` and transforms** at each stage (`img_size=112 → 224`).

That means at each resize stage:

* New transform pipelines are recreated (`RandomResizedCrop`, `Resize`, `ToTensor`, `Normalize`, etc.).
* PIL or torchvision’s transforms have to **resize every image again**.
* The DataLoader’s worker processes get reinitialized.

This introduces **significant CPU overhead**, especially if your dataset is large (e.g., ImageNet).

💡 In standard training, transforms and workers are **initialized once** and reused for all 30 epochs → zero rebuild cost per epoch.

---

### **2. GPU Kernel Cache Warmup**

When you switch image sizes dynamically:

* PyTorch’s CUDA kernels for convolutions, GEMMs, and batchnorm layers get re-tuned.
* CUDNN reselects the best algorithm for new input shapes.
* This causes the first few iterations after resizing to run slower.

💡 Standard training (fixed 224) stays fully optimized and cached from epoch 1 → consistent fast runtime.

---

### **3. Additional Augmentation (Mixup, CutMix, Label Smoothing, etc.)**

You’re using:

```python
use_mixup_fn = True
mixup_fn = get_mixup_fn(...)
```

That introduces **extra CPU–GPU tensor ops per batch**:

* Generates random λ values,
* Blends images and labels,
* Performs elementwise operations before feeding to the model.

This increases compute per iteration by **5–15%** even with the same batch size.

💡 Standard training doesn’t perform these operations unless you explicitly add them.

---

### **4. Learning Rate Scheduler Restarts + Warmup**

In progressive resizing, you likely reinitialize:

```python
scheduler = OneCycleLR(...)
```

per stage.

That reintroduces **warmup and cosine annealing** every few epochs.
This changes how long each optimizer step takes slightly —
because of extra Python-side scheduler calls and more bookkeeping (momentum updates, etc.).

---

### **5. IO Bottlenecks Increase**

When image size doubles (112 → 224):

* Image tensors are **4× larger** (since area ∝ size²).
* Each batch takes longer to copy from CPU → GPU.
* DataLoader workers take longer to preprocess and augment each batch.

💡 If your CPU or disk isn’t fast enough (common in progressive runs since workers restart each stage), the GPU spends time *waiting for data*.

You can confirm this by running:

```bash
nvidia-smi dmon
```

and noticing GPU utilization dips between batches.

---

### **6. Stage Transition Overhead**

When you move from one stage to another (e.g. after `img_size=112`), the first epoch of the next stage often does:

* Model reallocation on GPU (if you change layers or batch norm).
* DataLoader recreation.
* Optimizer state adjustments.

So the early epochs after a new stage are slower than stable ones.

---

## 🧠 TL;DR Summary Table

| Cause               | Standard Training | Progressive Resizing                     |
| ------------------- | ----------------- | ---------------------------------------- |
| Dataloader rebuild  | ❌ once only       | ✅ every stage                            |
| Transform cost      | low (cached)      | high (resizing new images)               |
| CUDNN kernel tuning | done once         | re-run at each stage                     |
| Mixup/CutMix        | optional          | often enabled                            |
| IO overhead         | stable            | increased (larger imgs + worker restart) |
| Scheduler warmup    | single curve      | multiple restarts                        |
| Epoch time          | ~1.8 min          | ~2.8 min                                 |

---

## ⚙️ How to Speed It Up

1. **Cache resized datasets**
   → Precompute images at 112 px and 224 px once, and load directly from disk (e.g., store in LMDB or WebDataset format).

2. **Persist workers across stages**
   → Instead of rebuilding DataLoader, reuse worker pool and only change the transform’s resize function dynamically.

3. **Use `torch.backends.cudnn.benchmark = True`**
   → Lets cuDNN cache optimal kernels even for dynamic shapes.

4. **Profile DataLoader time vs GPU compute time**

   ```python
   for i, batch in enumerate(dataloader):
       start = time.time()
       # your training step
       end = time.time()
       print("Step time:", end - start)
   ```

   If `GPU Utilization < 70%`, your bottleneck is CPU / IO.

5. **Use smaller augmentations or turn off mixup for early stages**
   → Early stages (56 px / 112 px) often don’t benefit much from Mixup but still add cost.

---

## ✅ In Short

Even though both runs say `img_size=224` and `batch_size=256`, progressive resizing’s extra costs (transform rebuild, mixup, IO, cudnn tuning) explain the **~50% longer per-epoch time**.

---

Would you like me to show a **CPU–GPU timing breakdown (ms per batch)** version of your training loop so you can *quantitatively measure* where time is being lost (DataLoader vs GPU compute)?
