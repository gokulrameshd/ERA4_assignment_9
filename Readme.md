# 🗂️ Project Folder Structure — Deep Learning Training Pipeline

This document provides an overview of the folder hierarchy, describing the purpose of each file and directory in the training pipeline.

---

## 📁 Folder Layout

```
training_pipeline/
│
├── data_loader.py
│   ├─ Loads training & validation datasets using torchvision
│   ├─ Applies GPU-accelerated transforms (if available)
│   └─ Returns DataLoaders + number of classes
│
├── model.py
│   ├─ Builds the model architecture (ResNet-18)
│   └─ Replaces the final fully connected layer with custom num_classes
│
├── cyclic_scheduler.py
│   ├─ Implements OneCycleLR scheduler
│   └─ Automatically sets LR schedule for each training step
│
├── lr_finder.py
│   ├─ Implements a modern Learning Rate Finder (Smith 2017)
│   ├─ Uses AMP + GradScaler for stability and speed
│   ├─ Detects optimal LR and saves loss vs LR plots
│   └─ Restores model/optimizer automatically
│
├── train.py
│   ├─ Main training pipeline
│   ├─ Integrates LR Finder + OneCycleLR
│   ├─ Tracks accuracy, loss, LR, and momentum
│   ├─ Supports AMP, torch.compile(), and TF32 optimization
│   ├─ Saves best and last model checkpoints
│   └─ Generates live plots for training progress
│
├── dataloader_optimization_readme.md
│   ├─ Describes DataLoader performance tuning
│   └─ Includes benchmark results and parameter explanations
│
├── README_LR_FINDER.md
│   ├─ Full documentation for the Learning Rate Finder module
│   └─ Explains usage, parameters, and interpretation
│
├── plots/
│   ├─ lr_finder_plot.png
│   ├─ accuracy_live.png
│   ├─ loss_live.png
│   ├─ lr_live.png
│   └─ momentum_live.png
│
├── training_log.txt
│   ├─ CSV-formatted training history (Loss, Acc, LR, etc.)
│
├── best_weights.pth
│   └─ Best model checkpoint (highest validation accuracy)
│
└── last_weights.pth
    └─ Last saved model after final epoch
```

---

## 🚀 Entry Point

| File | Purpose |
|------|----------|
| **`train.py`** | The **main entry point** for the entire pipeline. It orchestrates data loading, model creation, learning rate finding, scheduling, training, validation, and plotting. |

### ▶️ How to Run

You can launch the complete training workflow from the terminal:

```bash

# Run the main training script
python train.py
```

### Optional Arguments (to modify in script)

| Variable | Location | Description |
|-----------|-----------|-------------|
| `DATA_DIR` | inside `train.py` | Path to dataset root (`train/`, `val/` subfolders required) |
| `BATCH_SIZE` | inside `train.py` | Batch size for training and validation |
| `NUM_EPOCHS` | inside `train.py` | Total number of epochs |
| `DEVICE` | inside `train.py` | `"cuda"` or `"cpu"` |
| `PLOTS_DIR` | inside `train.py` | Directory to save plots (default: `"plots/"`) |

### Example Output

Running `python train.py` will:
- ✅ Run **Learning Rate Finder**  
- ⚙️ Configure **OneCycleLR scheduler**  
- 🧠 Train the **ResNet model**  
- 🖼️ Generate **live accuracy/loss/LR plots**  
- 💾 Save `best_weights.pth`, `last_weights.pth`, and `training_log.txt`

---

## 📖 Overview

| Component | Description |
|------------|--------------|
| **Data Loader** | Efficient dataset loader with pinned memory, prefetching, and worker optimization. |
| **Model** | ResNet-based model with a replaceable classifier head. |
| **LR Finder** | Finds optimal learning rate using controlled exponential LR sweeps. |
| **Scheduler** | OneCycleLR policy for smooth convergence. |
| **Training** | Full AMP-optimized pipeline with progress tracking, checkpoints, and visualization. |
| **Documentation** | Markdown files describing optimization principles and LR Finder internals. |
| **Outputs** | Logs, plots, and weight checkpoints for reproducible experiments. |

---
