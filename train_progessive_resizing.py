"""
train.py
--------------
End-to-end ResNet-34 training pipeline with:
✅ LR Finder (AMP + Auto reset + CSV export)
✅ OneCycleLR Scheduler
✅ Dynamic tqdm bars (train + val)
✅ ETA + GPU memory monitor
✅ Dynamic live plots (Acc/Loss/LR/Momentum)
✅ Best & Last weights saving
✅ Smart LR auto-scaling from LR Finder
Optimized: TF32, torch.compile, modern torch.amp usage
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from datetime import datetime

from data_loader import get_dataloaders
from model import create_model
from hyper_parameter_modules import create_onecycle_scheduler,make_optimizer_and_scheduler
from lr_finder_custom import LRFinder
# Modern AMP imports
from torch.amp import GradScaler
from train_test_modules import train_one_epoch_imagenet, validate_imagenet, save_plot

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils._sympy")


# ==============================================================
# ⚙️ CONFIG
# ==============================================================
DATA_DIR = "./sample_data"
NUM_CLASSES = 100
BATCH_SIZE = 256    
IMG_SIZE = 224
NUM_EPOCHS = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_BEST = "./train_progressive_resizing/best_weights.pth"
SAVE_LAST = "./train_progressive_resizing/last_weights.pth"
CSV_LOG_FILE = "./train_progressive_resizing/training_log.csv"
TXT_LOG_FILE = "./train_progressive_resizing/training_log.txt"
PLOTS_DIR = "./train_progressive_resizing/plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

# Performance flags
torch.backends.cudnn.benchmark = True
# enable TF32 on Ampere/Ada GPUs for faster matmuls (negligible accuracy change)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# prefer Tensor Cores where possible
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    print("Failed to set float32 matmul precision")
    pass
# ==============================================================




# ==============================================================
# 🚀 MAIN TRAIN FUNCTION
# ==============================================================
def train_1():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print(f"\n🚀 Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} on {DEVICE}")

    # Progressive resizing stages
    stages = [
        {"img_size": 64, "batch_size": 1024, "epochs": 5, "mixup": True},#512 10
        # {"img_size": 160, "batch_size": 512, "epochs": 8,  "mixup": True},#384
        {"img_size": 64, "batch_size": 512, "epochs": 5, "mixup": False},#256 12
    ]
    # stages = [
    #     {"img_size": 56, "batch_size": 512, "epochs": 10},   # Stage 0: learn coarse features - 8
    #     {"img_size": 112, "batch_size": 256, "epochs": 10},   # Stage 1: learn coarse features - 8
    #     {"img_size": 224, "batch_size": 128, "epochs": 10},  # Stage 2: standard ImageNet resolution
    #     # Optionally, add 320 stage if GPU memory allows
    #     {"img_size": 320, "batch_size": 64, "epochs": 10}, # Stage 3: fine-tune the model
    # ]
    # -----------------------------------------------------------
    # 🧠 Model setup
    model = create_model(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)

    # Try to compile the model (PyTorch 2.x). Safe to ignore failures.
    try:
        model = torch.compile(model)
        print("⚡ model compiled with torch.compile()")
    except Exception:
        print("Failed to compile model with torch.compile(), continuing without compilation")
        pass

    # Criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)



    # ============================================================
    # Prepare a single GradScaler to pass through epochs (keeps state)
    scaler = GradScaler(device="cuda" if DEVICE == "cuda" else "cpu")

    # -----------------------------------------------------------
    # 📊 Training Loop
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": [], "mom": [], "train_time": []}
    prev_weights = None
    global_epoch = 0
    NUM_EPOCHS = sum(stage["epochs"] for stage in stages)
    with open(CSV_LOG_FILE, "w") as log:
        log.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,Learning_Rate,Momentum\n")

    train_start_time = time.time()
    for stage_idx, stage in enumerate(stages):
        IMG_SIZE = stage["img_size"]
        BATCH_SIZE = stage["batch_size"]
        NUM_EPOCHS_STAGE = stage["epochs"]
        use_mixup_fn = stage["mixup"]
        print(f"\n🔹 Stage {stage_idx+1}: IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={NUM_EPOCHS_STAGE}")

        # ----------------------------
        # Load dataloaders with new size
        try:
            if train_loader is not None:
                train_loader.close()
            if val_loader is not None:
                val_loader.close()
        except Exception:
            pass
        train_loader, val_loader, num_classes = get_dataloaders(DATA_DIR, BATCH_SIZE, IMG_SIZE)

    #      # ----------------------------
    #     # Reset optimizer for new stage
        optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

    #     # LR Finder only for first stage or if desired
        if stage_idx == 0:
            print("\n🔍 Running Learning Rate Finder...")
            lr_finder = LRFinder(model, optimizer, criterion, device=DEVICE, cache_dir=PLOTS_DIR)
            lr_finder.range_test(train_loader, start_lr=1e-6, end_lr=10, num_iter=100)
            best_lr, safe_lr = lr_finder.plot(
                save_path=os.path.join(PLOTS_DIR, f"lr_finder_stage_{IMG_SIZE}.png"),
                save_csv=True,
                suggest=True,
                annotate=True
            )
            if best_lr is None or not math.isfinite(best_lr) or best_lr <= 0:
                print("LR Finder returned invalid value. Falling back to 1e-3.")
                best_lr, safe_lr = 1e-3, 1e-3 * 0.3
            use_lr = float(max(1e-6, min(safe_lr, 0.1)))
            use_lr = 0.1
        else:
            # Optionally reduce LR slightly in higher stages
            use_lr *= 0.5
            print(f"Stage {stage_idx+1} LR set to {use_lr:.6f}")

    # # ----------------------------
        scheduler = create_onecycle_scheduler(
            optimizer=optimizer,
            max_lr=use_lr,
            train_loader_len=len(train_loader),
            epochs=NUM_EPOCHS_STAGE,
        )
        # optimizer, scheduler = make_optimizer_and_scheduler(model, BATCH_SIZE, NUM_EPOCHS_STAGE, steps_per_epoch)
    # ----------------------------
        # Load previous stage weights if any
        if prev_weights:
            model.load_state_dict(prev_weights)
            print(f"🔄 Loaded weights from previous stage")
    

    # ----------------------------
        # Stage training loop
        for epoch in range(NUM_EPOCHS_STAGE):
            global_epoch += 1
            epoch_start = time.time()

            train_loss, train_acc, scaler = train_one_epoch_imagenet(model, train_loader, optimizer, criterion, DEVICE,
                                                             scheduler, scaler,use_mixup_fn= use_mixup_fn,num_classes=num_classes)
            val_loss, val_acc = validate_imagenet(model, val_loader, criterion, DEVICE,num_classes=num_classes)

            current_lr = scheduler.get_last_lr()[0] if scheduler else use_lr
            current_mom = optimizer.param_groups[0].get("momentum", None)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)
            history["mom"].append(current_mom)

            epoch_time = time.time() - epoch_start
            total_train_time = time.time() - train_start_time
            history["train_time"].append(total_train_time)
            print(
                f"[Epoch {global_epoch+1:03}/{NUM_EPOCHS}] | ⏱️ {epoch_time/60:.2f}m | "
                f"LR: {current_lr:.6f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%"
            )

            # ---- SAVE MODELS ----
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), SAVE_BEST)
                prev_weights = model.state_dict()  # carry forward
                print(f"New Best Accuracy: {best_acc*100:.2f}% (saved as {SAVE_BEST})\033[0m")
                with open(TXT_LOG_FILE, "a") as log:
                    log.write(f"New Best Accuracy: {best_acc*100:.2f}% (saved as {SAVE_BEST})\033[0m \n ")

            torch.save(model.state_dict(), SAVE_LAST)

            # ---- LOG ----
            with open(TXT_LOG_FILE, "a") as log:
                log.write(
                    # f"{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{current_lr:.6f},{current_mom:.4f}\n"
                    f"[Epoch {global_epoch+1:03}/{NUM_EPOCHS}] | ⏱️ {epoch_time/60:.2f}m | "
                    f"LR: {current_lr:.6f} | Train Acc: {train_acc*100:.2f}% | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Loss: {val_loss:.4f} | "
                    f"Momentum: {current_mom:.4f} \n"
                )
            with open(CSV_LOG_FILE, "a") as log:
                log.write(f"{global_epoch:03},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{current_lr:.6f},{current_mom:.4f}\n")
            # ---- DYNAMIC PLOTS ----
            # epochs_so_far = range(1, epoch + 2)
            epochs_so_far = range(1, global_epoch + 1)


            save_plot(epochs_so_far, {"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]}, "Accuracy", "Epoch", "Accuracy", "accuracy_live.png",PLOTS_DIR)
            save_plot(epochs_so_far, {"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]}, "Loss", "Epoch", "Loss", "loss_live.png",PLOTS_DIR)
            save_plot(epochs_so_far, {"Learning Rate": history["lr"]}, "Learning Rate", "Epoch", "LR", "lr_live.png",PLOTS_DIR)
            save_plot(epochs_so_far, {"Momentum": history["mom"]}, "Momentum", "Epoch", "Momentum", "momentum_live.png",PLOTS_DIR)
            save_plot(history["train_time"], {"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]}, "Accuracy", "Time(s)", "Accuracy", "accuracy_time.png",PLOTS_DIR)
            save_plot(history["train_time"], {"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]}, "Loss", "Time(s)", "Loss", "loss_time.png",PLOTS_DIR)
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    # -----------------------------------------------------------
    print(f"\n🏁 Training Complete — Best Val Acc: {best_acc*100:.2f}%")
    print(f"✅ Best model: {SAVE_BEST}")
    print(f"✅ Last model: {SAVE_LAST}")
    print(f"🖼️ Live plots in: {PLOTS_DIR}")
    print(f"🏁 Training Time: {train_time/60:.2f}m")
    with open(TXT_LOG_FILE, "a") as log:
        log.write(f"🏁 Training Complete — Best Val Acc: {best_acc*100:.2f}%\n")
        log.write(f"✅ Best model: {SAVE_BEST}\n")
        log.write(f"✅ Last model: {SAVE_LAST}\n")
        log.write(f"🖼️ Live plots in: {PLOTS_DIR}\n")
        log.write(f"🏁 Training Time: {train_time/60:.2f}m\n")


def train_2():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print(f"\n🚀 Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} on {DEVICE}")

    # Progressive resizing stages
    stages = [
        {"img_size": 128, "batch_size": 1024, "epochs": 10, "mixup": True},#512 10
        {"img_size": 160, "batch_size": 512, "epochs": 8,  "mixup": True},#384
        {"img_size": 224, "batch_size": 256, "epochs": 12, "mixup": False},#256 12
    ]
    # stages = [
    #     {"img_size": 56, "batch_size": 512, "epochs": 10},   # Stage 0: learn coarse features - 8
    #     {"img_size": 112, "batch_size": 256, "epochs": 10},   # Stage 1: learn coarse features - 8
    #     {"img_size": 224, "batch_size": 128, "epochs": 10},  # Stage 2: standard ImageNet resolution
    #     # Optionally, add 320 stage if GPU memory allows
    #     {"img_size": 320, "batch_size": 64, "epochs": 10}, # Stage 3: fine-tune the model
    # ]
    # -----------------------------------------------------------
    #data loader
    train_loaders = {}
    val_loaders = {}
    for stage in stages:
        train_loader, val_loader, num_classes = get_dataloaders(DATA_DIR, stage["batch_size"], stage["img_size"])
        train_loaders[stage["img_size"]] = train_loader
        val_loaders[stage["img_size"]] = val_loader
    
    # 🧠 Model setup
    model = create_model(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)

    # Try to compile the model (PyTorch 2.x). Safe to ignore failures.
    try:
        model = torch.compile(model)
        print("⚡ model compiled with torch.compile()")
    except Exception:
        print("Failed to compile model with torch.compile(), continuing without compilation")
        pass

    # Criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)



    # ============================================================
    # Prepare a single GradScaler to pass through epochs (keeps state)
    scaler = GradScaler(device="cuda" if DEVICE == "cuda" else "cpu")

    # -----------------------------------------------------------
    # 📊 Training Loop
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": [], "mom": [], "train_time": []}
    prev_weights = None
    global_epoch = 0
    NUM_EPOCHS = sum(stage["epochs"] for stage in stages)
    with open(CSV_LOG_FILE, "w") as log:
        log.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,Learning_Rate,Momentum\n")

    train_start_time = time.time()
    for stage_idx, stage in enumerate(stages):
        IMG_SIZE = stage["img_size"]
        BATCH_SIZE = stage["batch_size"]
        NUM_EPOCHS_STAGE = stage["epochs"]
        use_mixup_fn = stage["mixup"]
        print(f"\n🔹 Stage {stage_idx+1}: IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={NUM_EPOCHS_STAGE}")

        # ----------------------------
        # Load dataloaders with new size
        try:
            if train_loader is not None:
                train_loader.close()
            if val_loader is not None:
                val_loader.close()
        except Exception:
            pass
        train_loader, val_loader = train_loaders[IMG_SIZE], val_loaders[IMG_SIZE]

    #      # ----------------------------
    #     # Reset optimizer for new stage
        optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

    #     # LR Finder only for first stage or if desired
        if stage_idx == 0:
            print("\n🔍 Running Learning Rate Finder...")
            lr_finder = LRFinder(model, optimizer, criterion, device=DEVICE, cache_dir=PLOTS_DIR)
            lr_finder.range_test(train_loader, start_lr=1e-6, end_lr=10, num_iter=100)
            best_lr, safe_lr = lr_finder.plot(
                save_path=os.path.join(PLOTS_DIR, f"lr_finder_stage_{IMG_SIZE}.png"),
                save_csv=True,
                suggest=True,
                annotate=True
            )
            if best_lr is None or not math.isfinite(best_lr) or best_lr <= 0:
                print("LR Finder returned invalid value. Falling back to 1e-3.")
                best_lr, safe_lr = 1e-3, 1e-3 * 0.3
            use_lr = float(max(1e-6, min(safe_lr, 0.1)))
            use_lr = 0.1
        else:
            # Optionally reduce LR slightly in higher stages
            use_lr *= 0.5
            print(f"Stage {stage_idx+1} LR set to {use_lr:.6f}")

    # # ----------------------------
        scheduler = create_onecycle_scheduler(
            optimizer=optimizer,
            max_lr=use_lr,
            train_loader_len=len(train_loader),
            epochs=NUM_EPOCHS_STAGE,
        )
        # optimizer, scheduler = make_optimizer_and_scheduler(model, BATCH_SIZE, NUM_EPOCHS_STAGE, steps_per_epoch)
    # ----------------------------
        # Load previous stage weights if any
        if prev_weights:
            model.load_state_dict(prev_weights)
            print(f"🔄 Loaded weights from previous stage")
    

    # ----------------------------
        # Stage training loop
        for epoch in range(NUM_EPOCHS_STAGE):
            global_epoch += 1
            epoch_start = time.time()

            train_loss, train_acc, scaler = train_one_epoch_imagenet(model, train_loader, optimizer, criterion, DEVICE,
                                                             scheduler, scaler,use_mixup_fn= use_mixup_fn,num_classes=num_classes)
            val_loss, val_acc = validate_imagenet(model, val_loader, criterion, DEVICE,num_classes=num_classes)

            current_lr = scheduler.get_last_lr()[0] if scheduler else use_lr
            current_mom = optimizer.param_groups[0].get("momentum", None)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)
            history["mom"].append(current_mom)

            epoch_time = time.time() - epoch_start
            total_train_time = time.time() - train_start_time
            history["train_time"].append(total_train_time)
            print(
                f"[Epoch {global_epoch+1:03}/{NUM_EPOCHS}] | ⏱️ {epoch_time/60:.2f}m | "
                f"LR: {current_lr:.6f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%"
            )

            # ---- SAVE MODELS ----
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), SAVE_BEST)
                prev_weights = model.state_dict()  # carry forward
                print(f"New Best Accuracy: {best_acc*100:.2f}% (saved as {SAVE_BEST})\033[0m")
                with open(TXT_LOG_FILE, "a") as log:
                    log.write(f"New Best Accuracy: {best_acc*100:.2f}% (saved as {SAVE_BEST})\033[0m \n ")

            torch.save(model.state_dict(), SAVE_LAST)

            # ---- LOG ----
            with open(TXT_LOG_FILE, "a") as log:
                log.write(
                    # f"{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{current_lr:.6f},{current_mom:.4f}\n"
                    f"[Epoch {global_epoch+1:03}/{NUM_EPOCHS}] | ⏱️ {epoch_time/60:.2f}m | "
                    f"LR: {current_lr:.6f} | Train Acc: {train_acc*100:.2f}% | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Loss: {val_loss:.4f} | "
                    f"Momentum: {current_mom:.4f} \n"
                )
            with open(CSV_LOG_FILE, "a") as log:
                log.write(f"{global_epoch:03},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{current_lr:.6f},{current_mom:.4f}\n")
            # ---- DYNAMIC PLOTS ----
            # epochs_so_far = range(1, epoch + 2)
            epochs_so_far = range(1, global_epoch + 1)


            save_plot(epochs_so_far, {"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]}, "Accuracy", "Epoch", "Accuracy", "accuracy_live.png",PLOTS_DIR)
            save_plot(epochs_so_far, {"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]}, "Loss", "Epoch", "Loss", "loss_live.png",PLOTS_DIR)
            save_plot(epochs_so_far, {"Learning Rate": history["lr"]}, "Learning Rate", "Epoch", "LR", "lr_live.png",PLOTS_DIR)
            save_plot(epochs_so_far, {"Momentum": history["mom"]}, "Momentum", "Epoch", "Momentum", "momentum_live.png",PLOTS_DIR)
            save_plot(history["train_time"], {"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]}, "Accuracy", "Time(s)", "Accuracy", "accuracy_time.png",PLOTS_DIR)
            save_plot(history["train_time"], {"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]}, "Loss", "Time(s)", "Loss", "loss_time.png",PLOTS_DIR)
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    # -----------------------------------------------------------
    print(f"\n🏁 Training Complete — Best Val Acc: {best_acc*100:.2f}%")
    print(f"✅ Best model: {SAVE_BEST}")
    print(f"✅ Last model: {SAVE_LAST}")
    print(f"🖼️ Live plots in: {PLOTS_DIR}")
    print(f"🏁 Training Time: {train_time/60:.2f}m")
    with open(TXT_LOG_FILE, "a") as log:
        log.write(f"🏁 Training Complete — Best Val Acc: {best_acc*100:.2f}%\n")
        log.write(f"✅ Best model: {SAVE_BEST}\n")
        log.write(f"✅ Last model: {SAVE_LAST}\n")
        log.write(f"🖼️ Live plots in: {PLOTS_DIR}\n")
        log.write(f"🏁 Training Time: {train_time/60:.2f}m\n")

if __name__ == "__main__":
    train_1()
    # train_2()
