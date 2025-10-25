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
import matplotlib.pyplot as plt

from data_loader import get_dataloaders, set_seed, get_mixup_fn
from model import create_model
from hyper_parameter_modules import create_onecycle_scheduler
# Modern AMP imports
from torch.amp import GradScaler
from train_test_modules import train_one_epoch_imagenet, validate_imagenet, save_plot, save_checkpoint, load_checkpoint,ModelEMA
from lr_finder_custom import LRFinder

# ==============================================================
# ⚙️ CONFIG
# ==============================================================
DATA_DIR = "/home/deep/Documents/jeba/Classification_R_D/res/data"
BATCH_SIZE = 512
IMG_SIZE = 224
NUM_EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_BEST = "./hydrid_train/best_weights.pth"
SAVE_LAST = "./hydrid_train/last_weights.pth"
CSV_LOG_FILE = "./hydrid_train/training_log.csv"
TXT_LOG_FILE = "./hydrid_train/training_log.txt"
PLOTS_DIR = "./hydrid_train/plots"
USE_MIXUP = True
ENABLE_LR_FINDER = False
SAVE_FREQ_LAST = 5   # only overwrite last_weights every N epochs (reduce IO)
ENABLE_EMA = False
ENABLE_CHANNEL_LAST = True
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
def get_loaders(img_size, batch_size):
    return get_dataloaders(DATA_DIR, batch_size, img_size)
# ==============================================================
# 🚀 MAIN TRAIN FUNCTION
# ==============================================================
def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    set_seed(42)

    print(f"\n🚀 Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} on {DEVICE}")

    # ============================================================
    # ⚙️ Progressive Stage Configurations
    # ============================================================
    # TRAIN_STAGES = [
    #     {"fraction": 0.5, "img_size": 128, "batch_size": 1024, "epochs": 20},
    #     {"fraction": 1.0, "img_size": 224, "batch_size": 512, "epochs": 30},
    # ]
    TRAIN_STAGES = [
        {"fraction": 0.5, "img_size": 128, "batch_size": 1024, "epochs": 15},
        {"fraction": 0.5, "img_size": 224, "batch_size": 512, "epochs": 15},
        {"fraction": 0.75, "img_size": 224, "batch_size": 512, "epochs": 10},
        {"fraction": 1.0, "img_size": 224, "batch_size": 512, "epochs": 10},
    ]
    TOTAL_EPOCHS = sum(stage["epochs"] for stage in TRAIN_STAGES)

    # ============================================================
    # 🧠 Initial Data (Stage 1)
    # ============================================================
    current_stage = 0
    stage_cfg = TRAIN_STAGES[current_stage]
    print(f"\n📦 Stage 1 — {int(stage_cfg['fraction']*100)}% data | {stage_cfg['img_size']}px | batch={stage_cfg['batch_size']}")

    train_loader, val_loader, num_classes = get_dataloaders(
        DATA_DIR, stage_cfg["batch_size"], stage_cfg["img_size"], fraction=stage_cfg["fraction"]
    )


    # -----------------------------------------------------------
    # 🧠 Model setup
    model = create_model(num_classes=num_classes, pretrained=False).to(DEVICE)
    if ENABLE_CHANNEL_LAST:
        model = model.to(memory_format=torch.channels_last) 
    if ENABLE_EMA:
        ema = ModelEMA(model, decay=0.999,model_fn=lambda: create_model(num_classes, pretrained=False))
    else:
        ema = None

    # Try to compile the model (PyTorch 2.x). Safe to ignore failures.
    try:
        model = torch.compile(model,)#mode="max-autotune"
        print("⚡ model compiled with torch.compile()")
    except Exception:
        print("⚠️ torch.compile() failed/ignored, continuing without it")

    if USE_MIXUP:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    # -----------------------------------------------------------
    # 🔍 LR Finder
    print("\n🔍 Running Learning Rate Finder...")
    # Use a small temp optimizer copy and model copy to run LR finder without altering original optimizer
    # We use the model and optimizer provided (lr_finder will cache & reset)


        
    if ENABLE_LR_FINDER:
        lr_finder = LRFinder(model, optimizer, criterion, device=DEVICE, cache_dir=PLOTS_DIR)
        lr_finder.range_test(train_loader, start_lr=1e-6, end_lr=10, num_iter=100)

        # # Plot and get LR suggestions (plot returns suggested_lr, safe_lr)
        best_lr, safe_lr = lr_finder.plot(
            save_path=os.path.join(PLOTS_DIR, "lr_finder_plot.png"),
            save_csv=True,
            suggest=True,
            annotate=True
        )
        # best_lr, safe_lr = run_lr_finder(model, optimizer, criterion, train_loader, 
        #                                  device=DEVICE, start_lr=1e-6, end_lr=10, num_iter=100, 
        #                                  cache_dir=PLOTS_DIR, use_amp=True)
        # Fallback if LR finder fails
        if best_lr is None or not math.isfinite(best_lr) or best_lr <= 0:
            print("LR Finder returned invalid value. Falling back to 1e-3.")
            best_lr, safe_lr = 1e-3, 1e-3 * 0.3

        print(f"\nRaw Suggested LR: {best_lr:.6f}")
        print(f"Safe Max LR for OneCycleLR: {safe_lr:.6f}")

        # Clamp LR to safe bounds
        lr_floor, lr_ceiling = 1e-6, 0.1
        use_lr = float(max(lr_floor, min(safe_lr, lr_ceiling)))
    else:   
        use_lr = 0.1 #Hardcoded for now 
    
    mixup_fn = None
    if USE_MIXUP:
        mixup_fn = get_mixup_fn(mixup_alpha=0.2, cutmix_alpha=1.0, mixup_prob=1.0, 
                                label_smoothing=0.1, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    print(f"Final Selected LR → {use_lr:.6f}")

    # -----------------------------------------------------------
    # ♻️ Reset model + optimizer cleanly (recreate to clear any LR-finder state)
    print("Resetting model and optimizer after LR Finder...")
    if ENABLE_LR_FINDER:
        model = create_model(num_classes=num_classes, pretrained=False).to(DEVICE)
        try:
            model = torch.compile(model,)#mode="max-autotune"
            if ENABLE_CHANNEL_LAST:
                model = model.to(memory_format=torch.channels_last)
            if ENABLE_EMA:
                ema = ModelEMA(model, decay=0.999,model_fn=lambda: create_model(num_classes, pretrained=False))
            else:
                ema = None

        except Exception:
            pass
        optimizer = optim.SGD(model.parameters(), lr=use_lr, momentum=0.9, weight_decay=1e-4)
    

    # -----------------------------------------------------------
    # 🌀 OneCycleLR Scheduler (per step)
    scheduler = create_onecycle_scheduler(
        optimizer=optimizer,
        max_lr=use_lr,
        train_loader_len=len(train_loader),
        epochs=NUM_EPOCHS,
    )

    # ============================================================
    # Prepare a single GradScaler to pass through epochs (keeps state)
    device_type = "cuda" if "cuda" in str(DEVICE) else "cpu"
    scaler = GradScaler(device=device_type)

    # -----------------------------------------------------------
    # 📊 Training Loop
    # --- AUTO RESUME ---
    if os.path.exists(SAVE_LAST):
        start_epoch, best_acc, history = load_checkpoint(
            SAVE_LAST, model, optimizer, scheduler, scaler, device=DEVICE
        )
    else:
        start_epoch, best_acc, history = 0, 0.0, {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": [],
                  "mom": [] , "train_time": [], "val_time": [], "time_lapsed": []}
    print(f"🚀 Starting new training run from epoch {start_epoch}.")


    with open(CSV_LOG_FILE, "w") as log:
        log.write("Epoch,Train_Loss,Train_Acc,Train_Time,Val_Loss,Val_Acc,Val_Time,Learning_Rate,Momentum \n")
    start_time = time.time()

    # ============================================================
    # 🧠 TRAINING LOOP (Hybrid Progressive)
    # ============================================================
    for epoch in range(start_epoch, NUM_EPOCHS):

        # ---------------------------------------
        # Stage Transition Check
        # ---------------------------------------
        if current_stage < len(TRAIN_STAGES) - 1:
            stage_end = sum(s["epochs"] for s in TRAIN_STAGES[:current_stage + 1])
            if epoch == stage_end:
                current_stage += 1
                stage_cfg = TRAIN_STAGES[current_stage]
                print(f"\n📈 Switching to Stage {current_stage + 1} → "
                      f"{int(stage_cfg['fraction']*100)}% data | {stage_cfg['img_size']}px | batch={stage_cfg['batch_size']}")

                train_loader, val_loader, _ = get_dataloaders(
                    DATA_DIR, stage_cfg["batch_size"], stage_cfg["img_size"],
                    fraction=stage_cfg["fraction"]
                )
                # Optional LR dampening
                for g in optimizer.param_groups:
                    g["lr"] *= 0.5

                scheduler = create_onecycle_scheduler(
                    optimizer, max_lr=g["lr"],
                    train_loader_len=len(train_loader),
                    epochs=TOTAL_EPOCHS - epoch
                )
        # ---------------------------------------
        # 🏋️ Train
        # ---------------------------------------
        train_results = train_one_epoch_imagenet(model, train_loader, optimizer, criterion, DEVICE,
                                                scheduler, scaler, mixup_fn=mixup_fn, 
                                                enable_last_channel = ENABLE_CHANNEL_LAST,
                                                ema=ema,num_classes=num_classes)
        train_loss = train_results["loss"]
        train_acc = train_results["acc"]
        scaler = train_results["scaler"]
        
        train_time = train_results["time"]
        # ---------------------------------------
        # ✅ Validate
        # ---------------------------------------
        val_model = ema.ema if ema is not None else model
        val_results = validate_imagenet(val_model, val_loader, criterion, DEVICE,  num_classes=num_classes)
        val_loss = val_results["loss"]
        val_acc = val_results["acc"]
        val_time = val_results["time"]
        # ---------------------------------------
        # Log stats
        # ---------------------------------------
        current_lr = scheduler.get_last_lr()[0] if scheduler else use_lr
        current_mom = optimizer.param_groups[0].get("momentum", None)
        time_lapsed = (time.time() - start_time )/60

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["mom"].append(current_mom)
        history["time_lapsed"].append(time_lapsed)
        history["train_time"].append(train_time)
        history["val_time"].append(val_time)
        print(
            f"[Epoch {epoch+1:03}/{NUM_EPOCHS}] | ⏱️ {train_time:.2f}m | "
            f"LR: {current_lr:.6f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%"
        )

        # ---- SAVE MODELS ----
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_BEST)
            # save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_acc, history,
            #             path=SAVE_BEST)
            print(f"New Best Accuracy: {best_acc*100:.2f}% (saved as {SAVE_BEST})\033[0m")
            with open(TXT_LOG_FILE, "a") as log:
                log.write(f"New Best Accuracy: {best_acc*100:.2f}% (saved as {SAVE_BEST})\033[0m \n")

        # torch.save(model.state_dict(), SAVE_LAST)
        save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_acc, history,
                        path=SAVE_LAST)

        # ---- LOG ----
        with open(TXT_LOG_FILE, "a") as log:
            log.write(
                # f"{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{current_lr:.6f},{current_mom:.4f}\n"
                f"[Epoch {epoch+1:03}/{NUM_EPOCHS}] | ⏱️ {train_time:.2f}m | "
                f"LR: {current_lr:.6f} | Train Acc: {train_acc*100:.2f}% | Train Loss: {train_loss:.4f} | Train Time: {train_time:.2f}m  | Val Acc: {val_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Time: {val_time:.2f}m | "
                f"Momentum: {current_mom:.4f} \n"
            )
        with open(CSV_LOG_FILE, "a") as log:
            log.write(f"{epoch+1:03},{train_loss:.4f},{train_acc:.4f},{train_time:.2f},{val_loss:.4f},{val_acc:.4f},{val_time:.2f},{current_lr:.6f},{current_mom:.4f}\n")

        # ---- DYNAMIC PLOTS ----
        epochs_so_far = range(1, epoch + 2)

       
        save_plot(epochs_so_far, {"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]}, "Accuracy", "Epoch", "Accuracy", "accuracy_live.png",PLOTS_DIR)
        save_plot(epochs_so_far, {"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]}, "Loss", "Epoch", "Loss", "loss_live.png",PLOTS_DIR)
        save_plot(epochs_so_far, {"Learning Rate": history["lr"]}, "Learning Rate", "Epoch", "LR", "lr_live.png",PLOTS_DIR)
        save_plot(epochs_so_far, {"Momentum": history["mom"]}, "Momentum", "Epoch", "Momentum", "momentum_live.png",PLOTS_DIR)
        #plot train time vs accuracy
        
        save_plot(history["time_lapsed"], {"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]}, "Accuracy", "Time(m)", "Accuracy", "accuracy_time.png",PLOTS_DIR)
        save_plot(history["time_lapsed"], {"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]}, "Loss", "Time(m)", "Loss", "loss_time.png",PLOTS_DIR)


    train_end_time = time.time()
    train_time = train_end_time - start_time
    print(f"🏁 Training Complete — Best Val Acc: {best_acc*100:.2f}%")

    # -----------------------------------------------------------
    print(f"\n🏁 Training Complete — Best Val Acc: {best_acc*100:.2f}%")
    print(f"✅ Best model: {SAVE_BEST}")
    print(f"✅ Last model: {SAVE_LAST}")
    print(f"🖼️ Live plots in: {PLOTS_DIR}")
    print(f"🏁 Training Time: {train_time:.2f}m")
    with open(TXT_LOG_FILE, "a") as log:
        log.write(
            f"🏁 Training Complete — Best Val Acc: {best_acc*100:.2f}%\n"
            f"✅ Best model: {SAVE_BEST}\n"
            f"✅ Last model: {SAVE_LAST}\n"
            f"🖼️ Live plots in: {PLOTS_DIR}\n"
            f"🏁 Training Time: {train_time:.2f}m\n"
        )

if __name__ == "__main__":
    main()
