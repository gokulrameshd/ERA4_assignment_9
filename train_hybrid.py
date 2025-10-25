

import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from datetime import datetime
import matplotlib.pyplot as plt

from data_loader import get_dataloaders, set_seed, get_mixup_fn, get_total_steps
from model import create_model
from hyper_parameter_modules import create_onecycle_scheduler
# Modern AMP imports
from torch.amp import GradScaler
from train_test_modules import (train_one_epoch_imagenet, validate_imagenet, save_plot, save_checkpoint,
                                load_checkpoint,ModelEMA,set_trainable_layers,build_optimizer,)
from lr_finder_custom import LRFinder

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================
# ⚙️ CONFIG
# ==============================================================
ENABLE_HYBRID_TRAINING = True
DATA_DIR = "./sample_data"
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
ENABLE_LR_DAMPENING = True
ENABLE_STAGE_WISE_SCHEDULER = False
ENABLE_PROGRESSIVE_UNFREEZING = False
ENABLE_PROGRESSIVE_FREEZING = False
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
def main_1():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    set_seed(42)

    print(f"\n🚀 Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} on {DEVICE}")

    # ============================================================
    # ⚙️ Progressive Stage Configurations
    # ============================================================

    if ENABLE_HYBRID_TRAINING:
        if ENABLE_PROGRESSIVE_UNFREEZING:
            TRAIN_STAGES = [
                            {"fraction": 0.5, "img_size": 128, "batch_size": 1024, "epochs": 10, "unfreeze_to": "layer3"},
                            {"fraction": 0.75, "img_size": 160, "batch_size": 768, "epochs": 15, "unfreeze_to": "layer2"},
                            {"fraction": 1.0, "img_size": 224, "batch_size": 512, "epochs": 25, "unfreeze_to": "all"},
                        ]
        elif ENABLE_PROGRESSIVE_FREEZING:
            TRAIN_STAGES = [
                {"fraction": 0.5, "img_size": 128, "batch_size": 1024, "epochs": 15,"lr_scale": 1.0, "freeze_to": None},
                {"fraction": 0.5, "img_size": 160, "batch_size": 512, "epochs": 15, "lr_scale": 0.9, "freeze_to": None},
                {"fraction": 0.5, "img_size": 224, "batch_size": 512, "epochs": 10, "lr_scale": 0.8, "freeze_to": None},
                {"fraction": 0.75, "img_size": 224, "batch_size": 750, "epochs": 10, "lr_scale": 0.7, "freeze_to": "layer2"},
                {"fraction": 1.0, "img_size": 224, "batch_size": 1024, "epochs": 10, "lr_scale": 0.5, "freeze_to": "layer3"},
            ]
        else:
            TRAIN_STAGES = [
                {"fraction": 0.25, "img_size": 128, "batch_size": 512, "epochs": 10, "lr_scale": 1.0},  # Fast warmup "batch_size": 1024 15
                {"fraction": 0.25, "img_size": 160, "batch_size": 350, "epochs": 7, "lr_scale": 0.8},  # Mid-scale refinement "batch_size": 768 15
                {"fraction": 0.25, "img_size": 224, "batch_size": 256, "epochs": 5, "lr_scale": 0.6},   # Full fine-tune "batch_size": 512 20
            ]
        # NUM_EPOCHS = sum(stage["epochs"] for stage in TRAIN_STAGES)
    else:
        TRAIN_STAGES = [
                {"fraction": 1.0, "img_size": 224, "batch_size": 512, "epochs": 60, "lr_scale": 1.0},
        ]
    NUM_EPOCHS = sum(stage["epochs"] for stage in TRAIN_STAGES)
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
                  "mom": [] , "train_time": [], "val_time": [], "time_lapsed": [], "total_time_epoch": []}
    print(f"🚀 Starting new training run from epoch {start_epoch}.")


    with open(CSV_LOG_FILE, "w") as log:
        log.write("Epoch,Train_Loss,Train_Acc,Train_Time,Val_Loss,Val_Acc,Val_Time,Learning_Rate,Momentum \n")
    

    # ============================================================
    # 🧠 TRAINING LOOP (Hybrid Progressive)
    # ============================================================
    for current_stage,stage_cfg in enumerate(TRAIN_STAGES):
    
        start_time = time.time()
        # ---------------------------------------
        # Stage Transition Check
        # ---------------------------------------
        stage_cfg = TRAIN_STAGES[current_stage]
        print(f"\n📈 Switching to Stage {current_stage + 1} → "
                f"{int(stage_cfg['fraction']*100)}% data | {stage_cfg['img_size']}px | batch={stage_cfg['batch_size']}")
        if ENABLE_PROGRESSIVE_UNFREEZING:
            set_trainable_layers(model, "unfreeze", stage_cfg["unfreeze_to"])
            optimizer = build_optimizer(model, use_lr, weight_decay=1e-4, momentum=0.9)
        elif ENABLE_PROGRESSIVE_FREEZING:
            set_trainable_layers(model, "freeze", stage_cfg["freeze_to"])
            optimizer = build_optimizer(model, use_lr, weight_decay=1e-4, momentum=0.9)
        train_loader, val_loader, _ = get_dataloaders(
            DATA_DIR, stage_cfg["batch_size"], stage_cfg["img_size"],
            fraction=stage_cfg["fraction"]
        )
        if ENABLE_STAGE_WISE_SCHEDULER:
            if ENABLE_LR_FINDER:
                lr_finder = LRFinder(model, optimizer, criterion, device=DEVICE, cache_dir=PLOTS_DIR)
                lr_finder.range_test(train_loader, start_lr=1e-6, end_lr=1, num_iter=100)

                # # Plot and get LR suggestions (plot returns suggested_lr, safe_lr)
                best_lr, safe_lr = lr_finder.plot(
                    save_path=os.path.join(PLOTS_DIR, f"lr_finder_plot_stage_{current_stage+1}.png"),
                    save_csv=True,
                    suggest=True,
                    annotate=True
                )
                # Fallback if LR finder fails
                if best_lr is None or not math.isfinite(best_lr) or best_lr <= 0:
                    print("LR Finder returned invalid value. Falling back to 1e-3.")
                    best_lr, safe_lr = 1e-3, 1e-3 * 0.3

                print(f"\nRaw Suggested LR: {best_lr:.6f}")
                print(f"Safe Max LR for OneCycleLR: {safe_lr:.6f}")

                # Clamp LR to safe bounds
                lr_floor, lr_ceiling = 1e-6, 0.1
                use_lr = float(max(lr_floor, min(safe_lr, lr_ceiling)))
            # Optional LR dampening
            elif ENABLE_LR_DAMPENING:
                for g in optimizer.param_groups:
                    g["lr"] *= 0.5
                use_lr = optimizer.param_groups[0]["lr"]
            else:
                # scale lr by batch size (linear rule) OR apply dampening
                base_batch = TRAIN_STAGES[0]["batch_size"]
                use_lr = use_lr * (stage_cfg["batch_size"] / base_batch)
            scheduler = create_onecycle_scheduler(
                optimizer, max_lr=use_lr,
                train_loader_len=len(train_loader),
                epochs=stage_cfg["epochs"]
            )
        # ---------------------------------------
        # 🏋️ Train
        # ---------------------------------------
        for epoch in range(stage_cfg["epochs"]):
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

            # ---- DYNAMIC PLOTS ----
            epochs_so_far = range(1, epoch + 2)

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
            save_checkpoint(epochs_so_far, model, optimizer, scheduler, scaler, best_acc, history,
                            path=SAVE_LAST)
            # ---------------------------------------
            # Log stats
            # ---------------------------------------
            current_lr = scheduler.get_last_lr()[0] if scheduler else use_lr
            current_mom = optimizer.param_groups[0].get("momentum", None)
            total_time_epoch = (time.time() - start_time)/60
            try:
                time_lapsed = history["time_lapsed"][-1] + total_time_epoch
            except :
                time_lapsed = total_time_epoch
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)
            history["mom"].append(current_mom)
            history["time_lapsed"].append(time_lapsed)
            history["train_time"].append(train_time)
            history["val_time"].append(val_time)
            history["total_time_epoch"].append(total_time_epoch)

            print(
                f"[Epoch {epoch+1:03}/{NUM_EPOCHS}] | ⏱️ {train_time:.2f}m | "
                f"LR: {current_lr:.6f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%"
            )

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

            

        
            save_plot(epochs_so_far, {"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]}, "Accuracy", "Epoch", "Accuracy", "accuracy_live.png",PLOTS_DIR)
            save_plot(epochs_so_far, {"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]}, "Loss", "Epoch", "Loss", "loss_live.png",PLOTS_DIR)
            save_plot(epochs_so_far, {"Learning Rate": history["lr"]}, "Learning Rate", "Epoch", "LR", "lr_live.png",PLOTS_DIR)
            save_plot(epochs_so_far, {"Momentum": history["mom"]}, "Momentum", "Epoch", "Momentum", "momentum_live.png",PLOTS_DIR)
            #plot train time vs accuracy
            save_plot(history["time_lapsed"], {"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]}, "Accuracy", "Time(m)", "Accuracy", "accuracy_time.png",PLOTS_DIR)
            save_plot(history["time_lapsed"], {"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]}, "Loss", "Time(m)", "Loss", "loss_time.png",PLOTS_DIR)


        print(f"🏁 Training Complete — Best Val Acc: {best_acc*100:.2f}%")

    # -----------------------------------------------------------
    print(f"\n🏁 Training Complete — Best Val Acc: {best_acc*100:.2f}%")
    print(f"✅ Best model: {SAVE_BEST}")
    print(f"✅ Last model: {SAVE_LAST}")
    print(f"🖼️ Live plots in: {PLOTS_DIR}")
    print(f"🏁 Training Time: {history["time_lapsed"][-1]:.2f}m")
    with open(TXT_LOG_FILE, "a") as log:
        log.write(
            f"🏁 Training Complete — Best Val Acc: {best_acc*100:.2f}%\n"
            f"✅ Best model: {SAVE_BEST}\n"
            f"✅ Last model: {SAVE_LAST}\n"
            f"🖼️ Live plots in: {PLOTS_DIR}\n"
            f"🏁 Training Time: {history["time_lapsed"][-1]:.2f}m\n"
        )

def main_2():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    set_seed(42)

    print(f"\n🚀 Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} on {DEVICE}")

    # ============================================================
    # ⚙️ Progressive Stage Configurations
    # ============================================================

    if ENABLE_HYBRID_TRAINING:
        if ENABLE_PROGRESSIVE_UNFREEZING:
            TRAIN_STAGES = [
                            {"fraction": 0.5, "img_size": 128, "batch_size": 1024, "epochs": 10, "unfreeze_to": "layer3"},
                            {"fraction": 0.75, "img_size": 160, "batch_size": 768, "epochs": 15, "unfreeze_to": "layer2"},
                            {"fraction": 1.0, "img_size": 224, "batch_size": 512, "epochs": 25, "unfreeze_to": "all"},
                        ]
        elif ENABLE_PROGRESSIVE_FREEZING:
            TRAIN_STAGES = [
                {"fraction": 0.5, "img_size": 128, "batch_size": 1024, "epochs": 15,"lr_scale": 1.0, "freeze_to": None},
                {"fraction": 0.5, "img_size": 160, "batch_size": 512, "epochs": 15, "lr_scale": 0.9, "freeze_to": None},
                {"fraction": 0.5, "img_size": 224, "batch_size": 512, "epochs": 10, "lr_scale": 0.8, "freeze_to": None},
                {"fraction": 0.75, "img_size": 224, "batch_size": 750, "epochs": 10, "lr_scale": 0.7, "freeze_to": "layer2"},
                {"fraction": 1.0, "img_size": 224, "batch_size": 1024, "epochs": 10, "lr_scale": 0.5, "freeze_to": "layer3"},
            ]
        else:
            TRAIN_STAGES = [
                {"fraction": 0.25, "img_size": 128, "batch_size": 512, "epochs": 5, "lr_scale": 1.0},  # Fast warmup "batch_size": 1024 15
                {"fraction": 0.25, "img_size": 160, "batch_size": 350, "epochs": 5, "lr_scale": 0.8},  # Mid-scale refinement "batch_size": 768 15
                {"fraction": 0.25, "img_size": 224, "batch_size": 256, "epochs": 5, "lr_scale": 0.6},   # Full fine-tune "batch_size": 512 20
            ]
        # NUM_EPOCHS = sum(stage["epochs"] for stage in TRAIN_STAGES)
    else:
        TRAIN_STAGES = [
                {"fraction": 1.0, "img_size": 224, "batch_size": 512, "epochs": 60, "lr_scale": 1.0},
        ]
    NUM_EPOCHS = sum(stage["epochs"] for stage in TRAIN_STAGES)
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
    if ENABLE_STAGE_WISE_SCHEDULER == False:
        total_steps = get_total_steps(DATA_DIR, train_transforms=None, stages=TRAIN_STAGES)
        # 🌀 OneCycleLR Scheduler (per step)
        scheduler = create_onecycle_scheduler(
            optimizer=optimizer,
            max_lr=use_lr,
            train_loader_len=total_steps,
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
                  "mom": [] , "train_time": [], "val_time": [], "time_lapsed": [], "total_time_epoch": []}
    print(f"🚀 Starting new training run from epoch {start_epoch}.")


    with open(CSV_LOG_FILE, "w") as log:
        log.write("Epoch,Train_Loss,Train_Acc,Train_Time,Val_Loss,Val_Acc,Val_Time,Learning_Rate,Momentum \n")
    

    # ============================================================
    # 🧠 TRAINING LOOP (Hybrid Progressive)
    # ============================================================
    for epoch in range(start_epoch, NUM_EPOCHS):
        start_time = time.time()
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
                if ENABLE_PROGRESSIVE_UNFREEZING:
                    set_trainable_layers(model, "unfreeze", stage_cfg["unfreeze_to"])
                    optimizer = build_optimizer(model, use_lr, weight_decay=1e-4, momentum=0.9)
                elif ENABLE_PROGRESSIVE_FREEZING:
                    set_trainable_layers(model, "freeze", stage_cfg["freeze_to"])
                    optimizer = build_optimizer(model, use_lr, weight_decay=1e-4, momentum=0.9)
                train_loader, val_loader, _ = get_dataloaders(
                    DATA_DIR, stage_cfg["batch_size"], stage_cfg["img_size"],
                    fraction=stage_cfg["fraction"]
                )
                if ENABLE_STAGE_WISE_SCHEDULER:
                    if ENABLE_LR_FINDER:
                        lr_finder = LRFinder(model, optimizer, criterion, device=DEVICE, cache_dir=PLOTS_DIR)
                        lr_finder.range_test(train_loader, start_lr=1e-6, end_lr=1, num_iter=100)

                        # # Plot and get LR suggestions (plot returns suggested_lr, safe_lr)
                        best_lr, safe_lr = lr_finder.plot(
                            save_path=os.path.join(PLOTS_DIR, f"lr_finder_plot_stage_{current_stage}.png"),
                            save_csv=True,
                            suggest=True,
                            annotate=True
                        )
                        # Fallback if LR finder fails
                        if best_lr is None or not math.isfinite(best_lr) or best_lr <= 0:
                            print("LR Finder returned invalid value. Falling back to 1e-3.")
                            best_lr, safe_lr = 1e-3, 1e-3 * 0.3

                        print(f"\nRaw Suggested LR: {best_lr:.6f}")
                        print(f"Safe Max LR for OneCycleLR: {safe_lr:.6f}")

                        # Clamp LR to safe bounds
                        lr_floor, lr_ceiling = 1e-6, 0.1
                        use_lr = float(max(lr_floor, min(safe_lr, lr_ceiling)))
                    # Optional LR dampening
                    elif ENABLE_LR_DAMPENING:
                        for g in optimizer.param_groups:
                            g["lr"] *= 0.5
                        use_lr = optimizer.param_groups[0]["lr"]
                    else:
                        # scale lr by batch size (linear rule) OR apply dampening
                        base_batch = TRAIN_STAGES[0]["batch_size"]
                        use_lr = use_lr * (stage_cfg["batch_size"] / base_batch)
                    scheduler = create_onecycle_scheduler(
                        optimizer, max_lr=use_lr,
                        train_loader_len=len(train_loader),
                        epochs=stage_cfg["epochs"]
                    )
                else:
                    if current_stage == 0:
                        total_steps = get_total_steps(DATA_DIR, train_transforms=None, stages=TRAIN_STAGES)
                        # 🌀 OneCycleLR Scheduler (per step)
                        scheduler = create_onecycle_scheduler(
                            optimizer=optimizer,
                            max_lr=use_lr,
                            train_loader_len=total_steps,
                            epochs=NUM_EPOCHS)


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
        # ---------------------------------------
        # Log stats
        # ---------------------------------------
        current_lr = scheduler.get_last_lr()[0] if scheduler else use_lr
        current_mom = optimizer.param_groups[0].get("momentum", None)
        total_time_epoch = (time.time() - start_time)/60
        try:
            time_lapsed = history["time_lapsed"][-1] + total_time_epoch
        except :
            time_lapsed = total_time_epoch
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["mom"].append(current_mom)
        history["time_lapsed"].append(time_lapsed)
        history["train_time"].append(train_time)
        history["val_time"].append(val_time)
        history["total_time_epoch"].append(total_time_epoch)

        print(
            f"[Epoch {epoch+1:03}/{NUM_EPOCHS}] | ⏱️ {train_time:.2f}m | "
            f"LR: {current_lr:.6f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%"
        )

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


    print(f"🏁 Training Complete — Best Val Acc: {best_acc*100:.2f}%")

    # -----------------------------------------------------------
    print(f"\n🏁 Training Complete — Best Val Acc: {best_acc*100:.2f}%")
    print(f"✅ Best model: {SAVE_BEST}")
    print(f"✅ Last model: {SAVE_LAST}")
    print(f"🖼️ Live plots in: {PLOTS_DIR}")
    print(f"🏁 Training Time: {history["time_lapsed"][-1]:.2f}m")
    with open(TXT_LOG_FILE, "a") as log:
        log.write(
            f"🏁 Training Complete — Best Val Acc: {best_acc*100:.2f}%\n"
            f"✅ Best model: {SAVE_BEST}\n"
            f"✅ Last model: {SAVE_LAST}\n"
            f"🖼️ Live plots in: {PLOTS_DIR}\n"
            f"🏁 Training Time: {history["time_lapsed"][-1]:.2f}m\n"
        )
if __name__ == "__main__":
    # main_1()
    main_2()
