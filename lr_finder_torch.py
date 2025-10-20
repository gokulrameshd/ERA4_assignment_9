import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch_lr_finder import LRFinder
#pip install torch-lr-finder to install the library

def run_lr_finder(
    model,
    optimizer,
    criterion,
    train_loader,
    device="cuda",
    start_lr=1e-7,
    end_lr=1,
    num_iter=200,
    step_mode="exp",
    smooth_f=0.05,
    diverge_th=5.0,
    cache_dir="./plots",
    use_amp=True,
):
    """
    Runs Learning Rate Finder using torch-lr-finder + AMP + auto-plot & CSV save.

    Args:
        model (nn.Module): PyTorch model
        optimizer (Optimizer): model optimizer
        criterion: loss function
        train_loader: dataloader
        device (str): 'cuda' or 'cpu'
        start_lr (float): starting learning rate
        end_lr (float): maximum LR to test
        num_iter (int): number of iterations
        step_mode (str): 'exp' or 'linear'
        smooth_f (float): loss smoothing factor
        diverge_th (float): stop if loss diverges > threshold
        cache_dir (str): directory to save results
        use_amp (bool): enable AMP training
    """

    os.makedirs(cache_dir, exist_ok=True)
    model.to(device)
    scaler = GradScaler(enabled=use_amp)

    # Define custom training step to support AMP
    def amp_train_fn(inputs, labels):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=str(device), enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss

    # --- Initialize LR Finder ---
    lr_finder = LRFinder(model, optimizer, criterion, device=device)

    print(f"\n🚀 Starting LR range test from {start_lr} → {end_lr} ({num_iter} iters)\n")

    # tqdm wrapper to visualize progress
    pbar = tqdm(total=num_iter, desc="LR Finder", unit="iter")

    def callback(batch_idx, inputs, labels, loss, lr):
        """Progress bar and optional hooks"""
        if batch_idx % 10 == 0:
            pbar.set_postfix({"lr": f"{lr:.2E}", "loss": f"{loss:.4f}"})
        pbar.update(1)

    # --- Run LR test ---
    lr_finder.range_test(
    train_loader,
    start_lr=start_lr,
    end_lr=end_lr,
    num_iter=num_iter,
    step_mode=step_mode,
    smooth_f=smooth_f,
    diverge_th=diverge_th,
    # update_step=amp_train_fn,  # ✅ AMP-aware training step
    )
    pbar.close()

    # --- Plot and save ---
    plot_path = os.path.join(cache_dir, "lr_finder_plot.png")
    try:
        res = lr_finder.plot(
                            log_lr=True, skip_start=10, skip_end=5,
                            show_lr=True, suggest=True, annotate=True)
    except TypeError:
        print("Using old version of torch-lr-finder")
        res = lr_finder.plot(log_lr=True, skip_start=10, skip_end=5, show_lr=1.0e-3)
    print(res)
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    # Safely close figure if plot() returned it
    # Check what was returned
    # if isinstance(res, tuple) and len(res) > 0:
    #     fig = res[0]  # first element is the Figure
    #     plt.close(fig)
    # elif hasattr(res, "get_figure"):  # sometimes it's just Axes
    #     plt.close(res.get_figure())
    # else:
    plt.close()  # fallback: closes the current figure

    print(f"📈 Plot saved to: {os.path.abspath(plot_path)}")

    # --- Save CSV ---
    hist = lr_finder.history
    csv_path = os.path.join(cache_dir, "lr_finder_log.csv")
    np.savetxt(
        csv_path,
        np.column_stack((np.array(hist["lr"]), np.array(hist["loss"]))),
        delimiter=",",
        header="lr,loss",
        comments="",
    )
    print(f"🧾 LR log saved to: {os.path.abspath(csv_path)}")

    # --- Suggested LR ---
    min_loss_idx = np.argmin(hist["loss"])

    suggested_lr = hist["lr"][min_loss_idx]
    safe_lr = suggested_lr * 0.3

    print(f"💡 Suggested LR: {suggested_lr:.2E}")
    print(f"💡 Safe (max) LR for OneCycleLR: {safe_lr:.2E}")

    # --- Reset model and optimizer ---
    lr_finder.reset()
    print("♻️ Model and optimizer restored after LR test.\n")

    return suggested_lr, safe_lr
