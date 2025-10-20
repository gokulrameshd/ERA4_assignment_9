"""
lr_finder.py — Clean, reliable Learning Rate Finder for PyTorch

Features:
✅ AMP + GradScaler support
✅ Safe CPU caching of model/optimizer state
✅ Savitzky–Golay or moving average smoothing
✅ Gradient-based LR suggestion (valley method)
✅ Dual LR outputs (suggested + safe for OneCycleLR)
✅ CSV export + clean matplotlib plot
"""

import matplotlib
matplotlib.use("Agg")  # For headless servers (no GUI)
import os
import math
import copy
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler

# Optional dependencies
try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False


class LRFinder:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion,
        device: str | torch.device = None,
        memory_cache: bool = True,
        cache_dir: str | None = None,
        scaler: GradScaler | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device) if device else next(model.parameters()).device
        self.memory_cache = memory_cache
        self.cache_dir = cache_dir or "."
        os.makedirs(self.cache_dir, exist_ok=True)

        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.suggested_lr = None
        self.scaler = scaler if scaler else GradScaler()

        # Cache model/optimizer states on CPU
        if self.memory_cache:
            self._model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            opt_state = copy.deepcopy(optimizer.state_dict())
            for k, v in opt_state.get("state", {}).items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if torch.is_tensor(vv):
                            v[kk] = vv.cpu()
            self._opt_state = opt_state
        else:
            self._model_state = None
            self._opt_state = None

    # ---------------------------------------------------------------
    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _smooth_loss(self, losses, use_savgol=True, window=11, poly=3):
        """Smooth losses using Savitzky-Golay or moving average."""
        if use_savgol and _HAS_SAVGOL and len(losses) >= window:
            win = window if window % 2 == 1 else window - 1
            win = min(win, len(losses))
            if win >= 3:
                try:
                    return savgol_filter(losses, win, poly)
                except Exception:
                    pass
        kernel_size = min(window, len(losses))
        if kernel_size <= 1:
            return losses
        kernel = np.ones(kernel_size) / kernel_size
        padded = np.pad(losses, (kernel_size//2, kernel_size-1-kernel_size//2), mode='edge')
        return np.convolve(padded, kernel, mode='valid')

    # ---------------------------------------------------------------
    def range_test(
        self,
        train_loader,
        start_lr=1e-7,
        end_lr=10,
        num_iter=100,
        step_mode="exp",
        smooth_f=0.05,
        diverge_th=5.0,
        use_amp=True,
        log_every=1,
        adaptive_stop=True,
        warmup_skip=5,
    ):
        """Run LR range test."""
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.model.to(self.device)
        self.model.train()

        lr_schedule = (
            np.exp(np.linspace(np.log(start_lr), np.log(end_lr), num_iter))
            if step_mode.lower() == "exp"
            else np.linspace(start_lr, end_lr, num_iter)
        )

        self._set_lr(start_lr)
        iterator = iter(train_loader)
        avg_loss, recent_losses = None, []
        pbar = tqdm(range(num_iter), desc="Finding LR", unit="iter")

        for iteration in pbar:
            try:
                inputs, labels = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, labels = next(iterator)

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            # AMP forward/backward
            if use_amp:
                with autocast(device_type=str(self.device)):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            loss_val = float(loss.detach().cpu().item())
            if not math.isfinite(loss_val):
                print(f"⚠️ Non-finite loss detected at iter {iteration}. Stopping early.")
                break

            # Skip unstable warmup iterations
            if iteration < warmup_skip:
                continue

            avg_loss = loss_val if avg_loss is None else smooth_f * loss_val + (1 - smooth_f) * avg_loss
            self.best_loss = avg_loss if self.best_loss is None else min(self.best_loss, avg_loss)

            # ✅ Append before updating LR (fix reversed curve)
            lr = float(self.optimizer.param_groups[0]["lr"])
            self.history["lr"].append(lr)
            self.history["loss"].append(avg_loss)

            recent_losses.append(avg_loss)
            if len(recent_losses) > 10:
                recent_losses.pop(0)

            if iteration % log_every == 0:
                pbar.set_postfix({"lr": f"{lr:.2E}", "loss": f"{avg_loss:.4f}"})

            if avg_loss > diverge_th * max(1.0, self.best_loss):
                print(f"⛔️ Loss diverged at iter {iteration} (loss={avg_loss:.4f})")
                break
            if adaptive_stop and len(recent_losses) >= 8 and recent_losses[-1] > 2.0 * min(recent_losses):
                print(f"⛔️ Adaptive stop: recent loss doubled at iter {iteration}")
                break

            # Update LR *after* logging
            next_lr = lr_schedule[min(iteration + 1, len(lr_schedule) - 1)]
            self._set_lr(next_lr)

        print(f"✅ LR range test complete — {len(self.history['lr'])} points collected.")

    # ---------------------------------------------------------------
    def plot(
        self,
        skip_start=10,
        skip_end=5,
        log_lr=True,
        suggest=True,
        save_path=None,
        save_csv=True,
        auto_reset=True,
        annotate=True,
    ):
        """Plot LR vs loss, with stable valley-based LR detection."""
        if save_path is None:
            save_path = os.path.join(self.cache_dir, "lr_finder_plot.png")

        lrs = np.array(self.history["lr"])
        losses = np.array(self.history["loss"])
        if len(lrs) == 0:
            raise RuntimeError("No LR history found. Run range_test() first.")

        # Ensure left→right direction
        if lrs[0] > lrs[-1]:
            lrs, losses = lrs[::-1], losses[::-1]

        lrs_plot = lrs[skip_start: len(lrs) - skip_end]
        losses_plot = losses[skip_start: len(losses) - skip_end]
        losses_smoothed = self._smooth_loss(losses_plot)

        # --- Suggested LR (valley method) ---
        suggested_lr = None
        if suggest and len(losses_smoothed) >= 5:
            grads = np.gradient(np.log(losses_smoothed + 1e-12))
            min_grad_idx = np.argmin(grads)
            # valley-based method: pick minimum loss *after* steepest descent
            min_loss_idx = np.argmin(losses_smoothed[min_grad_idx:]) + min_grad_idx
            suggested_lr = float(lrs_plot[min_loss_idx])

        elif suggest:
            print("⚠️ Not enough points to compute suggested LR (skipped).")

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 6))
        if log_lr:
            ax.semilogx(lrs_plot, losses_smoothed, label="Smoothed Loss", lw=2)
        else:
            ax.plot(lrs_plot, losses_smoothed, label="Smoothed Loss", lw=2)

        ax.set_xlabel("Learning Rate (log scale)" if log_lr else "Learning Rate", fontsize=12)
        ax.set_ylabel("Smoothed Loss", fontsize=12)
        ax.set_title("Learning Rate Range Test", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Highlight suggested LR
        if annotate and suggested_lr is not None:
            ax.axvline(x=suggested_lr, color='r', linestyle='--', alpha=0.8, label=f"Suggested LR: {suggested_lr:.2E}")
            ymin, ymax = ax.get_ylim()
            ax.text(suggested_lr, ymin + 0.02 * (ymax - ymin), f"{suggested_lr:.2E}",
                    color='r', rotation=90, va='bottom', fontsize=9)

        ax.legend()
        fig.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"📈 LR Finder plot saved to: {os.path.abspath(save_path)}")

        # --- Save CSV ---
        if save_csv:
            hist = {"lr": lrs.tolist(), "loss": losses.tolist()}
            csv_path = os.path.join(self.cache_dir, "lr_finder_log.csv")
            if _HAS_PANDAS:
                pd.DataFrame(hist).to_csv(csv_path, index=False)
            else:
                np.savetxt(csv_path, np.vstack([lrs, losses]).T, delimiter=",", header="lr,loss", comments="")
            print(f"🧾 Saved LR history to: {os.path.abspath(csv_path)}")

        # --- Auto-reset model/optimizer ---
        if auto_reset and self.memory_cache:
            try:
                self.reset()
                print("♻️ Model and optimizer restored after LR test.")
            except Exception as e:
                print(f"⚠️ Could not reset model: {e}")

        # --- Return both LR suggestions ---
        if suggested_lr is not None:
            safe_lr = suggested_lr * 0.3
            print(f"💡 Recommended max LR (for OneCycleLR): {safe_lr:.2E}")
            return suggested_lr, safe_lr
        else:
            return None, None

    # ---------------------------------------------------------------
    def reset(self):
        """Restore model/optimizer state safely."""
        if not self.memory_cache or self._model_state is None or self._opt_state is None:
            raise RuntimeError("No cached state found.")
        self.model.load_state_dict(self._model_state)
        self.model.to(self.device)
        self.optimizer.load_state_dict(self._opt_state)
        for state in self.optimizer.state.values():
            if isinstance(state, dict):
                for k, v in state.items():
                    if torch.is_tensor(v):
                        try:
                            state[k] = v.to(self.device)
                        except Exception:
                            print(f"⚠️ Could not move optimizer tensor key={k}")
        self.scaler = GradScaler()
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print("✅ Model and optimizer safely reset on device:", self.device)
