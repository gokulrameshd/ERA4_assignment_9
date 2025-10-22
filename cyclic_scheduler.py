from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import SGD

def create_onecycle_scheduler(optimizer, max_lr, train_loader_len, epochs,
                              pct_start=0.3, div_factor=25.0, final_div_factor=1e4):
    """Return a configured OneCycleLR scheduler."""
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=train_loader_len,
        epochs=epochs,
        pct_start=pct_start,
        anneal_strategy="cos",
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )
    return scheduler

# ==========================
# ✅ OPTIMIZER + SCHEDULER
# ==========================
def make_optimizer_and_scheduler(model, batch_size, epochs, steps_per_epoch):
    base_lr = min(0.1 * (batch_size / 256), 0.4)  # Linear LR scaling rule
    print("base_lr:::",base_lr)
    optimizer = SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=base_lr,
        total_steps=epochs * steps_per_epoch,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4
    )
    return optimizer, scheduler