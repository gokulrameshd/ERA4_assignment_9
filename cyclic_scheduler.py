from torch.optim.lr_scheduler import OneCycleLR


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
