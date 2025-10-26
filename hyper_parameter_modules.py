import torch.optim as optim
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, CyclicLR,OneCycleLR
import torch.nn as nn
import torch

def get_base_hyper_parameters(model, learning_rate, weight_decay):
    """
    Returns the basic training components with a user-defined LR and Weight Decay.
    (This function combines your original ...cosine_annealing and ...cyclic_lr functions)
    
    Returns:
        - criterion: CrossEntropyLoss with label smoothing
        - optimizer: SGD with momentum
        - scaler:    AMP GradScaler
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = optim.SGD(
        model.parameters(), 
        lr=learning_rate, 
        momentum=0.9, 
        weight_decay=weight_decay
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    return criterion, optimizer, scaler

def get_resnet34_hyper_parameters(model):
    """
    Returns training components with hard-coded settings optimized 
    for ResNet34 on CIFAR-100.
    (This function combines your original ...resnet34_cifar100_cyclic_lr 
     and ...one_cycle_lr_with_cos_annealing functions)

    Settings:
        - LR: 0.05 (This is a placeholder; the scheduler will control it)
        - Weight Decay: 5e-4
        - Momentum: 0.9

    Returns:
        - criterion: CrossEntropyLoss with label smoothing
        - optimizer: SGD with momentum
        - scaler:    AMP GradScaler
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05,  # Placeholder LR; your scheduler will take over
        momentum=0.9,
        weight_decay=5e-4
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    # Return components *without* the scheduler
    return criterion, optimizer, scaler




def get_hyper_parameters_cifar100_cosine_annealing(model, epochs, learning_rate, weight_decay):
    # ========================
    # Loss, Optimizer, Scheduler
    # ========================
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()
    return criterion, optimizer, scheduler, scaler

def get_hyper_parameters_cifar100_cyclic_lr(model, epochs, learning_rate, weight_decay, trainloader):
    """
    This function returns the criterion, optimizer, scheduler, and scaler for the CIFAR-100 dataset with cyclic learning rate.
    The learning rate is cycled between 1e-4 and 0.1.
    The step size is 5 epochs up and 5 epochs down.
    The mode is triangular2.
    The cycle momentum is True.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = CyclicLR(
        optimizer,
        base_lr=1e-4,
        max_lr=0.1,
        step_size_up=len(trainloader)*5,  # 5 epochs up
        step_size_down=len(trainloader)*5,  # 5 epochs down
        mode='triangular2',
        cycle_momentum=True
    )
    scaler = torch.cuda.amp.GradScaler()
    return criterion, optimizer, scheduler, scaler

def get_hyper_parameters_resnet34_cifar100_cyclic_lr(model, trainloader):
    """
    This function returns the criterion, optimizer, scheduler, and scaler for the CIFAR-100 dataset with cyclic learning rate.
    The learning rate is cycled between 1e-4 and 0.05.
    The step size is 4 epochs up and 4 epochs down.
    The mode is triangular2.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = CyclicLR(
        optimizer, 
        base_lr=1e-4,
        max_lr=0.05,
        step_size_up=len(trainloader)*4, step_size_down=len(trainloader)*4,
        mode='triangular2'
    )
    scaler = torch.cuda.amp.GradScaler()
    return criterion, optimizer, scheduler, scaler

def get_hyper_parameters_one_cycle_lr_with_cos_annealing(model, trainloader,lr, epochs=30):
    """
    Returns criterion, optimizer, scheduler, and scaler for CIFAR-100 with OneCycleLR (cosine annealing).
    
    - Uses label smoothing for stable training.
    - OneCycleLR gradually warms up then cools down the LR with cosine decay.
    - Designed for ~30 epochs, modify `epochs` as needed.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,                # max LR, actual schedule handled by OneCycleLR
        momentum=0.9,
        weight_decay=5e-4
    )
    total_steps = len(trainloader) * epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,                         # peak LR
        epochs=epochs,                       # full training epochs
        steps_per_epoch=len(trainloader),     # batches per epoch
        # total_steps=total_steps - 1,  # ✅ minus one to prevent overshoot
        pct_start=0.3,                       # warmup fraction
        anneal_strategy='cos'                # cosine decay
    )
    
    # scaler = torch.cuda.amp.GradScaler()
    return criterion, optimizer, scheduler


def create_onecycle_scheduler(optimizer, max_lr, train_loader_len, epochs,
                              pct_start=0.15, div_factor=25.0, final_div_factor=1e4):
    """Return a configured OneCycleLR scheduler."""
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=train_loader_len,
        epochs=epochs,
        pct_start=pct_start,
        anneal_strategy="cos",
        # div_factor=div_factor,
        # final_div_factor=final_div_factor,
    )
    return scheduler


def create_onecycle_scheduler_global(optimizer, max_lr, total_steps, epochs,
                              pct_start=0.15, div_factor=25.0, final_div_factor=1e4):
    """Return a configured OneCycleLR scheduler."""
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        epochs=epochs,
        pct_start=pct_start,
        anneal_strategy="cos",
        # div_factor=div_factor,
        # final_div_factor=final_div_factor,
    )
    return scheduler

# ==========================
# ✅ OPTIMIZER + SCHEDULER
# ==========================
def make_optimizer_and_scheduler(model, batch_size, epochs, steps_per_epoch):
    base_lr = min(0.1 * (batch_size / 256), 0.4)  # Linear LR scaling rule
    print("base_lr:::",base_lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
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