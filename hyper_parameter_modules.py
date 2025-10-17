import torch.optim as optim
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