from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary
import os
import numpy as np
# Let's visualize some of the images


# ========================
# CutMix / Mixup Utils
# ========================
def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)

    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
    y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def mixup_data(x, y, alpha=1.0):
    """Apply Mixup augmentation."""
    if alpha <= 0:
        return x, y, y, 1
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation."""
    if alpha <= 0:
        return x, y, y, 1
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    y_a, y_b = y, y[index]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion):

    correct = 0
    processed = 0
    train_loss = 0
    model.train()
    pbar = tqdm(train_loader)

    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)#F.nll_loss
        # train_losses.append(loss)
        
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm

        # pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += GetCorrectPredCount(y_pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc = 100*correct/processed
    train_loss = train_loss/len(train_loader)
    return train_loss, train_acc

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            y_pred = model(data)

            # accumulate loss (sum)
            test_loss += criterion(y_pred, target).item()
            correct += GetCorrectPredCount(y_pred, target)

    # average loss per batch
    test_loss /= len(test_loader)
    test_acc = 100. * correct / len(test_loader.dataset)

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc:.2f}%)\n")

    return test_loss, test_acc


def train_one_epoch(model, device, train_loader, optimizer, criterion, scheduler, scaler, epoch,total_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ✅ Step scheduler only once per batch, inside loop
        scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.*correct/total:.2f}%",
            "lr": f"{scheduler.get_last_lr()[0]:.5f}"
        })

    # ❌ DO NOT call scheduler.step() again here
    return total_loss / total, 100. * correct / total

# ========================
# Training & Evaluation
# ========================
def train_one_epoch_scaler_cutmix_mixup(model, device, train_loader, optimizer, criterion, scheduler, scaler,epoch, epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Randomly choose augmentation: CutMix or Mixup (or none)
        r = np.random.rand()
        if r < 0.25:
            imgs, y_a, y_b, lam = cutmix_data(imgs, targets, alpha=1.0)
        elif r < 0.5:
            imgs, y_a, y_b, lam = mixup_data(imgs, targets, alpha=1.0)
        else:
            y_a, y_b, lam = targets, targets, 1.0

        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += (lam * preds.eq(y_a).sum().item() +
                    (1 - lam) * preds.eq(y_b).sum().item())
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.*correct/total:.2f}%",
            "lr": f"{scheduler.get_last_lr()[0]:.5f}"
        })

    
    return total_loss / total, 100. * correct / total

@torch.no_grad()
def evaluate(model, device, test_loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for imgs, targets in test_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()
    return total_loss / total, 100. * correct / total


# ========================
# Training Loop
# ========================
def training_loop_with_scaler(model, device, train_loader, test_loader, optimizer, criterion, scheduler, epochs,CHECKPOINT_DIR="./checkpoints"):
    best_acc = 0.0
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    Learning_Rates = []
    best_accuracies = []
    scaler = torch.cuda.amp.GradScaler()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion, scheduler, scaler,epoch,epochs)
        val_loss, val_acc = evaluate(model, device, test_loader, criterion)
        train_losses.append(train_loss)
        test_losses.append(val_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(val_acc)
        Learning_Rates.append(scheduler.get_last_lr()[0])

        print(f"\nEpoch {epoch+1:03d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.5f}")

        # Save best checkpoint if the accuracy is better
        if val_acc > best_acc:
            best_acc = val_acc
            best_accuracies.append(best_acc)
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_acc': best_acc
            }, os.path.join(CHECKPOINT_DIR, "best.pth"))
            print(f"✅ Saved new best model ({best_acc:.2f}%)")

    print(f"Training complete. Best accuracy: {best_acc:.2f}%")
    return train_losses, test_losses, train_accuracies, test_accuracies, Learning_Rates, best_accuracies 

def training_loop_with_scaler_cutmix_mixup(model, device, train_loader, test_loader, optimizer, criterion, scheduler, epochs,CHECKPOINT_DIR="./checkpoints"):
    best_acc = 0.0
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    Learning_Rates = []
    best_accuracies = []
    scaler = torch.cuda.amp.GradScaler()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_scaler_cutmix_mixup(model, device, train_loader, optimizer, criterion, scheduler, scaler,epoch,epochs)
        val_loss, val_acc = evaluate(model, device, test_loader, criterion)
        train_losses.append(train_loss)
        test_losses.append(val_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(val_acc)
        Learning_Rates.append(scheduler.get_last_lr()[0])

        print(f"\nEpoch {epoch+1:03d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.5f}")

        # Save best checkpoint if the accuracy is better
        if val_acc > best_acc:
            best_acc = val_acc
            best_accuracies.append(best_acc)
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_acc': best_acc
            }, os.path.join(CHECKPOINT_DIR, "best.pth"))
            print(f"✅ Saved new best model ({best_acc:.2f}%)")

    print(f"Training complete. Best accuracy: {best_acc:.2f}%")
    return train_losses, test_losses, train_accuracies, test_accuracies, Learning_Rates, best_accuracies 

def train_test_with_scheduler(model, device, train_loader, test_loader, optimizer, criterion, scheduler, epochs,save_path="./checkpoints/best_model.pth"):
    best_acc = 0
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    Learning_Rates = []
    best_accuracies = []
    for epoch in range(epochs):
        print("EPOCH:", epoch)
        train_loss, train_acc = train(model, device, train_loader, optimizer,criterion)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        scheduler.step(test_loss)
        print("Learning Rate:", scheduler.get_last_lr())
        train_losses.append(train_loss) 
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        Learning_Rates.append(scheduler.get_last_lr())
        if test_acc > best_acc:
            best_acc = test_acc
            best_accuracies.append(best_acc)
            torch.save(model.state_dict(), save_path)
            print(f"Saved model with accuracy: {best_acc:.2f}%")
    return train_losses, test_losses, train_accuracies, test_accuracies, Learning_Rates,best_accuracies

def plot_losses(train_losses, test_losses, train_accuracies, test_accuracies):
    t = [t_items.item() for t_items in train_losses]
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(t)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_accuracies)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_accuracies)
    axs[1, 1].set_title("Test Accuracy")

    plt.show()

def get_model_summary(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return summary(model, input_size=(1, 28, 28))






