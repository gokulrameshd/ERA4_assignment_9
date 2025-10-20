# data_pipeline.py
import os
import random
import torch
import numpy as np

import timm
import timm.data
from torchvision import datasets, transforms
try:
    from torchvision.transforms import v2 as tv2
except Exception:
    tv2 = None

# Prefer timm Mixup if available (recommended). Otherwise fallback to simple mixup function.
try:
    from timm.data import Mixup
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

    class SimpleMixup:
        def __init__(self, alpha=0.2, prob=1.0, num_classes=1000):
            self.alpha = alpha
            self.prob = prob
            self.num_classes = num_classes

        def __call__(self, x, y):
            if self.alpha <= 0 or random.random() > self.prob:
                # return one-hot labels for consistency with timm.Mixup usage
                y_onehot = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float()
                return x, y_onehot
            lam = np.random.beta(self.alpha, self.alpha)
            bs = x.size(0)
            perm = torch.randperm(bs, device=x.device)
            x2 = x[perm]
            y2 = y[perm]
            x = lam * x + (1 - lam) * x2
            y_onehot = lam * torch.nn.functional.one_hot(y, num_classes=self.num_classes).float() + \
                       (1 - lam) * torch.nn.functional.one_hot(y2, num_classes=self.num_classes).float()
            return x, y_onehot

# -------------------------
# CONFIG (change these)
# -------------------------
img_size = 224
batch_size = 64                  # per-process batch size
num_workers = 8                  # tune to your CPU cores
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 1000
mixup_alpha = 0.2
cutmix_alpha = 1.0
mixup_prob = 1.0
label_smoothing = 0.1
# -------------------------

# Seed/worker init for reproducibility and better randomness in workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_transforms(img_size=224, device=device):
    """
    Returns (train_transform, val_transform) where each transform:
    - if torchvision v2 available & ToDevice/ToDtype present => uses GPU transforms
    - else uses standard CPU transforms
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # Use v2 (GPU transforms) if available (fast)
    if tv2 is not None and hasattr(tv2, "ToDevice") and hasattr(tv2, "ToDtype"):
        print("⚡ Using torchvision v2 GPU transforms")
        train_t = tv2.Compose([
            tv2.RandomResizedCrop(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            tv2.RandomHorizontalFlip(),
            tv2.RandAugment(num_ops=2, magnitude=9),   # strong augmentation
            tv2.ColorJitter(0.4, 0.4, 0.4, 0.1),
            tv2.RandomErasing(p=0.25, scale=(0.02, 0.33)),
            tv2.ToDtype(torch.float32, scale=True),    # convert & scale to [0,1]
            tv2.Normalize(mean=mean, std=std),
            tv2.ToDevice(device=device),                # move to GPU early (speeds up training)
        ])

        val_t = tv2.Compose([
            tv2.Resize(int(img_size * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
            tv2.CenterCrop(img_size),
            tv2.ToDtype(torch.float32, scale=True),
            tv2.Normalize(mean=mean, std=std),
            tv2.ToDevice(device=device),
        ])
    else:
        print("🧠 Using CPU torchvision transforms (v2 not available)")
        train_t = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        val_t = transforms.Compose([
            transforms.Resize(int(img_size * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return train_t, val_t

def build_dataloaders(data_dir, img_size=224, batch_size=64, num_workers=8, distributed=False):
    """
    data_dir should have ImageNet structure:
      data_dir/train/<class>/*.JPEG
      data_dir/val/<class>/*.JPEG
    Returns: train_loader, val_loader, mixup_fn, train_sampler (if distributed)
    """
    train_transform, val_transform = make_transforms(img_size)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # ImageFolder (works with standard ImageNet layout)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    if distributed:
        # DDP - use DistributedSampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, drop_last=True, sampler=train_sampler, worker_init_fn=seed_worker,
        persistent_workers=True, prefetch_factor=4
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers//2,
        pin_memory=True, drop_last=False, worker_init_fn=seed_worker,
        persistent_workers=True, prefetch_factor=2
    )

    # Mixup/cutmix function
    if _HAS_TIMM:
        mixup_fn = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=mixup_prob,
            switch_prob=0.5,
            mode='batch',
            label_smoothing=label_smoothing,
            num_classes=num_classes
        )
    else:
        # fallback - returns one-hot labels when not mixing, consistent interface
        mixup_fn = SimpleMixup(alpha=mixup_alpha, prob=mixup_prob, num_classes=num_classes)

    return train_loader, val_loader, mixup_fn, train_sampler

# -------------------------
# USAGE EXAMPLE
# -------------------------
if __name__ == "__main__":
    # Example usage (non-distributed):
    DATA_DIR = "/path/to/imagenet"   # change this
    train_loader, val_loader, mixup_fn, _ = build_dataloaders(
        DATA_DIR, img_size=img_size, batch_size=batch_size, num_workers=num_workers, distributed=False
    )

    # In training loop:
    device = torch.device(device)
    model = ...  # create your resnet50 model and move to device if not using v2.ToDevice
    criterion = torch.nn.CrossEntropyLoss()  # if using timm.Mixup, it handles smoothing internally

    for epoch in range(0, 1):
        model.train()
        for images, targets in train_loader:
            # If using torchvision v2 + ToDevice, images are already on GPU.
            if tv2 is None or not (hasattr(tv2, "ToDevice") and hasattr(tv2, "ToDtype")):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            else:
                # v2 pipeline already moved images to GPU, but targets still CPU
                targets = targets.to(device, non_blocking=True)

            # Apply mixup
            mixed_images, mixed_targets = mixup_fn(images, targets)

            # If mixup returns one-hot soft labels, ensure your loss expects that (timm.Mixup works with CE)
            outputs = model(mixed_images)
            if _HAS_TIMM:
                loss = criterion(outputs, mixed_targets)  # timm Mixup may return soft targets, but timm's recipes use standard CE
            else:
                # fallback: mixed_targets are soft one-hot vectors => use KLDiv or soft-label aware CE
                # A common approach: use BCEWithLogits for multi-label or logits+soft-target cross-entropy:
                loss = soft_cross_entropy(outputs, mixed_targets)  # implement or use torch.nn.functional.kl_div approach

            # backward / optimizer / scheduler steps...
