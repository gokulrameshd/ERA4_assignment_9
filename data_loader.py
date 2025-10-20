import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import numpy as np
import torch.nn.functional as F

# Seed/worker init for reproducibility and better randomness in workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloaders(data_dir="data", batch_size=128, img_size=224):
    """Universal DataLoader setup.
    - Uses GPU transforms if torchvision v2 supports them
    - Falls back to CPU transforms otherwise
    """

    num_workers = min(8, max(4, torch.get_num_threads() // 2))
    torch.backends.cudnn.benchmark = True

    # Try importing torchvision.v2 transforms
    try:
        from torchvision.transforms import v2
        has_gpu_transforms = hasattr(v2, "ToDevice") and hasattr(v2, "ToDtype")
    except Exception:
        has_gpu_transforms = False

    if has_gpu_transforms:
        print("⚡ Using GPU-accelerated torchvision.v2 transforms")
        train_transforms = v2.Compose([
            v2.RandomResizedCrop(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandAugment(num_ops=2, magnitude=9),  # Core augment
            v2.ColorJitter(0.4, 0.4, 0.4, 0.1),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.33)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
            v2.ToDevice(device="cuda"),
        ])

        val_transforms = v2.Compose([
            v2.Resize(int(img_size * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
            v2.CenterCrop(img_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
            v2.ToDevice(device="cuda"),
        ])
    else:
        print("🧠 Using standard CPU transforms (no v2 GPU support detected)")
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(int(img_size * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    # Create datasets and loaders
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=4,
        drop_last=True,   # safe with mixup
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=4,
        drop_last=False,   # safe with mixup
    )

    num_classes = len(train_dataset.classes)
    print(f"✅ Loaded dataset with {num_classes} classes using {num_workers} workers.")
    return train_loader, val_loader, num_classes

class SimpleMixup:
    def __init__(self, alpha=0.2, prob=1.0, num_classes=1000):
        self.alpha = alpha
        self.prob = prob
        self.num_classes = num_classes

    def __call__(self, x, y):
        if self.alpha <= 0 or random.random() > self.prob:
            y_onehot = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float()
            return x, y_onehot, y_onehot, 1.0  # <- y_a, y_b, lam=1 (no mix)
        lam = np.random.beta(self.alpha, self.alpha)
        bs = x.size(0)
        perm = torch.randperm(bs, device=x.device)
        x2 = x[perm]
        y2 = y[perm]
        x = lam * x + (1 - lam) * x2
        y_a = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float()
        y_b = torch.nn.functional.one_hot(y2, num_classes=self.num_classes).float()
        return x, y_a, y_b, lam


def get_mixup_fn(mixup_alpha=0.2, cutmix_alpha=1.0, mixup_prob=1.0, label_smoothing=0.1, num_classes=1000):
    # Prefer timm Mixup if available (recommended). Otherwise fallback to simple mixup function.
    _HAS_TIMM = False
    try:
        from timm.data import Mixup
        _HAS_TIMM = True
    except Exception:
        _HAS_TIMM = False
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
    return mixup_fn