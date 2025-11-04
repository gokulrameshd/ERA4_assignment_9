import os
import torch
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, transforms
import random
import numpy as np
from math import ceil
import torch.nn.functional as F

# Seed/worker init for reproducibility and better randomness in workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_base_transforms():
    """Universal DataLoader setup.
    - Uses GPU transforms if torchvision v2 supports them
    - Falls back to CPU transforms otherwise
    """

    #num_workers = min(8, max(4, torch.get_num_threads() // 2))
    num_workers =4
    torch.backends.cudnn.benchmark = True

    # Try importing torchvision.v2 transforms
    try:
        from torchvision.transforms import v2
        has_gpu_transforms = hasattr(v2, "ToDevice") and hasattr(v2, "ToDtype")
    except Exception:
        has_gpu_transforms = False

    if has_gpu_transforms:
        print("⚡ Using GPU-accelerated torchvision.v2 transforms")
        train_base_transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandAugment(num_ops=2, magnitude=9),  # Core augment
            v2.ColorJitter(0.4, 0.4, 0.4, 0.1),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.33)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
            v2.ToDevice(device="cuda"),
        ])

        val_base_transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
            v2.ToDevice(device="cuda"),
        ])
    else:
        print("🧠 Using standard CPU transforms (no v2 GPU support detected)")
        train_base_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        val_base_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    return train_base_transforms, val_base_transforms

from torchvision.transforms import v2, InterpolationMode

def get_stage_transforms(img_size, use_mixup=True, stage_idx=0,has_gpu_transforms=True):
    """
    Dynamically adapts transform strength based on stage.
    """
    # Progressive reduction in augmentation magnitude and erasing
    aug_magnitude = max(5, 9 - stage_idx * 1.5)
    erase_prob = max(0.05, 0.25 - stage_idx * 0.04)

    color_jitter_strength = 0.4 if use_mixup else 0.2  # lighter if no mixup

    if has_gpu_transforms:
        print(f"⚡ Using GPU transforms — stage {stage_idx+1} | mag={aug_magnitude:.1f}, erase={erase_prob:.2f}")
        train_tf = v2.Compose([
            v2.RandomResizedCrop(img_size, interpolation=InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandAugment(num_ops=2, magnitude=int(aug_magnitude)),
            v2.ColorJitter(color_jitter_strength, color_jitter_strength, color_jitter_strength, 0.1),
            v2.RandomErasing(p=erase_prob, scale=(0.02, 0.33)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
            v2.ToDevice(device="cuda"),
        ])
        val_tf = v2.Compose([
            v2.Resize(int(img_size * 1.14), interpolation=InterpolationMode.BICUBIC),
            v2.CenterCrop(img_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
            v2.ToDevice(device="cuda"),
        ])
    else:
        print(f"🧠 Using CPU transforms — stage {stage_idx+1} | mag={aug_magnitude:.1f}, erase={erase_prob:.2f}")
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=int(aug_magnitude)),
            transforms.ColorJitter(color_jitter_strength, color_jitter_strength, color_jitter_strength, 0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=erase_prob, scale=(0.02, 0.33)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        val_tf = transforms.Compose([
            transforms.Resize(int(img_size * 1.14), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    return train_tf, val_tf


def get_dataloaders(data_dir="data", batch_size=128, img_size=224, fraction=1.0,stage_index = None,use_mixup = True,
                    use_stagewise_transforms = False,epoch = None,distributed: bool = False):
    """Universal DataLoader setup.
    - Uses GPU transforms if torchvision v2 supports them
    - Falls back to CPU transforms otherwise
    - If distributed=True, uses DistributedSampler and disables shuffle
    """

    num_workers = 4
    torch.backends.cudnn.benchmark = True

    # Try importing torchvision.v2 transforms
    try:
        from torchvision.transforms import v2
        has_gpu_transforms = hasattr(v2, "ToDevice") and hasattr(v2, "ToDtype")
    except Exception:
        has_gpu_transforms = False

    if use_stagewise_transforms:
        if stage_index == None :
            if epoch != None:
                stage_index = epoch//10
        train_transforms,val_transforms = get_stage_transforms(img_size, use_mixup=use_mixup, 
                                                            stage_idx=stage_index, has_gpu_transforms=has_gpu_transforms)
    else:
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
    base_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), val_transforms)
    num_classes = len(base_dataset.classes)
     # Fractional sampling
    
    if fraction < 1.0:
        subset_len = int(len(base_dataset) * fraction)
        train_subset, _ = random_split(base_dataset, [subset_len, len(base_dataset) - subset_len])
        train_dataset = train_subset
    else:
        train_dataset = base_dataset

    # DDP sampler
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        shuffle_train = False
        shuffle_val = False
    else:
        shuffle_train = True
        shuffle_val = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,   # safe with mixup
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle_val,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False,   # safe with mixup
        sampler=val_sampler,
    )

    
    print(f"✅ Loaded dataset with {num_classes} classes using {num_workers} workers.")
    return train_loader, val_loader, num_classes, train_sampler

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


class ProgressiveResizeDataset(torch.utils.data.Dataset):
    def __init__(self, root, base_transform, resize_schedule):
        """
        resize_schedule: dict mapping epoch ranges → image sizes.
            e.g. { (0,9): 128, (10,19): 160, (20,29): 224 }
        """
        self.dataset = datasets.ImageFolder(root)
        self.resize_schedule = resize_schedule
        self.base_transform = base_transform
        self.current_size = None
        self.resize_transform = None
        self.epoch = 0

    def set_epoch(self, epoch):
        """Update resize size dynamically based on epoch."""
        self.epoch = epoch
        for (start, end), size in self.resize_schedule.items():
            if start <= epoch <= end:
                if size != self.current_size:
                    self.current_size = size
                    self.resize_transform = transforms.Resize((size, size))
                break

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.resize_transform:
            img = self.resize_transform(img)
        img = self.base_transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

# --- 🧠 Build DataLoader pool (prebuilt loaders per stage) ---
def make_loader(dataset, img_size, batch_size, is_train=True, num_workers=8,transforms=None):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip() if is_train else transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    dataset.transform = transform
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # keep deterministic=False for speed (ImageNet)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def _get_total_steps(data_dir, stages):
    base_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"))
    total_images = len(base_dataset)
    total_steps = 0
    for s in stages:
        num_samples = ceil(s["fraction"] * total_images)
        steps_per_epoch = ceil(num_samples / s["batch_size"])
        total_steps += steps_per_epoch * s["epochs"]
    return total_steps

def get_total_steps(data_dir, train_transforms =None, stages=None):
    base_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transforms)
    total_steps = 0
    for s in stages:
        subset_len = int(len(base_dataset) * s["fraction"])
        train_subset, _ = random_split(base_dataset, [subset_len, len(base_dataset) - subset_len])
        train_dataset = train_subset
        train_loader = DataLoader(
            train_dataset,
            batch_size=s["batch_size"],
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=True,   # safe with mixup
        )
        # total_steps += len(train_loader)
        total_steps += len(train_loader)*s["epochs"]
    return total_steps

def get_total_steps_stagewise( train_loader, stage_cfg=None):
    # total_steps = len(train_loader)*stage_cfg["epochs"]
    total_steps = len(train_loader)
    return total_steps

def compute_total_steps(data_dir, stages):
    """Compute total optimizer steps across all progressive stages."""
    base_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"))
    total_images = len(base_dataset)
    total_steps = 0
    for s in stages:
        num_samples = int(s["fraction"] * total_images)
        steps_per_epoch = ceil(num_samples / s["batch_size"])
        total_steps += steps_per_epoch * s["epochs"]
    return total_steps
