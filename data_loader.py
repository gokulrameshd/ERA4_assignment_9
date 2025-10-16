import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
            v2.RandomResizedCrop(img_size),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
            v2.ToDevice(device="cuda"),
        ])

        val_transforms = v2.Compose([
            v2.Resize(int(img_size * 1.14)),
            v2.CenterCrop(img_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
            v2.ToDevice(device="cuda"),
        ])
    else:
        print("🧠 Using standard CPU transforms (no v2 GPU support detected)")
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
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
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    num_classes = len(train_dataset.classes)
    print(f"✅ Loaded dataset with {num_classes} classes using {num_workers} workers.")
    return train_loader, val_loader, num_classes
