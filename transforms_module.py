import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ========================
# Albumentations Dataset Wrapper
# ========================
class AlbumentationsDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img)  # convert PIL → numpy
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

def get_cifar100_albumentations_transforms_train_val_1():
    train_transform = A.Compose([
        A.PadIfNeeded(min_height=40, min_width=40, border_mode=0, value=(0,0,0)),
        A.RandomCrop(32, 32),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.7),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.7),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, p=0.5),
        A.Normalize(mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761)),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761)),
        ToTensorV2()
    ])
    return train_transform, test_transform


class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        # img is a PIL Image from torchvision dataset
        img = np.array(img)  # convert to numpy
        augmented = self.transform(image=img)  # Albumentations expects keyword arg
        return augmented["image"]  # return only the image


def get_cifar100_albumentations_transforms_train(CIFAR100_MEAN, CIFAR100_STD,CIFAR100_MEAN_255):
    """
    Creates an Albumentations Compose object for CIFAR-10 augmentation.
    """

    return A.Compose([
        A.PadIfNeeded(min_height=40, min_width=40, border_mode=0, value=(0,0,0)),
        A.RandomCrop(32, 32),
        # 1. Horizontal Flip
        A.HorizontalFlip(p=0.5),

        # 2. ShiftScaleRotate
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=10,
            p=0.5
        ),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.7),
        # 3. CoarseDropout (Cutout)
        A.CoarseDropout(
            p=0.5,
            max_holes=1,
            max_height=16,
            max_width=16,
            min_holes=1,
            min_height=16,
            min_width=16,
            fill_value=CIFAR100_MEAN_255,  # must match pre-normalization scale
            mask_fill_value=None
        ),

        # 4. Normalize and convert to Tensor
        A.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])
    # Train transforms


def get_cifar100_albumentations_transforms_test(CIFAR100_MEAN, CIFAR100_STD):
    """
    Creates an Albumentations Compose object for CIFAR-10 test augmentation.
    """

    return A.Compose([
        A.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])

def get_cifar100_albumentations_transforms_train_val_2(CIFAR100_MEAN, CIFAR100_STD,CIFAR100_MEAN_255):
    """
    Creates an Albumentations Compose object for CIFAR-100 train augmentation.
    """
    train_transforms = get_cifar100_albumentations_transforms_train(CIFAR100_MEAN, CIFAR100_STD,CIFAR100_MEAN_255)
    test_transforms = get_cifar100_albumentations_transforms_test(CIFAR100_MEAN, CIFAR100_STD)
    return train_transforms, test_transforms

