import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from albumentations import Compose, Normalize, Resize, HorizontalFlip, RandomRotate90, ColorJitter
from albumentations.pytorch import ToTensorV2

class SingleImageDataset(Dataset):
    """
    For a single image/mask pair.
    """
    def __init__(self, img_path, mask_path=None, transform=None):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = img
        self.orig_h, self.orig_w = img.shape[:2]
        self.transform = transform
        self.mask = None
        if mask_path:
            mask = cv2.imread(mask_path, 0)  # Load as 2D array
            if mask is None:
                raise FileNotFoundError(f"Mask not found at {mask_path}")
            self.mask = mask

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img = self.img
        mask = self.mask
        # Albumentations: handle case when mask=None (inference)
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=img, mask=mask)
                img_t = augmented["image"]
                mask_t = augmented["mask"].long()
                return img_t, mask_t
            else:
                img_t = self.transform(image=img)["image"]
                dummy_mask = torch.zeros((img_t.shape[1], img_t.shape[2]), dtype=torch.long)
                return img_t, dummy_mask
        # No transform
        return img, mask
    
# --- Augmentations for training and validation ---
TRAIN_TRANSFORM = Compose([
    Resize(512, 512),
    HorizontalFlip(p=0.5),
    RandomRotate90(p=0.5),
    ColorJitter(p=0.5),
    Normalize(),
    ToTensorV2()
])
VAL_TRANSFORM = Compose([
    Resize(512, 512),
    Normalize(),
    ToTensorV2()
])
