import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2

class SingleImageDataset(Dataset):
    """
    Dataset for a single land plan image with optional HSV-based pseudo-mask.
    """
    def __init__(self, img_path, pseudo_mask=True, transform=None):
        # Load image and convert from BGR (OpenCV default) to RGB
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.orig_h, self.orig_w = img.shape[:2]
        self.img = img
        self.transform = transform

        # Generate a two-channel mask: [background, red-pixels]
        if pseudo_mask:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            m1 = cv2.inRange(hsv, (0,30,50), (10,255,255))     # lower red range
            m2 = cv2.inRange(hsv, (160,30,50), (180,255,255))  # upper red range
            red = (m1 | m2).astype(np.uint8)
            self.mask = np.stack([1 - red, red], axis=-1)
        else:
            self.mask = None

    def __len__(self):
        return 1  # Single image dataset

    def __getitem__(self, idx):
        # Prepare image and mask for output
        data = {"image": self.img}
        if self.mask is not None:
            data["mask"] = self.mask

        # Apply Albumentations transform if provided
        if self.transform:
            aug = self.transform(image=data["image"], mask=data.get("mask"))
            return aug["image"], aug.get("mask")

        # Fallback: manually convert image and mask to tensors
        img_t = ToTensorV2()(image=self.img)["image"]
        mask_t = None
        if self.mask is not None:
            mask_t = torch.from_numpy(self.mask).permute(2, 0, 1).long()  # OpenCV (Height, Width, Channels) HWC → PyTorch CHW 
        return img_t, mask_t

# Albumentations transform pipeline: resize, normalize, convert to tensor
TRANSFORM = Compose([
    Resize(512, 512),
    Normalize(),
    ToTensorV2()
])