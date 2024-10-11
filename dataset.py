import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import cv2 

class RetinalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def preprocess_image(self, image):
        # Convert the image to a numpy array
        image_np = np.array(image)

        # Extract the green channel
        green_channel = image_np[:, :, 1]

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_green = clahe.apply(green_channel)

        # Stack the green channel to create an RGB image
        rgb_image = np.stack((enhanced_green, enhanced_green, enhanced_green), axis=-1)

        # Convert back to PIL Image
        return Image.fromarray(rgb_image)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Preprocess the image
        image = self.preprocess_image(image)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask