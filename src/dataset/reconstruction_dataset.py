import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class ReconstructionDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None, augment=False):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.augment = augment

        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.tif', '.png', '.jpg'))])
        self.mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith(('.tif', '.png', '.jpg'))])

        print(f"Found {len(self.image_files)} images and {len(self.mask_files)} masks.")
        
        image_basenames = set([os.path.splitext(f)[0] for f in self.image_files])
        mask_basenames = set([os.path.splitext(f)[0] for f in self.mask_files])
        matched_basenames = image_basenames.intersection(mask_basenames)

        self.image_files = [f for f in self.image_files if os.path.splitext(f)[0] in matched_basenames]
        self.mask_files = [f for f in self.mask_files if os.path.splitext(f)[0] in matched_basenames]
        
        print(f"Using {len(self.image_files)} matched images and masks.")

        if self.augment:
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ])
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_files[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        if self.augment and self.augmentation:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.augmentation(image)
            torch.manual_seed(seed)
            mask = self.augmentation(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask