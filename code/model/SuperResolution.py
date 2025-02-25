import os

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


class SuperResolutionDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, hr_transform=None, lr_transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform
        self.hr_image_names = [img_name for img_name in os.listdir(hr_dir) if img_name.endswith(('.png', '.jpg', '.jpeg'))]
        self.lr_image_names = [img_name for img_name in os.listdir(lr_dir) if img_name.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.hr_image_names)

    def __getitem__(self, idx):
        # print(self.image_names)
        hr_img_name = self.hr_image_names[idx]
        lr_img_name = self.lr_image_names[idx]

        hr_img_path = os.path.join(self.hr_dir, hr_img_name)
        lr_img_path = os.path.join(self.lr_dir, lr_img_name)

        hr_image = Image.open(hr_img_path).convert("L")
        # hr_image =np.array(hr_image)/255.0
        lr_image = Image.open(lr_img_path).convert("L")
        # lr_image = np.array(lr_image) / 255.0

        hr_image = np.array(hr_image).astype(np.float32)
        lr_image = np.array(lr_image).astype(np.float32)
        # # 归一化到 [0, 1]
        hr_image = hr_image / np.max(hr_image)
        lr_image = lr_image / np.max(lr_image)

        if self.hr_transform:
            hr_image = self.hr_transform(hr_image)

        if self.lr_transform:
            lr_image = self.lr_transform(lr_image)



        return lr_image, hr_image



