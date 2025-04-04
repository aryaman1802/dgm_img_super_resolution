import os
import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim

class DIV2KDataset(Dataset):
    """
    Custom dataset for DIV2K high-resolution images.
    It assumes images are stored in a directory (hr_dir) in a common image format.
    The dataset applies two transforms:
      - transform_hr: crops and converts the HR image to tensor.
      - transform_lr: downsamples the original PIL image to create the corresponding LR image.
    """
    def __init__(self, hr_dir, crop_size=96, upscale_factor=4):
        super(DIV2KDataset, self).__init__()
        self.hr_image_paths = glob.glob(os.path.join(hr_dir, '*.png'))
        if len(self.hr_image_paths) == 0:
            self.hr_image_paths = glob.glob(os.path.join(hr_dir, '*.jpg'))
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor

        # Transform for HR images: random crop and convert to tensor.
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor()
        ])
        # Transform for LR images: random crop the same region and then downscale.
        self.lr_transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.hr_image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.hr_image_paths[idx]).convert("RGB")
        # To ensure corresponding crops, we apply the same random crop parameters:
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.crop_size, self.crop_size))
        hr_image = transforms.functional.crop(img, i, j, h, w)
        lr_image = hr_image.resize((self.crop_size // self.upscale_factor, self.crop_size // self.upscale_factor), Image.BICUBIC)
        # Finally, convert both to tensors.
        hr_tensor = transforms.ToTensor()(hr_image)
        lr_tensor = transforms.ToTensor()(lr_image)
        return lr_tensor, hr_tensor

