import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, scale_factor=4, transform=None):
        """
        hr_dir: Path to the directory containing high resolution images.
        scale_factor: Factor by which to downscale the HR image to generate the LR image.
        transform: Optional additional transformations to be applied on the HR image.
        """
        self.hr_dir = hr_dir
        # Collect all image paths with common image extensions.
        self.hr_image_paths = [
            os.path.join(hr_dir, f) for f in os.listdir(hr_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.scale_factor = scale_factor
        self.transform = transform
        self.to_tensor = ToTensor()
    
    def __len__(self):
        return len(self.hr_image_paths)
    
    def __getitem__(self, idx):
        # Load HR image and ensure it's in RGB format.
        hr_path = self.hr_image_paths[idx]
        hr_image = Image.open(hr_path).convert("RGB")
        
        # Optionally apply additional transformations.
        if self.transform:
            hr_image = self.transform(hr_image)
        
        # Convert the HR image to a tensor.
        hr_tensor = self.to_tensor(hr_image)
        
        # Generate the corresponding LR image by downsampling using bicubic interpolation.
        w, h = hr_image.size
        lr_w, lr_h = w // self.scale_factor, h // self.scale_factor
        lr_image = hr_image.resize((lr_w, lr_h), Image.BICUBIC)
        lr_tensor = self.to_tensor(lr_image)
        
        return lr_tensor, hr_tensor

