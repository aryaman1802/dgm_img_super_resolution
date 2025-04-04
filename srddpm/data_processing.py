import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, lr_scale=4, transform=None):
        """
        Args:
            hr_dir (str): Directory with HR images.
            lr_scale (int): Downscaling factor for generating LR images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        super(DIV2KDataset, self).__init__()
        self.hr_dir = hr_dir
        self.hr_images = sorted([
            os.path.join(hr_dir, file) 
            for file in os.listdir(hr_dir) 
            if file.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.lr_scale = lr_scale

        # If a transform is provided, use it; otherwise, use a default set of augmentations.
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.hr_images)
    
    def __getitem__(self, idx):
        # Load HR image and convert to RGB
        hr_path = self.hr_images[idx]
        hr_image = Image.open(hr_path).convert('RGB')
        
        # Apply transformations to HR image
        hr_tensor = self.transform(hr_image)  # Tensor shape: [C, H, W]
        
        # Generate LR image by downsampling using bicubic interpolation
        # Convert tensor back to PIL image for resizing
        hr_pil = transforms.ToPILImage()(hr_tensor)
        C, H, W = hr_tensor.shape
        lr_w, lr_h = W // self.lr_scale, H // self.lr_scale
        lr_pil = hr_pil.resize((lr_w, lr_h), Image.BICUBIC)
        lr_tensor = transforms.ToTensor()(lr_pil)
        
        return {'lr': lr_tensor, 'hr': hr_tensor}

