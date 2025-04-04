import torch
import torch.nn as nn
import torchvision.models as models

# VGG Feature Extractor for Perceptual Loss
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=35, use_bn=False, use_input_norm=True, device=torch.device('cuda')):
        super(VGGFeatureExtractor, self).__init__()
        # Load pre-trained VGG19 (with or without batch normalization)
        if use_bn:
            vgg_model = models.vgg19_bn(pretrained=True)
        else:
            vgg_model = models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        # Register mean and std buffers for normalization (ImageNet stats)
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        # Extract features up to the specified layer
        self.features = nn.Sequential(*list(vgg_model.features.children())[:feature_layer])
        # Freeze the VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Normalize input if required
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        return self.features(x)

# Perceptual Loss using VGG Feature Extractor
class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=35, use_bn=False, weight=1.0, device=torch.device('cuda')):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatureExtractor(feature_layer, use_bn, use_input_norm=True, device=device)
        self.criterion = nn.MSELoss()
        self.weight = weight

    def forward(self, sr, hr):
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        loss = self.criterion(sr_features, hr_features) * self.weight
        return loss

# Pixel-wise Loss (either MSE or L1)
class PixelLoss(nn.Module):
    def __init__(self, mode='MSE', weight=1.0):
        super(PixelLoss, self).__init__()
        if mode == 'MSE':
            self.criterion = nn.MSELoss()
        elif mode == 'L1':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError("Unsupported mode: choose 'MSE' or 'L1'")
        self.weight = weight

    def forward(self, sr, hr):
        loss = self.criterion(sr, hr) * self.weight
        return loss

# Adversarial Loss using Binary Cross Entropy
class AdversarialLoss(nn.Module):
    def __init__(self, weight=1e-3):
        super(AdversarialLoss, self).__init__()
        self.criterion = nn.BCELoss()
        self.weight = weight

    def forward(self, pred, target):
        loss = self.criterion(pred, target) * self.weight
        return loss

