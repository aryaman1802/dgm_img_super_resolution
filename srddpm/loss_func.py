import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=35, use_bn=False):
        """
        VGG feature extractor to compute perceptual loss.
        Args:
            feature_layer (int): Index of the layer up to which features are extracted.
            use_bn (bool): Whether to use VGG with batch normalization.
        """
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            vgg19 = models.vgg19_bn(pretrained=True)
        else:
            vgg19 = models.vgg19(pretrained=True)
        # Extract features up to the specified layer.
        self.features = nn.Sequential(*list(vgg19.features.children())[:feature_layer+1])
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()  # set to evaluation mode

    def forward(self, x):
        return self.features(x)

class SRDDPMLosses(nn.Module):
    def __init__(self, reconstruction_weight=1.0, perceptual_weight=1.0, diffusion_weight=1.0):
        """
        Combines the diffusion loss, reconstruction loss, and perceptual loss.
        Args:
            reconstruction_weight (float): Weight for the reconstruction (L1) loss.
            perceptual_weight (float): Weight for the perceptual loss.
            diffusion_weight (float): Weight for the diffusion (MSE) loss.
        """
        super(SRDDPMLosses, self).__init__()
        self.reconstruction_loss_fn = nn.L1Loss()
        self.diffusion_loss_fn = nn.MSELoss()
        self.vgg_extractor = VGGFeatureExtractor(feature_layer=35)  # Extract features from an intermediate layer
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.diffusion_weight = diffusion_weight

    def forward(self, predicted_noise, true_noise, generated_hr, ground_truth_hr):
        """
        Computes the combined loss.
        Args:
            predicted_noise (torch.Tensor): Noise predicted by the model (from the reverse process).
            true_noise (torch.Tensor): Ground truth noise used in the forward diffusion.
            generated_hr (torch.Tensor): Generated high-resolution image.
            ground_truth_hr (torch.Tensor): Ground truth high-resolution image.
        Returns:
            total_loss (torch.Tensor): The weighted sum of diffusion, reconstruction, and perceptual losses.
            loss_dict (dict): A dictionary with individual loss components.
        """
        # Diffusion Loss: MSE between predicted and true noise
        diffusion_loss = self.diffusion_loss_fn(predicted_noise, true_noise)
        
        # Reconstruction Loss: L1 loss between generated HR and ground truth HR
        recon_loss = self.reconstruction_loss_fn(generated_hr, ground_truth_hr)
        
        # Perceptual Loss: Use VGG features to compare generated and ground truth images.
        # Normalize images using ImageNet statistics.
        normalization_mean = torch.tensor([0.485, 0.456, 0.406], device=ground_truth_hr.device).view(1, 3, 1, 1)
        normalization_std = torch.tensor([0.229, 0.224, 0.225], device=ground_truth_hr.device).view(1, 3, 1, 1)
        gt_norm = (ground_truth_hr - normalization_mean) / normalization_std
        gen_norm = (generated_hr - normalization_mean) / normalization_std
        
        # Extract VGG features (detach to ensure VGG remains fixed)
        vgg_features_gt = self.vgg_extractor(gt_norm).detach()
        vgg_features_gen = self.vgg_extractor(gen_norm)
        perceptual_loss = self.diffusion_loss_fn(vgg_features_gen, vgg_features_gt)
        
        # Combine losses with their respective weights
        total_loss = (self.diffusion_weight * diffusion_loss +
                      self.reconstruction_weight * recon_loss +
                      self.perceptual_weight * perceptual_loss)
        
        loss_dict = {
            'diffusion_loss': diffusion_loss,
            'recon_loss': recon_loss,
            'perceptual_loss': perceptual_loss
        }
        return total_loss, loss_dict
