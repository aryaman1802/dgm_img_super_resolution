import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Residual Dense Block (RDB)
class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels=32, num_layers=5):
        super(ResidualDenseBlock, self).__init__()
        self.num_layers = num_layers
        self.growth_channels = growth_channels
        
        # Create convolutional layers for dense connections
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Conv2d(in_channels + i * growth_channels, growth_channels, kernel_size=3, padding=1)
            )
        # Local feature fusion layer to combine features from all layers
        self.lff = nn.Conv2d(in_channels + num_layers * growth_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            # Concatenate previous features along the channel dimension
            concat_features = torch.cat(features, dim=1)
            out = F.relu(layer(concat_features))
            features.append(out)
        # Fuse all concatenated features
        concat_features = torch.cat(features, dim=1)
        fused = self.lff(concat_features)
        # Apply residual connection with scaling to stabilize training
        return fused * 0.2 + x

# Conditional Network using multiple Residual Dense Blocks
class ConditionalNet(nn.Module):
    def __init__(self, in_channels=3, num_features=64, num_blocks=5):
        super(ConditionalNet, self).__init__()
        # Initial convolution layer to extract basic features
        self.conv_first = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
        # Sequence of ResidualDenseBlocks
        self.rdb_blocks = nn.Sequential(
            *[ResidualDenseBlock(num_features) for _ in range(num_blocks)]
        )
        
        # Final convolution to produce the conditioned feature map
        self.conv_last = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
    
    def forward(self, x):
        out = self.conv_first(x)
        out = self.rdb_blocks(out)
        out = self.conv_last(out)
        return out


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal embeddings for the given timesteps.
    timesteps: a tensor of shape (N,)
    embedding_dim: dimension of the embedding
    """
    half_dim = embedding_dim // 2
    emb_factor = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb_factor)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x, t_emb):
        h = self.conv(x)
        # Incorporate time embedding: broadcast to spatial dimensions
        time_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h_down = self.downsample(h)
        return h_down, h  # Return downsampled feature and skip connection

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(UpBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
    
    def forward(self, x, skip, t_emb):
        h = self.upconv(x)
        # Concatenate skip connection from down path
        h = torch.cat([h, skip], dim=1)
        h = self.conv(h)
        time_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        return h

class UNetDiffusion(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=128):
        super(UNetDiffusion, self).__init__()
        self.time_emb_dim = time_emb_dim
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial convolution to process the noisy input
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Downsampling path
        self.down1 = DownBlock(base_channels, base_channels*2, time_emb_dim)
        self.down2 = DownBlock(base_channels*2, base_channels*4, time_emb_dim)
        
        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Upsampling path
        self.up1 = UpBlock(base_channels*4, base_channels*2, time_emb_dim)
        self.up2 = UpBlock(base_channels*2, base_channels, time_emb_dim)
        
        # Final convolution to produce the denoised output image
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t, cond_features=None):
        """
        x: Noisy image tensor of shape (B, C, H, W)
        t: Tensor containing the time step (e.g., shape (B,))
        cond_features: Optional conditional features from the Conditional Network
        """
        # Generate and process time embeddings
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Initial feature extraction
        h0 = self.init_conv(x)
        
        # Downsampling with skip connections
        h1, skip1 = self.down1(h0, t_emb)
        h2, skip2 = self.down2(h1, t_emb)
        
        # Bottleneck processing
        h_mid = self.bottleneck(h2)
        
        # Upsampling and merging with skip connections
        h_up1 = self.up1(h_mid, skip2, t_emb)
        h_up2 = self.up2(h_up1, skip1, t_emb)
        
        # Optionally add conditional features from the LR image
        if cond_features is not None:
            h_up2 = h_up2 + cond_features
        
        # Final convolution to produce the output image
        out = self.final_conv(h_up2)
        return out


class DiffusionSchedule:
    def __init__(self, num_timesteps, beta_start=1e-4, beta_end=0.02):
        """
        Initializes the diffusion schedule.
        num_timesteps: Total number of diffusion steps.
        beta_start, beta_end: Defines the linear schedule for beta.
        """
        self.num_timesteps = num_timesteps
        # Create a linear schedule for beta
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        # Calculate alphas: αₜ = 1 - βₜ
        self.alphas = 1.0 - self.betas
        # Compute cumulative product: \bar{α}_t = ∏_{s=1}^t α_s
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def get_alpha_bar(self, t):
        """
        Retrieve \bar{α}_t for a given time step t.
        t: A tensor of time steps (shape: [batch_size])
        Returns: Tensor of corresponding alpha_bar values (shape: [batch_size, 1, 1, 1])
        """
        # Indexing into alpha_bars for each t and reshape to broadcast over image dimensions
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return alpha_bar

def forward_diffusion_sample(x0, t, schedule):
    """
    Perform the forward diffusion process.
    x0: Original clean image tensor of shape (B, C, H, W)
    t: Tensor containing the diffusion time steps (shape: [B])
    schedule: Instance of DiffusionSchedule
    Returns: Noisy image x_t and the sampled noise epsilon.
    """
    # Get corresponding alpha_bar for each t
    alpha_bar = schedule.get_alpha_bar(t)
    # Sample random Gaussian noise
    noise = torch.randn_like(x0)
    # Compute noisy image using the forward process equation
    xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
    return xt, noise


# Define a simple Residual Block used in the Residual Prediction Module.
class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + identity

# Residual Prediction Module.
class ResidualPredictionNet(nn.Module):
    def __init__(self, in_channels=3, num_features=64, num_residual_blocks=5, scale_factor=4):
        """
        in_channels: Number of channels in the input image (e.g., 3 for RGB).
        num_features: Number of feature maps used in intermediate layers.
        num_residual_blocks: How many residual blocks to use.
        scale_factor: Upscaling factor to reach high-resolution.
        """
        super(ResidualPredictionNet, self).__init__()
        # Upsampling layer using bilinear interpolation.
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        # An initial convolution layer to process the upsampled image.
        self.entry_conv = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        # Residual blocks to learn the missing details.
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_residual_blocks)]
        )
        # A final convolution layer to predict the residual image.
        self.exit_conv = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1)
    
    def forward(self, lr_image):
        # Upsample the low-resolution image to the desired high-resolution size.
        upsampled = self.upsample(lr_image)
        x = self.entry_conv(upsampled)
        x = self.residual_blocks(x)
        residual = self.exit_conv(x)
        # The final prediction is the upsampled image plus the predicted residual.
        hr_pred = upsampled + residual
        return hr_pred

