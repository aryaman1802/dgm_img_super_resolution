import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiffusionProcess:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        """
        Initializes the diffusion process.
        Args:
            T (int): Number of diffusion timesteps.
            beta_start (float): Starting value for the beta schedule.
            beta_end (float): Ending value for the beta schedule.
            device (str): Device to run the computations ('cuda' or 'cpu').
        """
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
    
    def get_index_from_list(self, vals, t, x_shape):
        """
        Retrieves the appropriate scalar for each sample in the batch given timestep t.
        Args:
            vals (torch.Tensor): Tensor of shape [T] containing values for each timestep.
            t (torch.Tensor): Tensor of shape [batch_size] containing timestep indices.
            x_shape (tuple): Shape of the target tensor (for proper broadcasting).
        Returns:
            torch.Tensor: A tensor of shape [batch_size, 1, 1, 1] (if image) for broadcasting.
        """
        batch_size = t.shape[0]
        out = vals[t].view(batch_size, *((1,)*(len(x_shape)-1)))
        return out

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t|x_0)
        x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1 - alpha_bar_t)*noise
        
        Args:
            x_start (torch.Tensor): The original high-resolution image tensor.
            t (torch.Tensor): Timestep tensor of shape [batch_size].
            noise (torch.Tensor, optional): Pre-specified noise. If None, noise is sampled.
        Returns:
            torch.Tensor: The noisy image at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar_t = self.get_index_from_list(self.sqrt_alpha_bars, t, x_start.shape)
        sqrt_one_minus_alpha_bar_t = self.get_index_from_list(self.sqrt_one_minus_alpha_bars, t, x_start.shape)
        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
    
    @staticmethod
    def get_timestep_embedding(timesteps, embedding_dim):
        """
        Generates sinusoidal timestep embeddings.
        Args:
            timesteps (torch.Tensor): 1-D tensor of N indices, one per batch element.
            embedding_dim (int): Dimension of the embedding.
        Returns:
            torch.Tensor: Embeddings of shape [batch_size, embedding_dim].
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad if embedding_dim is odd
            emb = torch.cat([emb, torch.zeros(timesteps.size(0), 1, device=emb.device)], dim=1)
        return emb

    def p_sample(self, model, x, t, lr):
        """
        Reverse diffusion process for one timestep.
        Predicts the noise using the model and computes x_{t-1} based on the DDPM formulation.
        
        Args:
            model (nn.Module): The denoising model (conditional on LR) predicting noise.
            x (torch.Tensor): The current image tensor at timestep t.
            t (torch.Tensor): Timestep tensor of shape [batch_size].
            lr (torch.Tensor): Low-resolution image tensor used for conditioning.
        Returns:
            torch.Tensor: The image tensor at timestep t-1.
        """
        # Predict noise using the model
        predicted_noise = model(x, t, lr)
        
        # Retrieve schedule values for current timesteps
        beta_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alpha_bar_t = self.get_index_from_list(self.sqrt_one_minus_alpha_bars, t, x.shape)
        sqrt_recip_alphas = 1.0 / torch.sqrt(self.get_index_from_list(self.alphas, t, x.shape))
        
        # Compute the mean following DDPM reverse process:
        # mu = 1/sqrt(alpha_t) * (x - (beta_t/sqrt(1 - alpha_bar_t)) * predicted_noise)
        model_mean = sqrt_recip_alphas * (x - beta_t / sqrt_one_minus_alpha_bar_t * predicted_noise)
        
        # For t > 0, add noise; for t == 0, return the mean
        noise = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
        sigma_t = torch.sqrt(beta_t)
        x_prev = model_mean + sigma_t * noise
        return x_prev

    def sample(self, model, lr, shape):
        """
        Generates a high-resolution image from noise, conditioned on an LR image.
        Iteratively applies p_sample for each timestep.
        
        Args:
            model (nn.Module): The denoising model (conditional on LR).
            lr (torch.Tensor): The low-resolution conditioning image.
            shape (tuple): The shape of the output image (batch_size, C, H, W).
        Returns:
            torch.Tensor: The generated high-resolution image.
        """
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=self.device)
            x = self.p_sample(model, x, t_tensor, lr)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_embedding_dim, out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, t_emb):
        # x: (batch, channels, H, W)
        out = self.activation(self.conv1(x))
        # Process time embedding and add to feature map
        t_emb_processed = self.activation(self.time_mlp(t_emb)).unsqueeze(-1).unsqueeze(-1)
        out = out + t_emb_processed
        out = self.activation(self.conv2(out))
        return out + self.residual_conv(x)

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, cond_channels=3, base_channels=64, time_embedding_dim=128):
        """
        Args:
            in_channels (int): Number of channels in the noisy HR image.
            cond_channels (int): Number of channels in the LR conditioning image.
            base_channels (int): Base number of feature channels.
            time_embedding_dim (int): Dimension of the timestep embedding.
        """
        super().__init__()
        # Process the LR conditioning image
        self.lr_processor = nn.Sequential(
            nn.Conv2d(cond_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Time embedding MLP: expects a tensor of shape (batch, 1)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        
        # Encoder: concatenate LR features with noisy HR input
        self.enc1 = ResidualBlock(in_channels + base_channels, base_channels, time_embedding_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2, time_embedding_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4, time_embedding_dim)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 8, time_embedding_dim)
        
        # Decoder with skip connections
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base_channels * 8, base_channels * 4, time_embedding_dim)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels * 2, time_embedding_dim)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels, time_embedding_dim)
        
        # Final layer to predict noise residual
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)
    
    def forward(self, x, t, lr):
        """
        Args:
            x (torch.Tensor): Noisy HR image, shape [B, in_channels, H, W].
            t (torch.Tensor): Timestep tensor, shape [B].
            lr (torch.Tensor): LR conditioning image, shape [B, cond_channels, H, W].
        Returns:
            torch.Tensor: Predicted noise residual with shape [B, in_channels, H, W].
        """
        batch_size = x.shape[0]
        # Process timestep: reshape to [B, 1] then embed
        t = t.float().unsqueeze(1)
        t_emb = self.time_mlp(t)  # Shape: [B, time_embedding_dim]
        
        # Process LR image and concatenate with noisy HR image
        lr_features = self.lr_processor(lr)  # Shape: [B, base_channels, H, W]
        x = torch.cat([x, lr_features], dim=1)  # Shape: [B, in_channels + base_channels, H, W]
        
        # Encoder path
        enc1 = self.enc1(x, t_emb)         # [B, base_channels, H, W]
        enc1_pool = self.pool1(enc1)         # [B, base_channels, H/2, W/2]
        enc2 = self.enc2(enc1_pool, t_emb)   # [B, base_channels*2, H/2, W/2]
        enc2_pool = self.pool2(enc2)         # [B, base_channels*2, H/4, W/4]
        enc3 = self.enc3(enc2_pool, t_emb)   # [B, base_channels*4, H/4, W/4]
        enc3_pool = self.pool3(enc3)         # [B, base_channels*4, H/8, W/8]
        
        # Bottleneck
        bottleneck = self.bottleneck(enc3_pool, t_emb)  # [B, base_channels*8, H/8, W/8]
        
        # Decoder path with upsampling and skip connections
        up3 = self.up3(bottleneck)                     # [B, base_channels*4, H/4, W/4]
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1), t_emb)  # [B, base_channels*4, H/4, W/4]
        up2 = self.up2(dec3)                           # [B, base_channels*2, H/2, W/2]
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1), t_emb)  # [B, base_channels*2, H/2, W/2]
        up1 = self.up1(dec2)                           # [B, base_channels, H, W]
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1), t_emb)  # [B, base_channels, H, W]
        
        # Output noise prediction
        out = self.out_conv(dec1)
        return out


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        """
        Patch-based discriminator similar to SRGAN discriminator.
        Args:
            in_channels (int): Number of channels in the input image.
            base_channels (int): Base number of feature maps.
        """
        super(PatchDiscriminator, self).__init__()
        self.net = nn.Sequential(
            # First layer: no batch norm
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Downsampling blocks
            nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final conv layer outputs a patch map of scores
            nn.Conv2d(base_channels * 4, 1, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.net(x)

def discriminator_hinge_loss(real_scores, fake_scores):
    """
    Computes the discriminator hinge loss.
    For real images: loss = ReLU(1 - D(x))
    For fake images: loss = ReLU(1 + D(G(z)))
    """
    loss_real = torch.mean(F.relu(1.0 - real_scores))
    loss_fake = torch.mean(F.relu(1.0 + fake_scores))
    return loss_real + loss_fake

def generator_hinge_loss(fake_scores):
    """
    Computes the generator hinge loss:
    loss = -mean(D(G(z)))
    """
    return -torch.mean(fake_scores)
