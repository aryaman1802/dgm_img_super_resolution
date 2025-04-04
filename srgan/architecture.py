import torch
import torch.nn as nn

# Define a single Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # Skip connection adds the input to the block's output
        return out + residual

# Define an Upsample Block using sub-pixel convolution (PixelShuffle)
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.prelu(x)

# Define the Generator Network
class Generator(nn.Module):
    def __init__(self, num_residual_blocks=16, upscale_factor=4):
        super(Generator, self).__init__()
        # Initial convolution to extract low-level features
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.prelu1 = nn.PReLU()
        
        # Series of residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )
        
        # Post-residual block convolution to fuse features
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Upsampling layers: If upscale_factor is 4, we need two upsample blocks (each scaling by 2)
        upsample_blocks = []
        num_upsample = int(upscale_factor // 2)
        for _ in range(num_upsample):
            upsample_blocks.append(UpsampleBlock(64, scale_factor=2))
        self.upsample = nn.Sequential(*upsample_blocks)
        
        # Final output layer to generate the high-resolution image
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)
        
    def forward(self, x):
        # Initial feature extraction
        out1 = self.prelu1(self.conv1(x))
        # Deep feature extraction through residual blocks
        out = self.residual_blocks(out1)
        # Fuse features with an additional convolution and skip connection
        out = self.bn2(self.conv2(out))
        out = out1 + out
        # Upsample to the desired resolution
        out = self.upsample(out)
        # Produce final high-resolution image
        out = self.conv3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_size=96):
        super(Discriminator, self).__init__()
        # Convolutional layers that progressively extract features and reduce spatial dimensions.
        self.net = nn.Sequential(
            # Layer 1: Convolution without batch normalization
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: Downsample by stride=2
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: Increase channel count
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: Downsample
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 5: Further increasing channels
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 6: Downsample
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 7: Increase to 512 channels
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 8: Final downsampling layer
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # After 4 downsampling layers (stride=2 each), a 96x96 image becomes 6x6.
        # Flatten and use fully connected layers for classification.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()  # Output a probability (real vs. fake)
        )
        
    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

