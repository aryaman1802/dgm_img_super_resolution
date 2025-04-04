import os
import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
from data_processing import DIV2KDataset
from architecture import Generator, Discriminator, UpsampleBlock, ResidualBlock
from loss_functions import PerceptualLoss, PixelLoss, AdversarialLoss

# --- Settings ---
hr_dir = "../input/div2k/"   # Set this to the DIV2K high-resolution images directory.
crop_size = 96
upscale_factor = 4
batch_size = 16
num_pretrain_epochs = 10     # Number of epochs for pre-training the generator.
num_adv_epochs = 50          # Number of epochs for adversarial training.
learning_rate = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialize Dataset and DataLoader ---
dataset = DIV2KDataset(hr_dir=hr_dir, crop_size=crop_size, upscale_factor=upscale_factor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# --- Import previously defined components ---
# Assume the following classes are defined (or imported from another module):
#   Generator, Discriminator, PixelLoss, PerceptualLoss, AdversarialLoss
# For example:
# from srgan_models import Generator, Discriminator, PixelLoss, PerceptualLoss, AdversarialLoss

# Instantiate models
generator = Generator(num_residual_blocks=16, upscale_factor=upscale_factor).to(device)
discriminator = Discriminator(input_size=crop_size).to(device)

# Instantiate loss functions
pixel_loss = PixelLoss(mode='MSE', weight=1.0).to(device)
perceptual_loss = PerceptualLoss(feature_layer=35, use_bn=False, weight=1.0, device=device).to(device)
adversarial_loss = AdversarialLoss(weight=1e-3).to(device)

# Optimizers for pre-training and adversarial training
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# ---------------------------
# Stage 1: Pre-train Generator
# ---------------------------
print("Starting pre-training of the generator using pixel loss...")
generator.train()
for epoch in range(num_pretrain_epochs):
    epoch_loss = 0.0
    for lr_imgs, hr_imgs in dataloader:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        optimizer_G.zero_grad()
        sr_imgs = generator(lr_imgs)
        loss = pixel_loss(sr_imgs, hr_imgs)
        loss.backward()
        optimizer_G.step()

        epoch_loss += loss.item()

    print(f"Pretrain Epoch [{epoch+1}/{num_pretrain_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

# ---------------------------
# Stage 2: Adversarial Training
# ---------------------------
print("Starting adversarial training...")
for epoch in range(num_adv_epochs):
    g_loss_epoch = 0.0
    d_loss_epoch = 0.0
    for lr_imgs, hr_imgs in dataloader:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        # ---------------------------------
        # Update Discriminator: maximize log(D(HR)) + log(1-D(SR))
        # ---------------------------------
        discriminator.train()
        optimizer_D.zero_grad()

        # Real images label: 1, Fake images label: 0
        valid = torch.ones((lr_imgs.size(0), 1), device=device)
        fake = torch.zeros((lr_imgs.size(0), 1), device=device)

        # Loss for real HR images
        real_pred = discriminator(hr_imgs)
        d_loss_real = nn.BCELoss()(real_pred, valid)

        # Generate super-resolved images
        sr_imgs = generator(lr_imgs)
        fake_pred = discriminator(sr_imgs.detach())
        d_loss_fake = nn.BCELoss()(fake_pred, fake)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        # ---------------------------------
        # Update Generator: minimize perceptual + pixel-wise + adversarial losses
        # ---------------------------------
        optimizer_G.zero_grad()
        # Generate images for generator update
        sr_imgs = generator(lr_imgs)

        # Re-compute adversarial loss with the updated discriminator
        pred_fake = discriminator(sr_imgs)
        loss_adv = adversarial_loss(pred_fake, valid)

        loss_pixel = pixel_loss(sr_imgs, hr_imgs)
        loss_perceptual = perceptual_loss(sr_imgs, hr_imgs)
        # Total generator loss: weights can be tuned as needed
        g_loss = loss_pixel + loss_perceptual + loss_adv

        g_loss.backward()
        optimizer_G.step()

        g_loss_epoch += g_loss.item()
        d_loss_epoch += d_loss.item()

    print(f"Adv Epoch [{epoch+1}/{num_adv_epochs}]  Generator Loss: {g_loss_epoch/len(dataloader):.4f}, "
            f"Discriminator Loss: {d_loss_epoch/len(dataloader):.4f}")

# Save the final models (optional)
torch.save(generator.state_dict(), "srgan_generator.pth")
torch.save(discriminator.state_dict(), "srgan_discriminator.pth")
print("Training completed and models are saved.")
