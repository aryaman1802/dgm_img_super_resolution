import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from srddpm.loss_func import SRDDPMLosses
from srddpm.data_processing import DIV2KDataset
from srddpm.architecture import ConditionalUNet, PatchDiscriminator, DiffusionProcess

def save_checkpoint(generator, discriminator, epoch, stage, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'srddpm_stage{stage}_epoch{epoch}.pth')
    checkpoint = {'generator_state_dict': generator.state_dict(), 'epoch': epoch}
    if discriminator is not None:
        checkpoint['discriminator_state_dict'] = discriminator.state_dict()
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def train_diffusion_stage(generator, diffusion, dataloader, optimizer, device, epoch, total_epochs):
    generator.train()
    running_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        lr = batch['lr'].to(device)
        hr = batch['hr'].to(device)
        batch_size = hr.shape[0]
        # Sample random timesteps for each image
        t = torch.randint(0, diffusion.T, (batch_size,), device=device)
        # Sample noise and add it to HR image using the forward diffusion process
        noise = torch.randn_like(hr)
        x_noisy = diffusion.q_sample(hr, t, noise=noise)
        # Predict noise using the generator conditioned on lr
        predicted_noise = generator(x_noisy, t, lr)
        # Compute diffusion loss (MSE between predicted and true noise)
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        diffusion_loss.backward()
        optimizer.step()
        running_loss += diffusion_loss.item() * batch_size
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Diffusion Stage Epoch [{epoch}/{total_epochs}] - Loss: {epoch_loss:.4f}")
    return epoch_loss

def train_adversarial_stage(generator, discriminator, diffusion, dataloader, g_optimizer, d_optimizer, loss_module, device, epoch, total_epochs):
    generator.train()
    discriminator.train()
    running_g_loss = 0.0
    running_d_loss = 0.0
    for batch in dataloader:
        lr = batch['lr'].to(device)
        hr = batch['hr'].to(device)
        batch_size = hr.shape[0]
        t = torch.randint(0, diffusion.T, (batch_size,), device=device)
        noise = torch.randn_like(hr)
        x_noisy = diffusion.q_sample(hr, t, noise=noise)
        
        # Generator forward pass: predict noise and compute generated HR image
        predicted_noise = generator(x_noisy, t, lr)
        sqrt_alpha_bar_t = diffusion.get_index_from_list(diffusion.sqrt_alpha_bars, t, hr.shape)
        sqrt_one_minus_alpha_bar_t = diffusion.get_index_from_list(diffusion.sqrt_one_minus_alpha_bars, t, hr.shape)
        # Recover generated HR image using a single reverse diffusion step approximation
        generated_hr = (x_noisy - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
        
        # Generator adversarial loss: aim to fool the discriminator
        fake_scores = discriminator(generated_hr)
        g_adv_loss = -torch.mean(fake_scores)
        
        # Compute combined loss (diffusion, reconstruction, perceptual)
        total_loss, loss_dict = loss_module(predicted_noise, noise, generated_hr, hr)
        # Total generator loss: combination of losses (weights can be adjusted as needed)
        g_loss = total_loss + g_adv_loss
        
        g_optimizer.zero_grad()
        g_loss.backward(retain_graph=True)
        g_optimizer.step()
        
        # Discriminator training: compute hinge loss
        real_scores = discriminator(hr)
        fake_scores = discriminator(generated_hr.detach())
        d_loss = torch.mean(F.relu(1.0 - real_scores)) + torch.mean(F.relu(1.0 + fake_scores))
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        running_g_loss += g_loss.item() * batch_size
        running_d_loss += d_loss.item() * batch_size
        
    epoch_g_loss = running_g_loss / len(dataloader.dataset)
    epoch_d_loss = running_d_loss / len(dataloader.dataset)
    print(f"Adversarial Stage Epoch [{epoch}/{total_epochs}] - Generator Loss: {epoch_g_loss:.4f}, Discriminator Loss: {epoch_d_loss:.4f}")
    return epoch_g_loss, epoch_d_loss

def main_training(generator, discriminator, diffusion, train_loader, device, diffusion_epochs=5, adversarial_epochs=5):
    # Stage 1: Diffusion training only.
    print("Starting Stage 1: Diffusion Training")
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    
    for epoch in range(1, diffusion_epochs + 1):
        train_diffusion_stage(generator, diffusion, train_loader, g_optimizer, device, epoch, diffusion_epochs)
        # Save checkpoint (only generator is used in stage 1)
        save_checkpoint(generator, None, epoch, stage=1)
    
    # Stage 2: Adversarial (fine-tuning) training.
    print("Starting Stage 2: Adversarial Training")
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-5)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-5)
    loss_module = SRDDPMLosses()  # Using default loss weights
    
    for epoch in range(1, adversarial_epochs + 1):
        train_adversarial_stage(generator, discriminator, diffusion, train_loader, g_optimizer, d_optimizer, loss_module, device, epoch, adversarial_epochs)
        # Save checkpoint for both generator and discriminator
        save_checkpoint(generator, discriminator, epoch, stage=2)

