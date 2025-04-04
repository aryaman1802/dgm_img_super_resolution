import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from architecture import ConditionalNet, UNetDiffusion, DiffusionSchedule, forward_diffusion_sample
from data_preprocessing import DIV2KDataset


def train(diffusion_model, cond_net, diffusion_schedule, dataloader, num_epochs=10, device='cuda'):
    """
    Training loop for the diffusion model.
    
    Args:
      diffusion_model: The U-Net diffusion model.
      cond_net: The conditional network (pre-trained and fixed).
      diffusion_schedule: The noise schedule instance.
      dataloader: DataLoader providing (LR, HR) image pairs.
      num_epochs: Number of training epochs.
      device: 'cuda' or 'cpu'.
    """
    diffusion_model.train()
    # Set conditional network to evaluation mode if pre-trained
    cond_net.eval()  
    optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-4)
    mse_loss = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (lr, hr) in enumerate(dataloader):
            lr = lr.to(device)
            hr = hr.to(device)
            
            # Extract conditional features from LR images
            with torch.no_grad():
                cond_features = cond_net(lr)
            
            # Sample a random diffusion time step for each image in the batch
            batch_size = hr.size(0)
            t = torch.randint(0, diffusion_schedule.num_timesteps, (batch_size,), device=device).long()
            
            # Generate a noisy version of the HR image using the forward diffusion process
            xt, noise = forward_diffusion_sample(hr, t, diffusion_schedule)
            xt = xt.to(device)
            noise = noise.to(device)
            
            # Predict the noise from the diffusion model
            pred_noise = diffusion_model(xt, t, cond_features)
            
            # Compute loss: how well did the model predict the added noise?
            loss = mse_loss(pred_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

def evaluate(diffusion_model, cond_net, diffusion_schedule, lr_image, num_steps=1000, device='cuda'):
    """
    A sketch of the evaluation/inference procedure.
    
    Args:
      diffusion_model: The trained diffusion model.
      cond_net: The conditional network.
      diffusion_schedule: The noise schedule.
      lr_image: A low-resolution image tensor.
      num_steps: Total diffusion steps (should match training schedule).
      device: 'cuda' or 'cpu'.
    
    Returns:
      hr_pred: The generated high-resolution image.
    """
    diffusion_model.eval()
    cond_net.eval()
    
    with torch.no_grad():
        # Extract conditional features from the LR image
        cond_features = cond_net(lr_image.to(device))
        
        # Start from pure noise
        hr_pred = torch.randn(lr_image.size(0), 3, lr_image.size(2) * 4, lr_image.size(3) * 4, device=device)
        
        # Here we would iteratively apply the reverse diffusion process.
        # For brevity, this is a simplified loop.
        for t in reversed(range(num_steps)):
            t_tensor = torch.full((lr_image.size(0),), t, device=device, dtype=torch.long)
            # One step of the reverse diffusion process:
            hr_pred = diffusion_model(hr_pred, t_tensor, cond_features)
            # Additional noise correction and scaling would be applied here in a full implementation.
    
    return hr_pred

# Example usage:
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the models
    cond_net = ConditionalNet().to(device)
    diffusion_model = UNetDiffusion().to(device)
    # Assume the conditional network is pre-trained; here, we keep it fixed.
    
    # Create the diffusion schedule (e.g., 1000 timesteps)
    diffusion_schedule = DiffusionSchedule(num_timesteps=1000)
    
    # Initialize the DIV2K dataset and DataLoader
    dataset = DIV2KDataset(hr_dir="data/raw/DIV2K", scale_factor=4)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Train the model
    train(diffusion_model, cond_net, diffusion_schedule, dataloader, num_epochs=5, device=device)
    
    # For evaluation, assume we take one low-resolution image from the dataset
    lr_sample, _ = dataset[0]
    lr_sample = lr_sample.unsqueeze(0)  # Add batch dimension
    hr_generated = evaluate(diffusion_model, cond_net, diffusion_schedule, lr_sample, num_steps=1000, device=device)
    print("Generated HR image shape:", hr_generated.shape)
