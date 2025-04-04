import torch
import torch.nn as nn
import torch.optim as optim
from architecture import forward_diffusion_sample


# Define a custom loss module for the diffusion model.
class DiffusionLoss(nn.Module):
    def __init__(self):
        super(DiffusionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_noise, true_noise):
        """
        Computes the mean squared error between the predicted noise and the actual noise.
        
        Args:
          pred_noise: The noise predicted by the diffusion model.
          true_noise: The actual noise that was added during the forward process.
        
        Returns:
          loss: A scalar loss value.
        """
        loss = self.mse_loss(pred_noise, true_noise)
        return loss

# Example setup of the optimizer and usage of the loss in a training step.
def training_step(diffusion_model, cond_net, diffusion_schedule, lr_batch, hr_batch, device):
    # Set models to proper modes.
    diffusion_model.train()
    cond_net.eval()  # Assuming the conditional network is pre-trained.
    
    # Extract conditional features.
    with torch.no_grad():
        cond_features = cond_net(lr_batch.to(device))
    
    # Sample random time steps for each image in the batch.
    batch_size = hr_batch.size(0)
    t = torch.randint(0, diffusion_schedule.num_timesteps, (batch_size,), device=device).long()
    
    # Generate the noisy image and corresponding noise using the forward diffusion process.
    xt, noise = forward_diffusion_sample(hr_batch.to(device), t, diffusion_schedule)
    
    # Predict the noise using the diffusion model.
    pred_noise = diffusion_model(xt, t, cond_features)
    
    # Compute the loss.
    loss_fn = DiffusionLoss().to(device)
    loss = loss_fn(pred_noise, noise)
    
    # Set up the optimizer (here we use Adam).
    optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-4)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

