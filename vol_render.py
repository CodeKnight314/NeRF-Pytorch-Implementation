import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def vol_render(rgb : torch.Tensor, sigma : torch.Tensor, t_vals : torch.Tensor):
    """
    Perform volume rendering for NeRF to compute the final pixel colors.

    Args:
        points (torch.Tensor): 
        rgb (torch.Tensor): RGB colors of sampled points along rays. Shape: [img_height, img_width, num_samples, 3]
        sigma (torch.Tensor): Density values of sampled points along rays. Shape: [img_height, img_width, num_samples, 1]
        t_vals (torch.Tensor): Distances along each ray. Shape: [num_samples,]

    Returns:
        pixel_colors (torch.Tensor): Final rendered pixel colors. Shape: [img_height, img_width, 3]
    """
    
    deltas = t_vals[..., 1:] - t_vals[...,:-1]
    deltas = torch.cat([deltas, torch.Tensor[1e10].float()], dim=-1)
    deltas = deltas.view(1, 1, -1)
    
    alpha = 1.0 - torch.exp(-sigma * deltas) # [img_height, img_width, num_samples, 1]

    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-2)
    T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-2)

    weights = alpha * T  # [img_height, img_width, num_samples, 1]

    pixel_colors = torch.sum(weights * rgb, dim=-2)  # Shape: [img_height, img_width, 3]

    return pixel_colors

def vol_visual(rgb_tensor: torch.Tensor, save_path: str):
    """
    Visualize and save the rendered RGB image.

    Args:
        rgb_tensor (torch.Tensor): Rendered RGB image tensor. Shape: [img_height, img_width, 3]
        save_path (str, optional): Path to save the image. If None, the image won't be saved.
    """
    rgb_numpy = rgb_tensor.detach().cpu().numpy()

    rgb_numpy = np.clip(rgb_numpy, 0.0, 1.0)

    plt.imshow(rgb_numpy)
    plt.axis('off')
    plt.show()

    # Save the image if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist
        plt.imsave(os.path.join(save_path, "test.png"), rgb_numpy)
        
if __name__ == "__main__":
    # Example usage of vol_render and vol_visual
    img_width, img_height = 400, 400
    num_samples = 64

    # Assuming we have some dummy tensors for testing
    rgb = torch.rand((img_height, img_width, num_samples, 3))  # Random RGB colors for each point
    sigma = torch.rand((img_height, img_width, num_samples, 1))  # Random densities for each point
    t_vals = torch.linspace(0, 1, num_samples)  # Sample distances along rays

    # Perform volume rendering to get final pixel colors
    pixel_colors = vol_render(rgb, sigma, t_vals)

    # Visualize and save the rendered image
    vol_visual(pixel_colors, save_path='./render_output')
