import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def vol_render(rgb: torch.Tensor, sigma: torch.Tensor, t_vals: torch.Tensor):
    """
    Perform volume rendering for NeRF to compute the final pixel colors.

    Args:
        rgb (torch.Tensor): RGB colors of sampled points along rays. Shape: [img_height, img_width, num_samples, 3]
        sigma (torch.Tensor): Density values of sampled points along rays. Shape: [img_height, img_width, num_samples, 1]
        t_vals (torch.Tensor): Distances along each ray. Shape: [num_samples,]

    Returns:
        pixel_colors (torch.Tensor): Final rendered pixel colors. Shape: [img_height, img_width, 3]
    """

    deltas = t_vals[1:] - t_vals[:-1]  # Shape: [num_samples - 1]
    delta_inf = torch.tensor([1e10], dtype=t_vals.dtype, device=t_vals.device)
    deltas = torch.cat([deltas, delta_inf], dim=0)  # Shape: [num_samples]
    deltas = deltas.view(1, 1, -1, 1)  # Reshape to [1, 1, num_samples, 1] for broadcasting

    alpha = 1.0 - torch.exp(-sigma * deltas)  # Shape: [img_height, img_width, num_samples, 1]

    eps = 1e-10  # Small epsilon to prevent numerical issues
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :, :1, :]), 1.0 - alpha + eps], dim=2),
        dim=2
    )[:, :, :-1, :]  # Shape: [img_height, img_width, num_samples, 1]

    weights = alpha * transmittance  # Shape: [img_height, img_width, num_samples, 1]

    img_tensor = torch.sum(weights * rgb, dim=2)  # Shape: [img_height, img_width, 3]

    return img_tensor

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
    img_width, img_height = 400, 400
    num_samples = 64

    rgb = torch.rand((img_height, img_width, num_samples, 3))  # Random RGB colors for each point
    sigma = torch.rand((img_height, img_width, num_samples, 1))  # Random densities for each point
    t_vals = torch.linspace(0, 1, num_samples)  # Sample distances along rays

    pixel_colors = vol_render(rgb, sigma, t_vals)

    vol_visual(pixel_colors, save_path='./render_output')
