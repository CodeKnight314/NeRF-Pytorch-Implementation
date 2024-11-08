import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from model import NeRF
from tqdm import tqdm
from ray_generation import ray_generation, ray_sampling
from PIL import Image
import argparse

def extrinsic_matrix_generation(num_steps: int, radius: float):
    re_matrices = [] 
    step_size = 360 / num_steps
    
    theta = 0.0
    for i in range(num_steps):
        theta_rad = math.radians(theta)
        
        E_Matrix = torch.tensor([
            [math.cos(theta_rad), 0, math.sin(theta_rad), radius * math.cos(theta_rad)],
            [0, 1, 0, 0],
            [-math.sin(theta_rad), 0, math.cos(theta_rad), radius * math.sin(theta_rad)],
            [0, 0, 0, 1]
        ]).float()
        
        re_matrices.append(E_Matrix)
        theta += step_size

    return re_matrices

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
        
def rendering(weight_path: str, output_path: str, num_steps: int, img_h: int, img_w: int, radius: float = 4.0311):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = NeRF().to(device)
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    
    e_list = extrinsic_matrix_generation(num_steps=num_steps, radius=radius)
    
    frames = [] 
    
    focal_length = img_w / (2.0 * math.tan(0.6911112070083618 / 2))
    K = torch.tensor([[focal_length, 0, img_w / 2],
                        [0, focal_length, img_h / 2],
                        [0, 0, 1]], dtype=torch.float32)
    
    for E in tqdm(e_list):
        Oc, dir = ray_generation(img_height=100, img_width=100, K_matrix=K, E_matrix=E)
        points, z_vals = ray_sampling(Oc=Oc, ray_direction=dir, near_bound=2., far_bound=6., num_samples=64)
        
        points = points.to(device)
        dir = dir.to(device)
        
        with torch.no_grad():
            rgb_prediction, density_prediction = model(points, dir)
        
        rgb_tensor = vol_render(rgb_prediction, density_prediction, z_vals)
        
        rgb_numpy = (rgb_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
        frames.append(Image.fromarray(rgb_numpy))
    
    os.makedirs(output_path, exist_ok=True)
    
    gif_path = os.path.join(output_path, "rendered_360.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

    print(f"GIF saved at: {gif_path}")
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Render 360-degree GIF from NeRF model")

    parser.add_argument("--weight_path", type=str, required=True, help="Path to the NeRF model weights")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the output GIF")
    parser.add_argument("--num_steps", type=int, default=64, help="Number of frames for 360-degree rotation")
    parser.add_argument("--img_h", type=int, default=100, help="Height of the rendered image")
    parser.add_argument("--img_w", type=int, default=100, help="Width of the rendered image")
    parser.add_argument("--radius", type=float, default=4.0311, help="Radius of the camera orbit around the scene")

    args = parser.parse_args()

    rendering(
        weight_path=args.weight_path,
        output_path=args.output_path,
        num_steps=args.num_steps,
        img_h=args.img_h,
        img_w=args.img_w,
        radius=args.radius
    )
