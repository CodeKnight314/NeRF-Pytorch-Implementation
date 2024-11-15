import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from model import NeRF
from tqdm import tqdm
from rays import ray_generation, ray_sampling
from PIL import Image
import argparse

def extrinsic_matrix_generation(num_steps: int, radius: float):
    """
    Generates a list of extrinsic matrices for 360-degree rotation around the scene.

    Parameters:
    - num_steps (int): Number of steps (frames) in the 360-degree rotation.
    - radius (float): Radius of the orbit around the scene.

    Returns:
    - re_matrices (List[torch.Tensor]): List of 4x4 extrinsic matrices representing camera poses.
    """
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

def compute_transmittance(sigma: torch.Tensor, t_vals: torch.Tensor):
    """
    Compute cumulative transmittance along each sample in a ray.

    Parameters:
    - sigma (torch.Tensor): (B, H * W * num_steps, 1), density values for each sample point.
    - t_vals (torch.Tensor): (B, num_steps, 1), distance values along each ray.

    Returns:
    - transmittance (torch.Tensor): (B, H*W, num_steps, 1), cumulative transmittance values.
    """
    B, num_steps, _ = t_vals.shape
    diff = torch.diff(t_vals, dim=1)
    diff_inf = torch.full((B, 1, 1), 1e10, dtype=torch.float32, device=sigma.device)
    diff = torch.cat([diff, diff_inf], dim=1)
    sigma = sigma.view(B, -1, num_steps, 1)
    transmittance = torch.exp(-sigma * diff)
    transmittance = torch.cumprod(transmittance, dim=-2)

    return transmittance

def compute_absorption(sigma: torch.Tensor, t_vals: torch.Tensor):
    """
    Compute per-sample absorption (alpha) values for each ray.

    Parameters:
    - sigma (torch.Tensor): (B, H * W * num_steps, 1), density values for each sample point.
    - t_vals (torch.Tensor): (B, num_steps, 1), distance values along each ray.

    Returns:
    - absorption (torch.Tensor): (B, H, W, num_steps, 1), per-sample absorption values.
    """
    B, num_steps, _ = t_vals.shape
    diff = torch.diff(t_vals, dim=1)
    diff_inf = torch.full((B, 1, 1), 1e10, dtype=torch.float32, device=sigma.device)
    diff = torch.cat([diff, diff_inf], dim=1)
    sigma = sigma.view(B, -1, num_steps, 1)
    absorption = 1 - torch.exp(-sigma * diff)

    return absorption

def render_volume(rgb: torch.Tensor, sigma: torch.Tensor, t_vals: torch.Tensor):
    """
    Render the RGB image by calculating weighted contributions from each sample.

    Parameters:
    - rgb (torch.Tensor): (B, H * W * num_steps, 3), RGB color values for each sample.
    - sigma (torch.Tensor): (B, H * W * num_steps, 1), density values for each sample point.
    - t_vals (torch.Tensor): (B, num_steps, 1), distance values along each ray.

    Returns:
    - rgb_image (torch.Tensor): (B, H * W, 3), rendered RGB image.
    """
    B, num_steps, _ = t_vals.shape
    transmittance = compute_transmittance(sigma, t_vals)
    absorption = compute_absorption(sigma, t_vals)
    
    weights = transmittance * absorption
    weights = weights.view(B, -1, num_steps, 1)
    weights = weights / torch.norm(weights, dim=-2, keepdim=True)
    
    rgb = rgb.view(B, -1, num_steps, 3)
    
    weighted_rgb = torch.sum(rgb * weights, dim=-2)
    
    rgb_image = weighted_rgb.view(B, -1, 3)

    return rgb_image

def vol_visual(rgb_tensor: torch.Tensor, save_path: str):
    """
    Visualize and save the rendered RGB image.

    Parameters:
    - rgb_tensor (torch.Tensor): Rendered RGB image tensor with shape [img_height, img_width, 3].
    - save_path (str): Directory to save the image. If None, the image won't be saved.
    """
    rgb_numpy = rgb_tensor.detach().cpu().numpy()
    rgb_numpy = np.clip(rgb_numpy, 0.0, 1.0)
    plt.imshow(rgb_numpy)
    plt.axis('off')
    plt.show()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.imsave(os.path.join(save_path, "test.png"), rgb_numpy)
        
def rendering(weight_path: str, output_path: str, num_steps: int, img_h: int, img_w: int, radius: float = 4.0311):
    """
    Renders a 360-degree rotation around a scene using a NeRF model and saves each frame.

    Parameters:
    - weight_path (str): Path to the saved NeRF model weights.
    - output_path (str): Directory to save the rendered frames.
    - num_steps (int): Number of frames in the 360-degree rotation.
    - img_h (int): Height of the rendered image.
    - img_w (int): Width of the rendered image.
    - radius (float): Radius of the camera orbit around the scene.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_path, exist_ok=True)

    model = NeRF(hidden_units=128).to(device)
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    
    e_list = extrinsic_matrix_generation(num_steps=num_steps, radius=radius)
    
    focal_length = 800 / (2.0 * math.tan(0.6911112070083618 / 2))
    K = torch.tensor([[focal_length, 0, 800 / 2],
                      [0, focal_length, 800 / 2],
                      [0, 0, 1]], dtype=torch.float32)
    
    for idx, E in enumerate(tqdm(e_list)):
        direction, Oc = ray_generation(img_height=100, img_width=100, K_matrix=K, E_matrix=E)
        points, z_vals = ray_sampling(Oc=Oc, ray_direction=direction, near_bound=2., far_bound=6., num_samples=64)

        points = points.view(-1, 3)
        z_vals = z_vals.view(-1, 1).unsqueeze(0)
        direction = direction.view(-1, 3)
        direction = direction.unsqueeze(1).repeat(1, num_steps, 1)  
        direction = direction.view(-1, 3) 

        points = points.to(device)
        direction = direction.to(device)
        
        with torch.no_grad():
            rgb_prediction, density_prediction = model(points, direction)

        rgb_prediction = rgb_prediction.view(1, img_h, img_w, -1, 3)
        density_prediction = density_prediction.view(1, img_h, img_w, -1, 1)
        
        rgb_tensor = render_volume(rgb_prediction, density_prediction, z_vals)
        
        rgb_numpy = (rgb_tensor.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
        frame_image = Image.fromarray(rgb_numpy)
        frame_path = os.path.join(output_path, f"frame_{idx:03d}.png")
        frame_image.save(frame_path)
    
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