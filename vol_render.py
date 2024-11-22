import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from model import NeRF
from tqdm import tqdm
from rays import ray_generation
from PIL import Image
import argparse
from math import tan

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

def compute_accumulated_transmittance(alphas): 
    """
    Calculates the accumulated transmittance along rays for volume rendering.

    Args:
        alphas (torch.Tensor): Alpha values (opacity) at sampled points along rays. Shape: [num_rays, num_samples].

    Returns:
        torch.Tensor: Cumulative transmittance along the ray. Shape: [num_rays, num_samples].
    """
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat([torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]], dim=-1)
    
def render_volume(model, ray_o, ray_d, near=0, far=1, num_steps=192): 
    """
    Performs volume rendering by simulating ray marching and querying the NeRF model.

    Args:
        model (NeRF): Trained NeRF model to predict RGB and density values.
        ray_o (torch.Tensor): Ray origins. Shape: [num_rays, 3].
        ray_d (torch.Tensor): Ray directions. Shape: [num_rays, 3].
        near (float): Near clipping distance for the rays.
        far (float): Far clipping distance for the rays.
        num_steps (int): Number of depth samples per ray.

    Returns:
        torch.Tensor: Rendered RGB values for each ray. Shape: [num_rays, 3].
    """
    device = ray_o.device 
    
    t = torch.linspace(near, far, num_steps, device=device).expand(ray_o.shape[0], num_steps)
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat([t[:, :1], mid], -1)
    upper = torch.cat([mid, t[:, -1:]], -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u
    delta = torch.cat([t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device)])
    
    x = ray_o.unsqueeze(1) + t.unsqueeze(2) * ray_d.unsqueeze(1)
    
    ray_d = ray_d.expand(num_steps, ray_d.shape[0], 3).transpose(0, 1)
    
    colors, sigma = model(x.reshape(-1, 3), ray_d.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])
    
    alpha = 1 - torch.exp(-sigma * delta)
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)
    return c

def vol_visual(rgb_tensor: torch.Tensor, save_path: str):
    """
    Visualizes and optionally saves a rendered RGB image.

    Args:
        rgb_tensor (torch.Tensor): RGB tensor of the rendered image. Shape: [H, W, 3].
        save_path (str): Directory to save the rendered image. If None, the image is not saved.
    """
    rgb_numpy = rgb_tensor.detach().cpu().numpy()
    rgb_numpy = np.clip(rgb_numpy, 0.0, 1.0)
    plt.imshow(rgb_numpy)
    plt.axis('off')
    plt.show()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.imsave(os.path.join(save_path, "test.png"), rgb_numpy)
        
def rendering(weight_path: str, output_path: str, num_steps: int, img_h: int, img_w: int, radius: float = 4.0311, camera_angle: float=0.6911112070083618):
    """
    Renders a 360-degree rotation GIF using a trained NeRF model.

    Args:
        weight_path (str): Path to the trained NeRF model weights.
        output_path (str): Directory to save the rendered frames.
        num_steps (int): Number of frames for the 360-degree rotation.
        img_h (int): Height of the rendered images.
        img_w (int): Width of the rendered images.
        radius (float): Radius of the camera orbit around the scene.
        camera_angle (float): Camera's field of view in radians.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_path, exist_ok=True)

    model = NeRF(hidden_units=128).to(device)
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    
    e_list = extrinsic_matrix_generation(num_steps=num_steps, radius=radius)
    
    focal_length = img_w / (2.0 * tan(camera_angle / 2))
    K = torch.tensor([[focal_length, 0, img_w / 2],
                        [0, focal_length, img_h / 2],
                        [0, 0, 1]], dtype=torch.float32)
    
    for idx, E in enumerate(tqdm(e_list)):
        direction, origin = ray_generation(img_height=img_h, img_width=img_w, K_matrix=K, E_matrix=E)

        direction = direction.to(device)
        origin = origin.to(device)
        
        with torch.no_grad():
            rgb_tensor = render_volume(model, origin, direction, near=2., far=6., num_steps=192)
        
        rgb_tensor = torch.clamp(rgb_tensor, max=1.0, min=0.0)
        rgb_numpy = (rgb_tensor.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
        frame_image = Image.fromarray(rgb_numpy)
        frame_path = os.path.join(output_path, f"frame_{idx:03d}.png")
        frame_image.save(frame_path)
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Render 360-degree GIF from NeRF model")

    parser.add_argument("--weight_path", type=str, required=True, help="Path to the NeRF model weights")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the output GIF")
    parser.add_argument("--num_steps", type=int, default=192, help="Number of frames for 360-degree rotation")
    parser.add_argument("--img_h", type=int, default=128, help="Height of the rendered image")
    parser.add_argument("--img_w", type=int, default=128, help="Width of the rendered image")
    parser.add_argument("--radius", type=float, default=4.0311, help="Radius of the camera orbit around the scene")
    parser.add_argument("--angle", type=float, required=False, help="Defined camera angle for rendering. Advised to be consistent with training")

    args = parser.parse_args()

    rendering(
        weight_path=args.weight_path,
        output_path=args.output_path,
        num_steps=args.num_steps,
        img_h=args.img_h,
        img_w=args.img_w,
        radius=args.radius
    )