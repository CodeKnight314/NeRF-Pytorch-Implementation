import torch
import numpy as np
from math import tan
from rays import ray_generation, ray_sampling
from vol_render import render_volume
import argparse
import os
from PIL import Image

def extrinsic_matrix(num_views: int, radius: float, height: float = 1.0):
    extrinsic_matrices = []
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        R_y = torch.tensor([
            [cos_angle, 0, sin_angle],
            [0, 1, 0],
            [-sin_angle, 0, cos_angle]
        ], dtype=torch.float32)

        t_x = radius * cos_angle
        t_y = height
        t_z = radius * sin_angle
        t = torch.tensor([t_x, t_y, t_z], dtype=torch.float32)

        extrinsic_matrix = torch.eye(4, dtype=torch.float32)
        extrinsic_matrix[:3, :3] = R_y
        extrinsic_matrix[:3, 3] = t

        extrinsic_matrices.append(extrinsic_matrix)

    return extrinsic_matrices

def inference(model, camera_angle, img_dim, num_views, radius, height, near_bound, far_bound, num_samples, output_dir):
    # Image dimension
    img_h, img_w = img_dim
    
    # Compute focal length from the camera angle
    focal_length = img_w / (2.0 * tan(camera_angle / 2))
    
    # Construct the intrinsic matrix (K) using the focal length and image size
    K = torch.tensor([[focal_length, 0, img_w / 2],
                      [0, focal_length, img_h / 2],
                      [0, 0, 1]], dtype=torch.float32)
    
    E_matrices = extrinsic_matrix(num_views, radius, height)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, E in enumerate(E_matrices):
        ray_directions, Oc = ray_generation(img_h, img_w, K, E)
        
        points, t_vals = ray_sampling(Oc, ray_directions, near_bound, far_bound, num_samples)
        points = points.unsqueeze(0)
        t_vals = t_vals.unsqueeze(0)
        
        with torch.no_grad():
            rgb_pred, density_pred = model(points, ray_directions)
            
        rgb_pred = rgb_pred.view(img_h, img_w, -1, 3)
        density_pred = density_pred.view(img_h, img_w, -1, 1)
        
        rgb_tensor = render_volume(rgb_pred, density_pred, t_vals)
        rgb_image = (rgb_tensor.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(rgb_image)
        image.save(os.path.join(output_dir, f'rendered_view_{i:03d}.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference and Render Views")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--camera-angle', type=float, required=True, help="Camera angle in radians")
    parser.add_argument('--img-dim', type=int, nargs=2, required=True, help="Image dimensions (height, width)")
    parser.add_argument('--num-views', type=int, required=True, help="Number of views to render")
    parser.add_argument('--radius', type=float, required=True, help="Radius of the camera orbit")
    parser.add_argument('--height', type=float, default=1.0, help="Height of the camera (default: 1.0)")
    parser.add_argument('--near-bound', type=int, required=True, help="Near bound of ray sampling")
    parser.add_argument('--far-bound', type=int, required=True, help="Far bound of ray sampling")
    parser.add_argument('--num-samples', type=int, required=True, help="Number of samples per ray")
    parser.add_argument('--output-dir', type=str, required=True, help="Directory to save rendered images")
    
    args = parser.parse_args()
    
    # Load the model
    model = torch.load(args.model_path)
    model.eval()
    
    # Run inference
    inference(model, args.camera_angle, args.img_dim, args.num_views, args.radius, args.height, args.near_bound, args.far_bound, args.num_samples, args.output_dir)