import torch
from torch.utils.data import DataLoader, Dataset
import os 
import json
from PIL import Image
from torchvision import transforms as T
from math import tan
from rays import ray_generation, ray_sampling  # Custom functions for ray generation and sampling
import argparse
from typing import List

class SyntheticNeRF(Dataset):
    def __init__(self, root_dir: str, mode: str, t_near: int, t_far: int, num_steps: int, size: int):
        super().__init__()
        
        self.root_dir = root_dir
        self.mode = mode
        
        with open(os.path.join(root_dir, f"transforms_{mode}.json")) as f:
            self.ann_json = json.load(f)
        
        self.frames = self.ann_json['frames']
        self.camera_angle = self.ann_json["camera_angle_x"]
        
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor()
        ])
        
        self.t_near = t_near
        self.t_far = t_far
        self.num_steps = num_steps
        self.size = size
        
    def __len__(self):
        return len(self.frames)
    
    def __generate_input__(self, img_h, img_w, K_matrix, E_matrix):
        """
        Generate rays and sampled points for a given image.

        Args:
        - img_h: int, image height.
        - img_w: int, image width.
        - K_matrix: Tensor, camera intrinsic matrix.
        - E_matrix: Tensor, camera extrinsic matrix.

        Returns:
        - points: Tensor, points sampled along the rays.
        - t_vals: Tensor, corresponding t-values for sampling steps.
        """
        # Generate ray directions and camera origins
        ray_direction, Oc = ray_generation(img_h, img_w, K_matrix, E_matrix)
        
        # Sample points along the rays from near plane to far plane
        points, z_vals = ray_sampling(Oc, ray_direction, self.t_near, self.t_far, self.num_steps)
        return points, z_vals, ray_direction
    
    def __getitem__(self, index: int):
        """
        Retrieve the image and associated data for a given index.

        Args:
        - index: int, index of the frame to retrieve.

        Returns:
        - dict: Dictionary containing image, camera matrices, points, and t-values.
        """
        # Get the filename of the current frame
        filename = os.path.basename(self.frames[index]["file_path"])
        
        # Load the image and convert to RGB format
        img = Image.open(os.path.join(self.root_dir, self.mode, filename + ".png")).convert("RGB")
        img_w, img_h = img.size  # Get image width and height
        
        # Transform image into tensor and reshape to [num_pixels, 3]
        img = self.transform(img)
        
        # Compute focal length from the camera angle
        focal_length = img_w / (2.0 * tan(self.camera_angle / 2))
        
        # Construct the intrinsic matrix (K) using the focal length and image size
        K = torch.tensor([[focal_length, 0, img_w / 2],
                         [0, focal_length, img_h / 2],
                         [0, 0, 1]], dtype=torch.float32)
        
        # Load the camera extrinsic matrix (E) from the JSON file
        E = torch.tensor(self.frames[index]['transform_matrix'], dtype=torch.float32)
        
        # Generate the input points and corresponding t-values
        points, t_vals, direction = self.__generate_input__(self.size, self.size, K, E)
        points = points.view(-1, 3)  # Reshape points to [num_rays, 3]
        t_vals = t_vals.view(-1, 1)  # Reshape t-values to [num_rays, 1]
        direction = direction.view(-1, 3)
        # Repeat for each sample along the ray
        direction = direction.unsqueeze(1).repeat(1, self.num_steps, 1)  
        
        # Reshape to final desired shape (height * width * num_steps, 3)
        direction = direction.view(-1, 3) 
        
        return {
            "Image": img,  # Image tensor with shape [3, height, width]
            "height": self.size,  # Image height
            "width": self.size,  # Image width
            "K_matrix": K,  # Camera intrinsic matrix [3, 3]
            "E_matrix": E,  # Camera extrinsic matrix [4, 3]
            "points": points,  # Points sampled along the rays [height x width x num_steps, 3]
            "direction": direction,
            "t_vals": t_vals  # t-values for each point sampled along the rays [num_steps, 1]
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="root direction to nerf_synthetic")
    
    args = parser.parse_args()
    
    dataset = SyntheticNeRF(args.root_dir, "train", 1, 128, 64)
    
    nerf_dict = dataset.__getitem__(0)
    
    print(nerf_dict["Image"].shape)
    print(nerf_dict["height"], nerf_dict["width"])
    print(nerf_dict["K_matrix"].shape)
    print(nerf_dict["E_matrix"].shape)
    print(nerf_dict["direction"].shape)
    print(nerf_dict["points"].shape)
    print(nerf_dict["t_vals"].shape)