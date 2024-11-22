import torch
from torch.utils.data import Dataset
import os 
import json
from PIL import Image
from torchvision import transforms as T
from math import tan
from rays import ray_generation, ray_sampling
import argparse

class SyntheticNeRF(Dataset):
    def __init__(self, root_dir: str, mode: str, size: int):
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
        - ray_direction: 3, HW
        - Oc: 1, 3
        """
        ray_direction, Oc = ray_generation(img_h, img_w, K_matrix, E_matrix)
        
        return ray_direction, Oc
    
    def __getitem__(self, index: int):
        """
        Retrieve the image and associated data for a given index.

        Args:
        - index: int, index of the frame to retrieve.

        Returns:
        - dict: Dictionary containing image, camera matrices, points, and t-values.
        """
        filename = os.path.basename(self.frames[index]["file_path"])
        
        img = Image.open(os.path.join(self.root_dir, self.mode, filename + ".png")).convert("RGB")
        img_w, img_h = img.size 
        
        img = self.transform(img).view(-1, 3)
        
        focal_length = img_w / (2.0 * tan(self.camera_angle / 2))
        
        K = torch.tensor([[focal_length, 0, img_w / 2],
                         [0, focal_length, img_h / 2],
                         [0, 0, 1]], dtype=torch.float32)
        
        if self.size: 
            new_width = self.size / img_w 
            new_height = self.size / img_h
            
            K = torch.tensor([[focal_length * new_width, 0, img_w * new_width/ 2],
                         [0, focal_length * new_height, img_h * new_height/ 2],
                         [0, 0, 1]], dtype=torch.float32)
            
        
        E = torch.tensor(self.frames[index]['transform_matrix'], dtype=torch.float32)
        
        Direction, Oc = self.__generate_input__(self.size, self.size, K, E)
                
        Direction = Direction.view(-1, 3)
        Oc = Oc.expand(Direction.shape[0], 3)
        
        return {
            "Image": img,
            "height": self.size,
            "width": self.size,
            "Origin": Oc,
            "K_matrix": K,
            "E_matrix": E,
            "direction": Direction,
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