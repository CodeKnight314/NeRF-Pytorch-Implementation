import torch
from torch.utils.data import DataLoader, Dataset
import os 
import json
from PIL import Image
from torchvision import transforms as T
from math import tan
from ray_generation import ray_generation, ray_sampling

class SyntheticNeRF(Dataset):
    def __init__(self, root_dir : str, mode : str, t_near : int, t_far : int, num_steps : int):
        super().__init__()
        
        self.root_dir = root_dir
        self.mode = mode
        
        with open(os.path.join(root_dir, mode, f"transforms_{mode}.json")) as f:
            self.ann_json = json.load(f)
        
        self.frames = self.ann_json['frames']
        self.camera_angle = self.ann_json["camera_angle_x"]
        
        self.transform = T.Compose([
            T.ToTensor()
        ])
        
        self.t_near = t_near
        self.t_far = t_far
        self.num_steps = num_steps
        
    def __len__(self):
        return len(self.frames)
    
    def __generate_input__(self, img_h, img_w, K_matrix, E_matrix):
        ray_direction, Oc = ray_generation(img_h, img_w, K_matrix, E_matrix)
        return ray_sampling(ray_direction, Oc, self.t_near, self.t_far, self.num_steps)
    
    def __getitem__(self, index):
        filename = os.path.basename(self.frames[index]["file_path"])
        
        img = Image.open(os.path.join(self.root_dir, self.mode, filename+".png")).convert("RGB")
        img_w, img_h = img.size
        img = self.transform(img).view(-1, 3)
        
        focal_length = img_w / (2.0 * tan(self.camera_angle/2))
        
        K = torch.tensor([[focal_length, 0, img_w/2],
                         [0, focal_length, img_h/2],
                         [0, 0, 1]], dtype=torch.float32)
        
        E = torch.tensor(self.ann_json[index]['transform_matrix'], dtype=torch.float32)
        
        points, t_vals = self.__generate_input__(img_h, img_w, K, E)
        points = points.view(-1, 3)
        t_vals = t_vals.view(-1, 1)
        
        return {
            "Image": img,
            "height": img_h,
            "width": img_w,
            "K_matrix": K,
            "E_matrix": E,
            "points": points,
            "t_vals": t_vals
        }