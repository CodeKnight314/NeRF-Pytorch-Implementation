import torch
from torch.utils.data import DataLoader, Dataset
import os 
from glob import glob
import json
from PIL import Image
from torchvision import transforms as T
from math import tan

class SyntheticNeRF(Dataset):
    def __init__(self, root_dir : str, mode : str):
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
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index):
        filename = os.path.basename(self.frames[index]["file_path"])
        
        img = Image.open(os.path.join(self.root_dir, self.mode, filename+".png")).convert("RGB")
        img_h, img_w = img.size
        img = self.transform(img).view(-1, 3)
        
        focal_length = img_w / (2.0 * tan(self.camera_angle/2))
        
        K = torch.tensor([focal_length, 0, img_w/2],
                         [0, focal_length, img_h/2],
                         [0, 0, 1], dtype=torch.float32)
        
        E = torch.tensor(self.ann_json[index]['transform_matrix'], dtype=torch.float32)
        return {"Image":img, "height":img_h, "width":img_w, "K_matrix":K, "E_matrix":E}