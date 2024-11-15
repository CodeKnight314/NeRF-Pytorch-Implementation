import torch
import torch.nn as nn

class PhotometricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.MSELoss()
        
    def forward(self, predict_rgb, g_truth_rgb): 
        return self.l1_loss(predict_rgb, g_truth_rgb)

class NeRF(nn.Module):
    def __init__(self, pos_encoding_L: int = 10, dir_encoding_L: int = 4, 
                 hidden_units: int = 256, num_layers: int = 6):
        super().__init__()
        
        self.pos_encoding_L = pos_encoding_L
        self.dir_encoding_L = dir_encoding_L
        
        # Initial Linear layers with skip connections
        self.initLinear = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.initLinear.append(nn.Linear(2 * pos_encoding_L * 3 + 3, hidden_units)) 
            elif i == 4:  # This is where we will apply the skip connection
                self.initLinear.append(nn.Linear(hidden_units + 2 * pos_encoding_L * 3 + 3, hidden_units))
            else:
                self.initLinear.append(nn.Linear(hidden_units, hidden_units))
            self.initLinear.append(nn.ReLU())

        self.density = nn.Sequential(
            nn.Linear(hidden_units, 1),
            nn.ReLU()
        )
        
        self.color = nn.Sequential(
            nn.Linear(2 * dir_encoding_L * 3 + hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, 3),
            nn.Sigmoid()
        )

        self._initialize_weights()
        
    def pos_encoding(self, x, L: int = 10):
        """
        Positional encoding for input coordinates.
        """
        output = []
        for i in range(L):
            sin_component = torch.sin((2 ** i) * torch.pi * x)
            cos_component = torch.cos((2 ** i) * torch.pi * x)
            output.extend([sin_component, cos_component])
        return torch.cat(output, dim=-1)
    
    def forward(self, pos, direction):
        pos_encod = self.pos_encoding(pos, L=self.pos_encoding_L) 
        direction_encod = self.pos_encoding(direction, L=self.dir_encoding_L) 
        
        x = torch.cat([pos, pos_encod], dim=-1)
        
        for i in range(4 * 2):
            x = self.initLinear[i](x)
        
        x = torch.cat([x, pos, pos_encod], dim=-1)
        
        for i in range(4 * 2, len(self.initLinear)):
            x = self.initLinear[i](x)
        
        density = self.density(x)
        color_input = torch.cat([direction_encod, x], dim=-1)
        color = self.color(color_input)
        return color, density

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class TinyNeRF(nn.Module): 
    def __init__(self, pos_encoding_L: int = 6, dir_encoding_L: int = 4, hidden_units: int = 256):
        super().__init__() 
        
        pos_features = 3 * 2 * pos_encoding_L + 3 
        dir_features = 3 * 2 * dir_encoding_L
        
        self.L_pos = pos_encoding_L 
        self.L_dir = dir_encoding_L
        
        self.early_branch = nn.Sequential(*[
            nn.Linear(pos_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units + 1),
            nn.ReLU()
        ])
        
        self.late_branch = nn.Sequential(*[
            nn.Linear(hidden_units + dir_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 3),
            nn.Sigmoid(),
        ])
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def pos_encoding(self, x: torch.Tensor, L: int = 10):
        output = []
        for i in range(L):
            sin_component = torch.sin((2 ** i) * torch.pi * x)
            cos_component = torch.cos((2 ** i) * torch.pi * x)
            output.extend([sin_component, cos_component])
        return torch.cat(output, dim=-1)
        
    def forward(self, positions: torch.Tensor, direction: torch.Tensor): 
        pos_encod = torch.cat([positions, self.pos_encoding(positions, self.L_pos)], dim=-1)
        dir_encod = self.pos_encoding(direction, self.L_dir)
        
        output = self.early_branch(pos_encod)
        density = output[:, :, 0].unsqueeze(-1)
        combined = torch.cat([output[:, :, 1:], dir_encod], dim=-1)
        color = self.late_branch(combined)
        return color, density
        
if __name__ == "__main__":
    model = NeRF(pos_encoding_L=10, dir_encoding_L=4)

    batch_size = 16
    height = 100 
    width = 100 
    num_steps = 12 
    num_points = height * width * num_steps
    pos = torch.randn(batch_size, num_points, 3)
    direction = torch.randn(batch_size, num_points, 3)
    
    color, density = model(pos, direction)
    print(f"color output shape: {color.shape}")
    print(f"density output shape: {density.shape}")