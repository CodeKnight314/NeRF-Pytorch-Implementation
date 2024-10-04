import torch
import torch.nn as nn

class NeRF(nn.Module):
    def __init__(self, pos_encoding_L: int = 10, dir_encoding_L: int = 4, 
                 hidden_units: int = 256, num_layers: int = 4):
        super().__init__()
        
        self.pos_encoding_L = pos_encoding_L
        self.dir_encoding_L = dir_encoding_L
        
        # Initial Linear layers with skip connections
        self.initLinear = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.initLinear.append(nn.Linear(2 * pos_encoding_L * 3 + 3, hidden_units)) 
            else:
                self.initLinear.append(nn.Linear(hidden_units, hidden_units))
            self.initLinear.append(nn.ReLU())

        # Intermediate Linear layers with positional encoding and skip connections
        self.stemLinear = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.stemLinear.append(nn.Linear(2 * pos_encoding_L * 3 + hidden_units, hidden_units))
            else:
                self.stemLinear.append(nn.Linear(hidden_units, hidden_units))
            self.stemLinear.append(nn.ReLU())
        
        # Density prediction with Softplus activation
        self.density = nn.Sequential(
            nn.Linear(hidden_units, 1),
            nn.Softplus() 
        )
        
        # Color prediction
        self.color = nn.Sequential(
            nn.Linear(2 * dir_encoding_L * 3 + hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, 3), nn.Sigmoid()
        )
        
    def pos_encoding(self, x, L: int = 10):
        """
        Positional encoding for input coordinates.

        Args:
            x: Input tensor of shape (N, D) 
               N: Number of points
               D: Dimension of each point (e.g., 3 for (x, y, z))
            L: Number of frequencies 

        Returns:
            Tensor of shape (N, D * 2L)
        """
        output = []
        for i in range(L):
            sin_component = torch.sin((2 ** i) * torch.pi * x)
            cos_component = torch.cos((2 ** i) * torch.pi * x)
            output.extend([sin_component, cos_component])
        return torch.cat(output, dim=-1)
    
    def forward(self, pos, direction):
        # Positional Encoding
        pos_encod = self.pos_encoding(pos, L=self.pos_encoding_L) 
        direction_encod = self.pos_encoding(direction, L=self.dir_encoding_L) 
        
        # Initial layers with skip connections
        x = torch.cat([pos, pos_encod], dim=-1)
        for i in range(len(self.initLinear)):
            if isinstance(self.initLinear[i], nn.Linear) and i > 0:
                x = self.initLinear[i](x) + x  # Skip connection
            else:
                x = self.initLinear[i](x)
        
        # Intermediate layers with skip connections
        h = torch.cat([x, pos_encod], dim=-1)
        for i in range(len(self.stemLinear)):
            if isinstance(self.stemLinear[i], nn.Linear) and i > 0:
                h = self.stemLinear[i](h) + h  # Skip connection
            else:
                h = self.stemLinear[i](h)
        
        density = self.density(h)
        color_input = torch.cat([direction_encod, h], dim=-1)
        color = self.color(color_input)
        
        return color, density
    
if __name__ == "__main__":
    model = NeRF(pos_encoding_L=10, dir_encoding_L=4)

    batch_size = 16
    num_points = 10000
    pos = torch.randn(batch_size, num_points, 3)
    direction = torch.randn(batch_size, num_points, 3)
    
    color, density = model(pos, direction)
    print(f"color output shape: {color.shape}")
    print(f"density output shape: {density.shape}")