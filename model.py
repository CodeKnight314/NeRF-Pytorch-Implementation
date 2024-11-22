import torch
import torch.nn as nn

class PhotometricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, predict_rgb, g_truth_rgb): 
        return 0.5 * self.l1_loss(predict_rgb, g_truth_rgb) + 0.5 * self.mse_loss(predict_rgb, g_truth_rgb)

class NeRF(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=128):   
        super(NeRF, self).__init__()
        
        self.block1 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        
        self.block2 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1), )
        
        self.block3 = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2), nn.ReLU(), )
        self.block4 = nn.Sequential(nn.Linear(hidden_dim // 2, 3), nn.Sigmoid(), )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    def positional_encoding(self, x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos)
        emb_d = self.positional_encoding(d, self.embedding_dim_direction)
        h = self.block1(emb_x)
        tmp = self.block2(torch.cat((h, emb_x), dim=1))
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        h = self.block3(torch.cat((h, emb_d), dim=1))
        c = self.block4(h)
        return c, sigma

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