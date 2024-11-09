import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from model import NeRF, PhotometricLoss
from dataset import SyntheticNeRF
from vol_render import vol_render
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm 
from typing import List

def get_chunks(
  inputs: torch.Tensor,
  chunksize: int = 2**15
) -> List[torch.Tensor]:
  r"""
  Divide an input into chunks.
  """
  return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

def prepare_chunks(
  points: torch.Tensor,
  chunksize: int = 2**15
) -> List[torch.Tensor]:
    r"""
    Chunkify points to prepare for NeRF model without encoding them beforehand.
    """
    points = points.reshape((-1, 3))
    points = get_chunks(points, chunksize=chunksize)
    return points

def prepare_viewdirs_chunks(
  points: torch.Tensor,
  rays_d: torch.Tensor,
  chunksize: int = 2**15
) -> List[torch.Tensor]:
    r"""
    Chunkify viewdirs to prepare for NeRF model without encoding them beforehand.
    """
    # Assuming rays_d has shape [batch_size, 3]
    # and points has shape [batch_size, num_points, 3]

    # Expand viewdirs to match points along the number of sampled points
    # Shape will become [batch_size, num_points, 3]
    viewdirs = rays_d[:, None, :].expand(points.shape)
    
    # Reshape to match chunk format: [-1, 3]
    viewdirs = viewdirs.reshape((-1, 3))
    
    # Chunkify the view directions
    viewdirs = get_chunks(viewdirs, chunksize=chunksize)
    return viewdirs

def forward_with_chunks(model, points, directions, chunksize=2**15):
    """
    Perform a forward pass using chunked inputs to avoid memory issues.

    Args:
        model (nn.Module): The NeRF model.
        points (torch.Tensor): The input points tensor. Shape: [batch_size, num_points, 3].
        directions (torch.Tensor): The input view directions tensor. Shape: [batch_size, num_points, 3].
        chunksize (int): Size of each chunk to use for forward pass.

    Returns:
        torch.Tensor: The concatenated outputs from the model for all chunks.
    """
    # Chunk the inputs
    points_chunks = prepare_chunks(points, chunksize)
    viewdirs_chunks = prepare_viewdirs_chunks(directions, points, chunksize)

    # Initialize lists to store chunk results
    rgb_results = []
    density_results = []

    # Perform forward pass in chunks
    for pts_chunk, dirs_chunk in zip(points_chunks, viewdirs_chunks):
        # Pass through model
        rgb_chunk, density_chunk = model(pts_chunk, dirs_chunk)
        
        # Append results
        rgb_results.append(rgb_chunk)
        density_results.append(density_chunk)

    # Concatenate all chunks to get the full result
    rgb = torch.cat(rgb_results, dim=0)
    density = torch.cat(density_results, dim=0)

    return rgb, density

def train_nerf(args):
    dataset = SyntheticNeRF(args.root, mode="train", t_near=args.t_near, t_far=args.t_far, num_steps=args.num_steps, size=args.size)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = NeRF(pos_encoding_L=args.pos_encoding_L, dir_encoding_L=args.dir_encoding_L, hidden_units=128, num_layers=4)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    photometric_loss = PhotometricLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Mixed precision scaler
    scaler = GradScaler()

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_idx, data in enumerate(tqdm(dataloader, desc=f"[{epoch+1}/{args.epochs}]")):
            # Load data onto device
            img = data['Image'].to(device)  # [batch_size, 3, height, width]
            height = data['height'][0].item()
            width = data['width'][0].item()
            points = data['points'].to(device)  # [batch_size, num_points, 3]
            direction = data['direction'].to(device) 

            # Zero the gradient
            optimizer.zero_grad()
    
            with autocast():  # Enable mixed precision to reduce memory usage
                # Forward pass with chunking
                rgb_pred, density_pred = model(points, direction)
    
                # Reshape for rendering
                rgb_pred = rgb_pred.view(args.batch, height, width, -1, 3)
                density_pred = density_pred.view(args.batch, height, width, -1, 1)
    
                # Volume rendering
                rgb_tensor = vol_render(rgb_pred, density_pred, data['t_vals'].to(device)).permute(0, 3, 1, 2)


    
                # Calculate photometric loss
                loss = photometric_loss(rgb_tensor, img)
    
            # Backpropagation with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            # Update total loss
            total_loss += loss.item()
        
            # Free up memory by deleting unused variables
            del points, direction, rgb_pred, density_pred, rgb_tensor, loss
            torch.cuda.empty_cache()
    
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        print(f"[Epoch End] Total GPU Memory Usage (MB): {torch.cuda.memory_allocated() / 1024 / 1024:.2f}")

    # Save model
    torch.save(model.state_dict(), args.save)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NeRF model")
    parser.add_argument('--root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--batch', type=int, default=1, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=600, help='Number of epochs to train the model')
    parser.add_argument('--save', type=str, default='nerf_model.pth', help='Path to save the trained model')
    parser.add_argument('--pos_encoding_L', type=int, default=10, help='Number of frequencies for positional encoding')
    parser.add_argument('--dir_encoding_L', type=int, default=4, help='Number of frequencies for directional encoding')
    parser.add_argument('--t_near', type=int, default=2, help='Near plane for ray sampling')
    parser.add_argument('--t_far', type=int, default=6, help='Far plane for ray sampling')
    parser.add_argument('--num_steps', type=int, default=64, help='Number of steps for ray sampling')
    parser.add_argument('--size', type=int, default=100, help='Dimension of image for NeRF to sample rays from')
    args = parser.parse_args()

    train_nerf(args)