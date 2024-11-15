import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from model import NeRF, PhotometricLoss
from dataset import SyntheticNeRF
from vol_render import render_volume
from rays import sample_fine_points
from tqdm import tqdm 
from typing import List
import os
import math
import numpy as np
from PIL import Image

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0) - 10 * torch.log10(mse).item()

def create_mask(image, threshold=0.01):
    """
    Create a binary mask where non-background pixels are set to 1.
    
    Args:
        image (torch.Tensor): Input image tensor of shape [batch_size, channels, height, width].
        threshold (float): Threshold to consider a pixel as non-background.
    
    Returns:
        torch.Tensor: Binary mask of shape [batch_size, 1, height, width].
    """
    # Assuming the background is black, set a threshold to create the mask
    mask = (image > threshold).float().sum(dim=1, keepdim=True) > 0  # Sum over channels
    return mask.float()

def save_side_by_side_comparison(pred_img, gt_img, save_path, index):
    pred_np = (pred_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    gt_np = (gt_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    
    comparison = np.concatenate((gt_np, pred_np), axis=1)
    comparison_img = Image.fromarray(comparison)
    
    os.makedirs(save_path, exist_ok=True)
    comparison_img.save(os.path.join(save_path, f"comparison_{index:03d}.png"))

def get_chunks(inputs: torch.Tensor, chunksize: int = 2**15) -> List[torch.Tensor]:
    """
    Splits the input tensor into chunks of a specified size.

    Args:
        inputs (torch.Tensor): The input tensor to be split. Shape: (N, D), where N is the number of data points and D is the dimension.
        chunksize (int): The maximum size of each chunk along the first dimension (default is 2**15).

    Returns:
        List[torch.Tensor]: A list of chunked tensors, each of shape (chunksize, D), except possibly the last one which may be smaller.
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

def prepare_direction_vectors(direction: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
    Prepares direction vectors by expanding them along the steps dimension and flattening.

    Args:
        direction (torch.Tensor): Input direction vectors of shape (N, 3), where N is the number of rays.
        num_steps (int): The number of steps along each ray to repeat the direction vector.

    Returns:
        torch.Tensor: Prepared direction vectors of shape (N * num_steps, 3).
    """
    direction = direction.unsqueeze(1).repeat(1, num_steps, 1)  
    direction = direction.view(-1, 3)
    return direction

def prepare_data(points: torch.Tensor, direction_vectors: torch.Tensor, num_steps: int, chunksize: int = 2**10):
    """
    Prepares points and direction vectors for chunked processing.

    Args:
        points (torch.Tensor): Input points of shape (N * num_steps, 3), where N is the number of rays and num_steps is the number of points sampled along each ray.
        direction_vectors (torch.Tensor): Input direction vectors of shape (N, 3) or (N * num_steps, 3).
        num_steps (int): The number of steps along each ray.
        chunksize (int): The maximum size of each chunk along the first dimension (default is 2**15).

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]: Two lists of chunked tensors representing the points and direction vectors, respectively.
    """
    if direction_vectors.shape != points.shape:
        direction_vectors = prepare_direction_vectors(direction_vectors, num_steps)
    direction_vectors = get_chunks(direction_vectors, chunksize) 
    points = get_chunks(points, chunksize)
    return points, direction_vectors

def forward_with_chunks_coarse_fine(model_coarse, model_fine, points, direction_vectors, num_steps, fine_samples=64):
    """
    Processes input points and direction vectors in chunks through both coarse and fine models.
    Coarse model output is used to inform the sampling for the fine model.
    """
    points, directions = prepare_data(points, direction_vectors, num_steps)

    colors_coarse = []
    densities_coarse = []
    colors_fine = []
    densities_fine = []

    for chunk_points, chunk_direction_vectors in zip(points, directions):
        # Coarse model predictions
        chunk_colors_coarse, chunk_densities_coarse = model_coarse(chunk_points, chunk_direction_vectors)
        colors_coarse.append(chunk_colors_coarse)
        densities_coarse.append(chunk_densities_coarse)

        chunk_points = chunk_points.view(chunk_points.shape[0], -1, num_steps, 3)
        chunk_densities_coarse = chunk_densities_coarse.view(chunk_densities_coarse.shape[0], -1, num_steps, 1)
        
        fine_points = sample_fine_points(chunk_points, chunk_densities_coarse, fine_samples)  # Define this function
        fine_directions = chunk_direction_vectors
        
        chunk_colors_fine, chunk_densities_fine = model_fine(fine_points, fine_directions)
        colors_fine.append(chunk_colors_fine)
        densities_fine.append(chunk_densities_fine)

    # Concatenate results
    colors_coarse = torch.cat(colors_coarse, dim=0)
    densities_coarse = torch.cat(densities_coarse, dim=0)
    colors_fine = torch.cat(colors_fine, dim=0)
    densities_fine = torch.cat(densities_fine, dim=0)

    return (colors_coarse, densities_coarse), (colors_fine, densities_fine)

def train_nerf(args):
    dataset = SyntheticNeRF(args.root, mode="train", t_near=args.t_near, t_far=args.t_far, num_steps=args.num_steps, size=args.size)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    # Initialize coarse and fine models
    model_coarse = NeRF(pos_encoding_L=args.pos_encoding_L, dir_encoding_L=args.dir_encoding_L, hidden_units=256)
    model_fine = NeRF(pos_encoding_L=args.pos_encoding_L, dir_encoding_L=args.dir_encoding_L, hidden_units=256)
    model_coarse.train()
    model_fine.train()
    
    optimizer = optim.Adam(list(model_coarse.parameters()) + list(model_fine.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1e-5)
    photometric_loss = PhotometricLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_coarse = model_coarse.to(device)
    model_fine = model_fine.to(device)

    image_save_path = os.path.join(args.save, "imgs")
    os.makedirs(image_save_path, exist_ok=True)

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_psnr = 0.0
        
        for batch_idx, data in enumerate(tqdm(dataloader, desc=f"[{epoch+1}/{args.epochs}]")):
            img = data['Image'].to(device)
            height = data['height'][0].item()
            width = data['width'][0].item()
            points = data['points'].to(device)
            direction = data['direction'].to(device)

            optimizer.zero_grad()
    
            (rgb_pred_coarse, density_pred_coarse), (rgb_pred_fine, density_pred_fine) = forward_with_chunks_coarse_fine(
                model_coarse, model_fine, points, direction, num_steps=args.num_steps, fine_samples=args.fine_samples
            )

            rgb_pred_coarse = rgb_pred_coarse.view(rgb_pred_coarse.shape[0], -1, 3)
            density_pred_coarse = density_pred_coarse.view(density_pred_coarse.shape[0], -1, 1)
            rgb_pred_fine = rgb_pred_fine.view(rgb_pred_fine.shape[0], -1, 3)
            density_pred_fine = density_pred_fine.view(density_pred_fine.shape[0], -1, 1)

            # Coarse pass rendering
            rgb_tensor_coarse = render_volume(rgb_pred_coarse, density_pred_coarse, data['t_vals'].to(device))
            rgb_tensor_coarse = rgb_tensor_coarse.view(-1, 3, height, width)

            # Fine pass rendering
            rgb_tensor_fine = render_volume(rgb_pred_fine, density_pred_fine, data['t_vals'].to(device))
            rgb_tensor_fine = rgb_tensor_fine.view(-1, 3, height, width)

            # Losses for both passes
            loss_coarse = photometric_loss(rgb_tensor_coarse, img)
            loss_fine = photometric_loss(rgb_tensor_fine, img)
            loss = loss_coarse + loss_fine
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            
            psnr = calculate_psnr(rgb_tensor_fine, img)
            total_psnr += psnr
            
        avg_loss = total_loss / len(dataloader)
        avg_psnr = total_psnr / len(dataloader)
                
        rgb_tensor_min = rgb_tensor_fine.min().item()
        rgb_tensor_max = rgb_tensor_fine.max().item()
        img_min = img.min().item()
        img_max = img.max().item()

        total_coarse_norm = 0
        for p in model_coarse.parameters():
            param_norm = p.grad.data.norm(2)
            total_coarse_norm += param_norm.item() ** 2
        total_coarse_norm = total_coarse_norm ** 0.5

        total_fine_norm = 0 
        for p in model_fine.parameters(): 
            param_norm = p.grad.data.norm(2)
            total_fine_norm += param_norm.item() ** 2 
        total_fine_norm = total_fine_norm ** 0.5
        
        print(f"Gradient Coarse Norm: {total_coarse_norm}")
        print(f"Gradient Fine Norm: {total_fine_norm}")
        print(f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f}, Average PSNR: {avg_psnr:.2f} dB")
        print(f"rgb_tensor prediction - Min: {rgb_tensor_min:.4f}, Max: {rgb_tensor_max:.4f}")
        print(f"Ground truth image - Min: {img_min:.4f}, Max: {img_max:.4f}")
        
        scheduler.step()

        torch.save(model_coarse.state_dict(), os.path.join(args.save, "nerf_model_coarse.pth"))
        torch.save(model_fine.state_dict(), os.path.join(args.save, "nerf_model_fine.pth"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NeRF model")
    parser.add_argument('--root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--batch', type=int, default=1, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--save', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--pos_encoding_L', type=int, default=10, help='Number of frequencies for positional encoding')
    parser.add_argument('--dir_encoding_L', type=int, default=4, help='Number of frequencies for directional encoding')
    parser.add_argument('--t_near', type=int, default=2, help='Near plane for ray sampling')
    parser.add_argument('--t_far', type=int, default=6, help='Far plane for ray sampling')
    parser.add_argument('--num_steps', type=int, default=128, help='Number of steps for ray sampling')
    parser.add_argument('--fine_samples', type=int, default=128, help='Number of samples for the fine model')
    parser.add_argument('--size', type=int, default=128, help='Dimension of image for NeRF to sample rays from')
    args = parser.parse_args()

    train_nerf(args)