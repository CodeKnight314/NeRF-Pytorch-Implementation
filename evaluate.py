import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from model import NeRF, PhotometricLoss
from dataset import SyntheticNeRF 
from vol_render import render_volume

def evaluate_nerf(args):
    dataset = SyntheticNeRF(args.dataset_path, mode="val", t_near=args.t_near, t_far=args.t_far, num_steps=args.num_steps)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = NeRF(pos_encoding_L=args.pos_encoding_L, dir_encoding_L=args.dir_encoding_L)
    model.train()

    photometric_loss = PhotometricLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(args.num_epochs):
        total_loss = 0.0
        for batch_idx, data in enumerate(dataloader):
            # Load data onto device
            img = data['Image'].to(device)  # [3, height, width]
            height = data['height'][0].item()
            width = data['width'][0].item()
            points = data['points'].to(device)  # [num_points, 3]
            direction = data['direction'].to(device)

            # Predict RGB values and density from NeRF model
            with torch.no_grad(): 
                rgb_pred, density_pred = model(points, direction)
            
            rgb_pred = rgb_pred.view(args.batch_size, height, width, -1, 3)
            density_pred = density_pred.view(args.batch_size, height, width, -1, 1)
            
            rgb_tensor = render_volume(rgb_pred, density_pred, data['t_vals'].to(device))

            loss = photometric_loss(rgb_tensor, img)
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_loss:.4f}")

    # Save model checkpoint
    torch.save(model.state_dict(), args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NeRF model")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--save_path', type=str, default='nerf_model.pth', help='Path to save the trained model')
    parser.add_argument('--pos_encoding_L', type=int, default=10, help='Number of frequencies for positional encoding')
    parser.add_argument('--dir_encoding_L', type=int, default=4, help='Number of frequencies for directional encoding')
    parser.add_argument('--t_near', type=int, default=1, help='Near plane for ray sampling')
    parser.add_argument('--t_far', type=int, default=128, help='Far plane for ray sampling')
    parser.add_argument('--num_steps', type=int, default=64, help='Number of steps for ray sampling')
    args = parser.parse_args()

    evaluate_nerf(args)