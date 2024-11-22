# NeRF-Pytorch-Implementation

## Overview:
This is an educational repository on the implementation of NeRF for novel view synthesis from scratch. The primary dataset used is NeRF Synthetic which includes synthetic rendered images of objects at various angles. The main goal of this repository is to provide a demonstration of various technical concepts involved in traing NeRF and rendering images. 

## Results
<p align="center">
  <img src="images/Lego Model.gif" alt="Switching between Low-Resolution and Super-Resolved Image">
</p>

## Table of Contents
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
  - [Training](#training)
  - [Dataset Preparation](#dataset-preparation)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/CodeKnight314/ESRGAN-pytorch.git
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv nerf-env
    source nerf-env/bin/activate
    ```

3. cd to project directory: 
    ```bash 
    cd NeRF-Pytorch-Implementation/
    ```

4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Inference

Use the `vol_render.py` script to perform super-resolution on images in a specified directory.

**Arguments:**
- `--root_dir`: Directory containing input images.
- `--output_dir`: Directory to save super-resolved images.
- `--resize`: Optional flag to resize super-resolved images to the original image size.

**Example:**
```bash
python vol_render.py --weight_path ./dir/weights --output_path ./data/output --img_h HEIGHT --img_w WIDTH
```

You can optionally use the `--resize` flag if you want the super-resolved image to be resized to original image size.

```bash
python inference.py --root_dir ./data/input --output_dir ./data/output --resize 
```

### Training
Use the `train.py` script to train the ESRGAN model. It includes pretraining of the generator and full training with the discriminator.

**Arguments:**
- `--root` Root directory for images (requires train and val split).
- `--lr`: Initial Learning rate of the NeRF model.
- `--epochs`: Total epochs for running the model.
- `--save`: Output directory for saving model weights and images
- `--num_steps`: Number of samples per generated ray
- `--size`: The desired output size of rendered image.

**Example:**
```bash 
python NeRF-Pytorch-Implementation/train.py --root lego/ --lr 5e-4 --epochs 16 --save Outputs/ --num_steps 192 --size 128
```