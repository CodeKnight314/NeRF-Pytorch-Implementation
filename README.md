# NeRF-Pytorch-Implementation

## Overview:
This is an educational repository on the implementation of NeRF for novel view synthesis from scratch. The primary dataset used is NeRF Synthetic which includes synthetic rendered images of objects at various angles. The main goal of this repository is to provide a demonstration of various technical concepts involved in traing NeRF and rendering images. 

```python 
python NeRF-Pytorch-Implementation/train.py --root lego/ --batch 1 --lr 5e-4 --epochs 100 --save Outputs/ --pos_encoding_L 10 --dir_encoding_L 4 --num_steps 128 --size 128
```