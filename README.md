
# UNet Model for Microscopy Image Segmentation

This repository contains PyTorch Implementations for UNet-based segmentation of microscopy images. 

## Project Structure
├── check_tif.ipynb # Image visualization and verification  
├── loss_functions.py # Loss functions for UNet  
├── unet_model.py # UNet architecture  
├── utils_unet_dataset.py # Dataset loader  

## Note on Data
Raw data directories (`png/`, `tif/`, `UNet_train/`) are excluded from this repository.

If you want to reproduce the results:

1. Prepare the following structure:
UNet_train/  
├── img/  
└── mask/  

2. Place the folder at the root before running training or visualization.