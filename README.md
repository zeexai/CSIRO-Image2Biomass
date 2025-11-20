# CSIRO-Image2Biomass
This project builds an AI-based regression model that predicts five types of pasture biomass directly from images of grassland. This helps farmers estimate feed availability and plan grazing more efficiently, without manual measurement methods.
The model takes a pasture image as input and outputs:
*Dry_Green_g
*Dry_Dead_g
*Dry_Clover_g
*GDM_g (Green Dry Matter)
*Dry_Total_g (most important target)
These predictions can support sustainable agriculture, reduce manual labor, and scale to large farms.

# Model Architecture
This project uses a pretrained EfficientNet-B0 backbone (ImageNet weights) and adds a custom regression head:
Image → EfficientNetB0 → Feature Vector → Dense → Dense → 5 outputs

# Training Pipeline
1️⃣ Dataset Loading
Images and labels are read using dataset.py
Resize to 224×224
Normalize
Light augmentations (flip, rotation)

2️⃣ Model Training
train.py handles:
Train/validation split
MSE loss
Adam optimizer
Weighted R² evaluation
Saving best model (best_model.pth)

3️⃣ Inference
inference.py:
Loads saved model
Predicts 5 biomass values per test image
Formats output for Kaggle
Generates submission.csv
