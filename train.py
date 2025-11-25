import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import BiomassDataset
from model import BiomassModel

print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CSV paths (unchanged)
train_csv = "/kaggle/working/train_split.csv"
val_csv   = "/kaggle/working/val_split.csv"

# Image directories (unchanged)
train_img_dir = "/kaggle/working/image-data/train"
val_img_dir   = "/kaggle/working/image-data/train"

# Strongest offline augmentations
transform = A.Compose([
    A.Resize(224, 224),

    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(p=0.5, rotate_limit=25),
    A.RandomBrightnessContrast(p=0.5),
    A.ColorJitter(p=0.5),
    A.GaussianBlur(p=0.3),
    A.MotionBlur(p=0.3),
    A.ISONoise(p=0.4),
    A.RGBShift(p=0.3),
    A.CLAHE(p=0.3),
    A.CoarseDropout(max_holes=6, max_width=32, max_height=32, p=0.6),

    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Dataset & Dataloader
train_dataset = BiomassDataset(train_csv, train_img_dir, transform)
val_dataset   = BiomassDataset(val_csv, val_img_dir, transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

# -------- Model --------
model = BiomassModel(n_outputs=1, model_name="resnet18").to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)

# LR Scheduler: Cosine annealing for long training
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# SWA for better generalization
swa_model = optim.swa_utils.AveragedModel(model)
swa_scheduler = optim.swa_utils.SWALR(optimizer, swa_lr=1e-4)


def train_one_epoch():
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.view(-1).float().to(device)

        optimizer.zero_grad()
        preds = model(imgs).squeeze()
        loss = criterion(preds, labels)
        loss.backward()

        # gradient clipping reduces instability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate():
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.view(-1).float().to(device)

            preds = model(imgs).squeeze()
            loss = criterion(preds, labels)
            total_loss += loss.item()

    return total_loss / len(val_loader)


# ------------ Training Loop ------------
EPOCHS = 120

for epoch in range(EPOCHS):
    train_loss = train_one_epoch()
    val_loss = validate()

    scheduler.step()

    # Update SWA model after half training
    if epoch > EPOCHS // 2:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        swa_scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# SWA BN update improves metrics
optim.swa_utils.update_bn(train_loader, swa_model)

torch.save(swa_model.module.state_dict(), "/kaggle/working/resnet18_biomass.pth")
print("Model saved!")
