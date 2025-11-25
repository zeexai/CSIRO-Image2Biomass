import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class BiomassDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = str(row["image_path"])
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Missing image → {img_path}")
            return torch.zeros(3, 224, 224), torch.tensor([row["target"]], dtype=torch.float32)

        try:
            img = np.array(Image.open(img_path).convert("RGB"))
        except:
            print(f"Corrupted image → {img_path}")
            return torch.zeros(3, 224, 224), torch.tensor([row["target"]], dtype=torch.float32)

        if self.transform:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img).permute(2,0,1).float() / 255.0

        target = torch.tensor([row["target"]], dtype=torch.float32)

        return img, target
