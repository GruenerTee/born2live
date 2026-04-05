#!/usr/bin/env python3
import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
import numpy as np
from ai.ratio_models import ScatteringCNN_Ratios
from ai.trainer import ModelTrainer
from ai_data_utils import load_dataset

SHAPE_MAP = {"cylinder": 0, "sphere": 1, "box": 2, "cone": 3}

class RatioDataset(Dataset):
    """
    Dataset that calculates ratios:
    Targets: [a, a/radius, radius/height]
    """
    def __init__(self, file_path):
        # Load absolute values
        target_keys = ["a", "radius", "height"]
        X, Y_abs = load_dataset(file_path, target_keys)
        
        with np.load(file_path, allow_pickle=True) as loader:
            Y_dicts = loader['Y_dicts']
            
        Y_cls = []
        for d in Y_dicts:
            shape_str = d.get('shape', 'cylinder').lower()
            Y_cls.append(SHAPE_MAP.get(shape_str, 0))
            
        # 1. Pre-process image data
        self.X = torch.FloatTensor(X).unsqueeze(1)
        self.X = torch.log1p(self.X)
        if self.X.max() > 0:
            self.X /= self.X.max()
            
        # 2. Calculate Ratios
        # Y_abs columns: 0:a, 1:radius, 2:height
        a = Y_abs[:, 0]
        radius = Y_abs[:, 1]
        height = Y_abs[:, 2]
        
        # Avoid division by zero
        radius = np.where(radius == 0, 1e-6, radius)
        height = np.where(height == 0, 1e-6, height)
        
        a_r_ratio = a / radius
        r_h_ratio = radius / height
        
        # Combine targets: [a, a/r, r/h]
        Y_ratios = np.stack([a, a_r_ratio, r_h_ratio], axis=1)
        self.Y_reg = torch.FloatTensor(Y_ratios)
        
        # Normalize targets (Z-score)
        self.Y_reg_mean = self.Y_reg.mean(dim=0)
        self.Y_reg_std = self.Y_reg.std(dim=0) + 1e-6
        self.Y_reg = (self.Y_reg - self.Y_reg_mean) / self.Y_reg_std
        
        self.Y_cls = torch.LongTensor(Y_cls)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Return image, ratios, and shape label
        return self.X[idx], self.Y_reg[idx], self.Y_cls[idx]

def train_ratios():
    import argparse
    parser = argparse.ArgumentParser(description="Train ratio-based scattering model.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    args = parser.parse_args()

    # 1. Configuration
    dataset_dir = "large_dataset"
    model_output = "trained_ratio_model.pth"
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr

    # 2. Load Data
    chunk_files = sorted(glob.glob(os.path.join(dataset_dir, "chunk_*.npz")))
    if not chunk_files:
        print(f"Error: No chunk files found in {dataset_dir}/")
        return

    print(f"Found {len(chunk_files)} chunks. Loading ratio datasets...")
    individual_datasets = [RatioDataset(f) for f in chunk_files]
    full_dataset = ConcatDataset(individual_datasets)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # 3. Initialize Model
    model = ScatteringCNN_Ratios(num_ratios=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()

    # 4. Training Loop
    print(f"Starting Ratio-based training on {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, targets_reg, targets_cls in train_loader:
            images, targets_reg, targets_cls = images.to(device), targets_reg.to(device), targets_cls.to(device)
            
            optimizer.zero_grad()
            out_reg, out_cls = model(images)
            
            loss_reg = criterion_reg(out_reg, targets_reg)
            loss_cls = criterion_cls(out_cls, targets_cls)
            
            loss = loss_reg + loss_cls
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets_reg, targets_cls in val_loader:
                images, targets_reg, targets_cls = images.to(device), targets_reg.to(device), targets_cls.to(device)
                out_reg, out_cls = model(images)
                val_loss += (criterion_reg(out_reg, targets_reg) + criterion_cls(out_cls, targets_cls)).item()
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {train_loss/len(train_loader):.6f} | Val: {val_loss/len(val_loader):.6f}")

    # 5. Save Model
    # Extract stats for inference
    ds = individual_datasets[0]
    stats = {'reg_mean': ds.Y_reg_mean, 'reg_std': ds.Y_reg_std}
    torch.save({'model_state_dict': model.state_dict(), 'stats': stats}, model_output)
    print(f"Saved ratio model to {model_output}")

if __name__ == "__main__":
    train_ratios()
