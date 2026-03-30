#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from ai_data_utils import load_dataset

# 1. Dataset Class
class BornAgainDataset(Dataset):
    def __init__(self, file_path, target_keys):
        X, Y = load_dataset(file_path, target_keys)
        # Normalize X (0-1 range for grayscale pixels)
        # Assuming data is non-negative scattering intensity
        self.X = torch.FloatTensor(X).unsqueeze(1) # Add channel dimension: (N, 1, H, W)
        if self.X.max() > 0:
            self.X = self.X / self.X.max() 
            
        self.Y = torch.FloatTensor(Y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# 2. Small CNN Model
class ScatteringCNN(nn.Module):
    def __init__(self, num_outputs):
        super(ScatteringCNN, self).__init__()
        # Simple architecture for 200x200 or similar inputs
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 100x100 if input 200x200
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 50x50
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 25x25
        )
        
        # Adaptive pooling to handle any input size (e.g. 128x128 or 200x200)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_outputs) # Output layer for regression
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_model(dataset_path, target_keys, epochs=20, batch_size=8, lr=0.001):
    # Load data
    dataset = BornAgainDataset(dataset_path, target_keys)
    if len(dataset) < 2:
        print("Not enough samples for training. Please run more simulations.")
        return
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    
    # Init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScatteringCNN(num_outputs=len(target_keys)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, params in train_loader:
            images, params = images.to(device), params.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, params)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, params in val_loader:
                images, params = images.to(device), params.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, params).item()
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(val_loader):.6f}")
    
    # Save model
    torch.save(model.state_dict(), "bornagain_ai_model.pth")
    print("\nModel saved as bornagain_ai_model.pth")
    return model

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Find a dataset file to train on (e.g. from the last simulation)
    # Search for any simulation_dataset.npz in the sim/ directory
    import glob
    npz_files = glob.glob("sim/**/simulation_dataset.npz", recursive=True)
    
    if not npz_files:
        print("No simulation_dataset.npz found. Run a simulation first.")
    else:
        dataset_path = npz_files[0] # Take the first one found
        target_keys = ["radius", "height", "a", "ref"]
        
        print(f"Training on: {dataset_path}")
        train_model(dataset_path, target_keys)
