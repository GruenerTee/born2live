import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from ai_data_utils import load_dataset
from ai.model import ScatteringCNN

class BornAgainDataset(Dataset):
    """Custom Dataset for loading BornAgain simulation data."""
    def __init__(self, file_path, target_keys):
        from ai_data_utils import load_dataset
        X, Y = load_dataset(file_path, target_keys)
        
        # Pre-process image data: (N, 1, H, W) and normalize
        self.X = torch.FloatTensor(X).unsqueeze(1)
        if self.X.max() > 0:
            self.X /= self.X.max()
            
        self.Y = torch.FloatTensor(Y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class ModelTrainer:
    """Manages training and evaluation of the Scattering AI models."""
    def __init__(self, model, lr=0.001, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, train_loader, val_loader, epochs=20):
        print(f"Starting training on {self.device}...")
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for images, params in train_loader:
                images, params = images.to(self.device), params.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, params)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, params in val_loader:
                    images, params = images.to(self.device), params.to(self.device)
                    outputs = self.model(images)
                    val_loss += self.criterion(outputs, params).item()
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(val_loader):.6f}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
