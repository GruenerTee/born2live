import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ai_data_utils import load_dataset

SHAPE_MAP = {"cylinder": 0, "sphere": 1, "box": 2, "cone": 3}
REV_SHAPE_MAP = {v: k for k, v in SHAPE_MAP.items()}

class BornAgainDataset(Dataset):
    """Custom Dataset for loading BornAgain simulation data with shape classification."""
    def __init__(self, file_path, target_keys):
        # Load regression targets
        X, Y_reg = load_dataset(file_path, target_keys)
        
        # Load shape labels
        with np.load(file_path, allow_pickle=True) as loader:
            Y_dicts = loader['Y_dicts']
            
        Y_cls = []
        for d in Y_dicts:
            shape_str = d.get('shape', 'cylinder').lower()
            Y_cls.append(SHAPE_MAP.get(shape_str, 0))
            
        # Pre-process image data: (N, 1, H, W) and normalize
        self.X = torch.FloatTensor(X).unsqueeze(1)
        if self.X.max() > 0:
            self.X /= self.X.max()
            
        self.Y_reg = torch.FloatTensor(Y_reg)
        self.Y_cls = torch.LongTensor(Y_cls)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y_reg[idx], self.Y_cls[idx]

class ModelTrainer:
    """Manages multi-task training (Regression + Classification)."""
    def __init__(self, model, lr=0.001, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion_reg = nn.MSELoss()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, train_loader, val_loader, epochs=20):
        print(f"Starting training on {self.device}...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for images, targets_reg, targets_cls in train_loader:
                images = images.to(self.device)
                targets_reg = targets_reg.to(self.device)
                targets_cls = targets_cls.to(self.device)
                
                self.optimizer.zero_grad()
                outputs_reg, outputs_cls = self.model(images)
                
                loss_reg = self.criterion_reg(outputs_reg, targets_reg)
                loss_cls = self.criterion_cls(outputs_cls, targets_cls)
                
                loss = loss_reg + loss_cls
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.6f}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
