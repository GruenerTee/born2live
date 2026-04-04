import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ai_data_utils import load_dataset

SHAPE_MAP = {"cylinder": 0, "sphere": 1, "box": 2, "cone": 3}
REV_SHAPE_MAP = {v: k for k, v in SHAPE_MAP.items()}

class BornAgainDataset(Dataset):
    """Custom Dataset for loading BornAgain simulation data with shape classification and peak features."""
    def __init__(self, file_path, target_keys):
        # Load regression targets
        X, Y_reg = load_dataset(file_path, target_keys)
        
        # Load shape labels and peaks
        with np.load(file_path, allow_pickle=True) as loader:
            Y_dicts = loader['Y_dicts']
            
        Y_cls = []
        Y_peaks = []
        for d in Y_dicts:
            # Shape
            shape_str = d.get('shape', 'cylinder').lower()
            Y_cls.append(SHAPE_MAP.get(shape_str, 0))
            
            # Peaks (10 peaks * 2 coords = 20 values)
            peaks = d.get('peaks', np.zeros(20, dtype=np.float32))
            Y_peaks.append(peaks)
            
        # Pre-process image data: (N, 1, H, W), Log-transform and normalize
        self.X = torch.FloatTensor(X).unsqueeze(1)
        self.X = torch.log1p(self.X) # log(1 + x)
        if self.X.max() > 0:
            self.X /= self.X.max()
            
        self.Y_reg = torch.FloatTensor(Y_reg)
        # Normalize targets: Subtract mean and divide by std for each parameter
        self.Y_reg_mean = self.Y_reg.mean(dim=0)
        self.Y_reg_std = self.Y_reg.std(dim=0) + 1e-6
        self.Y_reg = (self.Y_reg - self.Y_reg_mean) / self.Y_reg_std
        self.Y_cls = torch.LongTensor(Y_cls)
        self.Y_peaks = torch.FloatTensor(np.array(Y_peaks))
        
        # Simple normalization for peak coordinates (assuming deg range ~0-3)
        self.Y_peaks /= 3.0 
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y_reg[idx], self.Y_cls[idx], self.Y_peaks[idx]

class ModelTrainer:
    """Manages multi-task training (Regression + Classification + Hybrid Inputs)."""
    def __init__(self, model, lr=0.001, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion_reg = nn.MSELoss()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.stats = {} # Store normalization stats here

    def train(self, train_loader, val_loader, epochs=20):
        # Capture stats from the dataset (if available)
        if hasattr(train_loader.dataset, 'dataset'): # If using random_split/Subset
            ds = train_loader.dataset.dataset
        else:
            ds = train_loader.dataset
            
        if hasattr(ds, 'Y_reg_mean'):
            self.stats['reg_mean'] = ds.Y_reg_mean
            self.stats['reg_std'] = ds.Y_reg_std

        print(f"Starting training on {self.device}...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for images, targets_reg, targets_cls, peaks in train_loader:
                images = images.to(self.device)
                targets_reg = targets_reg.to(self.device)
                targets_cls = targets_cls.to(self.device)
                peaks = peaks.to(self.device)
                
                self.optimizer.zero_grad()
                outputs_reg, outputs_cls = self.model(images, peaks)
                
                loss_reg = self.criterion_reg(outputs_reg, targets_reg)
                loss_cls = self.criterion_cls(outputs_cls, targets_cls)
                
                loss = loss_reg + loss_cls
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.6f}")

    def save(self, path):
        # Save both model weights and normalization stats
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'stats': self.stats
        }
        torch.save(checkpoint, path)
        print(f"Model and stats saved to {path}")
