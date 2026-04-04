import torch
import torch.nn as nn

class ScatteringCNN(nn.Module):
    """
    Convolutional Neural Network for Scattering Data.
    Performs Multi-Task Learning:
    1. Regression: Predicts continuous physical parameters (radius, height, a).
    2. Classification: Predicts the shape of the particle (cylinder, sphere, box, cone).
    """
    def __init__(self, num_reg_outputs, num_classes=4, verbose=False):
        super(ScatteringCNN, self).__init__()
        self.verbose = verbose
        
        # Shared Feature Extractor
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.feature_size = 64 * 8 * 8
        
        # Task 1: Regression Head (Physical Parameters)
        self.regression_head = nn.Sequential(
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_reg_outputs)
        )
        
        # Task 2: Classification Head (Shape)
        self.classification_head = nn.Sequential(
            nn.Linear(self.feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        if self.verbose:
            print(f"--- Dataflow Verbose ---")
            print(f"Input shape: {x.shape}")
        
        features = self.conv(x)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        
        if self.verbose:
            print(f"Features size: {features.shape}")
            
        reg_output = self.regression_head(features)
        cls_output = self.classification_head(features)
        
        if self.verbose:
            print(f"Regression output: {reg_output.shape}")
            print(f"Classification output: {cls_output.shape}")
            print(f"------------------------")
            self.verbose = False
            
        return reg_output, cls_output
