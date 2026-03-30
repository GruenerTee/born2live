import torch
import torch.nn as nn

class ScatteringCNN(nn.Module):
    """Convolutional Neural Network for Scattering Data Regression."""
    def __init__(self, num_outputs):
        super(ScatteringCNN, self).__init__()
        # CNN layers to extract spatial features
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
        
        # Adaptive pooling ensures compatibility with any input image size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Fully connected layers for the final regression output
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_outputs)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
