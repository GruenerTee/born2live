import torch
import torch.nn as nn

class ForwardSimulatorMLP(nn.Module):
    """
    Predicts scattering features (peaks) from physical parameters.
    Input: [shape (one-hot), lattice, lattice/radius, radius/height]
    Output: [10 peaks * 2 coordinates = 20 values]
    """
    def __init__(self, input_size=4+3, output_size=20):
        super(ForwardSimulatorMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, params):
        return self.mlp(params)

class ScatteringCNN_Ratios(nn.Module):
    """
    Inverse model: Predicts physical ratios from scattering images.
    Target variables: [lattice (a), lattice/radius (a/r), radius/height (r/h)]
    """
    def __init__(self, num_ratios=3, num_classes=4):
        super(ScatteringCNN_Ratios, self).__init__()
        
        # Reuse robust CNN architecture
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.feature_size = 128 * 4 * 4
        
        # Predict Lattice and Ratios
        self.ratio_head = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_ratios)
        )
        
        # Predict Shape
        self.shape_head = nn.Sequential(
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, img):
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        
        ratios = self.ratio_head(x)
        shape = self.shape_head(x)
        return ratios, shape
