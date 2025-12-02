"""
Simple CNN for MNIST classification.
This serves as our baseline model for federated learning experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for MNIST (28x28 grayscale images).
    Total parameters: ~104,000
    """
    
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Quick test to verifying it works
if __name__ == "__main__":
    model = SimpleCNN(num_classes=10)
    print(f"✅ Model created! Total params: {model.get_num_parameters():,}")
    
    # Test run on GPU
    if torch.cuda.is_available():
        model = model.cuda()
        dummy = torch.randn(1, 1, 28, 28).cuda()
        out = model(dummy)
        print(f"✅ Forward pass on GPU successful! Output shape: {out.shape}")
