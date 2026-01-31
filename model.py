"""
CNN Model Architecture for CIFAR-10 Classification

This module defines the convolutional neural network architecture
used for image classification on the CIFAR-10 dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A simple CNN architecture for CIFAR-10 classification.
    
    Architecture:
        - 3 Convolutional blocks (Conv → ReLU → BatchNorm → MaxPool)
        - 2 Fully connected layers with dropout
        - Final softmax layer for 10-class classification
    
    Args:
        num_classes (int): Number of output classes (default: 10)
        dropout_rate (float): Dropout probability (default: 0.5)
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # First conv block: 32x32 → 16x16
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv block: 16x16 → 8x8
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third conv block: 8x8 → 4x4
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten: (batch_size, 256, 4, 4) → (batch_size, 256*4*4)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def get_num_params(self):
        """
        Calculate the total number of trainable parameters.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for deeper networks.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride for first convolution (default: 1)
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """Forward pass with residual connection."""
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out


class ImprovedCNN(nn.Module):
    """
    Improved CNN with residual connections for better performance.
    
    This architecture uses residual blocks to enable deeper networks
    and achieve better accuracy on CIFAR-10.
    
    Args:
        num_classes (int): Number of output classes (default: 10)
    """
    
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a layer with multiple residual blocks."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def create_model(model_type='simple', **kwargs):
    """
    Factory function to create different CNN architectures.
    
    Args:
        model_type (str): Type of model ('simple' or 'improved')
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        nn.Module: Instantiated model
    """
    if model_type == 'simple':
        model = SimpleCNN(**kwargs)
    elif model_type == 'improved':
        model = ImprovedCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_model('simple')
    print(f"Model created with {model.get_num_params():,} parameters")
    
    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test improved model
    model_improved = create_model('improved')
    output_improved = model_improved(x)
    print(f"\nImproved model output shape: {output_improved.shape}")
