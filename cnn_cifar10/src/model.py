import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    """Basic CNN for CIFAR-10 (improved version of original Net)"""
    
    def __init__(self, num_classes=10):
        super(BasicCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)      # 32x32x32
        self.pool1 = nn.MaxPool2d(2, 2)                             # 16x16x32
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)    # 16x16x64
        self.pool2 = nn.MaxPool2d(2, 2)                             # 8x8x64
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)   # 8x8x128
        self.pool3 = nn.MaxPool2d(2, 2)                             # 4x4x128
        
        # Classifier layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Feature extraction
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten for classifier
        x = x.view(-1, 128 * 4 * 4)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedCNN(nn.Module):
    """Enhanced CNN with 50% more parameters for better CIFAR-10 performance"""
    
    def __init__(self, num_classes=10):
        super(EnhancedCNN, self).__init__()
        
        # More channels + additional conv layer
        # Block 1: 3 -> 48 (was 32)
        self.conv1a = nn.Conv2d(3, 48, kernel_size=3, padding=1)     # 32x32x48
        self.conv1b = nn.Conv2d(48, 48, kernel_size=3, padding=1)    # 32x32x48
        self.pool1 = nn.MaxPool2d(2, 2)                              # 16x16x48
        
        # Block 2: 48 -> 96 (was 64)  
        self.conv2a = nn.Conv2d(48, 96, kernel_size=3, padding=1)    # 16x16x96
        self.conv2b = nn.Conv2d(96, 96, kernel_size=3, padding=1)    # 16x16x96
        self.pool2 = nn.MaxPool2d(2, 2)                              # 8x8x96
        
        # Block 3: 96 -> 192 (was 128)
        self.conv3a = nn.Conv2d(96, 192, kernel_size=3, padding=1)   # 8x8x192
        self.conv3b = nn.Conv2d(192, 192, kernel_size=3, padding=1)  # 8x8x192
        self.pool3 = nn.MaxPool2d(2, 2)                              # 4x4x192
        
        # Batch normalization for better training
        self.bn1a = nn.BatchNorm2d(48)
        self.bn1b = nn.BatchNorm2d(48)
        self.bn2a = nn.BatchNorm2d(96)
        self.bn2b = nn.BatchNorm2d(96)
        self.bn3a = nn.BatchNorm2d(192)
        self.bn3b = nn.BatchNorm2d(192)
        
        # Enhanced classifier: 192*4*4 = 3072
        self.fc1 = nn.Linear(192 * 4 * 4, 1024)  # Larger: 512 -> 1024
        self.fc2 = nn.Linear(1024, 512)          # Additional layer
        self.fc3 = nn.Linear(512, 128)           # Additional layer  
        self.fc4 = nn.Linear(128, num_classes)   # Final classifier
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Block 1: Two conv layers + batch norm
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = self.pool1(F.relu(self.bn1b(self.conv1b(x))))
        
        # Block 2: Two conv layers + batch norm
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = self.pool2(F.relu(self.bn2b(self.conv2b(x))))
        
        # Block 3: Two conv layers + batch norm
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = self.pool3(F.relu(self.bn3b(self.conv3b(x))))
        
        # Flatten for classifier
        x = x.view(-1, 192 * 4 * 4)
        
        # Enhanced classifier with more layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return x
