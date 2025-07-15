import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50_Transfer(nn.Module):
    """ResNet-50 with transfer learning for CIFAR-10"""
    
    def __init__(self, num_classes=10, freeze_backbone=False):
        super(ResNet50_Transfer, self).__init__()
        
        # Load pretrained ResNet-50 (ImageNet weights)
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
        
        in_features = self.backbone.fc.in_features
        # Replace final classifier for CIFAR-10
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1024),  # Larger intermediate layer
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Ensure final layers are trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
            
        self.model_name = "ResNet-50 Transfer"
        
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True

