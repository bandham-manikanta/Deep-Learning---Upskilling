import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNet_Transfer(nn.Module):
    """EfficientNet with transfer learning for CIFAR-10"""
    
    def __init__(self, num_classes=10, freeze_backbone=False, variant='b3'):
        super(EfficientNet_Transfer, self).__init__()
        
        # Load pretrained EfficientNet (ImageNet weights)
        if variant == 'b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif variant == 'b3':
            self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        elif variant == 'b5':
            self.backbone = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        # Freeze backbone if specified
        if freeze_backbone:
            # YOUR STRATEGY: Freeze everything first
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Then unfreeze ONLY the last feature block
            for param in self.backbone.features[-1].parameters():
                param.requires_grad = True
            
        # Replace final classifier for CIFAR-10
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
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
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
            
        self.model_name = f"EfficientNet-{variant.upper()} Transfer"
        self.variant = variant
        
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True

