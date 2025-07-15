import torch
import torch.nn as nn
import torchvision.models as models

class VIT_Transfer(nn.Module):
    def __init__(self, num_classes=10, model_name='vit_l_16', freeze_backbone=False):
        super(VIT_Transfer, self).__init__()

        self.backbone = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        self.model_size = "Large"
        self.num_encoder_layers = 24

        for param in self.backbone.parameters():
            param.requires_grad = False
        if freeze_backbone:
            # Freeze ALL parameters first
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Unfreeze ONLY the last transformer encoder layer
            last_encoder_idx = self.num_encoder_layers - 1
            for param in self.backbone.encoder.layers[last_encoder_idx].parameters():
                param.requires_grad = True
        
        hidden_dim = self.backbone.heads.head.in_features

        # self.backbone.heads.head = nn.Sequential(
        #     nn.Dropout(0.3),
        #     nn.Linear(hidden_dim, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(512, num_classes)
        # )

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1024),  # Larger intermediate layer
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        for param in self.backbone.heads.head.parameters():
            param.requires_grad = True

        self.model_name = f"ViT-{model_name.split('_')[1].upper()}-Transfer"

    def forward(self, x):
        return self.backbone(x)
    