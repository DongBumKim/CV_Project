import torch
import torch.nn as nn
from torchvision import models

class CustomResNetModel(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomResNetModel, self).__init__()
        
        # Load pre-trained ResNet50 model as backbone
        self.backbone = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the original FC layer
        
        # Batch Normalization on the input to adjust for domain variations
        self.input_bn = nn.BatchNorm2d(3)
        
        # Group Normalization for generalization on backbone output
        self.gn_layer = nn.GroupNorm(num_groups=32, num_channels=num_features)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.3)
        
        # Final classification layer (with multi-scale feature aggregation)
        # Doubling the feature size by concatenating with itself after dropout
        self.fc = nn.Linear(num_features * 2, num_classes)

    def forward(self, x):
        # Initial input normalization for domain adaptation
        x = self.input_bn(x)
        
        # Extract features with the ResNet backbone (output shape: (B, num_features))
        features = self.backbone(x)
        
        # Apply Group Normalization
        features = self.gn_layer(features.unsqueeze(-1)).squeeze(-1)  # Applying GN on (B, num_features, 1)
        
        # Duplicate and concatenate features for multi-scale representation
        combined_features = torch.cat([features, features], dim=1)
        
        # Apply dropout for regularization
        combined_features = self.dropout(combined_features)
        
        # Final classification output
        outputs = self.fc(combined_features)
        
        return outputs
