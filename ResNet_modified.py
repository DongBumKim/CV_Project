import torch
import torch.nn as nn
from torchvision import models

class Custom_ResNet_Model(nn.Module):
    def __init__(self, num_classes=7, num_attention_layers=2):
        super(Custom_ResNet_Model, self).__init__()
        
        # Load pre-trained ResNet50 model as backbone
        self.backbone = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        
        # Replace the final fully connected layer to match the number of classes
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the original FC layer to add custom layers
        
        # SWIN-inspired input BatchNorm
        self.input_bn = nn.BatchNorm2d(3)
        
        # Domain-Specific Batch Normalization for enhanced generalization
        self.bn_layer = nn.BatchNorm1d(num_features)
        
        # Multiple SWIN-inspired Local Attention Layers with batch normalization after each
        self.attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim=num_features, num_heads=4, batch_first=True) for _ in range(num_attention_layers)]
        )
        self.attention_bn_layers = nn.ModuleList(
            [nn.BatchNorm1d(num_features) for _ in range(num_attention_layers)]
        )
        
        # Final fully connected layer for classification
        self.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        
        # Initial input batch normalization
        x = self.input_bn(x)  # Input shape: (B, 3, 64, 64)
        
        # Pass through the ResNet backbone
        features = self.backbone(x)  # Output shape: (B, 2048)
        
        # Reshape for attention: add a sequence length dimension (sequence length = 1 here)
        features = features.unsqueeze(1)  # Shape: (B, 1, 2048)
        
        # Apply multiple layers of attention with batch normalization after each layer
        for attention_layer, bn_layer in zip(self.attention_layers, self.attention_bn_layers):
            features, _ = attention_layer(features, features, features)  # Shape: (B, 1, 2048)
            features = features.squeeze(1)  # Temporarily remove sequence dimension for batch norm
            features = bn_layer(features)  # Apply batch normalization
            features = features.unsqueeze(1)  # Add sequence dimension back for next attention layer
        
        # Remove sequence dimension and apply final batch normalization
        features = features.squeeze(1)  # Shape: (B, 2048)
        features = self.bn_layer(features)  # Domain-Specific BatchNorm
        
        # Pass through the final fully connected layer for classification
        outputs = self.fc(features)  # Shape: (B, num_classes)
        
        return outputs
