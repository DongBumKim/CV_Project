import torch
import torch.nn as nn
from torchvision import models

import torch.nn.functional as F


class ResNet_Resize(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet_Resize, self).__init__()
        # self.backbone = models.resnet101(weights='ResNet101_Weights.DEFAULT')
        self.backbone = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Final classification layer
        self.fc = nn.Linear(num_features, num_classes)
        
    
    def forward(self, x):
        
        # Interpoliloation
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Feature extraction using ResNet backbone
        features = self.backbone(x)  # Output shape: [B, 2048]
        # Final classification layer
        outputs = self.fc(features)
        
        return outputs
    
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_planes, in_planes // ratio)
        self.fc2 = nn.Linear(in_planes // ratio, in_planes)
    
    def forward(self, x):
        avg_pool = torch.mean(x, dim=-1)  # Global average pooling over features
        attention = torch.sigmoid(self.fc2(F.relu(self.fc1(avg_pool))))
        return x * attention.unsqueeze(2)


class CustomModel(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomModel, self).__init__()
        # self.backbone = models.resnet101(weights='ResNet101_Weights.DEFAULT')
        self.backbone = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Normalization layers
        self.input_bn = nn.BatchNorm2d(3)
        self.bn_layer = nn.BatchNorm1d(num_features)
        self.in_layer = nn.LayerNorm(num_features, elementwise_affine=True)
        self.bn_fc = nn.BatchNorm1d(num_features)
        
        
        # Channel Attention
        self.channel_attention = ChannelAttention(num_features)
        
        # Multi-Head Self-Attention
        self.multihead_attention = nn.MultiheadAttention(embed_dim=num_features, num_heads=8, batch_first=True)
        
        # Final classification layer
        self.fc = nn.Linear(num_features, num_classes)
        
        # Additional dense attention
        self.attention_layer = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_features),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        
        # Interpoliloation
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        
        # Input normalization
        x = self.input_bn(x)
        
        # Feature extraction using ResNet backbone
        features = self.backbone(x)  # Output shape: [B, 2048]
        
        # Apply channel attention
        features = self.channel_attention(features.unsqueeze(2)).squeeze(2)  # Channel Attention
        
        # Normalize features
        bn_features = self.bn_layer(features)
        in_features = self.in_layer(features)
        features = 0.5 * bn_features + 0.5 * in_features
        
        # Apply multi-head self-attention
        attention_input = features.unsqueeze(1)  # Add a sequence dimension: [B, 1, 2048]
        attention_output, _ = self.multihead_attention(attention_input, attention_input, attention_input)
        features = features + attention_output.squeeze(1)  # Residual connection
        
        # Dense attention
        attention_weights = self.attention_layer(features)
        features = features * attention_weights  # Element-wise attention
        
        
        features = self.bn_fc(features)
        # Final classification layer
        outputs = self.fc(features)
        return outputs