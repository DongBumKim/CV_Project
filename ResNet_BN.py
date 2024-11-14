import torch
import torch.nn as nn
from torchvision import models

import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomModel, self).__init__()
        # Load pre-trained ResNet50 model as backbone
        self.backbone = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
            
        # Replace the final fully connected layer to match the number of classes
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the original FC to add custom layers
        
        # Add domain-adaptive layers after ResNet backbone
        self.fc = nn.Linear(num_features, 7)
        # input batch norm
        self.input_bn = nn.BatchNorm2d(3)
        # Domain-Specific Batch Normalization for enhanced generalization
        self.bn_layer = nn.BatchNorm1d(num_features)
    
    def forward(self, x):
        
        x = self.input_bn(x)
        
        # input x shape : B,3,64,64
        
        # Pass through the ResNet backbone
        
        features = self.backbone(x) # B,2048
        
        # Apply batch normalization for domain-specific feature adaptation
        features = self.bn_layer(features)
        
        # Pass through the domain-adaptive fully connected layers
        outputs = self.fc(features)
        
        return outputs
    