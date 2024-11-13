import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeiTModel
class Base_DeiTModel(nn.Module):
    def __init__(self, num_classes=7):
        super(Base_DeiTModel, self).__init__()
        # Load a pre-trained ViT model from Hugging Face
        self.backbone = DeiTModel.from_pretrained("facebook/deit-base-patch16-224")
        # Freeze backbone parameters if desired
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        # Get the number of features from the ViT's output
        num_features = self.backbone.config.hidden_size
        
        # Custom layers after ViT backbone
        self.fc = nn.Linear(num_features, num_classes)

    
    def forward(self, x):

        # Resize input to match ViT's expected size of 224x224
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        
        # De-IT Token ( CLS + Dilated / 2)
        features = (self.backbone(pixel_values=x).last_hidden_state[:, 0, :] + 
            self.backbone(pixel_values=x).last_hidden_state[:, 1, :]) / 2

        # Pass through the custom fully connected layer to get the final output
        outputs = self.fc(features)
        
        return outputs


class CustomDeiTModel(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomDeiTModel, self).__init__()
        
        # Load a pre-trained ViT model from Hugging Face
        self.backbone = DeiTModel.from_pretrained("facebook/deit-base-patch16-224")
        
        # Freeze backbone parameters if desired
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        # Get the number of features from the ViT's output
        num_features = self.backbone.config.hidden_size
        
        # Custom layers after ViT backbone
        self.fc = nn.Linear(num_features, num_classes)
        
        # Input batch normalization for domain-specific adaptation
        self.input_bn = nn.BatchNorm2d(3)
        self.bn_layer = nn.BatchNorm1d(num_features)
    
    def forward(self, x):
        # Normalize the input
        x = self.input_bn(x)
        
        # Resize input to match ViT's expected size of 224x224
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        
        # De-IT Token ( CLS + Dilated / 2)
        features = (self.backbone(pixel_values=x).last_hidden_state[:, 0, :] + 
            self.backbone(pixel_values=x).last_hidden_state[:, 1, :]) / 2
        
        # Apply batch normalization for domain-specific feature adaptation
        features = self.bn_layer(features)
        
        # Pass through the custom fully connected layer to get the final output
        outputs = self.fc(features)
        
        return outputs
