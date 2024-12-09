import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import SwinModel

class BaseSwinModel(nn.Module):
    def __init__(self, num_classes=7):
        super(BaseSwinModel, self).__init__()
        
        # Load a pre-trained Swin Transformer from Hugging Face (e.g., swin-base)
        self.backbone = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
        
        # Get the number of features from Swin Transformer's output
        num_features = self.backbone.config.hidden_size
        
        # Custom classification head
        self.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        # Resize input to match Swin's expected size of 224x224
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        
        # CLS Token
        features = self.backbone(pixel_values=x).pooler_output
        
        # Pass through the custom fully connected layer
        outputs = self.fc(features)
        
        return outputs



class CustomSwinModel(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomSwinModel, self).__init__()
        
        # Load a pre-trained ViT model from Hugging Face
        self.backbone = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
        
        # Get the number of features from Swin Transformer's output
        num_features = self.backbone.config.hidden_size
        
        # Custom classification head
        self.fc = nn.Linear(num_features, num_classes)
        
        # Input batch normalization for domain-specific adaptation
        self.bn_layer = nn.BatchNorm1d(num_features)
    
    def forward(self, x):

        
        # Resize input to match ViT's expected size of 224x224
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        
        # Class_Token
        features = self.backbone(pixel_values=x).pooler_output
        
        # Apply batch normalization for domain-specific feature adaptation
        features = self.bn_layer(features) 
        
        # Pass through the custom fully connected layer to get the final output
        outputs = self.fc(features)
        
        return outputs
