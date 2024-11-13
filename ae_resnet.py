import torch
import torch.nn as nn
from torchvision import models

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class AEBottleneck(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=512):
        super(AEBottleneck, self).__init__()
        
        # Encoder layers
        self.fc_encode1 = nn.Linear(input_dim, hidden_dim)
        self.bn_encode1 = nn.BatchNorm1d(hidden_dim)
        self.fc_encode2 = nn.Linear(hidden_dim, latent_dim)
        self.bn_encode2 = nn.BatchNorm1d(latent_dim)
        
        # Decoder layers
        self.fc_decode1 = nn.Linear(latent_dim, hidden_dim)
        self.bn_decode1 = nn.BatchNorm1d(hidden_dim)
        self.fc_decode2 = nn.Linear(hidden_dim, input_dim)
        self.bn_decode2 = nn.BatchNorm1d(input_dim)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoding phase
        x = self.relu(self.bn_encode1(self.fc_encode1(x)))
        z = self.relu(self.bn_encode2(self.fc_encode2(x)))
        
        # Decoding phase
        x_reconstructed = self.relu(self.bn_decode1(self.fc_decode1(z)))
        x_reconstructed = self.bn_decode2(self.fc_decode2(x_reconstructed))  # No ReLU on the final layer

        return x_reconstructed

class ResNet50WithAE(nn.Module):
    def __init__(self, latent_dim=128, num_classes=7, noise_std=0.1):
        super(ResNet50WithAE, self).__init__()
        
        # Load the ResNet50 model pre-trained on ImageNet
        self.backbone = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        
        # Extract the input dimension of the feature space before the fully connected layer
        self.feature_dim = self.backbone.fc.in_features # 2048
        self.latent_dim = latent_dim 

        # Replace the fully connected layer with our classifier
        self.backbone.fc = nn.Identity()  # Remove the final layer, we'll handle it ourselves
        self.fc_class = nn.Linear(self.feature_dim, num_classes)  # Classifier
        
        # AE bottleneck for feature regularization
        self.ae_bottleneck = AEBottleneck(self.feature_dim, self.latent_dim)
        
        # Gaussian noise layers
        self.input_noise = GaussianNoise(noise_std)  # Apply noise to input

    def forward(self, x):
        # Add Gaussian noise to the input
        x = self.input_noise(x)
        
        x = self.backbone(x)
    
        # AE bottleneck for feature regularization
        reconstructed_features = self.ae_bottleneck(x)
        
        # Classification output (use reconstructed features or original features)
        out_class = self.fc_class(reconstructed_features)
        
        return out_class
