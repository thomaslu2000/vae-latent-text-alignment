import torch
import torch.nn as nn
import torch.nn.functional as F

from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)

class FeatureVAE(nn.Module):
    def __init__(self, z_dim, num_features, freeze_pretrained=True, im_size=128):
        super(FeatureVAE, self).__init__()
        
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=z_dim,
            input_height=im_size,
            first_conv=False,
            maxpool1=False
        )
        
        if freeze_pretrained:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
        
        self.project_mu = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, z_dim-num_features)
        )
        self.project_var = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, z_dim-num_features)
        )
        
        self.z_dim = z_dim
        self.num_features = num_features

    
    def resample(self, mu, var):
        return mu + torch.exp(var / 2) * torch.randn_like(var)
    
    def forward(self, x, labels):
        
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.project_mu(x)
        var = self.project_var(x)
        
        z = self.resample(mu, var)
        z2 = torch.concat((z, labels), dim=-1)
        img = F.sigmoid(self.decoder(z2))
        return img, mu, var, z
    
    def generate(self, z, labels):
        z2 = torch.concat((z, labels), dim=-1)
        img = F.sigmoid(self.decoder(z2))
        return img