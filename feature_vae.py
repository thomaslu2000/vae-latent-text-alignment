import torch
import torch.nn as nn
from betavae import BVAE

class FeatureVAE(nn.Module):
    def __init__(self, z_dim, num_features):
        super(FeatureVAE, self).__init__()
        
        self.bvae = BVAE(z_dim)
        self.bvae.project_mu = nn.Linear(256, z_dim - num_features)
        self.bvae.project_var = nn.Linear(256, z_dim - num_features)
        
        self.z_dim = z_dim
        self.num_features = num_features
        
        self.bvae.init_weights()

    def forward(self, x, labels):
        mu, var = self.bvae.to_parameters(x)
        z = self.bvae.resample(mu, var)
        z2 = torch.concat((z, labels), dim=-1)
        img = self.bvae.to_img(z2)
        return img, mu, var, z
    
    def generate(self, z, labels):
        z2 = torch.concat((z, labels), dim=-1)
        img = self.bvae.to_img(z2)
        return img