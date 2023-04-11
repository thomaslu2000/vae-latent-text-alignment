import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as linalg
from betavae import BVAE

class SupervisedBVAE(BVAE):
    
    def __init__(self, z=32, supervised_out=3):
        super().__init__(z)
        self.z_to_out_map = nn.Linear(z, supervised_out) #unsure if this gets init-ed correctly
    
    def forward(self, x):
        mu, var = super().to_parameters(x)
        z = super().resample(mu, var)
        img = super().to_img(z)
        yhat = self.z_to_out_map(z)
        return img, mu, var, z, yhat
    
    def forward_with_supervision(self, x, y):
        mu, var = super().to_parameters(x)
        
        z = mu
        yhat = self.z_to_out_map(z)
        diff = F.linear((y-yhat), linalg.pinv(self.z_to_out_map.weight))
        z = z + diff # it will not necessarily be the case that Wz+bias = y
        
        
        img = super().to_img(z)
        return img