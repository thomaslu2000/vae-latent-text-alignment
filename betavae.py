import torch
import torch.nn as nn
import torch.nn.functional as F

class BVAE(nn.Module):

    def __init__(self, z=32):
        super(BVAE, self).__init__()
        self.z_dim = z
        
        self.random_z = None
        self.internal_dim = 256
        self.project_mu = nn.Linear(self.internal_dim, z)
        self.project_var = nn.Linear(self.internal_dim, z)
        self.project_z = nn.Linear(z, self.internal_dim)
        
        self.to_z = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.from_z = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, output_padding=0),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
                    
        self.init_weights()
    
    def init_weights(self):
        def init_weights2(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        self.apply(init_weights2)
        
    def to_parameters(self, x):
        x = self.to_z(x)
        x = x.view(x.size(0), -1)
        mu = self.project_mu(x)
        var = self.project_var(x)
        return mu, var
    
    def resample(self, mu, var):
        return mu + torch.exp(var / 2) * torch.randn_like(var)
    
    def to_img(self, z):
        z = self.project_z(z).view(z.size(0), -1, 2, 2)
        img = self.from_z(z)
        return img

    def forward(self, x):
        mu, var = self.to_parameters(x)
        z = self.resample(mu, var)
        img = self.to_img(z)
        return img, mu, var, z
    

class BVAE2(nn.Module):

    def __init__(self, z=100):
        super(BVAE2, self).__init__()
        self.z_dim = z
        
        self.random_z = None
        self.internal_dim = 3200
        self.project_mu = nn.Linear(self.internal_dim, z)
        self.project_var = nn.Linear(self.internal_dim, z)
        self.project_z = nn.Linear(z, self.internal_dim)
        
        self.to_z = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=6, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.from_z = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=5, stride=2, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2, output_padding=0),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
        self.init_weights()
    
    def init_weights(self):
        def init_weights2(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        self.apply(init_weights2)
        
    def to_parameters(self, x):
        x = self.to_z(x)
#         print(x.shape)
        x = x.view(x.size(0), -1)
#         print(x.shape)
        mu = self.project_mu(x)
        var = self.project_var(x)
        return mu, var
    
    def resample(self, mu, var):
        return mu + torch.exp(var / 2) * torch.randn_like(var)
    
    def to_img(self, z):
        z = self.project_z(z).view(z.size(0), 128, 5, 5)
        img = self.from_z(z)
        return img

    def forward(self, x):
        mu, var = self.to_parameters(x)
        z = self.resample(mu, var)
        img = self.to_img(z)
        return img, mu, var, z