from torch import nn
import torch

class Encoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=[512, 256], latent_dim=128):
        super().__init__()
        layers = []
        
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(dims[-1], latent_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Decoder(nn.Module):
    def __init__(self, output_dim=768, hidden_dims=[256, 512], latent_dim=128):
        super().__init__()
        layers = []
        dims = [latent_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, z):
        return torch.sigmoid(self.network(z))