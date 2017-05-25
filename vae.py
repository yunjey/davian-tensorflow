import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torchvision

class VAE(nn.Module):
    def __init__(self, z_dim=100):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, z_dim*2))  # 2 for mean and variance.
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid())        
        
    def forward(self, x):
        out = self.encoder(x)
        mu, log_var = torch.chunk(out, 2, dim=1)  # mean and log variance.
        eps = Variable(torch.randn(mu.size(0), mu.size(1)).cuda(), requires_grad=False)
        z = mu + eps * torch.exp(log_var/2) # For backprop, we use reparameterization trick instead of sampling.
        out = self.decoder(z)
        return out, mu, log_var
    
    def sample(self, z):
        return self.decoder(z)
        
model = VAE()

iter_per_epoch = len(data_loader)

fixed_z = Variable(torch.randn(100, 50).cuda())
for epoch in range(50):
    for i, (images, _) in enumerate(data_loader):
        images = images.view(images.size(0), -1)
        images = Variable(images.cuda())
        out, mu, log_var = model(images)
        
        reconst_loss = nn.functional.binary_cross_entropy(out, images, size_average=False)
        kl_divergence = torch.mean(torch.sum(0.5 * (mu**2 + torch.exp(log_var) - log_var -1), 1))
        
        total_loss = reconst_loss + kl_divergence
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
