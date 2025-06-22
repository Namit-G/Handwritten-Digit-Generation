# train_cvae_mnist.ipynb

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset ---
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# --- Model ---
class CVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.embed = nn.Embedding(10, 10)

        self.encoder = nn.Sequential(
            nn.Linear(784 + 10, 400),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + 10, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def encode(self, x, y):
        y_embed = self.embed(y)
        x = torch.cat([x.view(-1, 784), y_embed], dim=1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_embed = self.embed(y)
        z = torch.cat([z, y_embed], dim=1)
        return self.decoder_fc(z)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# --- Loss ---
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- Training ---
model = CVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x, y)
        loss = loss_function(recon, x, mu, logvar)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader.dataset):.4f}")

# Save model
torch.save(model.state_dict(), 'cvae_mnist.pth')
