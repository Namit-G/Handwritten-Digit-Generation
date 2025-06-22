# app.py

import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cpu")

# Model definition (same as training)
class CVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.embed = nn.Embedding(10, 10)

        self.encoder = nn.Sequential(nn.Linear(784 + 10, 400), nn.ReLU())
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + 10, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def decode(self, z, y):
        y_embed = self.embed(y)
        z = torch.cat([z, y_embed], dim=1)
        return self.decoder_fc(z)

model = CVAE().to(device)
model.load_state_dict(torch.load("cvae_mnist.pth", map_location=device))
model.eval()

st.title("MNIST Handwritten Digit Generator")
digit = st.selectbox("Select a digit to generate (0-9)", list(range(10)))

if st.button("Generate 5 Images"):
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    with torch.no_grad():
        for i in range(5):
            z = torch.randn(1, model.latent_dim).to(device)
            y = torch.tensor([digit]).long().to(device)
            generated = model.decode(z, y)
            img = generated.view(28, 28).cpu().numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
    st.pyplot(fig)
