# src/genotype_vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenotypeVAE(nn.Module):
    """
    Simple fully-connected VAE for genotype reconstruction.

    Architecture:
      - Input: (N_samples, num_snps)
      - Encoder: depth 'depth', width 'width', ELU activations
      - Latent: mean + logvar of size latent_dim
      - Decoder: mirrored MLP with ELU, final sigmoid
      - Loss: MSE reconstruction + beta * KL
    """

    def __init__(self, input_dim, width=128, depth=6, latent_dim=32, beta=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        self.depth = depth
        self.latent_dim = latent_dim
        self.beta = beta

        # --------- encoder ---------
        enc_layers = []
        in_dim = input_dim
        for _ in range(depth):
            enc_layers.append(nn.Linear(in_dim, width))
            enc_layers.append(nn.ELU())
            in_dim = width
        self.encoder_net = nn.Sequential(*enc_layers)

        self.mu_layer = nn.Linear(width, latent_dim)
        self.logvar_layer = nn.Linear(width, latent_dim)

        # --------- decoder ---------
        dec_layers = []
        in_dim = latent_dim
        for _ in range(depth):
            dec_layers.append(nn.Linear(in_dim, width))
            dec_layers.append(nn.ELU())
            in_dim = width
        self.decoder_net = nn.Sequential(*dec_layers)

        self.out_layer = nn.Linear(width, input_dim)
        # sigmoid at the end since genotypes are in [0,1]
        self.out_act = nn.Sigmoid()

    def encode(self, x):
        h = self.encoder_net(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_net(z)
        out = self.out_act(self.out_layer(h))
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(self, recon, x, mu, logvar, beta=None):
        """
        Reconstruction + beta * KL.

        x, recon: (batch, input_dim) in [0,1]
        """
        if beta is None:
            beta = self.beta

        # MSE over features, averaged over batch
        recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)

        # KL divergence term
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        loss = recon_loss + beta * kl
        return loss, recon_loss, kl
