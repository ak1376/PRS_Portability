# src/genotype_vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenotypeVAE(nn.Module):
    """
    Fully-connected β-VAE for genotype *dosages* scaled to [0, 1].

    Assumptions:
      - Input X is (batch_size, num_snps) with values in [0, 1]
        e.g. genotypes 0/1/2 scaled as 0, 0.5, 1.0.
      - Decoder outputs are in (0,1) via sigmoid.
      - Reconstruction loss is binary cross-entropy (BCE) over SNPs.
      - KL term is weighted by beta (β-VAE).

    Architecture:
      - Encoder: depth 'depth', width 'width', ELU activations
      - Latent: mean + logvar of size latent_dim
      - Decoder: mirrored MLP with ELU, final sigmoid
      - Loss: BCE reconstruction + beta * KL
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
        # sigmoid at the end since *scaled* genotypes are in [0,1]
        self.out_act = nn.Sigmoid()

    # -------------------------
    # Core VAE ops
    # -------------------------
    def encode(self, x):
        """
        x: (batch, input_dim) in [0,1]
        returns:
          mu, logvar: (batch, latent_dim)
        """
        h = self.encoder_net(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick:
          z = mu + eps * exp(0.5 * logvar)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        z: (batch, latent_dim)
        returns:
          recon: (batch, input_dim) in (0,1)
        """
        h = self.decoder_net(z)
        out = self.out_act(self.out_layer(h))
        return out

    def forward(self, x):
        """
        x: (batch, input_dim) in [0,1]
        returns:
          recon, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    # -------------------------
    # Loss (BCE + β * KL)
    # -------------------------
    def loss_function(self, recon, x, mu, logvar, beta=None):
        """
        Reconstruction + beta * KL.

        x, recon: (batch, input_dim) in [0,1]

        - Reconstruction: binary cross-entropy over all SNPs,
          summed over features and averaged over batch.
        - KL: standard VAE KL(q(z|x) || N(0, I)), averaged over batch.
        """
        if beta is None:
            beta = self.beta

        # BCE over all SNPs, summed over features, averaged over batch
        # (same spirit as popVAE: per-site BCE scaled by num_snps)
        recon_loss = F.binary_cross_entropy(
            recon, x, reduction="sum"
        ) / x.size(0)

        # KL divergence term (per-sample, averaged over batch)
        kl = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        ) / x.size(0)

        loss = recon_loss + beta * kl
        return loss, recon_loss, kl
