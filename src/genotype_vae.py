# src/genotype_vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenotypeVAE(nn.Module):
    """
    Fully-connected VAE for diploid genotype reconstruction.

    Data:
      - Input x: diploid genotypes per SNP, ideally in {0,1,2}.
        (We allow small numerical noise; values are clamped into [0,2].)

    Architecture:
      - Encoder: depth 'depth', width 'width', ELU activations
      - Latent: mean + logvar, dim = latent_dim
      - Decoder: mirrored MLP with ELU, final sigmoid giving p in (0,1)
      - Likelihood: Binomial(2, p)
      - Loss: NLL_binomial(2,p | x) + beta * KL

    Notes:
      - This is *not* Bernoulli; we are modeling diploid counts correctly.
      - You should *not* standardize x for this model; keep them as 0/1/2
        (or 0,0.5,1 scaled that we rescale back internally).
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
        # output allele prob p in (0,1)
        self.out_act = nn.Sigmoid()

    # ----- core VAE pieces -----

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
        p = self.out_act(self.out_layer(h))  # allele probs
        return p

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        p = self.decode(z)
        return p, mu, logvar

    # ----- Binomial(2,p) loss -----

    @staticmethod
    def _binomial_nll(x, p, n: float = 2.0, eps: float = 1e-7):
        """
        Negative log-likelihood for Binomial(n=2, p) up to an additive constant.

        x: (batch, D)  diploid genotype counts, ideally 0,1,2.
        p: (batch, D)  allele probabilities in (0,1) from decoder.

        We compute:
            -log P(x | p) ∝ -[ x log p + (n - x) log(1-p) ]

        The combinatorial term log C(n, x) is dropped since it does not
        depend on p, so it does not affect gradients.
        """
        # clamp probabilities for numerical stability
        p = p.clamp(eps, 1.0 - eps)

        # clamp genotypes into [0, n]
        x = x.clamp(0.0, n)

        nll = -(x * torch.log(p) + (n - x) * torch.log(1.0 - p))
        # sum over features, average over batch
        return nll.sum(dim=1).mean()

    def loss_function(self, recon_p, x, mu, logvar, beta=None):
        """
        Binomial(2,p) NLL + beta * KL.

        Inputs:
          recon_p: decoder output (batch, D), allele probabilities in (0,1)
          x:       raw diploid genotypes (batch, D), expected ~ {0,1,2}
          mu, logvar: latent parameters

        Returns:
          loss, recon_nll, kl
        """
        if beta is None:
            beta = self.beta

        # Reconstruction: Binomial(2,p) NLL
        recon_nll = self._binomial_nll(x, recon_p, n=2.0)

        # KL divergence term (standard VAE)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        loss = recon_nll + beta * kl
        return loss, recon_nll, kl
