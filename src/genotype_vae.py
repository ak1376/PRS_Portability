# src/genotype_vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenotypeVAE(nn.Module):
    """
    Fully-connected VAE for diploid genotype reconstruction.

    Data:
      - Input x: diploid genotypes per SNP.
        We support:
          * raw dosages in {0,1,2}, OR
          * scaled values in {0.0, 0.5, 1.0} (i.e. dosage / 2).
        The loss will internally convert scaled inputs back to 0/1/2.

    Architecture:
      - Encoder: MLP, depth `depth`, width `width`, ELU activations
      - Latent: mean + logvar, dim = latent_dim
      - Decoder: mirrored MLP with ELU, final sigmoid giving p in (0,1)
      - Likelihood: Binomial(n=2, p)
      - Loss: NLL_binomial(2,p | x) + beta * KL

    Knobs:
      - beta: KL weight
      - deterministic_latent:
          * False (default): standard VAE, reparameterization sampling
          * True: use z = mu (no sampling); behaves like a deterministic AE
    """

    def __init__(
        self,
        input_dim: int,
        width: int = 128,
        depth: int = 6,
        latent_dim: int = 32,
        beta: float = 1.0,
        deterministic_latent: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        self.depth = depth
        self.latent_dim = latent_dim
        self.beta = beta
        self.deterministic_latent = deterministic_latent

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

    def encode(self, x: torch.Tensor):
        """
        x: (batch, D) — can be 0/1/2 or scaled [0,1].
        """
        h = self.encoder_net(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Standard reparameterization trick, with an option for deterministic latent.

        If self.deterministic_latent is True, returns mu (no sampling).
        Otherwise, returns mu + eps * std.
        """
        if self.deterministic_latent:
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        """
        z: (batch, latent_dim)
        Returns:
          p: (batch, D), allele probabilities in (0,1)
        """
        h = self.decoder_net(z)
        p = self.out_act(self.out_layer(h))  # allele probs
        return p

    def forward(self, x: torch.Tensor):
        """
        Forward pass:
          - encode x -> (mu, logvar)
          - reparameterize -> z
          - decode z -> p (allele probabilities)

        Returns:
          p, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        p = self.decode(z)
        return p, mu, logvar

    # ----- Binomial(2,p) loss -----

    @staticmethod
    def _prepare_dosage(x: torch.Tensor, n: float = 2.0) -> torch.Tensor:
        """
        Ensure x is in [0,n] as dosage counts.

        Supports:
          - x ~ {0,1,2} (already dosage)
          - x ~ {0.0, 0.5, 1.0} (scaled dosage / 2) -> multiply by 2.
        """
        with torch.no_grad():
            x_max = x.max()

        # If max <= 1, assume scaled dosage in [0,1], multiply by n/1 (=2).
        if x_max <= 1.0 + 1e-6:
            x_dosage = x * n
        else:
            x_dosage = x

        # Clamp into [0, n] to be safe
        x_dosage = x_dosage.clamp(0.0, n)
        return x_dosage

    @staticmethod
    def _binomial_nll(x_dosage: torch.Tensor, p: torch.Tensor, n: float = 2.0, eps: float = 1e-7):
        """
        Negative log-likelihood for Binomial(n=2, p) up to an additive constant.

        x_dosage: (batch, D)  diploid genotype counts (0,1,2)
        p:        (batch, D)  allele probabilities in (0,1) from decoder.

        We compute (dropping log C(n,x) term):
            -log P(x | p) ∝ -[ x log p + (n - x) log(1-p) ]

        Returns scalar NLL averaged over batch.
        """
        # clamp probabilities for numerical stability
        p = p.clamp(eps, 1.0 - eps)

        # clamp x for safety
        x_dosage = x_dosage.clamp(0.0, n)

        nll = -(x_dosage * torch.log(p) + (n - x_dosage) * torch.log(1.0 - p))
        # sum over features, average over batch
        return nll.sum(dim=1).mean()

    def loss_function(self, recon_p: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta=None):
        """
        Binomial(2,p) NLL + beta * KL.

        Inputs:
          recon_p: decoder output (batch, D), allele probabilities in (0,1)
          x:       diploid genotypes (batch, D), either:
                     - raw dosages ~ {0,1,2}, OR
                     - scaled dosages ~ {0.0,0.5,1.0}
          mu, logvar: latent parameters

        Returns:
          loss, recon_nll, kl
        """
        if beta is None:
            beta = self.beta

        # Make sure x is in dosage units
        x_dosage = self._prepare_dosage(x, n=2.0)

        # Reconstruction: Binomial(2,p) NLL
        recon_nll = self._binomial_nll(x_dosage, recon_p, n=2.0)

        # KL divergence term (standard VAE)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        loss = recon_nll + beta * kl
        return loss, recon_nll, kl
