# src/genotype_vae.py

import math
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
    def _binomial_nll(
        x_dosage: torch.Tensor,
        p: torch.Tensor,
        n: float = 2.0,
        eps: float = 1e-7,
        mask: torch.Tensor | None = None,
    ):
        """
        Negative log-likelihood for Binomial(n=2, p).

        x_dosage: (batch, D)  diploid genotype counts (0,1,2)
        p:        (batch, D)  allele probabilities in (0,1) from decoder.
        mask:     optional boolean tensor (batch, D). If provided, NLL is
                  averaged only over entries where mask == True.
        """
        # clamp probabilities for numerical stability
        p = p.clamp(eps, 1.0 - eps)

        # clamp x for safety
        x_dosage = x_dosage.clamp(0.0, n)

        # elementwise NLL
        nll = -(x_dosage * torch.log(p) + (n - x_dosage) * torch.log(1.0 - p))

        if mask is not None:
            # restrict to masked positions only
            mask_f = mask.to(dtype=nll.dtype)
            total = mask_f.sum()
            if total > 0:
                return (nll * mask_f).sum() / total
            # if no masked entries (edge case), fall back to full loss

        # sum over features, average over batch
        return nll.sum(dim=1).mean()

    def loss_function(
        self,
        recon_p: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta=None,
        mask: torch.Tensor | None = None,
    ):
        """
        Binomial(2,p) NLL + beta * KL.

        Inputs:
          recon_p: decoder output (batch, D), allele probabilities in (0,1)
          x:       diploid genotypes (batch, D), either:
                     - raw dosages ~ {0,1,2}, OR
                     - scaled dosages ~ {0.0,0.5,1.0}
          mu, logvar: latent parameters
          mask:    optional boolean tensor (batch, D). If provided, the
                   reconstruction term is computed **only** on entries where
                   mask == True (i.e., masked positions).

        Returns:
          loss, recon_nll, kl
        """
        if beta is None:
            beta = self.beta

        # Make sure x is in dosage units
        x_dosage = self._prepare_dosage(x, n=2.0)

        # Reconstruction: Binomial(2,p) NLL (optionally masked)
        recon_nll = self._binomial_nll(x_dosage, recon_p, n=2.0, mask=mask)

        # KL divergence term (standard VAE)
        kl = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1,
        ).mean()

        loss = recon_nll + beta * kl
        return loss, recon_nll, kl

# src/genotype_cnn_vae.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GenotypeCNNVAE(nn.Module):
    """
    1D-CNN VAE for diploid genotype reconstruction.

    Input:
      x: (batch, D) genotypes per SNP, either:
        - raw dosages in {0,1,2}, OR
        - scaled in {0.0, 0.5, 1.0} (dosage/2)

    Output:
      recon_p: (batch, D) allele probabilities in (0,1)

    Likelihood:
      Binomial(n=2, p)

    Loss:
      NLL_binomial(2,p | x) + beta * KL
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        beta: float = 1.0,
        deterministic_latent: bool = False,
        # CNN knobs
        channels: tuple[int, ...] = (32, 64, 128, 256),
        kernel_size: int = 7,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.beta = float(beta)
        self.deterministic_latent = bool(deterministic_latent)

        assert kernel_size % 2 == 1, "Use an odd kernel_size for 'same-ish' padding."
        padding = kernel_size // 2

        # --------- encoder (Conv1d downsampling) ---------
        enc_layers = []
        in_ch = 1
        L = self.input_dim
        self._enc_lengths = [L]  # track lengths after each block

        for out_ch in channels:
            # stride=2 downsamples length
            enc_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=padding))
            if use_batchnorm:
                enc_layers.append(nn.BatchNorm1d(out_ch))
            enc_layers.append(nn.ELU())
            if dropout and dropout > 0:
                enc_layers.append(nn.Dropout(dropout))

            # compute new length (Conv1d formula)
            L = self._conv1d_out_len(L, kernel_size, padding, stride=2, dilation=1)
            self._enc_lengths.append(L)
            in_ch = out_ch

        self.encoder_net = nn.Sequential(*enc_layers)

        self.enc_out_channels = channels[-1]
        self.enc_out_len = L
        self.enc_flat_dim = self.enc_out_channels * self.enc_out_len

        self.mu_layer = nn.Linear(self.enc_flat_dim, self.latent_dim)
        self.logvar_layer = nn.Linear(self.enc_flat_dim, self.latent_dim)

        # --------- decoder (upsample + Conv1d) ---------
        # Project z back to the encoder feature map shape
        self.dec_proj = nn.Linear(self.latent_dim, self.enc_flat_dim)

        dec_layers = []
        rev_channels = list(channels)[::-1]  # e.g. 256,128,64,32
        in_ch = rev_channels[0]

        for out_ch in rev_channels[1:]:
            # upsample by 2, then conv to refine + change channels
            dec_layers.append(_Upsample1d(scale_factor=2))
            dec_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding))
            if use_batchnorm:
                dec_layers.append(nn.BatchNorm1d(out_ch))
            dec_layers.append(nn.ELU())
            if dropout and dropout > 0:
                dec_layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.decoder_net = nn.Sequential(*dec_layers)

        # Final refinement + map to 1 channel, then sigmoid to allele prob
        self.out_conv = nn.Conv1d(in_ch, 1, kernel_size=1)
        self.out_act = nn.Sigmoid()

    # ---------------- core VAE pieces ----------------

    def encode(self, x: torch.Tensor):
        """
        x: (batch, D)
        """
        x1 = x.unsqueeze(1)  # (B, 1, D)
        h = self.encoder_net(x1)  # (B, C, Lenc)
        h_flat = h.flatten(1)     # (B, C*Lenc)
        mu = self.mu_layer(h_flat)
        logvar = self.logvar_layer(h_flat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        if self.deterministic_latent:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        """
        z: (batch, latent_dim)
        returns p: (batch, D)
        """
        h = self.dec_proj(z)  # (B, C*Lenc)
        h = h.view(-1, self.enc_out_channels, self.enc_out_len)  # (B, C, Lenc)

        h = self.decoder_net(h)  # (B, ch, ~D) (length may not match exactly)
        # force exact length == input_dim
        if h.shape[-1] != self.input_dim:
            h = F.interpolate(h, size=self.input_dim, mode="nearest")

        p = self.out_act(self.out_conv(h))  # (B, 1, D)
        return p.squeeze(1)                 # (B, D)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        p = self.decode(z)
        return p, mu, logvar

    # ---------------- Binomial(2,p) loss (same as yours) ----------------

    @staticmethod
    def _prepare_dosage(x: torch.Tensor, n: float = 2.0) -> torch.Tensor:
        with torch.no_grad():
            x_max = x.max()
        if x_max <= 1.0 + 1e-6:
            x_dosage = x * n
        else:
            x_dosage = x
        return x_dosage.clamp(0.0, n)

    @staticmethod
    def _binomial_nll(
        x_dosage: torch.Tensor,
        p: torch.Tensor,
        n: float = 2.0,
        eps: float = 1e-7,
        mask: torch.Tensor | None = None,
    ):
        p = p.clamp(eps, 1.0 - eps)
        x_dosage = x_dosage.clamp(0.0, n)
        nll = -(x_dosage * torch.log(p) + (n - x_dosage) * torch.log(1.0 - p))

        if mask is not None:
            mask_f = mask.to(dtype=nll.dtype)
            total = mask_f.sum()
            if total > 0:
                return (nll * mask_f).sum() / total

        return nll.sum(dim=1).mean()

    def loss_function(
        self,
        recon_p: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta=None,
        mask: torch.Tensor | None = None,
    ):
        if beta is None:
            beta = self.beta

        x_dosage = self._prepare_dosage(x, n=2.0)
        recon_nll = self._binomial_nll(x_dosage, recon_p, n=2.0, mask=mask)

        kl = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1,
        ).mean()

        loss = recon_nll + beta * kl
        return loss, recon_nll, kl

    @staticmethod
    def _conv1d_out_len(L: int, k: int, p: int, stride: int, dilation: int = 1) -> int:
        # PyTorch Conv1d output length:
        # floor((L + 2p - d*(k-1) - 1)/stride + 1)
        return int(math.floor((L + 2 * p - dilation * (k - 1) - 1) / stride + 1))


class _Upsample1d(nn.Module):
    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
