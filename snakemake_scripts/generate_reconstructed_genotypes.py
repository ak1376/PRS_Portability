#!/usr/bin/env python3
"""
Generate reconstructed genotypes from a trained VAE model.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import pickle

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.genotype_vae import GenotypeVAE, GenotypeCNNVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genotype", required=True, help="Original genotype .npy file")
    parser.add_argument("--meta", required=True, help="Metadata .pkl file")
    parser.add_argument("--model", required=True, help="Trained model .pt file")
    parser.add_argument("--output", required=True, help="Output .npy file for reconstructed genotypes")
    parser.add_argument("--arch", required=True, choices=["mlp", "cnn"], help="Architecture type")
    parser.add_argument("--latent-dim", type=int, required=True, help="Latent dimension")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension (MLP only)")
    parser.add_argument("--depth", type=int, default=6, help="Depth (MLP only)")
    parser.add_argument("--channels", type=str, default="32,64,128", help="CNN channels (comma-separated)")
    parser.add_argument("--kernel-size", type=int, default=5, help="CNN kernel size")
    parser.add_argument("--dropout", type=float, default=0.0, help="CNN dropout")
    parser.add_argument("--use-batchnorm", type=int, default=1, help="Use batch normalization (0 or 1)")
    
    args = parser.parse_args()

    # Load data
    genotypes = np.load(args.genotype)
    with open(args.meta, 'rb') as f:
        meta = pickle.load(f)

    print(f'Loaded genotypes shape: {genotypes.shape}')

    # Create model
    use_batchnorm = bool(args.use_batchnorm)
    
    if args.arch == 'mlp':
        model = GenotypeVAE(
            input_dim=genotypes.shape[1], 
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            depth=args.depth
        )
    elif args.arch == 'cnn':
        channels = [int(x) for x in args.channels.split(',')]
        model = GenotypeCNNVAE(
            input_dim=genotypes.shape[1],
            latent_dim=args.latent_dim, 
            channels=channels,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            use_batchnorm=use_batchnorm
        )
    else:
        raise ValueError(f'Unknown architecture: {args.arch}')

    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print('Model loaded successfully')

    # Generate reconstructions
    genotypes_tensor = torch.FloatTensor(genotypes).to(device)
    batch_size = 100  # Process in batches to avoid memory issues

    reconstructions = []
    with torch.no_grad():
        for i in range(0, len(genotypes_tensor), batch_size):
            batch = genotypes_tensor[i:i+batch_size]
            recon, _, _ = model(batch)
            reconstructions.append(recon.cpu().numpy())

    recon_genotypes = np.vstack(reconstructions)
    print(f'Generated reconstructions shape: {recon_genotypes.shape}')

    # Save reconstructed genotypes
    np.save(args.output, recon_genotypes)
    print('Saved reconstructed genotypes')


if __name__ == "__main__":
    main()