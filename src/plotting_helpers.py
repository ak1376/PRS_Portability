# src/plotting_helpers.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def plot_latent_pca(latent, populations, outfile):
    pca = PCA(n_components=2)
    Z = pca.fit_transform(latent)

    pops = np.array(populations)
    uniq = np.unique(pops)

    plt.figure(figsize=(6,5))
    for u in uniq:
        mask = pops == u
        plt.scatter(Z[mask,0], Z[mask,1], label=u, alpha=0.7, s=20)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Latent space PCA by population")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
