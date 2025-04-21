import argparse
import numpy as np
import pandas as pd
import tskit

def main(args):
    # Load tree sequence and merged_sorted CSV
    ts = tskit.load(args.ts_file)
    merged_sorted = pd.read_csv(args.sorted_individuals_csv)

    num_inds = ts.num_individuals
    num_sites = ts.num_sites

    ##############################################
    # Build diploid genotype matrix (sorted)
    ##############################################

    # 1. Map each node to individual ID
    node_to_ind_id = np.full(ts.num_nodes, -1, dtype=int)
    for ind in ts.individuals():
        for node in ind.nodes:
            node_to_ind_id[node] = ind.id

    # 2. Map individual ID to row index in G
    ind_id_to_row = {
        ind_id: i for i, ind_id in enumerate(merged_sorted["individual_id"])
    }

    # 3. Map node to row index
    node_to_row = np.full(ts.num_nodes, -1, dtype=int)
    for node_id in range(ts.num_nodes):
        ind_id = node_to_ind_id[node_id]
        if ind_id != -1:
            node_to_row[node_id] = ind_id_to_row[ind_id]

    # 4. Preallocate genotype matrix
    G = np.zeros((num_inds, num_sites), dtype=np.int8)

    # 5. Accumulate diploid genotypes
    for site_idx, var in enumerate(ts.variants()):
        for node_idx, genotype in enumerate(var.genotypes):
            row_idx = node_to_row[node_idx]
            if row_idx != -1:
                G[row_idx, site_idx] += genotype

    print("Genotype matrix shape:", G.shape)
    np.save(args.output_npy, G)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build diploid genotype matrix from tree sequence")
    parser.add_argument("--ts_file", type=str, required=True, help="Input .ts file from simulation")
    parser.add_argument("--sorted_individuals_csv", type=str, required=True, help="CSV file with sorted individual_id column")
    parser.add_argument("--output_npy", type=str, required=True, help="Output .npy file to store diploid genotype matrix")
    args = parser.parse_args()
    main(args)
