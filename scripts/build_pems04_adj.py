"""Build PEMS04 adjacency pickle from distance.csv (ASTGCN canonical format).

Produces data/pems04/adj_mx.pkl as a plain (N,N) float32 numpy array suitable for
stgnn_backbone_bank.load_adjacency() which now accepts both 3-tuple and array pickles.

Uses the standard ASTGCN/DCRNN thresholded Gaussian kernel:
    W[i,j] = exp(-d_ij^2 / sigma^2) if exp(...) >= epsilon else 0
with sigma = std of non-self distances and epsilon = 0.1.
"""
import numpy as np
import pandas as pd
import pickle

N = 307
df = pd.read_csv("data/pems04/distance.csv")
dist_mx = np.full((N, N), np.inf, dtype=np.float32)
np.fill_diagonal(dist_mx, 0.0)

edges = 0
for _, row in df.iterrows():
    i, j, c = int(row["from"]), int(row["to"]), float(row["cost"])
    dist_mx[i, j] = c
    edges += 1

finite = dist_mx[~np.isinf(dist_mx)]
finite = finite[finite > 0]
sigma = finite.std()
adj = np.exp(-(dist_mx ** 2) / (sigma ** 2))
adj[adj < 0.1] = 0.0
adj = adj.astype(np.float32)

nnz = (adj > 0).sum()
print(f"PEMS04 adjacency: N={N}, edges={edges}, nnz={nnz}, sigma={sigma:.2f}")
print(f"adj[0,:5] = {adj[0,:5]}")

with open("data/pems04/adj_mx.pkl", "wb") as f:
    pickle.dump(adj, f)
print("Saved data/pems04/adj_mx.pkl")
