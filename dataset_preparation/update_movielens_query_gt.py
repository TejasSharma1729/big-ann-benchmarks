import numpy as np
import os
import subprocess
from scipy.sparse import csr_matrix
from benchmark.dataset_io import read_sparse_matrix, write_sparse_matrix

DATA_DIR = "data/movielens"
X_PATH = os.path.join(DATA_DIR, "X.csr")
Q_PATH = os.path.join(DATA_DIR, "Q.csr")
GT_PATH = os.path.join(DATA_DIR, "groundtruth.gt")
N_QUERIES = 10000
K = 10

def main():
    if not os.path.exists(X_PATH):
        print(f"Error: {X_PATH} not found.")
        return

    print(f"Reading {X_PATH}...")
    X = read_sparse_matrix(X_PATH)
    print(f"Loaded X: {X.shape}")

    print(f"Selecting {N_QUERIES} queries...")
    np.random.seed(42)
    if X.shape[0] < N_QUERIES:
        print(f"Warning: Dataset smaller than requested queries. Taking all {X.shape[0]}.")
        indices = np.arange(X.shape[0])
    else:
        indices = np.random.choice(X.shape[0], N_QUERIES, replace=False)
    indices.sort()
    
    Q = X[indices]
    print(f"Q shape: {Q.shape}")
    
    print(f"Writing {Q_PATH}...")
    write_sparse_matrix(Q, Q_PATH)
    
    print("Computing Ground Truth...")
    cmd = [
        "python3", "dataset_preparation/make_sparse_groundtruth.py",
        "--base_csr_file", X_PATH,
        "--query_csr_file", Q_PATH,
        "--output_file", GT_PATH,
        "--k", str(K),
        "--nt", "8"
    ]
    subprocess.run(cmd, check=True)
    print("Done.")

if __name__ == "__main__":
    main()
