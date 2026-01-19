
import os
import numpy as np
from scipy.sparse import csr_matrix

def read_sparse_matrix(fname):
    # Simplified reader based on the context
    # Assuming standard CSR format logic if not using the repo's loader
    # But better to use the repo's loader if possible. 
    # Let's try to load using the library function if I can find it, 
    # otherwise, assume standard scipy load if it's a .npz, but the file is .csr.
    # The repo uses a custom `read_sparse_matrix`.
    pass

# I'll rely on the codebase's read_sparse_matrix
import sys
sys.path.append('/home/tejassharma/big-ann-benchmarks')
from benchmark.dataset_io import read_sparse_matrix

try:
    path = "data/sparse/base_small.csr"
    if not os.path.exists(path):
         # Try the one mentioned in the user's test.py if small doesn't exist
         path = "data/sparse/base_1M.csr"
    
    if os.path.exists(path):
        print(f"Checking {path}")
        X, dim = read_sparse_matrix(path)
        
        # Check norms of first 100 vectors
        norms = []
        for i in range(100):
            row = X.getrow(i)
            norm = np.linalg.norm(row.data)
            norms.append(norm)
        
        norms = np.array(norms)
        print(f"Mean Norm: {np.mean(norms)}")
        print(f"Min Norm: {np.min(norms)}")
        print(f"Max Norm: {np.max(norms)}")
        print(f"Are they normalized? {np.allclose(norms, 1.0, atol=1e-3)}")
    else:
        print("No sparse data file found to check.")

except Exception as e:
    print(e)
