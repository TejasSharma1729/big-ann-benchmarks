#!/usr/bin/env python3
"""
Generate groundtruth files for kddb, movielens, and avazu datasets.

The groundtruth file format is:
- 2 x uint32: (n_queries, k)
- n_queries * k x int32: neighbor indices
- n_queries * k x float32: distances

This uses brute-force search to compute exact k-NN results.
"""
import os
import sys
import numpy as np
from scipy.sparse import csr_matrix
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark.datasets import DATASETS

def compute_sparse_groundtruth(dataset_name, k=100):
    """Compute exact k-NN groundtruth for a sparse dataset using brute force.
    Args:
        dataset_name (str): Name of the dataset (key in DATASETS).
        k (int): Number of nearest neighbors to store. Standard is 100.
    """
    print(f"Computing groundtruth for {dataset_name} with k={k}")
    
    try:
        ds = DATASETS[dataset_name]()
    except KeyError:
        print(f"Error: Dataset '{dataset_name}' not found in DATASETS.")
        return None, None, 0
    
    # Load queries
    print("Loading queries...")
    queries = ds.get_queries()
    nq = queries.shape[0]
    print(f"  Loaded {nq} queries")
    
    # Result arrays
    I = np.zeros((nq, k), dtype='int32')  # Indices
    D = np.zeros((nq, k), dtype='float32')  # Distances (inner products)
    
    # Initialize with very negative values (since we want max inner product)
    D[:] = -np.inf
    I[:] = -1
    
    bs = 100000  # Process database in chunks
    iterator = ds.get_dataset_iterator(bs=bs)
    
    base_idx = 0
    t0 = time.time()
    
    # Batch size for queries to avoid OOM with large dense matrices
    query_bs = 1000
    
    try:
        for chunk_num, X_chunk in enumerate(iterator):
            if not isinstance(X_chunk, csr_matrix):
                # Fallback if iterator returns something else, though LocalSparseDataset usually returns csr
                pass

            chunk_size = X_chunk.shape[0]
            
            # Process queries in batches
            for q_start in range(0, nq, query_bs):
                q_end = min(q_start + query_bs, nq)
                Q_batch = queries[q_start:q_end]
                
                # Compute inner products: queries @ X_chunk.T
                # For sparse matrices, this returns a sparse matrix
                # Converting to dense is efficient enough for small query batches
                scores = Q_batch.dot(X_chunk.T)  # (q_batch, chunk_size)
                
                if hasattr(scores, 'toarray'):
                    scores = scores.toarray()
                
                # For each query in this batch
                for i in range(q_end - q_start):
                    qi = q_start + i
                    
                    # Combine current top-k with new scores
                    current_scores = D[qi]
                    current_indices = I[qi]
                    
                    new_scores = scores[i]
                    # Indices in the global dataset
                    new_indices = np.arange(base_idx, base_idx + chunk_size, dtype='int32')
                    
                    # We only need to keep top-k from (current + new)
                    # Optimization: Filter roughly before full sort? 
                    # For robust exactness, let's just merge and sort.
                    
                    all_scores = np.concatenate([current_scores, new_scores])
                    all_indices = np.concatenate([current_indices, new_indices])
                    
                    # Get top-k largest elements
                    # argpartition is faster than full sort
                    if len(all_scores) > k:
                        top_k_pos = np.argpartition(-all_scores, k)[:k]
                        # Sort the top k explicitly
                        top_k_pos = top_k_pos[np.argsort(-all_scores[top_k_pos])]
                        
                        D[qi] = all_scores[top_k_pos]
                        I[qi] = all_indices[top_k_pos]
                    else:
                        # Should not happen if dataset > k
                        pass

            base_idx += chunk_size
            elapsed = time.time() - t0
            rate = base_idx / elapsed if elapsed > 0 else 0
            print(f"  Processed {base_idx:,} vectors ({rate:.0f} vecs/s)", end='\r', flush=True)
                
    except (StopIteration, IndexError):
        pass
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print(f"Total time: {time.time() - t0:.2f}s")
    print(f"Processed {base_idx:,} vectors")
    
    return I, D, nq

def write_groundtruth_file(I, D, filename):
    """Write groundtruth in the expected binary format."""
    n, k = I.shape
    print(f"Writing groundtruth to {filename}")
    print(f"  n={n} queries, k={k}")
    
    # Ensure correct dtypes
    I = np.ascontiguousarray(I, dtype='int32')
    D = np.ascontiguousarray(D, dtype='float32')
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'wb') as f:
        # Header: n_queries, k
        np.array([n, k], dtype='uint32').tofile(f)
        # Indices
        I.tofile(f)
        # Distances
        D.tofile(f)
    
    expected_size = 8 + n * k * (4 + 4)
    actual_size = os.path.getsize(filename)
    print(f"  Expected size: {expected_size:,} bytes")
    print(f"  Actual size: {actual_size:,} bytes")
    
    if expected_size == actual_size:
        print(f"  ✓ Groundtruth file valid.")
    else:
        print(f"  X Size mismatch!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate groundtruth files for sparse datasets')
    parser.add_argument('--dataset', type=str, nargs='+', 
                        default=['kddb', 'movielens', 'avazu'],
                        help='Dataset name(s) to process')
    parser.add_argument('--k', type=int, default=100, 
                        help='Number of nearest neighbors (default: 100)')
    args = parser.parse_args()
    
    for ds_name in args.dataset:
        print(f"\n[{ds_name.upper()}]")
        I, D, nq = compute_sparse_groundtruth(ds_name, k=args.k)
        
        if I is not None:
            # Determine output path relative to repo root
            # Assuming standard structure data/{name}/groundtruth.gt
            output_file = f'data/{ds_name}/groundtruth.gt'
            write_groundtruth_file(I, D, output_file)
