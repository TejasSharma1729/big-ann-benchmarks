import argparse
import os
import shutil
import numpy as np
import scipy.sparse
from benchmark.dataset_io import write_sparse_matrix

def write_gt(ids, dists, fname):
    """
    Write ground truth in the benchmark format.
    Format:
    [n (uint32)] [d (uint32)]
    [ids (n*d int32)]
    [dists (n*d float32)]
    """
    n, d = ids.shape
    with open(fname, "wb") as f:
        np.array([n, d], dtype='uint32').tofile(f)
        ids.astype('int32').tofile(f)
        dists.astype('float32').tofile(f)

def compute_groundtruth(dataset, queries, k=100, batch_size=100):
    print(f"Computing ground truth for {queries.shape[0]} queries against {dataset.shape[0]} base vectors...")
    n_queries = queries.shape[0]
    
    # Pre-allocate results
    gt_I = np.zeros((n_queries, k), dtype='int32')
    gt_D = np.zeros((n_queries, k), dtype='float32')
    
    # Process in batches to save memory
    for i in range(0, n_queries, batch_size):
        end = min(i + batch_size, n_queries)
        batch_queries = queries[i:end]
        
        # Inner product: Q * D^T
        # Note: sparse matrix multiplication returns sparse or dense depending on density. 
        # Usually dense result is manageable for batch_size * nb
        
        # Assuming format is CSR.
        # dataset: (nb, dim)
        # queries: (nq, dim)
        # dot: (nq_batch, nb)
        
        scores = batch_queries.dot(dataset.T)
        
        # If scores is sparse, convert to dense
        if scipy.sparse.issparse(scores):
            scores = scores.toarray()
            
        # Find top k
        # We want MAX inner product.
        # argpartition is faster than sort
        # But we need sorted top k
        
        # Invert scores for min-heap style operations if using distance, but this is IP (maximization)
        # -scores so smallest is best for generic logic? No, let's jus use argsort or argpartition
        
        # argsort is easiest
        # We want indices of largest elements.
        
        # full sort is slow. argpartition for top k
        ind = np.argpartition(scores, -k, axis=1)[:, -k:]
        
        # Takes top k, but they are not sorted.
        # We need to sort them by score descending.
        
        # Extract the values
        row_indices = np.arange(scores.shape[0])[:, None]
        top_k_scores = scores[row_indices, ind]
        
        # Sort these top k (descending)
        sort_order = np.argsort(top_k_scores, axis=1)[:, ::-1]
        
        sorted_ind = ind[row_indices, sort_order]
        sorted_scores = top_k_scores[row_indices, sort_order]
        
        gt_I[i:end] = sorted_ind
        gt_D[i:end] = sorted_scores
        
        if i % 1000 == 0:
            print(f"Processed {i}/{n_queries} queries...")

    return gt_I, gt_D

def main():
    parser = argparse.ArgumentParser(description="Convert standard sparse matrices to benchmark format and optionally compute GT.")
    parser.add_argument("--dataset", required=True, help="Path to dataset file (scipy.sparse.load_npz compatible)")
    parser.add_argument("--queries", required=True, help="Path to queries file (scipy.sparse.load_npz compatible)")
    parser.add_argument("--groundtruth", help="Path to groundtruth file (numpy format or similar), if available.")
    parser.add_argument("--compute-gt", action="store_true", help="Compute ground truth using Inner Product brute force.")
    parser.add_argument("--gt-k", type=int, default=100, help="Number of neighbors for GT (default 100)")
    parser.add_argument("--outdir", default="data/custom-sparse", help="Output directory")

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        print(f"Creating directory: {args.outdir}")
        os.makedirs(args.outdir)

    print(f"Loading dataset from {args.dataset}...")
    dataset = scipy.sparse.load_npz(args.dataset)
    print(f"Loaded dataset shape: {dataset.shape}")
    
    out_ds = os.path.join(args.outdir, "dataset.csr")
    print(f"Writing dataset to {out_ds}...")
    write_sparse_matrix(dataset, out_ds)

    print(f"Loading queries from {args.queries}...")
    queries = scipy.sparse.load_npz(args.queries)
    print(f"Loaded queries shape: {queries.shape}")

    out_qs = os.path.join(args.outdir, "queries.csr")
    print(f"Writing queries to {out_qs}...")
    write_sparse_matrix(queries, out_qs)

    gt_I = None
    gt_D = None

    if args.groundtruth:
        print(f"Loading groundtruth from {args.groundtruth}...")
        # Try loading as npy
        try:
             # Assume it contains indices? Or a tuple (I, D)?
             # Let's assume the user provides I (indices).
             # If they have D, great.
             
             # If .npy
             data = np.load(args.groundtruth)
             if isinstance(data, dict) or hasattr(data, "files"): # .npz
                 if 'I' in data: gt_I = data['I']
                 if 'D' in data: gt_D = data['D']
             else:
                 # Assume it's just indices if one array
                 if np.issubdtype(data.dtype, np.integer):
                     gt_I = data
                 else:
                     print("Warning: Loaded GT is not integer indices. Skipping.")
        except Exception as e:
            print(f"Failed to load GT: {e}")

    elif args.compute_gt:
        gt_I, gt_D = compute_groundtruth(dataset, queries, k=args.gt_k)
    
    if gt_I is not None:
        if gt_D is None:
            # If we only have indices, we should maybe compute distances or fill with zeros?
            # Benchmark might need distances for range search, but usually KNN mostly needs indices for recall.
            # However `knn_result_read` reads D.
            print("Warning: No distances provided/computed for GT. Filling with zeros.")
            gt_D = np.zeros_like(gt_I, dtype='float32')

        out_gt = os.path.join(args.outdir, "groundtruth.gt")
        print(f"Writing groundtruth to {out_gt}...")
        write_gt(gt_I, gt_D, out_gt)
    else:
        print("No groundtruth generated. Running benchmark might fail if it expects GT for recall computation.")

if __name__ == "__main__":
    main()
