#!/usr/bin/env python3
try:
    from ..sparse_gtnn import KNNIndexDataset, read_sparse_matrix
except:
    from sparse_gtnn import KNNIndexDataset, read_sparse_matrix
import time
import statistics

print("Loading Data...")
X, dim = read_sparse_matrix("../../../data/sparse/base_1M.csr")
Q, dim = read_sparse_matrix("../../../data/sparse/queries.dev.csr")

print(f"Dataset Size: {len(X)}")

print("Building Baseline Index...")
base_index = KNNIndexDataset(X, 10, True)

# Reduced query set for quick debugging
queries = Q[:1000]

print("\n--- Double Group Testing ---")
# Verify search_double_group signature and output
start = time.perf_counter()
try:
    # search_double_group returns (results, num_dots)
    res_pair = base_index.search_double_group(queries)
    if isinstance(res_pair, tuple):
        res = res_pair[0]
        num_dots = res_pair[1]
    else:
        print("Unexpected return type")
        res = res_pair
        num_dots = 0
except Exception as e:
    import sys
    print(f"Search failed: {e}")
    sys.exit(1)

end = time.perf_counter()
avg_parallel = (end - start) * 1000 / len(queries)
print(f"Double-Group Testing (Threaded): Avg {avg_parallel:0.4f} ms")

print(f"Total Dot Products: {num_dots}")
if len(queries) > 0:
    print(f"Average Dot Products per Query: {num_dots / len(queries)}")

# verification
try:
    recalls = []
    print("Verifying results (top 5)...")
    for i in range(len(queries)):
        # verify_results returns [time, recall, precision]
        metrics = base_index.verify_results(queries[i], res[i])
        recalls.append(metrics[1])
        if i < 5:
            print(f"Query {i} Recall: {metrics[1]}")
    
    mean_recall = statistics.mean(recalls)
    print(f"Mean Recall: {mean_recall:0.4f}")
except Exception as e:
    print(f"Verification failed: {e}")
