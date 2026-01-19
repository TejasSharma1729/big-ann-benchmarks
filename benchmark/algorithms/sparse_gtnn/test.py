#!/usr/bin/env python3
from sparse_gtnn import KNNIndexDataset, read_sparse_matrix
import time
import statistics

print("Loading Data...")
X, dim = read_sparse_matrix("../../../data/sparse/base_1M.csr")
Q, dim = read_sparse_matrix("../../../data/sparse/queries.dev.csr")

print(f"Dataset Size: {len(X)}")

print("Building Baseline Index...")
base_index = KNNIndexDataset(X, 10, True)

queries = Q[:100]

def benchmark(name, func, qs):
    times = []
    recalls = []
    # Warmup
    results, _ = func(qs[0])
    
    for q in qs:
        start = time.perf_counter()
        results, _ = func(q)
        end = time.perf_counter()
        times.append((end - start) * 1000)

        _, _, recall = base_index.verify_results(q, results)
        recalls.append(recall)
    
    avg = statistics.mean(times)
    recall = statistics.mean(recalls)
    print(f"{name}: Avg {avg:0.4f} ms, Recall {recall:0.4f}")

print("\n--- Single Threaded Search Comparison ---")
benchmark("Current Implementation", base_index.search, queries)

print("\n--- Parallelized Search Comparison ---")
benchmark("Parallel Implementation", base_index.search_parallel, queries)

print("\n--- Parallel Across Searches Comparison ---")
start = time.perf_counter()
res = base_index.search_parallel(queries)
end = time.perf_counter()
avg_parallel = (end - start) * 1000 / len(queries)
print(f"Parallel Implementation: Avg {avg_parallel:0.4f} ms")