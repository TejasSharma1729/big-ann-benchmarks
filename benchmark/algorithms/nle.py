from __future__ import absolute_import
import time
import numpy as np
from benchmark.algorithms.base import BaseANN
from pylinscancufe import LinscanIndex
from benchmark.datasets import DATASETS
from scipy.sparse import csr_matrix

class NLE(BaseANN):
    def __init__(self, metric, index_params):
        self._metric = metric
        self._index_params = index_params
        self.name = "NLE"
        self.index = LinscanIndex()
        # Parse params if any. LinscanIndex() constructor in the example didn't take args,
        # but let's assume we might pass some in future or just ignore them.
        print(f"Initialized NLE with {index_params}")

    def fit(self, dataset):
        """
        Build the index for the data points given in dataset name.
        """
        ds = DATASETS[dataset]()
        print(f"Building index for {dataset}...")
        
        # Iterate over dataset blocks
        iterator = ds.get_dataset_iterator(bs=100000) # Process in chunks
        
        count = 0
        t0 = time.time()
        
        try:
            for block in iterator:
                # Block is likely a csr_matrix or similar given it's a sparse dataset
                # We need to iterate rows and insert them.
                if isinstance(block, csr_matrix):
                    for i in range(block.shape[0]):
                        row = block.getrow(i)
                        # Create dict {idx: val}
                        # Cast to int because pylinscancufe expects u32 values
                        # Scale by 1000 to preserve some precision from float32
                        data_scaled = (row.data * 1000).astype(int)
                        # Ensure non-negative? Data seems positive.
                        # data_scaled = np.maximum(data_scaled, 0)
                        vec_dict = dict(zip(row.indices.tolist(), data_scaled.tolist()))
                        self.index.insert(vec_dict)
                        count += 1
                        if count % 10000 == 0:
                            print(f"Inserted {count} vectors...", end='\r')
                else:
                    # Fallback or assumption if it's not csr_matrix directly (e.g. numpy array of opaque objects?)
                    # But usually it's a matrix.
                    pass
        except IndexError:
             # StopIteration is handled by the loop, but sometimes custom iterators raise IndexError
             pass
        except Exception as e:
            print(f"Error during fitting: {e}")
            raise e

        print(f"\nFinished inserting {count} vectors in {time.time()-t0:.2f}s")
        # Optimization step if required (pylinscancufe doesn't seem to have explicit build/optimize call in the example, 
        # but check if there's one. The example just did insert then retrieve).
        # print(self.index) # Printing index usually shows status

    def set_query_arguments(self, query_args):
        self._query_args = query_args
        # If budget is in query_args, update it. 
        # Otherwise use what was in index_params or default
        if "budget" in query_args:
            self._index_params["budget"] = query_args["budget"]

    def load_index(self, dataset):
        # We don't have load/save mechanism yet, so return False to force rebuild (fit)
        return False

    def query(self, X, k):
        """Carry out a batch query for k-NN of query set X."""
        # X is the queries matrix. For sparse dataset, it should be a sparse matrix (csr_matrix).
        
        q_vecs = []
        if isinstance(X, csr_matrix):
            for i in range(X.shape[0]):
                qc = X.getrow(i)
                data_scaled = (qc.data * 1000).astype(int)
                q = dict(zip(qc.indices.tolist(), data_scaled.tolist()))
                q_vecs.append(q)
        else:
            # Maybe it's a dense numpy array if the dataset wrapper converted it? 
            # But SparseDataset likely returns sparse queries.
            # If it is dense, we might need to densify? No, sparse index expects sparse.
            raise ValueError(f"Expected csr_matrix for queries, got {type(X)}")

        budget = self._index_params.get("budget", 1000) # Default budget
        
        # retrieve_parallel(queries, k, budget) 
        # based on test.py: res_vec = index.retrieve_parallel(q_vec, 10, 10)
        # It seems the arguments are (queries, k, budget/ef?)
        
        print(f"Querying with k={k}, budget={budget}")
        start = time.time()
        # The return type of retrieve_parallel needs to be checked. 
        # The example implies it returns one list/vector per query?
        # "res_vec = index.retrieve_parallel(...)"
        # We need to return (nq, k) array of IDs.
        
        results = self.index.retrieve_parallel(q_vecs, k, budget)
        self.res = np.array(results, dtype=np.int32)
        
        # Verify shape
        # retrieve_parallel likely returns a list of lists of IDs.
        # We need to ensure it's (nq, k). 
        # If it returns distances too, we need to separate them.
        # Let's assume for now it returns just IDs as per standard "retrieve" naming, 
        # but wait - usually benchmark expects IDs.
        
        total_time = time.time() - start
        print(f"Query done in {total_time:.2f}s")

    def range_query(self, X, radius):
        raise NotImplementedError("Range query not supported")
