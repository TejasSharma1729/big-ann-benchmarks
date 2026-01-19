from __future__ import absolute_import
import os
import time
import numpy as np
from numpy import ndarray, array, linalg
from scipy.sparse import csr_matrix, vstack
from typing import Literal, Optional, Union, Any, List, Tuple, Dict, Set, Callable, Iterable

from benchmark.algorithms.base import BaseANN
from benchmark.datasets import DATASETS

import sparse_gtnn
from sparse_gtnn import SparseElem, KNNIndexDataset
from sparse_gtnn import read_sparse_matrix

def create_sparse_elem(idx, val):
    elem = SparseElem()
    elem.index = int(idx)
    elem.value = float(val)
    return elem

class KNNSBase(BaseANN):
    """Base class for KNNS variants"""
    def __init__(self, metric, index_params, grouping="double"):
        self._metric = metric
        self._index_params = index_params
        self.k = index_params.get("k", 10)
        self.use_threading = index_params.get("use_threading", False)
        self.grouping = grouping
        self.name = f"KNNS-{grouping}"
        print(f"Initialized {self.name} with k={self.k}, threading={self.use_threading}")
        self.index = None
        self.id_map = None

    def _convert_matrix(self, matrix):
        print(f"Converting matrix {matrix.shape} to SparseVec format...")
        t0 = time.time()
        res = []
        indptr = matrix.indptr
        indices = matrix.indices
        data = matrix.data
        
        for i in range(matrix.shape[0]):
            start = indptr[i]
            end = indptr[i+1]
            vec = [create_sparse_elem(idx, val) for idx, val in zip(indices[start:end], data[start:end])]
            res.append(vec)
            if (i + 1) % 100000 == 0:
                print(f"Converted {i+1} vectors", end='\r')
        print(f"\nConversion finished in {time.time()-t0:.2f}s")
        return res

    def fit(self, dataset):
        """
        Build the index for the data points given in dataset name.
        """
        ds = DATASETS[dataset]()
        print(f"Building index for {dataset}...")
        
        # Optimization: Try loading via C++ extension first
        if hasattr(ds, 'get_dataset_fn'):
            try:
                fn = ds.get_dataset_fn()
                if fn and os.path.exists(fn):
                    print(f"Attempting fast load from {fn}...")
                    try:
                        # read_sparse_matrix returns (matrix, dimension)
                        # This expects the specific binary format native to the competition
                        matrix, dim = read_sparse_matrix(fn)
                        print(f"Fast load successful: {len(matrix)} vectors")
                        
                        print("Building KNNIndexDataset (Fast)...")
                        t1 = time.time()
                        self.index = KNNIndexDataset(matrix, self.k, self.use_threading)
                        print(f"Index built in {time.time()-t1:.2f}s")
                        return
                    except Exception as e:
                        print(f"Fast load check failed: {e}. Falling back to standard loader.")
            except Exception as e:
                print(f"Could not determine dataset filename: {e}")

        # Load the entire dataset
        iterator = ds.get_dataset_iterator(bs=ds.nb) 
        
        full_matrix: Optional[csr_matrix] = None
        for chunk in iterator:
            if full_matrix is None:
                full_matrix = chunk
            else:
                full_matrix = vstack([full_matrix, chunk]) # type: ignore
        
        assert full_matrix is not None
        print(f"Dataset loaded: {full_matrix.shape}")
        
        # Convert to list of list of SparseElem
        self.data_points = self._convert_matrix(full_matrix)
        
        print("Building KNNIndexDataset...")
        t1 = time.time()
        self.index = KNNIndexDataset(self.data_points, self.k, self.use_threading)
        print(f"Index built in {time.time()-t1:.2f}s")

    def setup(self, dtype, max_pts, ndims):
        self.index = KNNIndexDataset([], self.k, self.use_threading)
        self.id_map = []
        print("Algorithm set up (empty index)")

    def insert(self, data, ids):
        # data is csr_matrix
        new_vecs = self._convert_matrix(data)
        self.index.update(new_vecs)
        self.id_map.extend(ids)
        
    def delete(self, ids):
        print(f"Warning: delete({len(ids)} ids) requested but not implemented in sparse_gtnn.")
        
    def replace(self, data, tags_to_replace):
        print(f"Warning: replace({len(tags_to_replace)} tags) requested but not implemented fully. Inserting new version.")
        self.insert(data, tags_to_replace)

    def set_query_arguments(self, query_args):
        self.query_args = query_args
        print(f"Setting query arguments: {query_args}")

    def load_index(self, dataset):
        return False

    def query(self, X, k):
        """Carry out a batch query for k-NN of query set X."""
        # Convert queries to SparseVec format
        # X is a csr_matrix
        assert self.index is not None, "Index not built. Call fit() first."
        query_vecs = self._convert_matrix(X)
            
        print("Searching...")
        t0 = time.time()
        
        if self.grouping == "single":
            print(f"Using Single-Group Testing (standard loop, threading={self.use_threading})...")
            results_list = []
            ops = 0
            for i, vec in enumerate(query_vecs):
                if self.use_threading:
                    res, op = self.index.search_parallel(vec)
                else:
                    res, op = self.index.search(vec)
                results_list.append(res)
                ops += op
                if (i+1) % 100 == 0:
                     print(f"Processed {i+1} queries", end='\r')
            print(f"\nTotal Dot Products (Single): {ops}")
        else:
            print("Using Double-Group Testing (search_double_group)...")
            # Use search_double_group for batch processing (Double-group-testing)
            results_list, ops = self.index.search_double_group(query_vecs)
            print(f"Total Dot Products (Double): {ops}")
            
        print(f"Search completed in {time.time()-t0:.2f}s")
        
        I = np.array(results_list, dtype=np.int32) 
        
        if self.id_map is not None and len(self.id_map) > 0:
             mapping = np.array(self.id_map, dtype=np.int32)
             self.res = mapping[I]
        else:
             self.res = I


class KNNSBinarySplitting(KNNSBase):
    """KNNS with binary splitting (single-group testing) - processes queries one by one"""
    def __init__(self, metric, index_params):
        super().__init__(metric, index_params, grouping="single")
        self.name = "binary-splitting"


class KNNSDoubleGroupTesting(KNNSBase):
    """KNNS with double-group testing - batch query processing with dual hierarchy"""
    def __init__(self, metric, index_params):
        super().__init__(metric, index_params, grouping="double")
        self.name = "double-group-testing"
