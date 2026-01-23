#!/usr/bin/env python3
import os
import sys
import requests
import zipfile
import tarfile
import bz2
import gzip
import shutil
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from benchmark.dataset_io import write_sparse_matrix
import argparse
from sklearn.datasets import load_svmlight_file
from scipy import sparse

# Configuration
DATA_DIR = "data"
MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
KDDB_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kddb.bz2"
AVAZU_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2"

def save_custom_csr(X: sparse.csr_matrix, file_name: str):
    num_vectors = np.uint64(X.shape[0])
    dim = np.uint64(X.shape[1])
    data_f32 = X.data.astype(np.float32)
    indices_u32 = X.indices.astype(np.uint32)
    indptr_u64 = X.indptr.astype(np.uint64)
    num_nonzero = np.uint64(data_f32.size)
    with open(file_name, 'wb') as f:
        f.write(num_vectors.tobytes())
        f.write(dim.tobytes())
        f.write(num_nonzero.tobytes())
        f.write(indptr_u64.tobytes())
        f.write(indices_u32.tobytes())
        f.write(data_f32.tobytes())

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"File {dest} already exists. Skipping download.")
        return
    print(f"Downloading {url} to {dest}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        sys.exit(1)

def install_movielens(base_dir):
    print("=== Installing Movielens ===")
    ml_dir = os.path.join(base_dir, "movielens")
    os.makedirs(ml_dir, exist_ok=True)
    
    zip_path = os.path.join(ml_dir, "ml-20m.zip")
    download_file(MOVIELENS_URL, zip_path)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(ml_dir)
        
    csv_path = os.path.join(ml_dir, "ml-20m/ratings.csv")
    
    print("Processing ratings.csv...")
    rows = []
    cols = []
    
    # Use a safe mapping for movieIds as they are not contiguous
    movie_id_map = {}
    next_movie_id = 0
    
    # Read file
    with open(csv_path, 'r') as f:
        header = next(f) # userId,movieId,rating,timestamp
        for line in f:
            parts = line.strip().split(',')
            uid = int(parts[0]) - 1
            if uid >= 1000000:
                continue

            raw_mid = int(parts[1])
            
            if raw_mid not in movie_id_map:
                movie_id_map[raw_mid] = next_movie_id
                next_movie_id += 1
            
            mid = movie_id_map[raw_mid]
            
            rows.append(uid)
            cols.append(mid)
            
    print(f"Constructing CSR matrix (Users: {max(rows)+1}, Items: {next_movie_id})...")
    data = np.ones(len(rows), dtype=np.float32)
    # users are rows, movies are columns
    X = csr_matrix((data, (rows, cols)), shape=(max(rows)+1, next_movie_id), dtype=np.float32)
    
    # Create random queries
    print("Selecting quries...")
    np.random.seed(42) # Reproducibility
    n_queries = 1000
    indices = np.random.choice(X.shape[0], n_queries, replace=False)
    indices.sort()
    Q = X[indices]
    
    print("Writing files...")
    write_sparse_matrix(X, os.path.join(ml_dir, "X.csr"))
    write_sparse_matrix(Q, os.path.join(ml_dir, "Q.csr"))
    print("Movielens Done.\n")

def install_kddb(base_dir):
    print("=== Installing KDDB (KDD12) ===")
    kddb_dir = os.path.join(base_dir, "kddb")
    os.makedirs(kddb_dir, exist_ok=True)
    
    bz2_path = os.path.join(kddb_dir, "kddb.bz2")
    
    if not os.path.exists(bz2_path):
        print(f"Downloading {KDDB_URL}...")
        os.system(f"wget {KDDB_URL} -O {bz2_path}")
    
    raw_path = os.path.join(kddb_dir, "kddb")
    if not os.path.exists(raw_path):
        print(f"Decompressing {bz2_path}...")
        os.system(f"bunzip2 -k {bz2_path}") 

    print(f"Loading {raw_path}...")
    data = load_svmlight_file(raw_path)
    X_all = data[0]
    
    print("Splitting data...")
    # First 10000 rows into Q, rest into X
    Q = X_all[:10000]
    X = X_all[10000:]
    
    print("Writing files...")
    save_custom_csr(X, os.path.join(kddb_dir, "X.csr"))
    save_custom_csr(Q, os.path.join(kddb_dir, "Q.csr"))
    print("KDDB Done.\n")

def install_avazu(base_dir):
    print("=== Installing Avazu ===")
    av_dir = os.path.join(base_dir, "avazu")
    os.makedirs(av_dir, exist_ok=True)
    
    bz2_path = os.path.join(av_dir, "avazu-app.bz2")
    if not os.path.exists(bz2_path):
        print(f"Downloading {AVAZU_URL}...")
        os.system(f"wget {AVAZU_URL} -O {bz2_path}")
        
    raw_path = os.path.join(av_dir, "avazu-app")
    if not os.path.exists(raw_path):
        print(f"Decompressing {bz2_path}...")
        os.system(f"bunzip2 -k {bz2_path}")
        
    print(f"Loading {raw_path}...")
    data = load_svmlight_file(raw_path)
    X_all = data[0]
    
    print("Splitting data...")
    # First 10000 rows into Q, rest into X
    Q = X_all[:10000]
    X = X_all[10000:10010000]
    
    print("Writing files...")
    save_custom_csr(X, os.path.join(av_dir, "X.csr"))
    save_custom_csr(Q, os.path.join(av_dir, "Q.csr"))
    print("Avazu Done.\n")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--movielens", action="store_true")
    parser.add_argument("--kddb", action="store_true")
    parser.add_argument("--avazu", action="store_true")
    parser.add_argument("--all", action="store_true")
    
    args = parser.parse_args()
    
    if args.all or args.movielens:
        install_movielens(DATA_DIR)
    if args.all or args.kddb:
        install_kddb(DATA_DIR)
    if args.all or args.avazu:
        install_avazu(DATA_DIR)
        
    if not (args.all or args.movielens or args.kddb or args.avazu):
        print("Please specify --movielens, --kddb, --avazu, or --all")
