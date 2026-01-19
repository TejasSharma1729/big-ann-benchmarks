import numpy as np
import os
import sys

def truncate_csr(input_path, output_path, max_rows=1000000):
    print(f"Reading {input_path}...")
    with open(input_path, "rb") as f:
        sizes = np.fromfile(f, dtype='int64', count=3)
        n_rows, n_cols, n_nnz = sizes
        print(f"Original: {n_rows} rows, {n_cols} cols, {n_nnz} nnz")
        
        if n_rows <= max_rows:
            print(f"File has fewer rows ({n_rows}) than limit ({max_rows}). No action needed.")
            return

        # Read indptr
        indptr = np.fromfile(f, dtype='uint64', count=n_rows + 1)
        
        # Determine new nnz
        new_nnz = indptr[max_rows]
        
        # Read indices and data
        # We need to read all then slice, or just read needed amount? 
        # The file is sequential: header, indptr, indices, data.
        # But wait, python's fromfile reads from current position? Yes.
        
        # We have read indptr (n_rows + 1) items.
        # Now we are at start of indices.
        # We only need to read 'new_nnz' indices.
        indices = np.fromfile(f, dtype='uint32', count=new_nnz)
        
        # But we need to SKIP the rest of indices to get to data?
        # Yes.
        # Number of skipped indices = n_nnz - new_nnz.
        # Seek?
        current_pos = f.tell()
        remaining_indices = n_nnz - new_nnz
        f.seek(int(remaining_indices) * 4, 1) # 4 bytes per uint32
        
        # Now read data
        data = np.fromfile(f, dtype='float32', count=new_nnz)
        
    # Construct new indptr
    new_indptr = indptr[:max_rows+1]
    
    print(f"Truncated: {max_rows} rows, {n_cols} cols, {new_nnz} nnz")
    
    print(f"Writing to {output_path}...")
    with open(output_path, 'wb') as f:
        np.array([max_rows, n_cols, new_nnz], dtype='int64').tofile(f)
        new_indptr.tofile(f)
        indices.tofile(f)
        data.tofile(f)
    print("Done.")

if __name__ == "__main__":
    input_fn = "data/movielens/X.csr"
    if len(sys.argv) > 1:
        input_fn = sys.argv[1]
        
    if not os.path.exists(input_fn):
        print(f"Input file {input_fn} not found.")
        sys.exit(1)
        
    # We overwrite the file or create a temporary one and rename?
    # User said "modifies the X.csr".
    # I'll write to temp then rename.
    temp_fn = input_fn + ".tmp"
    truncate_csr(input_fn, temp_fn, 1000000)
    os.replace(temp_fn, input_fn)
