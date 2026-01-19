import os
import h5py
from benchmark.results import get_result_filename

# Recursively find all hdf5 files
locked_files = []
readable_files = []
corrupt_files = []

for root, _, files in os.walk('results'):
    for fn in files:
        if fn.endswith('.hdf5'):
            path = os.path.join(root, fn)
            try:
                # Try read-write first
                with h5py.File(path, 'r+') as f:
                    pass
            except OSError:
                locked_files.append(path)
                # Try read-only
                try:
                    with h5py.File(path, 'r') as f:
                        readable_files.append(path)
                except OSError:
                    corrupt_files.append(path)

print(f"Total HDF5: {len(locked_files) + len(readable_files) + len(corrupt_files)}") # logic slightly off if it succeeds r+, it is not in any list. readjusting.

print("---")
print("Files failing r+ (Write mode):")
for f in locked_files:
    print(f)
    
print("\nCan they be opened in 'r' (Read-Only) mode?")
intersection = set(locked_files).intersection(set(readable_files))
for f in intersection:
    print(f"YES: {f}")
    
print("\nFiles failing 'r' (Completely Unusable/Corrupt):")
for f in corrupt_files:
    print(f)
