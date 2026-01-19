import argparse
import os
import numpy as np

import sys
[sys.path.append(i) for i in ['.', '..']]

from benchmark.datasets import DATASETS
from benchmark.streaming.load_runbook import load_runbook

def get_range_start_end(entry, tag_to_id):
    for i in range(entry['end'] - entry['start']):
        tag_to_id[i+entry['start']] = i+entry['start']
    return tag_to_id

def get_next_set(tag_to_id: np.ndarray, entry):
    match entry['operation']:
        case 'insert':
            for i in range(entry['end'] - entry['start']):
                tag_to_id[i+entry['start']] = i+entry['start']
            return tag_to_id
        case 'delete':
            # delete is by key 
            for i in range(entry['end'] - entry['start']):
                tag_to_id.pop(i + entry['start'])
            return tag_to_id
        case 'replace':
            # replace key with value
            for i in range(entry['tags_end'] - entry['tags_start']):
                tag_to_id[i + entry['tags_start']] = entry['ids_start'] + i
            return tag_to_id
        case 'search':
            return tag_to_id
        case _:       
            raise ValueError('Undefined entry in runbook')
        
def gt_dir(ds, runbook_path):
    runbook_filename = os.path.split(runbook_path)[1]
    return os.path.join(ds.basedir, str(ds.nb), runbook_filename)

def compute_gt_internal(ds, ids, tags, step, runbook_path):
    print(f"Computing GT internally for step {step} with {len(ids)} points...")
    data = ds.get_dataset()
    sub_data = data[ids]
    queries = ds.get_queries()
    
    n_queries = queries.shape[0]
    k = 100
    batch_size = 100
    
    gt_I = []
    gt_D = []
    
    # Check metric
    metric = ds.distance()
    
    for i in range(0, n_queries, batch_size):
        q_batch = queries[i:i+batch_size]
        if metric == 'ip':
            sims = sub_data.dot(q_batch.T)
            if hasattr(sims, "toarray"):
                sims = sims.toarray()
            # Dense dims: (N_subset, Batch)
            # Transpose to (Batch, N_subset) for easier row-wise topk
            sims = sims.T
        else: # Euclidean - not optimized here but fallback
             # Implement if needed, for sparse-1M it is IP
             raise NotImplementedError("Internal L2 GT not implemented for sparse yet")

        for j in range(sims.shape[0]):
            scores = sims[j]
            # Max score for IP
            best_k_idx = np.argpartition(scores, -k)[-k:]
            sorted_k_idx = best_k_idx[np.argsort(-scores[best_k_idx])]
            
            gt_I.append(tags[sorted_k_idx])
            gt_D.append(scores[sorted_k_idx])
            
    gt_I = np.array(gt_I, dtype=np.int32)
    gt_D = np.array(gt_D, dtype=np.float32)
    
    dir = gt_dir(ds, runbook_path)
    os.makedirs(dir, exist_ok=True)
    gt_file = os.path.join(dir, 'step' + str(step) + '.gt100')
    
    with open(gt_file, 'wb') as f:
        np.array([n_queries, k], dtype=np.int32).tofile(f)
        gt_I.tofile(f)
        gt_D.tofile(f)
    print(f"Written {gt_file}")

def output_gt(ds, tag_to_id, step, gt_cmdline, runbook_path):
    ids_list = []
    tags_list = []
    for tag, id in tag_to_id.items():
        ids_list.append(id)
        tags_list.append(tag)

    ids = np.array(ids_list, dtype = np.uint32)
    tags = np.array(tags_list, dtype = np.uint32)

    if gt_cmdline is None:
        compute_gt_internal(ds, ids, tags, step, runbook_path)
        return

    data = ds.get_data_in_range(0, ds.nb)
    data_slice = data[np.array(ids)]

    dir = gt_dir(ds, runbook_path)
    prefix = os.path.join(dir, 'step') + str(step) 
    os.makedirs(dir, exist_ok=True)

    tags_file = prefix + '.tags'
    data_file = prefix + '.data'
    gt_file = prefix + '.gt100'

    with open(tags_file, 'wb') as tf:
        one = 1
        tf.write(tags.size.to_bytes(4, byteorder='little'))
        tf.write(one.to_bytes(4, byteorder='little'))
        tags.tofile(tf)    
    with open(data_file, 'wb') as f:
        f.write(ids.size.to_bytes(4, byteorder='little')) #npts
        f.write(ds.d.to_bytes(4, byteorder='little'))
        data_slice.tofile(f)
    
    gt_cmdline += ' --base_file ' + data_file 
    gt_cmdline += ' --gt_file ' + gt_file
    gt_cmdline += ' --tags_file ' + tags_file
    print("Executing cmdline: ", gt_cmdline)
    os.system(gt_cmdline)
    print("Removing data file")
    rm_cmdline = "rm " + data_file
    os.system(rm_cmdline)
    

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        help=f'Dataset to benchmark on.',
        required=True)
    parser.add_argument(
        '--runbook_file',
        help='Runbook yaml file path'
    )
    parser.add_argument(
        '--private_query',
        action='store_true'
    )
    parser.add_argument(
        '--gt_cmdline_tool',
        required=False,
        default=None
    )
    parser.add_argument(
        '--download',
        action='store_true'
    )
    args = parser.parse_args()

    ds = DATASETS[args.dataset]()
    max_pts, runbook = load_runbook(args.dataset, ds.nb, args.runbook_file)
    query_file = ds.qs_fn if args.private_query else ds.qs_fn
    
    if args.gt_cmdline_tool:
        common_cmd = args.gt_cmdline_tool + ' --dist_fn ' 
        match ds.distance():
            case 'euclidean':
                common_cmd += 'l2'
            case 'ip':
                common_cmd += 'mips'
            case _:
                raise RuntimeError('Invalid metric')
        common_cmd += ' --data_type '
        match ds.dtype:
            case 'float32':
                common_cmd += 'float'
            case 'int8':
                common_cmd += 'int8'
            case 'uint8':
                commond_cmd += 'uint8'
            case _:
                raise RuntimeError('Invalid datatype')
        common_cmd += ' --K 100'
        common_cmd += ' --query_file ' + os.path.join(ds.basedir, query_file)
    else:
        common_cmd = None

    step = 1
    ids = np.empty(0, dtype=np.uint32)

    for entry in runbook:
        # the first step must be an insertion
        if step == 1:
            tag_to_id = get_range_start_end(entry, {})
        else:
            tag_to_id = get_next_set(tag_to_id, entry)
        if (entry['operation'] == 'search'):
            output_gt(ds, tag_to_id, step, common_cmd, args.runbook_file)
        step += 1

if __name__ == '__main__':
    main()