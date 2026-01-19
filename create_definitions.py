import yaml
import json

datasets = ['sparse-1M', 'sparse-full', 'kddb', 'movielens', 'avazu']
definitions = {}

# Common configurations

# Linscan (Rust)
linscan_config = {
    'docker-tag': None,
    'module': 'neurips23.sparse.linscan.linscan',
    'constructor': 'Linscan',
    'base-args': ['@metric'],
    'run-groups': {
        'base': {
            'args': [{}],
            'query-args': json.dumps([
                [{'budget': b}] for b in [50, 4000]
            ])
        }
    }
}

# Cufe Linscan (CUDA/Accelerated)
cufe_config = {
    'docker-tag': None,
    'module': 'neurips23.sparse.cufe.linscan',
    'constructor': 'LinscanCUFE',
    'base-args': ['@metric'],
    'run-groups': {
        'base': {
            'args': [{}],
            'query-args': json.dumps([
                [{'budget': b}] for b in [50, 4000]
            ])
        }
    }
}

# NLE config (Legacy/Wrapper) - DISABLED per request
# If we wanted to run the neurips23 version of NLE:
# module: neurips23.sparse.nle.nle
# constructor: NLE

# SHNSW
shnsw_config = {
    'docker-tag': None,
    'module': 'neurips23.sparse.shnsw.shnsw',
    'constructor': 'SparseHNSW',
    'base-args': ['@metric'],
    'run-groups': {
        'base': {
            'args': [{'efConstruction': 200, 'M': 32, 'buildthreads': -1}],
            'query-args': json.dumps([
                [{'efSearch': ef}] for ef in [40, 500]
            ])
        }
    }
}

# KNNS Double Group Testing
knns_double_config = {
    'docker-tag': None,
    'module': 'benchmark.algorithms.knns',
    'constructor': 'KNNSDoubleGroupTesting',
    'base-args': ['@metric'],
    'run-groups': {
        'base': {
            'args': [{'k': '@count', 'use_threading': True}],
            'query-args': json.dumps([ [{'mode': 'batch'}] ])
        }
    }
}

# KNNS Binary Splitting
knns_binary_config = {
    'docker-tag': None,
    'module': 'benchmark.algorithms.knns',
    'constructor': 'KNNSBinarySplitting',
    'base-args': ['@metric'],
    'run-groups': {
        'base': {
            'args': [{'k': '@count', 'use_threading': True}],
            'query-args': json.dumps([ [{'mode': 'batch'}] ])
        }
    }
}

for ds in datasets:
    definitions[ds] = {
        'linscan': linscan_config,
        'cufe': cufe_config,
        'shnsw': shnsw_config,
        'knns_double': knns_double_config,
        'knns_binary': knns_binary_config
    }

with open('algos-all-variants.yaml', 'w') as f:
    yaml.dump(definitions, f)

print("Generated algos-all-variants.yaml")
