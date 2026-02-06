import yaml
import json
import copy

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
            # single index build config; multiple query argument variants
            'args': [{}],
            'query-args': json.dumps([
                [{'budget': 50}], [{'budget': 200}], [{'budget': 1000}], [{'budget': 4000}]
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
            # single index build config; multiple query argument variants
            'args': [{}],
            'query-args': json.dumps([
                [{'budget': 50}], [{'budget': 200}], [{'budget': 1000}], [{'budget': 4000}]
            ])
        }
    }
}

# NLE config (Legacy/Wrapper) - DISABLED per request
# NLE config (neurips23 sparse NLE)
nle_config = {
    'docker-tag': 'neurips23-sparse-nle',
    'module': 'neurips23.sparse.nle.nle',
    'constructor': 'NLE',
    'base-args': ['@metric'],
    'run-groups': {
        'base': {
            # single index build config; multiple query argument variants
            'args': [{'t1': 32, 't2': 128}],
            'query-args': json.dumps([
                {"k1": 4, "k2": 40, "k3": 1},
                {"k1": 4, "k2": 40, "k3": 10},
                {"k1": 4, "k2": 40, "k3": 100},
                {"k1": 4, "k2": 40, "k3": 1000}
            ])
        }
    }
}

# SHNSW
shnsw_config = {
    'docker-tag': None,
    'module': 'neurips23.sparse.shnsw.shnsw',
    'constructor': 'SparseHNSW',
    'base-args': ['@metric'],
    'run-groups': {
        'base': {
            # single index build config; multiple query argument variants
            'args': [{'efConstruction': 200, 'M': 32, 'buildthreads': -1}],
            'query-args': json.dumps([
                [{'efSearch': 40}], [{'efSearch': 200}], [{'efSearch': 500}], [{'efSearch': 1000}]
            ])
        }
    }
}

# Double Group Testing
double_group_testing_config = {
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

# Binary Splitting
binary_splitting_config = {
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
    # dataset-specific budgets for Linscan/CUFE (reflect dataset sizes/sparsity)
    # Reduced budgets (smaller than prior large values)
    if ds == 'movielens':
        budgets = [5, 20, 50, 100]
    elif ds == 'avazu':
        budgets = [1, 4, 6, 10]
    elif ds == 'kddb':
        budgets = [5, 20, 50, 100]
    elif ds == 'sparse-full':
        budgets = [5, 50, 60, 500]
    else:  # sparse-1M and default
        budgets = [1, 4, 6, 10]

    linscan_ds = copy.deepcopy(linscan_config)
    linscan_ds['run-groups']['base']['query-args'] = json.dumps([
        [{'budget': b}] for b in budgets
    ])

    cufe_ds = copy.deepcopy(cufe_config)
    cufe_ds['run-groups']['base']['query-args'] = json.dumps([
        [{'budget': b}] for b in budgets
    ])

    # PyANNS definitions: single index build, multiple ef query settings
    pyanns_config = {
        'docker-tag': None,
        'module': 'neurips23.ood.pyanns.pyanns',
        'constructor': 'Pyanns',
        'base-args': ['@metric'],
        'run-groups': {
            'base': {
                'args': [
                    {'R': 48, 'L': 500}
                ],
                'query-args': json.dumps([
                    {'ef': 50}, {'ef': 100}, {'ef': 200}, {'ef': 400}
                ])
            }
        }
    }

    definitions[ds] = {
        'linscan': linscan_ds,
        'cufe': cufe_ds,
        'shnsw': shnsw_config,
        'nle': nle_config,
        'pyanns': pyanns_config,
        'double-group-testing': double_group_testing_config,
        'binary-splitting': binary_splitting_config
    }

with open('algos-all-variants.yaml', 'w') as f:
    yaml.dump(definitions, f)

print("Generated algos-all-variants.yaml")
