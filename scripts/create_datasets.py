import os

import networkx as nx

from gnn4bcprediction.dataset_generation import create_datasets


def load_topologies(folder_path):
    topologies = []
    top_names = []

    for top_name in os.listdir(folder_path):
        topologies.append(nx.read_gml(f'{folder_path}/{top_name}', label='id'))
        top_names.append(top_name[:-4])

    return topologies, top_names


seed = 37
simulation_steps = 1000000
mc = 10
per_val = 0.2
per_test = 0.2
num_configs = 20
max_threshold = 0.5

## 1. Synthetic dataset #######################################################

syn_topologies, syn_names = load_topologies('data/topologies/synthetic')

for scenario in [False, True]:
    create_datasets(topologies=syn_topologies, top_names=syn_names,
                    dataset_name='synthetic', steps=simulation_steps, mc=mc,
                    per_val=per_val, per_test=per_test,
                    num_configs=num_configs, seed=seed,
                    max_threshold=max_threshold, mix=True,
                    communities=scenario, save_nx=False)

## 2. Real-world test graphs ##################################################

real_topologies, real_names = load_topologies('data/topologies/real')

# First configuration: Homogeneous thresholds
for scenario in [False, True]:
    create_datasets(topologies=real_topologies, top_names=real_names,
                    dataset_name='', steps=simulation_steps, mc=mc, per_val=0,
                    per_test=0, num_configs=num_configs, seed=seed,
                    max_threshold=max_threshold, mix=False,
                    communities=scenario, save_nx=False)
