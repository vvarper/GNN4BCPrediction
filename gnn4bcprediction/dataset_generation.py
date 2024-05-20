import copy
import json
import os
import random
import time

import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import from_networkx

from gnn4bcprediction.bc_models import generate_random_uniform_values, \
    run_hk_model_mc


def generate_threshold_per_community(graph, max_threshold, generator):
    # Get the number of communities
    num_communities = len(
        set(nx.get_node_attributes(graph, 'community').values()))

    # Generate a random threshold for each community
    thresholds = generate_random_uniform_values(num_communities,
                                                generator=generator,
                                                max_val=max_threshold)

    # Create a list of thresholds, where each node has the threshold of
    # its community
    threshold_bc = [thresholds[graph.nodes[node]['community']] for node in
                    graph.nodes()]

    return threshold_bc


def generate_attribute_graph(base_graph, initial_opinions, threshold_bc,
                             simulation_steps, mc, save_path=None):
    G = copy.deepcopy(base_graph)
    G.graph['mc'] = mc
    G.graph['simulation_steps'] = simulation_steps

    data_plot, final_opinions = run_hk_model_mc(mc=mc,
                                                initial_op=initial_opinions,
                                                graph=G,
                                                threshold_bc=threshold_bc,
                                                simulation_steps=simulation_steps)

    mean_final_opinions = np.mean(np.array(final_opinions), axis=0)

    for i, nodo in enumerate(G.nodes()):
        G.nodes[nodo]['initial_opinion'] = initial_opinions[i]
        G.nodes[nodo]['final_opinion'] = mean_final_opinions[i]
        G.nodes[nodo]['threshold'] = threshold_bc[i]

    if save_path is not None:
        with open(save_path, 'w') as f:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            json.dump(nx.node_link_data(G), f)

    return G


def generate_multiple_attribute_graph(base_graph, initial_opinions,
                                      simulation_steps, mc, num_graphs,
                                      max_threshold, communities=False,
                                      generator=None, save_path=None):
    n = base_graph.number_of_nodes()

    if not communities:
        thresholds = np.linspace(0.1, max_threshold, num_graphs)
    G_list = []

    for i in range(num_graphs):
        print('Generating graph {}'.format(i))
        if communities:
            threshold_bc = generate_threshold_per_community(base_graph,
                                                            max_threshold,
                                                            generator)
        else:
            threshold_bc = np.ones(n) * thresholds[i]

        G = generate_attribute_graph(base_graph=base_graph,
                                     initial_opinions=initial_opinions,
                                     threshold_bc=threshold_bc,
                                     simulation_steps=simulation_steps, mc=mc)
        G_list.append(G)

    # Combine all graphs as separate components
    G_complete = nx.disjoint_union_all(G_list)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(nx.node_link_data(G_complete), f)

    return G_complete


def create_pygdataset(graphs, per_val, per_test, seed, save_path):
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.manual_seed(seed)

    # Separate connected components
    subgraphs = [graph.subgraph(c).copy() for graph in graphs for c in
                 nx.connected_components(graph)]

    # Transform the graphs to PyG format
    full_data = []
    for graph in subgraphs:
        data = from_networkx(graph, group_node_attrs=['initial_opinion',
                                                      'final_opinion'])
        data.y = data.threshold
        data.x = data.x.to(torch.float32)
        data.y = data.y.to(torch.float32)
        del data.threshold
        if per_val > 0 or per_test > 0:
            data = RandomNodeSplit(num_val=per_val, num_test=per_test)(data)
        full_data.append(data)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if per_val > 0 or per_test > 0:
        temp_data, test_data = train_test_split(full_data, test_size=per_test,
                                                random_state=seed)
        train_data, val_data = train_test_split(temp_data, test_size=per_val,
                                                random_state=seed)

        torch.save(train_data, f'{save_path}_train.pt')
        torch.save(test_data, f'{save_path}_test.pt')
        torch.save(val_data, f'{save_path}_val.pt')

    else:
        torch.save(full_data, f'{save_path}.pt')

    return full_data


def create_datasets(topologies, top_names, dataset_name, steps, mc, per_val,
                    per_test, num_configs, seed, max_threshold=0.5, mix=True,
                    communities=False, save_nx=False):
    generator = random.Random(seed)
    initial_opinions = [
        generate_random_uniform_values(topology.number_of_nodes(),
                                       generator=generator) for topology in
        topologies]
    thr_scenario = 'com' if communities else 'hom'
    datasets_path = f'data/datasets/'
    sim_attributes = f'{steps}_{mc}_{num_configs}_{max_threshold}_{thr_scenario}'

    t1 = time.time()
    graphs = [generate_multiple_attribute_graph(base_graph=topologies[i],
                                                initial_opinions=
                                                initial_opinions[i],
                                                simulation_steps=steps, mc=mc,
                                                num_graphs=num_configs,
                                                max_threshold=max_threshold,
                                                communities=communities,
                                                generator=generator,
                                                save_path=f'data/nx_graphs/{top_names[i]}_{sim_attributes}.json' if save_nx else None)
              for i in range(len(topologies))]
    t2 = time.time()

    print(f'Graph generation time: {t2 - t1}')

    t1 = time.time()

    if mix:
        pyg_path = f'{datasets_path}{dataset_name}/{dataset_name}_{sim_attributes}'
        if per_val != 0 or per_test != 0:
            pyg_path = f'{pyg_path}_{per_val}-{per_test}'
        create_pygdataset(graphs, per_val=per_val, per_test=per_test,
                          seed=seed, save_path=pyg_path)
    else:
        for i, graph in enumerate(graphs):
            pyg_path = f'{datasets_path}{top_names[i]}/{top_names[i]}_{sim_attributes}'
            if per_val != 0 or per_test != 0:
                pyg_path = f'{pyg_path}_{per_val}-{per_test}'
            create_pygdataset([graph], per_val=per_val, per_test=per_test,
                              seed=seed, save_path=pyg_path)

    t2 = time.time()

    print(f'PyG generation time: {t2 - t1}')
