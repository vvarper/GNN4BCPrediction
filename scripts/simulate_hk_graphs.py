import json
import os

import networkx as nx
from matplotlib import pyplot as plt

from gnn4bcprediction.bc_models import plot_opinions, hk_model

graphs_folder = 'data/nx_graphs/'

topologies = {
    'synthetic': ['barabasi_1000_2', 'barabasi_1000_4', 'barabasi_1000_6',
                  'erdos_1000_0.1', 'erdos_1000_0.2', 'erdos_1000_0.3',
                  'newman_1000_3_0.3', 'newman_1000_5_0.3',
                  'newman_1000_7_0.3'],
    'real': ['cora', 'cora_ml', 'citeseer', 'pubmed', 'dblp']}

for top_type in topologies.keys():
    for scenario in ['com']:
        simul_output_folder = f'{graphs_folder}{top_type}/{scenario}/'
        os.makedirs(simul_output_folder, exist_ok=True)

        for topology in topologies[top_type]:
            print(f'Processing {topology}')

            with open(
                    f'{graphs_folder}{topology}_1000000_10_20_0.5_{scenario}.json') as f:
                data = json.load(f)

            G = nx.node_link_graph(data)
            graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

            for graph_id, graph in enumerate(graphs):
                mapping = {n: i for i, n in enumerate(graph.nodes)}
                graph = nx.relabel_nodes(graph, mapping)

                initial_opinions = [graph.nodes[n]['initial_opinion'] for n in
                                    graph.nodes]
                thresholds = [graph.nodes[n]['threshold'] for n in graph.nodes]

                data_plot, final_opinions_mc = hk_model(
                    initial_op=initial_opinions, graph=graph,
                    threshold_bc=thresholds,
                    simulation_steps=graph.graph['simulation_steps'])

                if scenario == 'hom':
                    title = f'HK execution with homogeneous threshold {thresholds[0]:.2f}'
                    filename = f'{simul_output_folder}hom_{topology}_{thresholds[0]:.2f}.png'
                else:
                    title = f'HK execution with threshold by community ({graph_id})'
                    filename = f'{simul_output_folder}com_{topology}_{graph_id}.png'

                plot_opinions(initial_opinions, data_plot, final_opinions_mc,
                              title, filename, graph.graph['simulation_steps'])
                plt.clf()
