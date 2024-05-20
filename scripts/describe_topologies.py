import os

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

for top_type in ['synthetic', 'real']:
    topologies_folder = f'data/topologies/{top_type}/'
    description_folder = f'{topologies_folder}description/'
    os.makedirs(description_folder, exist_ok=True)

    description = pd.DataFrame(
        columns=['topology', 'nodes', 'edges', 'density', 'avg_degree',
                 'diameter', 'avg_clustering', 'num_communities',
                 'modularity'])

    # For files in topologies folder (not including subfolders)
    for topology in os.listdir(topologies_folder):
        if os.path.isfile(f'{topologies_folder}{topology}'):
            print(f'Processing {topology}')

            G = nx.read_gml(f'{topologies_folder}{topology}')

            # Get communities and modularity
            communities_id = set([G.nodes[n]['community'] for n in G.nodes])
            communities_nodes = [
                set([n for n in G.nodes if G.nodes[n]['community'] == c]) for c
                in communities_id]
            modularity = nx.algorithms.community.modularity(G,
                                                            communities_nodes)

            # Log topology description
            description = pd.concat([description, pd.DataFrame(
                {'topology': topology, 'nodes': G.number_of_nodes(),
                 'edges': G.number_of_edges(), 'density': nx.density(G),
                 'avg_degree': np.mean([G.degree(n) for n in G.nodes]),
                 'diameter': nx.diameter(G),
                 'avg_clustering': nx.average_clustering(G),
                 'num_communities': len(communities_id),
                 'modularity': modularity}, index=[0])], ignore_index=True)

            # Save degree distribution plot
            degree_hist = nx.degree_histogram(G)
            plt.bar(range(len(degree_hist)), degree_hist)
            plt.xlabel('Degree')
            plt.ylabel('Frequency')
            plt.title(f'{topology} degree distribution')
            plt.savefig(
                f'{description_folder}degree_distribution_{topology}.png')
            plt.clf()

    # Save description
    description = description.sort_values(by='topology')
    description.to_csv(f'{description_folder}{top_type}_description.csv',
                       index=False)
