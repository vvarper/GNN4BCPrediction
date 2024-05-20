import os

import networkx as nx

n = 1000
seeds = [37, 25, 42]

## 1. Topologies generation ###################################################

# Erdos Renyi topologies
p_list = [0.1, 0.2, 0.3]
G_erdos = []

for p, seed in zip(p_list, seeds):
    G_erdos.append(nx.erdos_renyi_graph(n=n, p=p, seed=seed))

# Newman-Watts-Strogatz topologies with p=0.3
k_list = [5, 3, 7]
p_newman = 0.3
G_newman = []

for k, seed in zip(k_list, seeds):
    G_newman.append(nx.newman_watts_strogatz_graph(n=n, k=k, p=p, seed=seed))

# Barabasi-Albert topologies
m_list = [4, 2, 6]
G_barabasi = []

for m, seed in zip(m_list, seeds):
    G_barabasi.append(nx.barabasi_albert_graph(n=n, m=m, seed=seed))

## 2. Community detection #####################################################

for G in G_erdos + G_newman + G_barabasi:
    communities = nx.algorithms.community.louvain_communities(G, seed=seeds[0])
    communities = {node: cid for cid, nodes in enumerate(communities) for node
                   in nodes}
    nx.set_node_attributes(G, communities, 'community')

## 3. Save graphs as GML files ################################################
base_folder = 'data/topologies/synthetic/'
os.makedirs(os.path.dirname(base_folder), exist_ok=True)

for i in range(len(seeds)):
    nx.write_gml(G_erdos[i], f'{base_folder}erdos_{n}_{p_list[i]}.gml')
    nx.write_gml(G_newman[i],
                 f'{base_folder}newman_{n}_{k_list[i]}_{p_newman}.gml')
    nx.write_gml(G_barabasi[i], f'{base_folder}barabasi_{n}_{m_list[i]}.gml')
