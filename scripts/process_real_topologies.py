import os

import networkx as nx
from torch_geometric.datasets import CitationFull
from torch_geometric.utils import to_networkx

base_folder = 'data/topologies/real/'
os.makedirs(os.path.dirname(base_folder), exist_ok=True)

## 1. Download real topologies ################################################
cora_ml = to_networkx(
    CitationFull(root='data/CitationFull', name='cora_ml')[0],
    to_undirected=True)
cora = to_networkx(CitationFull(root='data/CitationFull', name='cora')[0],
                   to_undirected=True)
citeseer = to_networkx(CitationFull(root='data/CitationFull', name='citeseer')[0],
                       to_undirected=True)
pubmed = to_networkx(CitationFull(root='data/CitationFull', name='pubmed')[0],
                     to_undirected=True)
dblp = to_networkx(CitationFull(root='data/CitationFull', name='dblp')[0],
                   to_undirected=True)

## 2. Get the largest connected component of each topology ####################
cora_ml = cora_ml.subgraph(
    sorted(nx.connected_components(cora_ml), key=len, reverse=True)[0])
cora = cora.subgraph(
    sorted(nx.connected_components(cora), key=len, reverse=True)[0])
citeseer = citeseer.subgraph(
    sorted(nx.connected_components(citeseer), key=len, reverse=True)[0])
pubmed = pubmed.subgraph(
    sorted(nx.connected_components(pubmed), key=len, reverse=True)[0])
dblp = dblp.subgraph(
    sorted(nx.connected_components(dblp), key=len, reverse=True)[0])

## 3. Community detection #####################################################
for G in [cora_ml, cora, citeseer, pubmed, dblp]:
    communities = nx.algorithms.community.louvain_communities(G, seed=42)
    communities = {node: cid for cid, nodes in enumerate(communities) for node
                   in nodes}
    nx.set_node_attributes(G, communities, 'community')

## 4. Save graphs as GML files ################################################
for G, name in zip([cora_ml, cora, citeseer, pubmed, dblp],
                   ['cora_ml', 'cora', 'citeseer', 'pubmed', 'dblp']):
    nx.write_gml(G, f'{base_folder}{name}.gml')
