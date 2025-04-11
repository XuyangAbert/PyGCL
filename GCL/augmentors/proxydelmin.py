# import random
from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import add_edge
from torch_geometric.utils import to_networkx,from_networkx,homophily
from torch_geometric.data import Data
from numba import jit, prange
import numpy as np
import torch
import networkx as nx
from math import inf
from tqdm import tqdm

# from fastrewiringKupdates import *
from .MinGapKupdates import *
from .spectral_utils import *

def maximize_modularity(G):
  return nx.community.greedy_modularity_communities(G)

def proxydelmin(data, nxgraph, seed, max_iterations):
    # Track the original edges
    original_edges = set(nxgraph.edges())
    # # Perform community detection before rewiring
    # clustermod_before = maximize_modularity(nxgraph)
    # cluster_dict_before = {node: i for i, cluster in enumerate(clustermod_before) for node in cluster}
    # # Initialize counters before rewiring
    # same_class_same_community_before = 0
    # same_class_diff_community_before = 0
    # diff_class_same_community_before = 0
    # diff_class_diff_community_before = 0
    # start_algo = time.time()
    newgraph = min_and_update_edges(nxgraph, rank_by_proxy_delete_min, "proxydeletemin", updating_period=1, max_iter=max_iterations)
    newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
    # end_algo = time.time()
    return newgraph

class PROXYDELMIN(Augmentor):
    def __init__(self, max_iterations, seed):
        super(PROXYDELMIN, self).__init__()
        self.max_iterations = max_iterations
        self.seed = seed
        # self.tau = tau

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        data = Data(x=x, edge_index=edge_index)
        nxgraph = to_networkx(data, to_undirected=True)  
        # newgraph = sdrf(g, self.max_iterations, self.removal_bound, self.tau)
        newgraph = proxydelmin(data, nxgraph, self.seed, self.max_iterations)
        data.edge_index = torch.tensor(list(newgraph.edges())).t()
        device = torch.device('cuda')
        data = data.to(device)

        # modified_graph = Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
        # modified_graph = modified_graph.to(device)
        # return modified_graph
        return Graph(x=data.x, edge_index=data.edge_index, edge_weights=edge_weights)
