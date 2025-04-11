import time
import torch
import networkx as nx
from dataloader import *
from tqdm import tqdm
from .fastrewiringKupdates import *
#from rewiring.fastrewiringmax import *
from .MinGapKupdates import *
from .spectral_utils import *
from torch_geometric.utils import to_networkx,from_networkx,homophily
import random
# from clustering import *
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI

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
    newgraph = min_and_update_edges(nxgraph, rank_by_proxy_delete_min, "proxydeletemin",seed, max_iter=max_iterations, updating_period=1)
    newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
    # end_algo = time.time()
