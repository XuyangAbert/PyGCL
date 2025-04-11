import time
import torch
import networkx as nx
from dataloader import *
from tqdm import tqdm
# from rewiring.fastrewiringKupdates import *
#from rewiring.fastrewiringmax import *
from .MinGapKupdates import *
from .spectral_utils import *
from torch_geometric.utils import to_networkx,from_networkx,homophily
import random
# from clustering import *
# from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
def maximize_modularity(G):
  return nx.community.greedy_modularity_communities(G)

def proxydelmax(data, nxgraph,seed, max_iterations):
    print("Deleting edges to maximize the gap...")
    start_algo = time.time()
    
    # Track the original edges
    original_edges = set(nxgraph.edges())
    
    # Perform community detection before rewiring
    clustermod_before = maximize_modularity(nxgraph)
    cluster_dict_before = {node: i for i, cluster in enumerate(clustermod_before) for node in cluster}
    
    # Initialize counters before rewiring
    same_class_same_community_before = 0
    same_class_diff_community_before = 0
    diff_class_same_community_before = 0
    diff_class_diff_community_before = 0
    
    # Assuming `data.y` contains the node labels
    labels = data.y.cpu().numpy()
    
    # Count same-class and different-class edges before rewiring
    # for edge in original_edges:
    #     node1, node2 = edge
    #     same_class = labels[node1] == labels[node2]
    #     same_community = cluster_dict_before[node1] == cluster_dict_before[node2]
        
    #     if same_class:
    #         if same_community:
    #             same_class_same_community_before += 1
    #         else:
    #             same_class_diff_community_before += 1
    #     else:
    #         if same_community:
    #             diff_class_same_community_before += 1
    #         else:
    #             diff_class_diff_community_before += 1
    start_algo = time.time()
    newgraph = process_and_update_edges(nxgraph, rank_by_proxy_delete, "proxydeletemax", max_iter=max_iterations)
    newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
