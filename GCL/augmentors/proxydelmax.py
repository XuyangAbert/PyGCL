import time
import torch
import networkx as nx
# from dataloader import *
from tqdm import tqdm
from .fastrewiringKupdates import *
from .fastrewiringmax import *
from .spectral_utils import *
from torch_geometric.utils import to_networkx,from_networkx,homophily
import random
# from clustering import *
# from sklearn.metrics.cluster import normalized_mutual_info_score as NMI

def maximize_modularity(G):
  return nx.community.greedy_modularity_communities(G)

def proxydelmax(data, nxgraph, seed, max_iterations):
  # print("Deleting edges to maximize the gap...")
  # start_algo = time.time()
  
  # # Track the original edges
  # original_edges = set(nxgraph.edges())
  
  # # Perform community detection before rewiring
  # clustermod_before = maximize_modularity(nxgraph)
  # cluster_dict_before = {node: i for i, cluster in enumerate(clustermod_before) for node in cluster}
  
  # # Initialize counters before rewiring
  # same_class_same_community_before = 0
  # same_class_diff_community_before = 0
  # diff_class_same_community_before = 0
  # diff_class_diff_community_before = 0
  
  newgraph = process_and_update_edges(nxgraph, rank_by_proxy_delete, "proxydeletemax", max_iter=max_iterations)
  newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
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
    newgraph = proxydelmax(data, nxgraph, self.seed, self.max_iterations)
    data.edge_index = torch.tensor(list(newgraph.edges())).t()
    device = torch.device('cuda')
    data = data.to(device)

    # modified_graph = Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
    # modified_graph = modified_graph.to(device)
    # return modified_graph
    return Graph(x=data.x, edge_index=data.edge_index, edge_weights=edge_weights)

