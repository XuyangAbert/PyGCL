from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import add_edge
from torch_geometric.utils import to_networkx,from_networkx,homophily
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from math import inf
from tqdm import tqdm
import torch

def choose_edge_to_add(x, edge_index, degrees):
	# chooses edge (u, v) to add which minimizes y[u]*y[v]
	n = x.size
	m = edge_index.shape[1]
	y = x / ((degrees + 1) ** 0.5)
	products = np.outer(y, y)
	for i in range(m):
		u = edge_index[0, i]
		v = edge_index[1, i]
		products[u, v] = inf
	for i in range(n):
		products[i, i] = inf
	smallest_product = np.argmin(products)
	return smallest_product, smallest_product % n, smallest_product // n

def compute_degrees(edge_index, num_nodes=None):
	# returns array of degrees of all nodes
	if num_nodes is None:
		num_nodes = np.max(edge_index) + 1
	degrees = np.zeros(num_nodes)
	m = edge_index.shape[1]
	for i in range(m):
		degrees[edge_index[0, i]] += 1
	return degrees

def add_edge(edge_index, u, v):
	new_edge = np.array([[u, v],[v, u]])
	return np.concatenate((edge_index, new_edge), axis=1)


def adj_matrix_multiply(edge_index, x):
	# given an edge_index, computes Ax, where A is the corresponding adjacency matrix
	n = x.size
	y = np.zeros(n)
	m = edge_index.shape[1]
	for i in range(m):
		u = edge_index[0, i]
		v = edge_index[1, i]
		y[u] += x[v]
	return y


def compute_spectral_gap(edge_index, x):
	m = edge_index.shape[1]
	n = np.max(edge_index) + 1
	degrees = compute_degrees(edge_index, num_nodes=n)
	y = adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
	for i in range(n):
		if x[i] > 1e-9:
			return 1 - y[i]/x[i]
	return 0.

def _edge_rewire(edge_index, edge_type, x=None, num_iterations=50, initial_power_iters=50):
	m = edge_index.shape[1]
	n = np.max(edge_index) + 1
	if x is None:
		x = 2 * np.random.random(n) - 1
	degrees = compute_degrees(edge_index, num_nodes=n)
	for i in range(initial_power_iters):
		x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5)/sum(degrees)
		y = x + adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
		x = y / np.linalg.norm(y)
	for I in range(num_iterations):
		prod, i, j = choose_edge_to_add(x, edge_index, degrees=degrees)
		edge_index = add_edge(edge_index, i, j)
		degrees[i] += 1
		degrees[j] += 1
		edge_type = np.append(edge_type, 1)
		edge_type = np.append(edge_type, 1)
		x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5)/sum(degrees)
		y = x + adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
		x = y / np.linalg.norm(y)
	return edge_index, edge_type, x, prod

def edge_rewire(edge_index, x=None, edge_type=None, num_iterations=50, initial_power_iters=5):
	m = edge_index.shape[1]
	n = np.max(edge_index) + 1
	if x is None:
		x = 2 * np.random.random(n) - 1
	if edge_type is None:
		edge_type = np.zeros(m, dtype=np.int64)
	return _edge_rewire(edge_index, edge_type=edge_type, x=x, num_iterations=num_iterations, initial_power_iters=initial_power_iters)
 
def maximize_modularity(G):
  return nx.community.greedy_modularity_communities(G)
 
def fosr(data, max_iterations = 10):
	# Convert to NetworkX graph
	nxgraph = to_networkx(data, to_undirected=True)
	# Track the original edges
	original_edges = set(nxgraph.edges())
	# Perform community detection before rewiring
	clustermod_before = maximize_modularity(nxgraph)
	cluster_dict_before = {node: i for i, cluster in enumerate(clustermod_before) for node in cluster}
	# Assuming `data.y` contains the node labels
	for j in range(max_iterations):
		edge_index, edge_type, _, prod = edge_rewire(data.edge_index.cpu().numpy(), num_iterations=1)      
		data.edge_index = torch.tensor(edge_index)
	data.edge_index = torch.cat([data.edge_index])
	# Convert back to NetworkX graph after rewiring
	# newgraph = to_networkx(data, to_undirected=True)
	# print(newgraph.edge_weights)
	# return newgraph
	return data

class FOSR(Augmentor):
	def __init__(self, max_iterations):
		super(FOSR, self).__init__() 
		self.max_iterations = max_iterations
	def augment(self, g: Graph) -> Graph:
		x, edge_index, edge_weights = g.unfold()
		# print("Shape before sparisification:", edge_index.shape)
		data = Data(x=x, edge_index=edge_index)
		# new_graph = fosr(data, self.max_iterations)
		# edge_index = torch.tensor(list(new_graph.edges()))
		data = fosr(data, self.max_iterations)
		# print("Shape after sparisification:", data.edge_index.shape)
		device = torch.device('cuda')
		data = data.to(device)
		return Graph(x=data.x, edge_index=data.edge_index, edge_weights=edge_weights)
