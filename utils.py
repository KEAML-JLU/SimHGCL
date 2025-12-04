import torch
import random
import numpy as np
from torch import Tensor
from torch_scatter import scatter_add
from torch_geometric.typing import OptTensor
from itertools import permutations, combinations


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def clique_expansion_weight(edge_index):
    edge_weight_dict = {}
    for he in np.unique(edge_index[1, :]):
        nodes_in_he = np.sort(edge_index[0, :][edge_index[1, :] == he])
        if len(nodes_in_he) == 1:
            continue  # skip self loops
        combs = combinations(nodes_in_he, 2)
        for comb in combs:
            if not comb in edge_weight_dict.keys():
                edge_weight_dict[comb] = 1 / len(nodes_in_he)
            else:
                edge_weight_dict[comb] += 1 / len(nodes_in_he)
                
    new_edge_index = np.zeros((2, len(edge_weight_dict)))
    new_norm = np.zeros((len(edge_weight_dict)))
    cur_idx = 0
    for edge in edge_weight_dict:
        new_edge_index[:, cur_idx] = edge
        new_norm[cur_idx] = edge_weight_dict[edge]
        cur_idx += 1

    edge_index = torch.tensor(new_edge_index).type(torch.LongTensor)
    norm = torch.tensor(new_norm).type(torch.FloatTensor)
    return edge_index, norm


def clique_expansion(hyperedge_index: Tensor):
    edge_set = set(hyperedge_index[1].tolist())
    adjacency_matrix = []
    for edge in edge_set:
        mask = hyperedge_index[1] == edge
        nodes = hyperedge_index[:, mask][0].tolist()
        for e in permutations(nodes, 2):
            adjacency_matrix.append(e)
    
    adjacency_matrix = list(set(adjacency_matrix))
    adjacency_matrix = torch.LongTensor(adjacency_matrix).T.contiguous()
    return adjacency_matrix.to(hyperedge_index.device)


def clique_expansion_weight_two(hyperedge_index: Tensor):
    edge_set = set(hyperedge_index[1].tolist())
    edge_dict = {}
    adjacency_matrix = []
    for edge in edge_set:
        mask = hyperedge_index[1] == edge
        nodes = hyperedge_index[:, mask][0].tolist()
        edge_dict[edge] = nodes
        for e in permutations(nodes, 2):
            adjacency_matrix.append(e)
    
    adjacency_matrix = list(set(adjacency_matrix))
    adjacency_matrix = torch.LongTensor(adjacency_matrix).T.contiguous()
    length = adjacency_matrix.shape[1]
    edge_weights = {}
    for i in range(length):
      t = list(adjacency_matrix[:, i].numpy())
      for each in edge_dict.values():
        if set(t).issubset(set(each)):
          if i not in edge_weights.keys():
            edge_weights[i] = 1/len(each)
          else:
            edge_weights[i] += 1/len(each)

    return adjacency_matrix.to(hyperedge_index.device), torch.FloatTensor(list(edge_weights.values()))

