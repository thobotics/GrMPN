# -*- coding: utf-8 -*-

"""
    misc.py
    
    Created on  : August 11, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

import itertools
import numpy as np
import networkx as nx
import scipy.sparse as sp
from graph_nets import utils_tf
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def set_diff(seq0, seq1):
    """Return the set difference between 2 sequences as a list."""
    return list(set(seq0) - set(seq1))


def to_one_hot(indices, max_value, axis=-1):
    one_hot = np.eye(max_value)[indices]
    if axis not in (-1, one_hot.ndim):
        one_hot = np.moveaxis(one_hot, -1, axis)
    return one_hot


def get_node_dict(graph, attr):
    """Return a `dict` of node:attribute pairs from a graph."""
    return {k: v[attr] for k, v in graph.node.items()}


def create_feature(attr, fields):
    return np.hstack([np.array(attr[field], dtype=float) for field in fields])


def make_all_runnable_in_session(*args):
    """Lets an iterable of TF graphs be output from a session as NP graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]


def to_relative_coord_matrices(coords, end=0):
    coord_matrix = []
    for i in range(coords.shape[0]):
        # coord_tmp = []
        # for j in range(coord.shape[1]):
        #     coord_tmp.append(coord[i, j, :] - coord[i, :, :])
        coord_tmp = coords[i, end, :] - coords[i, :, :]
        coord_matrix.append(np.array(coord_tmp))
    coord_matrix = np.array(coord_matrix)
    return coord_matrix


def compute_euclidian_matrix(adj, coord, sparse_return=True):
    e_dist = squareform(pdist(coord)) * adj.toarray()
    if sparse_return:
        return sp.csc_matrix(e_dist)
    else:
        return e_dist


def map_equal(graph_a, graph_b, c_struct=True, c_goal=True):
    assert c_struct or c_goal, "Either struct or goal checking must be True"

    adj_a = nx.to_numpy_array(graph_a)
    adj_b = nx.to_numpy_array(graph_b)
    goal_a = [graph_a.node[n]["pos"] for n in graph_a.node if graph_a.node[n]["end"]][0]
    goal_b = [graph_b.node[n]["pos"] for n in graph_b.node if graph_b.node[n]["end"]][0]

    struct_equal = np.array_equal(adj_a, adj_b)
    goal_equal = np.array_equal(goal_a, goal_b)

    if c_struct and c_goal:
        return struct_equal and goal_equal
    elif c_struct:
        return struct_equal
    elif c_goal:
        return goal_equal


def grid_from_nxgraph(graph, dim):
    grid = np.zeros((2,) + dim)
    graph_nodes = graph.nodes

    for n in graph_nodes:
        y, x = graph_nodes[n]["pos"]
        grid[0][y][x] = 1.0

        if graph_nodes[n]["end"]:
            grid[1][y][x] = 10.0

    return grid


def convert_to_policy(mechanism, graph, map_design, value_dict):
    pred_policy = np.zeros((1,) + map_design.shape, dtype=np.int64)

    graph_nodes = graph.nodes
    value_nodes = value_dict["nodes"]
    pos_value = {graph_nodes[n]["pos"]: value_nodes[n] for n in graph_nodes}

    for y in range(map_design.shape[0]):
        for x in range(map_design.shape[1]):
            if map_design[y][x] == 1.:
                # MOORE
                actions = mechanism.neighbors_func(map_design, 1, y, x)
                q_sa = [pos_value[(j, i)] for _, j, i in actions]
                pred_policy[0][y][x] = np.argmax(q_sa)

    return pred_policy


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p
