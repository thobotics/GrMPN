# -*- coding: utf-8 -*-

"""
    nav_generator.py
    
    Created on  : August 11, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

import pickle
import numpy as np
import networkx as nx

from lib.misc import *
from envs.expert_generator import ExpertGenerator


def graph_to_input_target(graph, full_sol=False, end_only=True):
    """Returns 2 graphs with input and target feature vectors for training.

    Args:
      graph: An `nx.DiGraph` instance.

    Returns:
      The input `nx.DiGraph` instance.
      The target `nx.DiGraph` instance.

    Raises:
      ValueError: unknown node type
    """

    if end_only:
        input_node_fields = ("end",)
    else:
        input_node_fields = ("start", "end")

    # input_node_fields += ("obstacle", )

    input_edge_fields = ("distance",)
    target_node_fields = ("solution",) if full_sol else ("next_step",)
    target_edge_fields = ("solution",)

    input_graph = graph.copy()
    target_graph = graph.copy()

    solution_length = 0
    for node_index, node_feature in graph.nodes(data=True):
        input_graph.add_node(
            node_index, features=create_feature(node_feature, input_node_fields))
        target_node = to_one_hot(
            create_feature(node_feature, target_node_fields).astype(int), 2)[0]
        target_graph.add_node(node_index, features=target_node)
        solution_length += int(node_feature["solution"])
    solution_length /= graph.number_of_nodes()

    for receiver, sender, features in graph.edges(data=True):
        input_graph.add_edge(
            sender, receiver, features=create_feature(features, input_edge_fields))
        target_edge = to_one_hot(
            create_feature(features, target_edge_fields).astype(int), 2)[0]
        target_graph.add_edge(sender, receiver, features=target_edge)

    input_graph.graph["features"] = np.array([0.0])
    target_graph.graph["features"] = np.array([solution_length], dtype=float)

    return input_graph, target_graph
