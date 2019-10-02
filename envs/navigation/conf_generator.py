# -*- coding: utf-8 -*-

"""
    conf_generator.py
    
    Created on  : September 14, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

import os
import pickle
import collections
import numpy as np
import networkx as nx
from lib.misc import *
from envs.expert_generator import ExpertGenerator

DISTANCE_WEIGHT_NAME = "distance"  # The name for the distance edge attribute.


class ConfigurationsGenerator(ExpertGenerator):
    def __init__(self, mode="offline", file_name=None, test_set=False, preload_pkl=None,
                 mem_traj_size=20,
                 **kwargs):

        super().__init__(**kwargs)
        self.mem_traj_size = mem_traj_size

        self.mode = mode
        assert self.mode in ["online", "offline"]

        self._use_coord = True

        if self.mode == "offline":
            with open(file_name, "rb") as f:
                self._offline_data = pickle.load(f)
                data_size = (len(self._offline_data) // self.mem_size) * self.mem_size
                self.max_mem = min(self.max_mem, data_size) if self.max_mem > 0 else data_size

    def update_memory(self):

        if self.mode == "offline":
            graphs = self._offline_data
            idx = slice(self._mem_idx, self._mem_idx + self.mem_size)

            self._data = self._generate_graph_paths(graphs[idx], self.rand, self.mem_traj_size)

        self._mem_idx += self.mem_size
        if (self._mem_idx + self.mem_size) > self.max_mem:
            self._stop_flag = 1

        print("Updated %d" % self._mem_idx)

        self.reset_idx()

    def reset_idx(self, idx=0, all=False):
        super().reset_idx(idx)
        if all:
            self._stop_flag = False
            self._mem_idx = 0
            self._data = None  # Clear data

    def gen_sample_data(self, end_only=True):
        input_graphs, raw_graphs, _, _ = self.__next__()
        self.reset_idx(all=True)

        return input_graphs, raw_graphs

    def _generate_graph_paths(self, nx_graphs, rand, num_instance=1):
        """Generate graphs for training.

        Args:
          rand: A random seed (np.RandomState instance).
          num_examples: Total number of graphs to generate.
          num_nodes_min_max: A 2-tuple with the [lower, upper) number of nodes per
            graph. The number of nodes for a graph is uniformly sampled within this
            range.
          theta: (optional) A `float` threshold parameters for the geographic
            threshold graph's threshold. Default= the number of nodes.

        Returns:
          input_graphs: The list of input graphs.
          target_graphs: The list of output graphs.
          graphs: The list of generated graphs.
        """
        graphs = []
        paths = []
        gidxs = []

        for ng, ori_graph in enumerate(nx_graphs):
            g_rand = np.random.RandomState(seed=rand.randint(num_instance))

            # TODO: GVIN consistency
            end_idx = 0
            ori_graph.add_node(end_idx, end=True)
            ori_graph.add_nodes_from(set_diff(ori_graph.nodes(), [end_idx]), end=False)

            coords = [ori_graph.nodes[n]["pos"] for n in ori_graph.nodes()]
            # coords = to_relative_coord_matrices(np.array([coords]))
            for n in ori_graph.nodes():
                # ori_graph.nodes[n]["rel_pos"] = coords[end_idx, n]
                ori_graph.nodes[n]["rel_pos"] = coords[n]

            graphs.append(ori_graph.to_directed())

            for ni in range(num_instance):
                path = self._add_shortest_path(g_rand, ori_graph, end_idx, min_length=1)

                paths.append(path)
                gidxs.append(ng)

        return graphs, paths, gidxs

    def _add_shortest_path(self, rand, graph, end_idx=0, min_length=1):
        """Samples a shortest path from A to B and adds attributes to indicate it.

        Args:
          rand: A random seed for the graph generator. Default= None.
          graph: A `nx.Graph`.
          min_length: (optional) An `int` minimum number of edges in the shortest
            path. Default= 1.

        Returns:
          The `nx.DiGraph` with the shortest path added.

        Raises:
          ValueError: All shortest paths are below the minimum length
        """
        # Map from node pairs to the length of their shortest path.
        pair_to_length_dict = {}
        try:
            # This is for compatibility with older networkx.
            # lengths = nx.all_pairs_shortest_path_length(graph).items()
            lengths = nx.single_target_shortest_path_length(graph, end_idx).items()
        except AttributeError:
            # This is for compatibility with newer networkx.
            # lengths = list(nx.all_pairs_shortest_path_length(graph))
            lengths = list(nx.single_target_shortest_path_length(graph, end_idx))
        # for x, yy in lengths:
        #     for y, l in yy.items():
        #         if l >= min_length:
        #             pair_to_length_dict[x, y] = l
        for y, l in lengths:
            if l >= min_length:
                pair_to_length_dict[y, end_idx] = l

        if max(pair_to_length_dict.values()) < min_length:
            raise ValueError("All shortest paths are below the minimum length")
        # The node pairs which exceed the minimum length.
        node_pairs = list(pair_to_length_dict)

        # Computes probabilities per pair, to enforce uniform sampling of each
        # shortest path lengths.
        # The counts of pairs per length.
        counts = collections.Counter(pair_to_length_dict.values())
        prob_per_length = 1.0 / len(counts)
        probabilities = [
            prob_per_length / counts[pair_to_length_dict[x]] for x in node_pairs
        ]

        # Choose the start and end points.
        # i = rand.choice(len(node_pairs), p=probabilities)
        # start, end = node_pairs[i]

        # TODO: For GVIN consistency only. Change this soon !!!
        end_0 = np.where(np.array(node_pairs)[:, 1] == end_idx)[0]
        i = rand.randint(len(end_0))
        start, end = node_pairs[end_0[i]]

        path = nx.shortest_path(
            graph, source=start, target=end, weight=DISTANCE_WEIGHT_NAME)

        return path
