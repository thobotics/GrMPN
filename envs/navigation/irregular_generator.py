# -*- coding: utf-8 -*-

"""
    irregular_generator.py
    
    Created on  : August 13, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

import pickle
import collections
import numpy as np
import networkx as nx

from envs.navigation.nav_generator import graph_to_input_target
from lib.misc import *
from scipy import spatial
from envs.expert_generator import ExpertGenerator

DISTANCE_WEIGHT_NAME = "distance"  # The name for the distance edge attribute.


class IrregularGenerator(ExpertGenerator):
    def __init__(self, rand, mem_size, rand_batch=True,
                 mode="offline", test_set=False,
                 min_max_nodes=None, theta=20, **kwargs):
        super().__init__(**kwargs)

        self.rand = rand
        self.rand_batch = rand_batch
        self.mem_size = mem_size
        self.min_max_nodes = min_max_nodes
        self.theta = theta
        self.test_set = test_set

        self.mode = mode
        assert self.mode in ["online", "offline"]

        self._data = None
        self._mem_idx = 0

        if self.mode == "offline":
            with open("./data/irregular_graph/irregular_100.pkl", "rb") as f:
                self._offline_data = pickle.load(f)
                self.max_mem = len(self._offline_data[0])

    def __next__(self):
        if self._data is None:
            self.update_memory()

        if self.rand_batch:
            start = self.rand.randint(0, (len(self._data[0]) - self.batch_size + 1))
            idx = slice(start, start + self.batch_size)
        else:
            idx = slice(self.current_idx, self.current_idx + self.batch_size)
            if self.current_idx >= self.mem_size:
                self.reset_idx(idx=0)

            self.current_idx = self.current_idx + self.batch_size

        inputs, targets, graphs, paths = self._data
        data = inputs[idx], targets[idx], graphs[idx], paths[idx]

        return data

    def update_memory(self):

        if self.mode == "offline":
            adj_train, coord_train, start_train, label_train, \
            adj_test, coord_test, start_test, label_test = self._offline_data
            idx = slice(self._mem_idx, self._mem_idx + self.mem_size)
            # l_idx = slice(self._mem_idx, self._mem_idx + (start_train.shape[1] * self.mem_size))

            if not self.test_set:
                label_train = label_train.reshape((start_train.shape[:2]))
                self._data = self._generate_networkx_from_pkl(adj_train[idx], coord_train[idx],
                                                              start_train[idx], label_train[idx])
            else:
                label_test = label_test.reshape((start_test.shape[:2]))
                self._data = self._generate_networkx_from_pkl(adj_test[idx], coord_test[idx],
                                                              start_test[idx], label_test[idx])

            self._mem_idx = (self._mem_idx + self.mem_size) % self.max_mem
            if self._mem_idx == 0 or (self._mem_idx + self.mem_size) > self.max_mem:  # Check next step
                self.reset_idx(all=True)

        else:
            self._data = self._generate_networkx_graphs(
                self.rand, self.mem_size, self.min_max_nodes, self.theta)

        # self._data = self._generate_networkx_graphs(
        #     self.rand, self.mem_size, self.min_max_nodes, self.theta)

        self._mem_idx = self._mem_idx + self.mem_size

        self.reset_idx()

    def iter_call_back(self):
        pass

    def reset_idx(self, idx=0, all=False):
        super().reset_idx(idx)
        if all:
            self._mem_idx = 0

    def gen_sample_data(self, end_only=True, full_grid=False):
        inputs, targets, graphs, _ = self.__next__()
        # samples, _ = self._convert_to_graphs(data, data_generator.sample_graph, data_generator.path_idx, full_grid)

        input_graphs, target_graphs, raw_graphs = [], [], []
        for input, target, graph in zip(inputs, targets, graphs):
            if end_only:
                # TODO: Load new graph with full_sol=False, current pkl file doesn't has next_step
                input, target = graph_to_input_target(graph, full_sol=True, end_only=end_only)

            input_graphs.append(input)
            target_graphs.append(target)
            raw_graphs.append(graph)

        self.reset_idx(all=True)

        return input_graphs, target_graphs, raw_graphs

    def _generate_networkx_from_pkl(self, adj_matrices, coords, starts, labels):
        input_graphs = []
        target_graphs = []
        graphs = []
        paths = []

        for i, (adj, coord) in enumerate(zip(adj_matrices, coords)):
            graph = nx.from_scipy_sparse_matrix(adj)
            for n, p in enumerate(coord):
                graph.node[n]['pos'] = p

            graph = self._add_vin_graphs(graph, starts[i, :, 0], labels[i])

            for j, g in enumerate(graph):
                input_graph, target_graph = graph_to_input_target(g, full_sol=False)
                input_graphs.append(input_graph)
                target_graphs.append(target_graph)

                graphs.append(g)
                paths.append([starts[i, j, 0], labels[i, j]])

        return input_graphs, target_graphs, graphs, paths

    def _generate_networkx_graphs(self, rand, num_graphs, num_nodes_min_max, theta, num_instance=1):
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
        input_graphs = []
        target_graphs = []
        graphs = []
        paths = []

        for _ in range(num_graphs):
            ori_graphs = self._generate_graph(rand, num_nodes_min_max, theta=theta)[0]
            g_rand = np.random.RandomState(seed=rand.randint(num_instance))
            for _ in range(num_instance):
                stp_graph, path = self._add_shortest_path(g_rand, ori_graphs, min_length=1)

                graph = stp_graph[0]
                input_graph, target_graph = graph_to_input_target(graph, full_sol=True)
                input_graphs.append(input_graph)
                target_graphs.append(target_graph)
                graphs.append(graph)
                paths.append(path)

                # for graph in stp_graph:
                #   input_graph, target_graph = graph_to_input_target(graph)
                #   input_graphs.append(input_graph)
                #   target_graphs.append(target_graph)
                #   graphs.append(graph)

        return input_graphs, target_graphs, graphs, paths

    def _generate_graph(self, rand,
                        num_nodes_min_max,
                        dimensions=2,
                        theta=1000.0,
                        rate=1.0):
        """Creates a connected graph.

        The graphs are geographic threshold graphs, but with added edges via a
        minimum spanning tree algorithm, to ensure all nodes are connected.

        Args:
          rand: A random seed for the graph generator. Default= None.
          num_nodes_min_max: A sequence [lower, upper) number of nodes per graph.
          dimensions: (optional) An `int` number of dimensions for the positions.
            Default= 2.
          theta: (optional) A `float` threshold parameters for the geographic
            threshold graph's threshold. Large values (1000+) make mostly trees. Try
            20-60 for good non-trees. Default=1000.0.
          rate: (optional) A rate parameter for the node weight exponential sampling
            distribution. Default= 1.0.

        Returns:
          The graph.
        """
        # Sample num_nodes.
        num_nodes = rand.randint(*num_nodes_min_max)

        # Create geographic threshold graph.
        pos_array = rand.uniform(size=(num_nodes, dimensions))
        pos = dict(enumerate(pos_array))
        weight = dict(enumerate(rand.exponential(rate, size=num_nodes)))
        geo_graph = nx.geographical_threshold_graph(
            num_nodes, theta, pos=pos, weight=weight)

        # Create minimum spanning tree across geo_graph's nodes.
        distances = spatial.distance.squareform(spatial.distance.pdist(pos_array))
        i_, j_ = np.meshgrid(range(num_nodes), range(num_nodes), indexing="ij")
        weighted_edges = list(zip(i_.ravel(), j_.ravel(), distances.ravel()))
        mst_graph = nx.Graph()
        mst_graph.add_weighted_edges_from(weighted_edges, weight=DISTANCE_WEIGHT_NAME)
        mst_graph = nx.minimum_spanning_tree(mst_graph, weight=DISTANCE_WEIGHT_NAME)
        # Put geo_graph's node attributes into the mst_graph.
        for i in mst_graph.nodes():
            mst_graph.node[i].update(geo_graph.node[i])

        # Compose the graphs.
        combined_graph = nx.compose_all((mst_graph, geo_graph.copy()))
        # Put all distance weights into edge attributes.
        for i, j in combined_graph.edges():
            combined_graph.get_edge_data(i, j).setdefault(DISTANCE_WEIGHT_NAME,
                                                          distances[i, j])
        return combined_graph, mst_graph, geo_graph

    def _add_vin_graphs(self, graph, starts, labels):

        # next_path = path[i:]
        graphs = []
        end = 0  # TODO: Change this

        for start, label in zip(starts, labels):
            next_path = [start, label]

            # Creates a directed graph, to store the directed path from start to end.
            digraph = graph.to_directed()

            # Add the "start", "end", and "solution" attributes to the nodes and edges.
            digraph.add_node(start, start=True)
            digraph.add_node(end, end=True)
            digraph.add_node(label, next_step=True)
            digraph.add_nodes_from(set_diff(digraph.nodes(), [start]), start=False)
            digraph.add_nodes_from(set_diff(digraph.nodes(), [end]), end=False)
            digraph.add_nodes_from(set_diff(digraph.nodes(), [label]), next_step=False)
            digraph.add_nodes_from(set_diff(digraph.nodes(), next_path), solution=False)
            digraph.add_nodes_from(next_path, solution=True)
            path_edges = list(pairwise(next_path))
            digraph.add_edges_from(set_diff(digraph.edges(), path_edges), solution=False)
            digraph.add_edges_from(path_edges, solution=True)
            digraph.add_edges_from(digraph.edges, distance=1.0)  # For completeness
            graphs.append(digraph)

        return graphs

    def _add_shortest_path(self, rand, graph, min_length=1):
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
            lengths = nx.all_pairs_shortest_path_length(graph).items()
        except AttributeError:
            # This is for compatibility with newer networkx.
            lengths = list(nx.all_pairs_shortest_path_length(graph))
        for x, yy in lengths:
            for y, l in yy.items():
                if l >= min_length:
                    pair_to_length_dict[x, y] = l
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
        i = rand.choice(len(node_pairs), p=probabilities)
        start, end = node_pairs[i]
        path = nx.shortest_path(
            graph, source=start, target=end, weight=DISTANCE_WEIGHT_NAME)

        stp_graph = []
        for i, start in enumerate(path[:-1]):
            next_path = path[i:]
            # Creates a directed graph, to store the directed path from start to end.
            digraph = graph.to_directed()

            # Add the "start", "end", and "solution" attributes to the nodes and edges.
            digraph.add_node(start, start=True)
            digraph.add_node(end, end=True)
            digraph.add_node(path[i + 1], next_step=True)
            digraph.add_nodes_from(set_diff(digraph.nodes(), [start]), start=False)
            digraph.add_nodes_from(set_diff(digraph.nodes(), [end]), end=False)
            digraph.add_nodes_from(set_diff(digraph.nodes(), [path[i + 1]]), next_step=False)
            digraph.add_nodes_from(set_diff(digraph.nodes(), next_path), solution=False)
            digraph.add_nodes_from(next_path, solution=True)
            path_edges = list(pairwise(next_path))
            digraph.add_edges_from(set_diff(digraph.edges(), path_edges), solution=False)
            digraph.add_edges_from(path_edges, solution=True)
            stp_graph.append(digraph)

        return stp_graph, path
