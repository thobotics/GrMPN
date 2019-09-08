# -*- coding: utf-8 -*-

"""
    irregular_generator.py
    
    Created on  : August 13, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

import os
import pickle
import collections
import numpy as np
import networkx as nx

from lib.misc import *
from scipy import spatial
from envs.expert_generator import ExpertGenerator
from pygsp import graphs as gsp_graphs

import scipy.sparse as sp

DISTANCE_WEIGHT_NAME = "distance"  # The name for the distance edge attribute.


class IrregularGenerator(ExpertGenerator):
    def __init__(self,
                 mode="offline", file_name="", test_set=False, preload_pkl=None,
                 min_max_nodes=None, mem_traj_size=5, theta=20,  # For online
                 **kwargs):
        super().__init__(**kwargs)

        self.mem_traj_size = mem_traj_size
        self.min_max_nodes = min_max_nodes
        self.theta = theta
        self.test_set = test_set

        self.mode = mode
        assert self.mode in ["online", "offline"]

        self.preload_pkl = preload_pkl
        if self.preload_pkl is not None and not os.path.exists(self.preload_pkl):
            os.makedirs(self.preload_pkl)

        self._use_coord = True

        if self.mode == "offline":
            with open(file_name, "rb") as f:
                self._offline_data = pickle.load(f)
                if not self.test_set:
                    data_size = (len(self._offline_data[0]) // self.mem_size) * self.mem_size
                else:
                    data_size = (len(self._offline_data[5]) // self.mem_size) * self.mem_size
                self.max_mem = min(self.max_mem, data_size) if self.max_mem > 0 else data_size

    def update_memory(self):

        preload_pkl = "%s/data_%d.pkl" % (self.preload_pkl, self._mem_idx + self.mem_size) \
            if self.preload_pkl else None

        if preload_pkl and os.path.isfile(preload_pkl):
            with open(preload_pkl, "rb") as file:
                self._data = pickle.load(file)
                print("Loaded %s " % preload_pkl)
        elif self.mode == "offline":
            adj_train, coord_train, start_train, label_train, dist_train, \
            adj_test, coord_test, start_test, label_test, dist_test = self._offline_data
            idx = slice(self._mem_idx, self._mem_idx + self.mem_size)
            # l_idx = slice(self._mem_idx, self._mem_idx + (start_train.shape[1] * self.mem_size))

            if not self.test_set:
                label_train = label_train.reshape((start_train.shape[:2]))
                self._data = self._generate_networkx_from_pkl(adj_train[idx], coord_train[idx],
                                                              start_train[idx], label_train[idx], dist_train[idx])
            else:
                label_test = label_test.reshape((start_test.shape[:2]))
                self._data = self._generate_networkx_from_pkl(adj_test[idx], coord_test[idx],
                                                              start_test[idx], label_test[idx], dist_test[idx])

        else:
            self._data = self._generate_networkx_graphs(
                self.rand, self.mem_size, self.min_max_nodes, self.theta, num_instance=self.mem_traj_size)

        if preload_pkl and not os.path.isfile(preload_pkl):
            with open(preload_pkl, "wb") as file:
                pickle.dump(self._data, file)
                print("Dumped %s " % preload_pkl)

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

    def gen_sample_data(self, end_only=True, full_grid=False):
        input_graphs, raw_graphs, _, _ = self.__next__()
        self.reset_idx(all=True)

        return input_graphs, raw_graphs

    def _generate_networkx_from_pkl(self, adj_matrices, coords, starts, labels, dists):
        input_graphs = []
        target_graphs = []
        graphs = []
        paths = []
        gidxs = []
        starts = starts[:, :, 0]
        rel_coords = to_relative_coord_matrices(coords)

        for i, (adj, coord, rel_coord, dist) in enumerate(zip(adj_matrices, coords, rel_coords, dists)):
            # graph = nx.from_scipy_sparse_matrix(adj)
            # e_dist = compute_euclidian_matrix(adj, coord, sparse_return=True)
            e_dist = dist
            graph = nx.from_scipy_sparse_matrix(e_dist, edge_attribute=DISTANCE_WEIGHT_NAME)
            for n, (p, rp) in enumerate(zip(coord, rel_coord)):
                graph.node[n]['pos'] = p
                graph.node[n]['rel_pos'] = rp

            # Storable graph
            end = 0  # GVIN consistency
            digraph = graph.to_directed()
            digraph.add_node(end, end=True)
            digraph.add_nodes_from(set_diff(digraph.nodes(), [end]), end=False)

            graphs.append(digraph)
            for j in range(len(starts[i])):
                paths.append([starts[i, j], labels[i, j]])
                gidxs.append(i)

            # graph = self._add_vin_graphs(graph, starts[i, :], labels[i])
            #
            # for j, g in enumerate(graph):
            #     # input_graph, target_graph = graph_to_input_target(g, full_sol=False, coord=True)
            #     # input_graphs.append(input_graph)
            #     # target_graphs.append(target_graph)
            #
            #     graphs.append(g)
            #     paths.append([starts[i, j], labels[i, j]])

        return graphs, paths, gidxs

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
        gidxs = []

        for ng in range(num_graphs):
            ori_graph = self._generate_graph(rand, num_nodes_min_max, theta=theta)[0]
            # ori_graph = self._generate_sensor_graph(rand, num_nodes_min_max)
            g_rand = np.random.RandomState(seed=rand.randint(num_instance))

            # TODO: GVIN consistency
            end_idx = 0
            ori_graph.add_node(end_idx, end=True)
            ori_graph.add_nodes_from(set_diff(ori_graph.nodes(), [end_idx]), end=False)

            coords = [ori_graph.nodes[n]["pos"] for n in ori_graph.nodes()]
            coords = to_relative_coord_matrices(np.array([coords]))
            for n in ori_graph.nodes():
                ori_graph.nodes[n]["rel_pos"] = coords[end_idx, n]

            # TODO: GVIN consistency
            # Randomly block edges
            # block_prob = 0.1
            # n_blocks = int(block_prob * ori_graph.number_of_edges())
            # uvs = [(u, v) for u, v, e in ori_graph.edges(data=True)]
            #
            # for i in range(n_blocks):
            #     r_idx = uvs[g_rand.randint(0, len(uvs))]
            #     ori_graph.edges[r_idx][DISTANCE_WEIGHT_NAME] = 99.

            # e_idx = 0
            # neighs = [n for n in ori_graph.neighbors(e_idx)]
            # n_blocks = int(block_prob * len(neighs))
            # for i in range(n_blocks):
            #     ori_graph.edges[e_idx, g_rand.choice(neighs)][DISTANCE_WEIGHT_NAME] = 99.

            graphs.append(ori_graph.to_directed())

            for ni in range(num_instance):
                path = self._add_shortest_path(g_rand, ori_graph, end_idx, min_length=1)

                paths.append(path)
                gidxs.append(ng)

        return graphs, paths, gidxs

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

    def _generate_sensor_graph(self, rand, num_nodes_min_max, max_distance=0.75):
        # Sample num_nodes.
        num_nodes = rand.randint(*num_nodes_min_max)

        connected = False
        while not connected:
            seed = rand.randint(0, 99999)
            ssg = gsp_graphs.Sensor(N=num_nodes, Nc=2, seed=seed,
                                    connected=True, distribute=False, regular=False)

            # Truncate edges whose distance exceeds maximum distance
            # new_w = ssg.W
            # x, y, _ = sp.find(ssg.W > max_distance)
            # new_w[x, y] = 0
            # new_G = gsp_graphs.Graph(new_w)

            connected = ssg.is_connected()

        coords = dict(enumerate(ssg.coords))
        e_dist = compute_euclidian_matrix(ssg.A, ssg.coords, sparse_return=True)
        nxGraph = nx.from_scipy_sparse_matrix(e_dist, edge_attribute=DISTANCE_WEIGHT_NAME)
        for n, p in coords.items():
            nxGraph.node[n]['pos'] = p

        return nxGraph

    def _add_vin_graphs(self, graph, starts, labels):

        # next_path = path[i:]
        graphs = []
        end = 0  # TODO: For GVIN consistency only. Change this soon !!!

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
            # digraph.add_edges_from(digraph.edges, distance=1.0)  # For completeness
            graphs.append(digraph)

        return graphs

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
        # i = rand.choice(len(node_pairs), p=probabilities)
        # start, end = node_pairs[i]

        # TODO: For GVIN consistency only. Change this soon !!!
        end_0 = np.where(np.array(node_pairs)[:, 1] == end_idx)[0]
        i = rand.randint(len(end_0))
        start, end = node_pairs[end_0[i]]

        path = nx.shortest_path(
            graph, source=start, target=end, weight=DISTANCE_WEIGHT_NAME)

        return path
