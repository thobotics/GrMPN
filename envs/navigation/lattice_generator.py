# -*- coding: utf-8 -*-

"""
    lattice_generator.py
    
    Created on  : August 13, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

import pickle
import numpy as np
import networkx as nx
from envs.navigation.nav_generator import graph_to_input_target
from lib.misc import *
from envs.expert_generator import ExpertGenerator


class LatticeGenerator(ExpertGenerator):
    def __init__(self, rand, mem_size=0, rand_batch=True,
                 mode="offline", npz_file=None, test_set=False, full_grid=False, range=None, **kwargs):

        super().__init__(**kwargs)

        self.rand = rand
        self.rand_batch = rand_batch
        self.test_set = test_set
        self.mode = mode

        assert self.mode in ["online", "offline"]

        self._data = None
        self._mem_idx = 0
        self._path_idx = 0  # For online

        if self.mode == "online":

            self.full_grid = full_grid
            self._X_out, self._S1_out, self._S2_out, self._Labels_out, self._Gs = \
                self._load_grids(npz_file, test_set, range)
            self.sample_graph = self._sample_graph(self._X_out)

            self.max_mem = len(self._Gs)

        else:
            self.data_path = "./data/lattice_navigation/parsed_graphs"

            self.max_mem = 800

        self.mem_size = mem_size if mem_size > 0 else self.max_mem

    def __next__(self):

        if self._data is None:
            self.update_memory()

        if self.rand_batch:
            start = self.rand.randint(0, (self.mem_size - self.batch_size + 1))
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
            gtype = "test" if self.test_set else "train"
            with open("%s/%s_grid_16x16_%d.pkl" % (self.data_path, gtype, self._mem_idx), "rb") as f:
                self._data = pickle.load(f)

            # Hard-coded for offline training
            self._mem_idx = (self._mem_idx + self.mem_size) % self.max_mem
        else:
            idx = slice(self._mem_idx, self._mem_idx + self.mem_size)
            self._data, self._path_idx = self._convert_to_graphs(idx, self._path_idx)
            self._mem_idx = (self._mem_idx + self.mem_size) % self.max_mem
            if self._mem_idx == 0 or (self._mem_idx + self.mem_size) > self.max_mem:  # Check next step
                self.reset_idx(all=True)

        self.reset_idx()

    def iter_call_back(self):
        pass

    def reset_idx(self, idx=0, all=False):
        super().reset_idx(idx)
        if all:
            self._mem_idx = 0
            self._path_idx = 0

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

    def _sample_graph(self, grid):
        """ Turn NEWS to MOORE"""
        dim_x, dim_y = grid.shape[2:]
        Go = nx.grid_2d_graph(dim_x, dim_y)
        G = Go.copy()
        for n in Go:
            x, y = n
            if x > 0 and y < dim_y - 1:
                G.add_edge(n, (x - 1, y + 1))
            if x < dim_x - 1 and y > 0:
                G.add_edge(n, (x + 1, y - 1))

            if x > 0 and y > 0:
                G.add_edge(n, (x - 1, y - 1))
            if x < dim_x - 1 and y < dim_y - 1:
                G.add_edge(n, (x + 1, y + 1))
        return G

    def _load_grids(self, npz_file, test_set=False, range=None):
        data = np.load(npz_file, allow_pickle=True)

        if not test_set:
            X_out, S1_out, S2_out, Labels_out, Gs = data["arr_0"], data["arr_1"], \
                                                    data["arr_2"], data["arr_3"], data["arr_4"]
        else:
            X_out, S1_out, S2_out, Labels_out, Gs = data["arr_5"], data["arr_6"], \
                                                    data["arr_7"], data["arr_8"], data["arr_9"]

        G = Gs[slice(*range)] if range is not None else Gs
        return X_out, S1_out, S2_out, Labels_out, G

    def _convert_to_graphs(self, idx, next_gidx):
        X_out, S1_out, S2_out, Labels_out, Gs = self._X_out, self._S1_out, self._S2_out, self._Labels_out, self._Gs

        input_graphs, target_graphs, graphs, paths = [], [], [], []

        for g_idx in Gs[idx]:
            graph_map = self.sample_graph.copy()
            obstacles = [(x, y) for x, y in zip(*np.where(X_out[next_gidx][0] == 1))]

            if not self.full_grid:
                graph_map.remove_nodes_from(obstacles)

            pos = []
            pos_dict = {}
            for k, n in enumerate(graph_map.nodes):
                graph_map.add_node(n, pos=n)
                pos.append(n)
                pos_dict[n] = k

            graph_map = nx.convert_node_labels_to_integers(graph_map)

            # fig = plt.figure()
            # plt.imshow(X_out_tr[next_gidx][0])
            # fig.savefig('grid.png')
            #
            # fig = plt.figure()
            # nx.draw(graph_map, pos, font_size=4)
            # fig.savefig('graph.png')
            #
            # break

            path_idx = next_gidx
            for j in g_idx:
                # Creates a directed graph, to store the directed path from start to end.
                digraph = graph_map.to_directed()

                start = pos_dict[(S1_out[path_idx][0], S2_out[path_idx][0])]
                end = [pos_dict[(x, y)] for x, y in zip(*np.where(X_out[path_idx][1] > 0))][0]

                # Plus end
                path = (S1_out[path_idx:path_idx + j], S2_out[path_idx:path_idx + j])
                path = [pos_dict[(x[0], y[0])] for x, y in zip(*path)]
                path.append(end)

                # # Add the "start", "end", and "solution" attributes to the nodes and edges.
                digraph.add_node(start, start=True)
                digraph.add_node(end, end=True)
                digraph.add_nodes_from(set_diff(digraph.nodes(), [start]), start=False)
                digraph.add_nodes_from(set_diff(digraph.nodes(), [end]), end=False)
                digraph.add_nodes_from(set_diff(digraph.nodes(), path), solution=False)
                digraph.add_nodes_from(path, solution=True)
                if self.full_grid:
                    obs = [pos_dict[(x, y)] for x, y in obstacles]
                    digraph.add_nodes_from(obs, obstacle=True)
                    digraph.add_nodes_from(set_diff(digraph.nodes(), obs), obstacle=False)
                path_edges = list(pairwise(path))
                digraph.add_edges_from(set_diff(digraph.edges(), path_edges), solution=False)
                digraph.add_edges_from(path_edges, solution=True)
                digraph.add_edges_from(digraph.edges, distance=1.0)

                input_graph, target_graph = graph_to_input_target(digraph, full_sol=True)
                input_graphs.append(input_graph)
                target_graphs.append(target_graph)
                graphs.append(digraph)
                paths.append(path)

                path_idx += j

            next_gidx += int(np.sum(g_idx))

        return (input_graphs, target_graphs, graphs, paths), next_gidx
