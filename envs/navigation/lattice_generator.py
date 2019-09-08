# -*- coding: utf-8 -*-

"""
    lattice_generator.py
    
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
from envs.expert_generator import ExpertGenerator


class LatticeGenerator(ExpertGenerator):
    def __init__(self, mode="offline", npz_file=None, test_set=False, full_grid=False, preload_pkl=None, **kwargs):

        super().__init__(**kwargs)

        self.test_set = test_set
        self.mode = mode

        assert self.mode in ["online", "offline"]

        self.preload_pkl = preload_pkl
        if self.preload_pkl is not None and not os.path.exists(self.preload_pkl):
            os.makedirs(self.preload_pkl)

        self._use_coord = False
        self._path_idx = 0
        self.grid_dim = (0, 0)

        if self.mode == "offline":

            self.full_grid = full_grid
            self._X_out, self._S1_out, self._S2_out, self._Labels_out, self._Gs = \
                self._load_grids(npz_file, test_set)
            self.sample_graph = self._sample_graph(self._X_out)
            self.grid_dim = self._X_out.shape[2:]
            # self._Gs_arr = np.concatenate(self._Gs)

            self.max_mem = min(self.max_mem, len(self._Gs)) if self.max_mem > 0 else len(self._Gs)

        self.mem_size = self.mem_size if self.mem_size > 0 else self.max_mem

    def update_memory(self):

        preload_pkl = "%s/data_%d.pkl" % (self.preload_pkl, self._mem_idx + self.mem_size) \
            if self.preload_pkl else None

        if preload_pkl and os.path.isfile(preload_pkl):
            with open(preload_pkl, "rb") as file:
                self._data = pickle.load(file)
                print("Loaded %s " % preload_pkl)
        elif self.mode == "offline":
            idx = slice(self._mem_idx, self._mem_idx + self.mem_size)
            self._data, self._path_idx = self._convert_to_graphs(idx, self._path_idx)

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
            self._path_idx = 0

    def gen_sample_data(self, end_only=True, full_grid=False):
        input_graphs, raw_graphs, _, _ = self.__next__()
        self.reset_idx(all=True)

        return input_graphs, raw_graphs

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

    def _load_grids(self, npz_file, test_set=False):
        data = np.load(npz_file, allow_pickle=True)

        if not test_set:
            X_out, S1_out, S2_out, Labels_out, Gs = data["arr_0"], data["arr_1"], \
                                                    data["arr_2"], data["arr_3"], data["arr_4"]
        else:
            X_out, S1_out, S2_out, Labels_out, Gs = data["arr_5"], data["arr_6"], \
                                                    data["arr_7"], data["arr_8"], data["arr_9"]

        # G = Gs[slice(*range)] if range is not None else Gs
        return X_out, S1_out, S2_out, Labels_out, Gs

    def _convert_to_graphs(self, idx, next_gidx):
        X_out, S1_out, S2_out, Labels_out, Gs = self._X_out, self._S1_out, self._S2_out, self._Labels_out, self._Gs

        graphs, paths, gidxs = [], [], []

        for i, g_idx in enumerate(Gs[idx]):
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

            graph_map = nx.convert_node_labels_to_integers(graph_map).to_directed()

            path_idx = next_gidx

            end_idx = [pos_dict[(x, y)] for x, y in zip(*np.where(X_out[path_idx][1] > 0))][0]

            # Add relative positions
            for n in graph_map.nodes():
                graph_map.nodes[n]["rel_pos"] = tuple(np.subtract(graph_map.nodes[n]["pos"],
                                                                  graph_map.nodes[end_idx]["pos"]))
            graph_map.add_node(end_idx, end=True)
            graph_map.add_nodes_from(set_diff(graph_map.nodes(), [end_idx]), end=False)
            graph_map.add_edges_from(graph_map.edges, distance=1.0)
            graphs.append(graph_map)

            for j in g_idx:
                # Creates a directed graph, to store the directed path from start to end.

                # start = pos_dict[(S1_out[path_idx][0], S2_out[path_idx][0])]
                # end = [pos_dict[(x, y)] for x, y in zip(*np.where(X_out[path_idx][1] > 0))][0]

                # Plus end
                path = (S1_out[path_idx:path_idx + j], S2_out[path_idx:path_idx + j])
                path = [pos_dict[(x[0], y[0])] for x, y in zip(*path)]
                path.append(end_idx)

                paths.append(path)
                gidxs.append(i)

                path_idx += j

            next_gidx += int(np.sum(g_idx))

        return (graphs, paths, gidxs), next_gidx
