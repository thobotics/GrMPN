# -*- coding: utf-8 -*-

"""
    expert_generator.py
    
    Created on  : August 11, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

from envs.navigation.nav_generator import graph_to_input


class ExpertGenerator(object):

    def __init__(self, rand, rand_batch=True, max_mem=0, mem_size=0, batch_size=10):
        self.rand = rand
        self.rand_batch = rand_batch
        self.batch_size = batch_size
        self.max_mem = max_mem
        self.mem_size = mem_size

        self.current_idx = 0
        self._data = None
        self._use_coord = False
        self._mem_idx = 0
        self._stop_flag = False

    def __iter__(self):
        return self

    def __next__(self):

        if self._stop_flag == 2:
            raise StopIteration()

        if self._data is None:
            self.update_memory()

        graphs, paths, gidxs = self._data
        total_paths = len(paths)  # number of graphs (included rolled out trajectories)

        if self.rand_batch:
            start = self.rand.randint(0, (total_paths - self.batch_size + 1))
            idx = slice(start, start + self.batch_size)
            self.current_idx += self.batch_size
        else:
            idx = slice(self.current_idx, self.current_idx + self.batch_size)
            self.current_idx += self.batch_size

        """ Post process """
        inputs_data = []
        graphs_data = []
        gidxs_data = []
        paths_data = paths[idx]
        last_gid = -1
        offset = gidxs[idx][0]
        for gid in gidxs[idx]:
            gidxs_data.append(gid - offset)

            if not (gid - offset) == last_gid:
                cur_graph = graphs[gid]
                graphs_data.append(cur_graph)
                inputs_data.append(graph_to_input(cur_graph, coord=self._use_coord))
                last_gid = gid - offset

        data = inputs_data, graphs_data, paths_data, gidxs_data

        if (self.current_idx + self.batch_size) > total_paths:
            if self._stop_flag == 1:
                # Increase stopper level
                self._stop_flag = 2
            else:
                self.update_memory()

        return data

    def update_memory(self, **kwargs):
        pass

    def reset_idx(self, idx=0):
        self.current_idx = idx

    def gen_sample_data(self, **kwargs):
        raise NotImplementedError("Must override gen_sample_data")
