# -*- coding: utf-8 -*-

"""
    navigation.py
    
    Created on  : August 11, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
import networkx as nx
import numpy as np
from scipy import spatial
import tensorflow as tf
from graph_nets.modules import _unsorted_segment_softmax

from lib.misc import *
from envs.domain import Domain


class Navigation(Domain):

    def __init__(self, domain_type, end_only=True, max_path_length=1.0):
        super().__init__(domain_type=domain_type, end_only=end_only)
        self.end_only = end_only
        self.max_path_length = max_path_length

    def create_loss_ops(self, output_ops, npossible_ph, segment_ph, label_ph):
        loss_ops = []
        debug = []

        for output_op in output_ops:
            px = _unsorted_segment_softmax(
                data=tf.gather(output_op.nodes, npossible_ph),
                segment_ids=segment_ph,
                num_segments=tf.shape(label_ph)[0])

            log_likelihood = -tf.log(tf.gather(px, label_ph))
            loss_ops.append(tf.reduce_mean(log_likelihood))

            # debug.append((
            #     tf.gather(output_op.nodes, npossible_ph),
            #     px, tf.gather(px, label_ph), log_likelihood))

        return loss_ops

    def fetch_inputs(self, inputs, list_paths, input_ph, neigh_ph=None, segment_ph=None, label_ph=None,
                     full_grid=False):
        input_graphs = utils_np.networkxs_to_graphs_tuple(inputs)

        start_idxs = []
        nstep_idxs = []

        offset = 0
        for i, paths in enumerate(list_paths):
            for path in paths:
                for j, start in enumerate(path[:-1]):
                    cur = start + offset
                    next = path[j + 1] + offset
                    start_idxs.append(cur)
                    nstep_idxs.append(next)

            offset += input_graphs.n_node[i]

        neigh_idxs = [input_graphs.receivers[np.where(input_graphs.senders == idx)[0]] for idx in start_idxs]
        segment_idxs = np.concatenate([[i] * len(ne_idx) for i, ne_idx in enumerate(neigh_idxs)])

        label_idxs = []
        cnt = 0
        for ne, label in zip(neigh_idxs, nstep_idxs):
            for n in ne:
                if n == label:
                    label_idxs.append(cnt)
                cnt += 1

        input_graphs = input_graphs.replace(edges=np.ones_like(input_graphs.edges))
        feed_dict = {input_ph: input_graphs,
                     neigh_ph: np.concatenate(neigh_idxs),
                     segment_ph: segment_idxs,
                     label_ph: label_idxs}

        return feed_dict, neigh_idxs, label_idxs

    def compute_accuracy(self, output, graphs, path_idxs, neigh_idxs, label_idxs, use_nodes=True, use_edges=False):

        if not use_nodes and not use_edges:
            raise ValueError("Nodes or edges (or both) must be used")

        # tdds = utils_np.graphs_tuple_to_data_dicts(target)
        odds = utils_np.graphs_tuple_to_data_dicts(output)
        cs = []
        ss = []
        diff = []
        offset = 0
        label_offset = 0
        n_offset = 0
        pid = 0
        next_path_id = 0
        od = odds[0]

        for i, ne in enumerate(neigh_idxs):

            if i > 0 and i == next_path_id:
                od = odds[pid]
                offset = n_offset

            """ Action predicted accuracy """
            yn = label_idxs[i] - label_offset
            xn = np.argmax(od["nodes"][ne - offset])
            label_offset += len(ne)
            c = []
            if use_nodes:
                c.append(xn == yn)
            c = np.array(c)

            """ Success rate & Path difference """
            if i == next_path_id:
                original_path = np.array(path_idxs[pid])
                graph = graphs[pid]
                goal = None
                for n in graph.nodes:
                    if graph.nodes[n]["end"]:
                        goal = n
                        break
                val = od["nodes"][:, 0]
                edge_info = dict(graph.edges())

                # for start, path in zip(original_path[:, 0], original_path):
                for start in graph.nodes():
                    try:
                        path = nx.shortest_path(
                            graph, source=start, target=goal, weight="distance")
                    except nx.NetworkXNoPath:
                        continue

                    p_len = len(path) - 1
                    s_idx = path[0]
                    e_idx = path[-1]

                    """ Optimal len """
                    o_len = 0
                    for l in range(len(path) - 1):
                        o_len += edge_info[(path[l], path[l + 1])]['distance']

                    path_run = []
                    pred_len = 0
                    l = 0

                    while (pred_len - o_len) < self.max_path_length:

                        path_run.append(s_idx)
                        if s_idx == e_idx:
                            break

                        neighs = [(ne, val[ne]) for ne in graph.neighbors(s_idx)]
                        a, b = zip(*neighs)
                        s_idx = a[np.argmax(b)]

                        # if len(b) > 1:
                        #     s_idx = a[np.argmax(b)]
                        #     if len(path_run) > 2 and s_idx == path_run[-2]:
                        #         sorted_action = np.argsort(-np.array(b))
                        #         tmp_a = np.array(a)[sorted_action[1:]]
                        #
                        #         if len(tmp_a) > 1:
                        #             p = softmax(sorted_action[1:])
                        #             s_idx = np.random.choice(tmp_a, 1, p=p)[0]
                        #         else:
                        #             s_idx = tmp_a[0]
                        # else:
                        #     s_idx = a[0]

                        pred_len += edge_info[(path_run[l], s_idx)]['distance']
                        l += 1

                    ss.append(1 if s_idx == e_idx else 0)
                    diff.append(pred_len - o_len)
                    # diff.append(len(o_path) - len(path) if s_idx == e_idx else 0)

                n_offset += od["n_node"]
                next_path_id += sum([len(p) - 1 for p in path_idxs[pid]])
                pid += 1

            # s = np.all(c)
            cs.append(c)
            # ss.append(s)

        correct = np.mean(np.concatenate(cs, axis=0))
        solved = sum(ss) / len(ss)  # len(path_idxs)  # np.mean(np.stack(ss))
        difference = sum(diff) / (sum(ss) + 1e-9)
        return correct, solved, difference
