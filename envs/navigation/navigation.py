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

    def __init__(self, end_only=True):
        super().__init__(end_only=end_only)
        self.end_only = end_only

    def create_loss_ops(self, target_ph, output_ops, npossible_ph, segment_ph, label_ph):
        loss_ops = []
        # debug = []

        for output_op in output_ops:
            px = _unsorted_segment_softmax(
                data=tf.gather(output_op.nodes, npossible_ph),
                segment_ids=segment_ph,
                num_segments=tf.shape(output_op.n_node)[0])

            log_likelihood = -tf.log(tf.gather(px, label_ph))
            loss_ops.append(tf.reduce_mean(log_likelihood))

            # debug.append((px, tf.gather(px, label_ph), log_likelihood))

        return loss_ops  # , debug

    def fetch_pred_data(self, data_generator, input_ph, target_ph):
        inputs, targets, graphs, paths = data_generator.__next__()
        input_graphs = utils_np.networkxs_to_graphs_tuple(inputs)
        target_graphs = utils_np.networkxs_to_graphs_tuple(targets)

        feed_dict = {input_ph: input_graphs,
                     target_ph: target_graphs}

        return feed_dict, graphs

    def fetch_data(self, data_generator, input_ph, target_ph, neigh_ph, segment_ph, label_ph, full_grid=False):
        inputs, targets, graphs, paths = data_generator.__next__()
        inputs, targets = self._generate_nstep_graphs(inputs, targets, paths, full_grid)

        input_graphs = utils_np.networkxs_to_graphs_tuple(inputs)
        target_graphs = utils_np.networkxs_to_graphs_tuple(targets)

        # Create other variables to compute loss
        start_idxs = np.where(input_graphs.nodes[:, 0] == 1)[0]
        neigh_idxs = [input_graphs.receivers[np.where(input_graphs.senders == idx)[0]] for idx in start_idxs]
        segment_idxs = np.concatenate([[i] * len(ne_idx) for i, ne_idx in enumerate(neigh_idxs)])

        label_idxs = np.where(np.argmax(target_graphs.nodes, axis=-1) == 1)[0]
        offset = 0
        for i, ne_idx in enumerate(neigh_idxs):
            new_idx = np.where(ne_idx == label_idxs[i])[0][0]
            label_idxs[i] = new_idx + offset
            offset += len(ne_idx)

        if self.end_only:
            input_graphs = input_graphs.replace(nodes=input_graphs.nodes[:, 1][:, None])
        # input_graphs = input_graphs.replace(nodes=input_graphs.nodes[:, 1:])

        feed_dict = {input_ph: input_graphs,
                     target_ph: target_graphs,
                     neigh_ph: np.concatenate(neigh_idxs),
                     segment_ph: segment_idxs,
                     label_ph: label_idxs}

        return feed_dict, graphs, paths, neigh_idxs

    def compute_accuracy(self, target, output, path_idxs, neigh_idxs, use_nodes=True, use_edges=False):

        if not use_nodes and not use_edges:
            raise ValueError("Nodes or edges (or both) must be used")

        tdds = utils_np.graphs_tuple_to_data_dicts(target)
        odds = utils_np.graphs_tuple_to_data_dicts(output)
        cs = []
        ss = []
        diff = []
        offset = 0
        pid = 0
        next_path_id = 0
        for i, (td, od, ne) in enumerate(zip(tdds, odds, neigh_idxs)):

            """ Action predicted accuracy """
            xn = np.argmax(np.argmax(td["nodes"][ne - offset], axis=-1))
            yn = np.argmax(od["nodes"][ne - offset])
            c = []
            if use_nodes:
                c.append(xn == yn)
            c = np.array(c)

            """ Success rate & Path difference """
            if i == next_path_id:
                path = path_idxs[pid]
                p_len = len(path) - 1
                s_idx = path[0]

                # for j in range(p_len):  # Penalty 3 more step
                for j in range(p_len + 5):
                    ne_idx = od["receivers"][np.where(od["senders"] == s_idx)[0]]
                    tmp_nodes = np.zeros_like(od["nodes"]) - 9999.
                    tmp_nodes[ne_idx] = od["nodes"][ne_idx]
                    s_idx = np.argmax(tmp_nodes)

                    if s_idx == path[-1]:
                        break

                ss.append(1 if s_idx == path[-1] else 0)
                diff.append(j + 1 - p_len if s_idx == path[-1] else 0)

                next_path_id += p_len
                pid += 1

            # s = np.all(c)
            cs.append(c)
            # ss.append(s)
            offset += td["n_node"]

        correct = np.mean(np.concatenate(cs, axis=0))
        solved = sum(ss) / len(path_idxs)  # np.mean(np.stack(ss))
        difference = sum(diff) / (sum(ss) + 1e-3)
        return correct, solved, difference

    def _generate_nstep_graphs(self, inputs, targets, paths, full_grid):

        inp_feature = ("start", "end",)
        if full_grid:
            inp_feature += ("obstacle",)

        nstep_inputs = []
        nstep_targets = []
        for path, igraph, tgraph in zip(paths, inputs, targets):
            for i, start in enumerate(path[:-1]):
                input = igraph.to_directed()
                input.add_node(start, start=True)
                input.add_nodes_from(set_diff(input.nodes(), [start]), start=False)

                target = tgraph.to_directed()
                target.add_node(path[i + 1], next_step=True)
                target.add_nodes_from(set_diff(target.nodes(), [path[i + 1]]), next_step=False)

                for node_index, node_feature in input.nodes(data=True):
                    input.add_node(
                        node_index, features=create_feature(node_feature, inp_feature))

                for node_index, node_feature in target.nodes(data=True):
                    target_node = to_one_hot(
                        create_feature(node_feature, ("next_step",)).astype(int), 2)[0]
                    target.add_node(node_index, features=target_node)

                nstep_inputs.append(input)
                nstep_targets.append(target)

        return nstep_inputs, nstep_targets
