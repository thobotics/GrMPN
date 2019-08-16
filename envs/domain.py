# -*- coding: utf-8 -*-

"""
    domain.py
    
    Created on  : August 11, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf


class Domain(object):

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def create_placeholders(self, data_generator):
        input_graphs, target_graphs, _ = data_generator.gen_sample_data(**self._kwargs)
        input_ph = utils_tf.placeholders_from_networkxs(input_graphs)
        target_ph = utils_tf.placeholders_from_networkxs(target_graphs)
        return input_ph, target_ph

    def create_loss_ops(self, target_ph, output_ops, npossible_ph, segment_ph, label_ph):
        pass

    def fetch_pred_data(self, data_generator, input_ph, target_ph):
        pass

    def fetch_data(self, data_generator, input_ph, target_ph, neigh_ph, segment_ph, label_ph):
        pass

    def compute_accuracy(self, target, output, path_idxs, neigh_idxs, use_nodes, use_edges):
        pass
