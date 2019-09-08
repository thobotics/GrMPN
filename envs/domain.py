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

    def __init__(self, domain_type, **kwargs):
        self._domain_type = domain_type
        self._kwargs = kwargs

    def create_placeholders(self, data_generator):
        input_graphs, _ = data_generator.gen_sample_data(**self._kwargs)
        input_ph = utils_tf.placeholders_from_networkxs(input_graphs)

        return input_ph

    def create_loss_ops(self, output_ops, npossible_ph, segment_ph, label_ph):
        pass

    def fetch_inputs(self, inputs, list_paths, input_ph, neigh_ph, segment_ph, label_ph):
        pass

    def compute_accuracy(self, target, output, path_idxs, neigh_idxs, use_nodes, use_edges):
        pass
