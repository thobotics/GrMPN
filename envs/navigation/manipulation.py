# -*- coding: utf-8 -*-

"""
    manipulation.py
    
    Created on  : October 01, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

import tensorflow as tf
from graph_nets import utils_np
from scipy import sparse
from lib.gcn.gcn.utils import *
from lib.misc import *
from envs.navigation.navigation import Navigation


class Manipulation(Navigation):

    def __init__(self, embed_type, **kwargs):
        super().__init__(**kwargs)
        self._embed_type = embed_type

    def create_placeholders(self, data_generator):
        input_graphs, _ = data_generator.gen_sample_data(**self._kwargs)
        input_ph = utils_tf.placeholders_from_networkxs(input_graphs)
        input_emb_ph = utils_tf.placeholders_from_networkxs(input_graphs)
        input_emb_ph = input_emb_ph.replace(
            nodes=tf.placeholder(shape=(None, 1), name=input_emb_ph.nodes.op.name, dtype=input_emb_ph.nodes.dtype))

        return input_ph, input_emb_ph

    def fetch_inputs(self, inputs, list_paths, input_ph, input_embed_ph,
                     neigh_ph=None, segment_ph=None, label_ph=None, full_grid=False):

        results = super().fetch_inputs(inputs, list_paths, input_ph, input_embed_ph,
                                       neigh_ph, segment_ph, label_ph, full_grid)

        if self._embed_type == "MLP":
            return results

        feed_dict = results[0]

        kinematic_trees, kinematic_supports, kinematic_feats, segment_idxs = self.embed(inputs)

        if self._embed_type == "GN":

            input_emb = utils_np.networkxs_to_graphs_tuple(kinematic_trees)
            feed_dict[input_embed_ph] = input_emb

        elif self._embed_type == "GCN":

            st_feature = sparse_to_tuple(sparse.vstack(kinematic_feats))
            st_support = [sparse_to_tuple(sparse.block_diag(kinematic_supports))]

            feed_dict[input_embed_ph["segment_ph"]] = np.hstack(segment_idxs)
            feed_dict[input_embed_ph["features"]] = st_feature
            for i in range(len(st_support)):
                feed_dict[input_embed_ph["support"][i]] = st_support[i]
            feed_dict[input_embed_ph["dropout"]] = 0.0
            feed_dict[input_embed_ph['num_features_nonzero']] = st_feature[1].shape

        return results

    @staticmethod
    def embed(inputs):
        kinematic_trees = []
        kinematic_supports = []
        kinematic_feats = []
        segment_idxs = []
        n_count = 0

        for graph in inputs:
            for g_node in graph.nodes():
                features = graph.nodes[g_node]["features"][1:]

                kinematic_tree = nx.Graph()
                kinematic_tree.graph["features"] = np.array([0.0])
                for i, feat in enumerate(features):
                    kinematic_tree.add_node(i, features=np.array([feat]))

                kinematic_tree.add_edge(0, 1, features=np.array([1.0]))
                kinematic_tree.add_edge(1, 2, features=np.array([1.0]))
                kinematic_tree.add_edge(2, 3, features=np.array([1.0]))
                kinematic_tree.add_edge(3, 4, features=np.array([1.0]))
                kinematic_tree.add_edge(4, 5, features=np.array([1.0]))
                kinematic_tree.add_edge(5, 6, features=np.array([1.0]))

                kinematic_adj = nx.adjacency_matrix(kinematic_tree)
                kinematic_feat = sparse.coo_matrix(features.reshape((-1, 1)))

                # kinematic_feat = sparse_to_tuple(kinematic_feat)
                kinematic_support = preprocess_adj_sp(kinematic_adj)

                kinematic_trees.append(kinematic_tree.to_directed())
                kinematic_supports.append(kinematic_support)
                kinematic_feats.append(kinematic_feat)
                segment_idxs.append([n_count] * len(features))

                n_count += 1

        return kinematic_trees, kinematic_supports, kinematic_feats, segment_idxs
