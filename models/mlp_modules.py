# -*- coding: utf-8 -*-

"""
    attention_modules.py
    
    Created on  : July 03, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import blocks
import sonnet as snt
import tensorflow as tf
from graph_nets.modules import _make_default_node_block_opt, _make_default_edge_block_opt, \
    _make_default_global_block_opt


class GraphNetwork(snt.AbstractModule):
    """Implementation of a Graph Network.

    See https://arxiv.org/abs/1806.01261 for more details.
    """

    def __init__(self,
                 edge_model_fn,
                 node_model_fn,
                 global_model_fn,
                 reducer=tf.unsorted_segment_sum,
                 edge_block_opt=None,
                 node_block_opt=None,
                 global_block_opt=None,
                 name="graph_network"):
        """Initializes the GraphNetwork module.

        Args:
          edge_model_fn: A callable that will be passed to EdgeBlock to perform
            per-edge computations. The callable must return a Sonnet module (or
            equivalent; see EdgeBlock for details).
          node_model_fn: A callable that will be passed to NodeBlock to perform
            per-node computations. The callable must return a Sonnet module (or
            equivalent; see NodeBlock for details).
          global_model_fn: A callable that will be passed to GlobalBlock to perform
            per-global computations. The callable must return a Sonnet module (or
            equivalent; see GlobalBlock for details).
          reducer: Reducer to be used by NodeBlock and GlobalBlock to aggregate
            nodes and edges. Defaults to tf.unsorted_segment_sum. This will be
            overridden by the reducers specified in `node_block_opt` and
            `global_block_opt`, if any.
          edge_block_opt: Additional options to be passed to the EdgeBlock. Can
            contain keys `use_edges`, `use_receiver_nodes`, `use_sender_nodes`,
            `use_globals`. By default, these are all True.
          node_block_opt: Additional options to be passed to the NodeBlock. Can
            contain the keys `use_received_edges`, `use_sent_edges`, `use_nodes`,
            `use_globals` (all set to True by default), and
            `received_edges_reducer`, `sent_edges_reducer` (default to `reducer`).
          global_block_opt: Additional options to be passed to the GlobalBlock. Can
            contain the keys `use_edges`, `use_nodes`, `use_globals` (all set to
            True by default), and `edges_reducer`, `nodes_reducer` (defaults to
            `reducer`).
          name: The module name.
        """
        super(GraphNetwork, self).__init__(name=name)
        edge_block_opt = _make_default_edge_block_opt(edge_block_opt)
        node_block_opt = _make_default_node_block_opt(node_block_opt, reducer)
        global_block_opt = _make_default_global_block_opt(global_block_opt, reducer)

        # edge_block_opt["use_edges"] = True
        # edge_block_opt["use_globals"] = False
        # node_block_opt["use_nodes"] = True
        # node_block_opt["use_globals"] = False
        global_block_opt["use_edges"] = False
        # global_block_opt["use_globals"] = False

        with self._enter_variable_scope():
            self._edge_block = blocks.EdgeBlock(
                edge_model_fn=edge_model_fn, **edge_block_opt)
            self._node_block = blocks.NodeBlock(
                node_model_fn=node_model_fn, **node_block_opt)
            self._global_block = blocks.GlobalBlock(
                global_model_fn=global_model_fn, **global_block_opt)

    def _build(self, graph):
        """Connects the GraphNetwork.

        Args:
          graph: A `graphs.GraphsTuple` containing `Tensor`s. Depending on the block
            options, `graph` may contain `None` fields; but with the default
            configuration, no `None` field is allowed. Moreover, when using the
            default configuration, the features of each nodes, edges and globals of
            `graph` should be concatenable on the last dimension.

        Returns:
          An output `graphs.GraphsTuple` with updated edges, nodes and globals.
        """
        return self._global_block(self._node_block(self._edge_block(graph)))
