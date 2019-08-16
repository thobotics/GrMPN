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
from graph_nets.modules import _unsorted_segment_softmax, _received_edges_normalizer


def _sent_edges_normalizer(graph,
                           normalizer,
                           name="received_edges_normalizer"):
    """Performs elementwise normalization for all received edges by a given node.

    Args:
      graph: A graph containing edge information.
      normalizer: A normalizer function following the signature of
        `modules._unsorted_segment_softmax`.
      name: A name for the operation (optional).

    Returns:
      A tensor with the resulting normalized edges.

    """
    with tf.name_scope(name):
        return normalizer(
            data=graph.edges,
            segment_ids=graph.senders,
            num_segments=tf.reduce_sum(graph.n_node))


class VIntention(snt.AbstractModule):
    """Multi-head self-attention module.

    The module is based on the following three papers:
     * A simple neural network module for relational reasoning (RNs):
         https://arxiv.org/abs/1706.01427
     * Non-local Neural Networks: https://arxiv.org/abs/1711.07971.
     * Attention Is All You Need (AIAYN): https://arxiv.org/abs/1706.03762.

    The input to the modules consists of a graph containing values for each node
    and connectivity between them, a tensor containing keys for each node
    and a tensor containing queries for each node.

    The self-attention step consist of updating the node values, with each new
    node value computed in a two step process:
    - Computing the attention weights between each node and all of its senders
     nodes, by calculating sum(sender_key*receiver_query) and using the softmax
     operation on all attention weights for each node.
    - For each receiver node, compute the new node value as the weighted average
     of the values of the sender nodes, according to the attention weights.
    - Nodes with no received edges, get an updated value of 0.

    Values, keys and queries contain a "head" axis to compute independent
    self-attention for each of the heads.

    """

    def __init__(self, edge_model_fn, name="value_iteration_attention"):
        """Inits the module.

        Args:
          name: The module name.
        """
        super(VIntention, self).__init__(name=name)
        self._normalizer = _unsorted_segment_softmax
        self._edge_model = edge_model_fn

    def _build(self, node_values, attention_weights_logits, attention_graph):
        """Connects the multi-head self-attention module.

        The self-attention is only computed according to the connectivity of the
        input graphs, with receiver nodes attending to sender nodes.

        Args:
          node_values: Tensor containing the values associated to each of the nodes.
            The expected shape is [total_num_nodes, num_heads, key_size].
          node_keys: Tensor containing the key associated to each of the nodes. The
            expected shape is [total_num_nodes, num_heads, key_size].
          node_queries: Tensor containing the query associated to each of the nodes.
            The expected shape is [total_num_nodes, num_heads, query_size]. The
            query size must be equal to the key size.
          attention_graph: Graph containing connectivity information between nodes
            via the senders and receivers fields. Node A will only attempt to attend
            to Node B if `attention_graph` contains an edge sent by Node A and
            received by Node B.

        Returns:
          An output `graphs.GraphsTuple` with updated nodes containing the
          aggregated attended value for each of the nodes with shape
          [total_num_nodes, num_heads, value_size].

        Raises:
          ValueError: if the input graph does not have edges.
        """

        # TODO: Using edges features and max operator to fully form a Bellman update equation

        # [total_num_edges, num_heads, value_size]
        receiver_values = blocks.broadcast_receiver_nodes_to_edges(
            attention_graph.replace(nodes=node_values))

        # Attention weight for each edge.
        # [total_num_edges, num_heads]
        normalized_attention_weights = _sent_edges_normalizer(
            attention_graph.replace(edges=attention_weights_logits),
            normalizer=self._normalizer)

        edges_features = tf.tile(tf.expand_dims(attention_graph.edges, axis=1), [1, tf.shape(receiver_values)[1], 1])

        # # @ For testing: ATTENTION
        # # [total_num_edges, num_heads, key_size]
        # receiver_values = blocks.broadcast_sender_nodes_to_edges(
        #     attention_graph.replace(nodes=node_values))
        # # Attention weight for each edge.
        # # [total_num_edges, num_heads]
        # normalized_attention_weights = _received_edges_normalizer(
        #     attention_graph.replace(edges=attention_weights_logits),
        #     normalizer=self._normalizer)

        # Attending to sender values according to the weights.
        # [total_num_edges, num_heads, embedding_size]
        attented_edges = (receiver_values + edges_features) * normalized_attention_weights[..., None]

        # Summing all of the attended values from each node.
        # [total_num_nodes, num_heads, embedding_size]
        received_edges_aggregator = blocks.ReceivedEdgesToNodesAggregator(
            reducer=tf.unsorted_segment_sum)
        aggregated_attended_values = received_edges_aggregator(
            attention_graph.replace(edges=attented_edges))

        return attention_graph.replace(nodes=aggregated_attended_values)


class TAttention(snt.AbstractModule):
    """Multi-head self-attention module.

    The module is based on the following three papers:
     * A simple neural network module for relational reasoning (RNs):
         https://arxiv.org/abs/1706.01427
     * Non-local Neural Networks: https://arxiv.org/abs/1711.07971.
     * Attention Is All You Need (AIAYN): https://arxiv.org/abs/1706.03762.

    The input to the modules consists of a graph containing values for each node
    and connectivity between them, a tensor containing keys for each node
    and a tensor containing queries for each node.

    The self-attention step consist of updating the node values, with each new
    node value computed in a two step process:
    - Computing the attention weights between each node and all of its senders
     nodes, by calculating sum(sender_key*receiver_query) and using the softmax
     operation on all attention weights for each node.
    - For each receiver node, compute the new node value as the weighted average
     of the values of the sender nodes, according to the attention weights.
    - Nodes with no received edges, get an updated value of 0.

    Values, keys and queries contain a "head" axis to compute independent
    self-attention for each of the heads.

    """

    def __init__(self, edge_model_fn, name="dynamics_attention"):
        """Inits the module.

        Args:
          name: The module name.
        """
        super(TAttention, self).__init__(name=name)
        self._normalizer = _unsorted_segment_softmax
        self._edge_model = edge_model_fn

    def _build(self, node_values, node_keys, node_queries, attention_graph):
        """Connects the multi-head self-attention module.

        The self-attention is only computed according to the connectivity of the
        input graphs, with receiver nodes attending to sender nodes.

        Args:
          node_values: Tensor containing the values associated to each of the nodes.
            The expected shape is [total_num_nodes, num_heads, key_size].
          node_keys: Tensor containing the key associated to each of the nodes. The
            expected shape is [total_num_nodes, num_heads, key_size].
          node_queries: Tensor containing the query associated to each of the nodes.
            The expected shape is [total_num_nodes, num_heads, query_size]. The
            query size must be equal to the key size.
          attention_graph: Graph containing connectivity information between nodes
            via the senders and receivers fields. Node A will only attempt to attend
            to Node B if `attention_graph` contains an edge sent by Node A and
            received by Node B.

        Returns:
          An output `graphs.GraphsTuple` with updated nodes containing the
          aggregated attended value for each of the nodes with shape
          [total_num_nodes, num_heads, value_size].

        Raises:
          ValueError: if the input graph does not have edges.
        """

        # Sender nodes put their keys and values in the edges.
        # [total_num_edges, num_heads, query_size]
        sender_keys = blocks.broadcast_sender_nodes_to_edges(
            attention_graph.replace(nodes=node_keys))
        # [total_num_edges, num_heads, value_size]
        sender_values = blocks.broadcast_sender_nodes_to_edges(
            attention_graph.replace(nodes=node_values))

        # Receiver nodes put their queries in the edges.
        # [total_num_edges, num_heads, key_size]
        receiver_queries = blocks.broadcast_receiver_nodes_to_edges(
            attention_graph.replace(nodes=node_queries))

        # Attention weight for each edge.
        # [total_num_edges, num_heads]
        edges_features = tf.tile(tf.expand_dims(attention_graph.edges, axis=1), [1, tf.shape(sender_values)[1], 1])
        # edges_features = []
        # for i, edge_fn in enumerate(self._edge_model):
        #     edges_features.append(edge_fn(attention_graph.edges))
        # edges_features = tf.stack(edges_features, axis=1)

        attention_weights_logits = tf.reduce_sum(
            sender_keys * receiver_queries, axis=-1)
        # attention_weights_logits = []
        # for i, edge_fn in enumerate(self._edge_model):
        #     attention_weights_logits.append(tf.reduce_sum(
        #         edge_fn(
        #             tf.concat([sender_keys[:, i], receiver_queries[:, i], attention_graph.edges], axis=-1)
        #         ), axis=-1))
        # attention_weights_logits = tf.stack(attention_weights_logits, axis=1)
        normalized_attention_weights = _received_edges_normalizer(
            attention_graph.replace(edges=attention_weights_logits),
            normalizer=self._normalizer)

        # Attending to sender values according to the weights.
        # [total_num_edges, num_heads, embedding_size]
        attented_edges = (sender_values + edges_features) * normalized_attention_weights[..., None]

        # Summing all of the attended values from each node.
        # [total_num_nodes, num_heads, embedding_size]
        received_edges_aggregator = blocks.ReceivedEdgesToNodesAggregator(
            reducer=tf.unsorted_segment_sum)
        aggregated_attended_values = received_edges_aggregator(
            attention_graph.replace(edges=attented_edges))

        return attention_graph.replace(nodes=aggregated_attended_values), attention_weights_logits
        # return attention_graph.replace(nodes=aggregated_attended_values, edges=tf.reduce_mean(edges_features, axis=1)), attention_weights_logits
