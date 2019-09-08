# -*- coding: utf-8 -*-

"""
    graph_plotter.py
    
    Created on  : August 13, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

import collections
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graph_nets import utils_np

from lib.misc import *


def softmax_prob_last_dim(x):  # pylint: disable=redefined-outer-name
    e = np.exp(x)
    return e[:, -1] / np.sum(e, axis=-1)


def zero_one_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def draw_sols(draw_maps, num_processing_steps_ge,
              data_path="./results", fig_name="solution",
              max_graphs_to_plot=3, num_steps_to_plot=4):
    raw_graphs, outputs = draw_maps
    node_size = 20.0  # 10  # default: 120
    min_c = 0.05
    num_graphs = len(raw_graphs)
    # targets = utils_np.graphs_tuple_to_data_dicts(values["target"])
    step_indices = np.floor(
        np.linspace(0, num_processing_steps_ge - 1,
                    num_steps_to_plot)).astype(int).tolist()
    # outputs = list(
    #     zip(*(utils_np.graphs_tuple_to_data_dicts(values["outputs"][i])
    #           for i in step_indices)))
    h = min(num_graphs, max_graphs_to_plot)
    outputs = [[o[i] for i in step_indices] for o in outputs[:h]]

    w = num_steps_to_plot + 1
    fig = plt.figure(101, figsize=(18, h * 3))
    fig.clf()
    ncs = []

    for j, (graph, output) in enumerate(zip(raw_graphs, outputs)):
        if j >= h:
            break
        pos = get_node_dict(graph, "pos")

        # Prediction.
        for k, outp in enumerate(output):
            iax = j * (1 + num_steps_to_plot) + 2 + k
            ax = fig.add_subplot(h, w, iax)
            plotter = GraphPlotter(ax, graph, pos)
            color = {}
            non_end_nodes = []
            outp_nodes = []
            for n in graph.nodes:
                if not graph.nodes[n].get("end"):
                    non_end_nodes.append(n)
                    outp_nodes.append(outp["nodes"][n])

            # prob = softmax_prob_last_dim(outp["nodes"])
            # plotter.draw_graph_with_solution(node_size=node_size, node_color=color)

            prob = np.squeeze(zero_one_normalize(np.array(outp_nodes)))
            for i, n in enumerate(non_end_nodes):
                color[n] = prob[i]
            plotter.draw_graph_value(non_end_nodes, node_size=node_size, node_color=color)
            ax.set_title("Model-predicted\nStep {:02d} / {:02d}".format(
                step_indices[k] + 1, step_indices[-1] + 1))

    fig.savefig('%s/%s.png' % (data_path, fig_name), dpi=100, bbox_inches='tight', transparent=False)


class GraphPlotter(object):

    def __init__(self, ax, graph, pos):
        self._ax = ax
        self._graph = graph
        self._pos = pos
        self._base_draw_kwargs = dict(G=self._graph, pos=self._pos, ax=self._ax)
        self._solution_length = None
        self._nodes = None
        self._edges = None
        self._start_nodes = None
        self._end_nodes = None
        self._solution_nodes = None
        self._obstacle_nodes = None
        self._intermediate_solution_nodes = None
        self._solution_edges = None
        self._non_solution_nodes = None
        self._non_solution_edges = None
        self._ax.set_axis_off()

    @property
    def solution_length(self):
        if self._solution_length is None:
            self._solution_length = len(self._solution_edges)
        return self._solution_length

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = self._graph.nodes()
        return self._nodes

    @property
    def edges(self):
        if self._edges is None:
            self._edges = self._graph.edges()
        return self._edges

    @property
    def start_nodes(self):
        if self._start_nodes is None:
            self._start_nodes = [
                n for n in self.nodes if self._graph.node[n].get("start", False)
            ]
        return self._start_nodes

    @property
    def end_nodes(self):
        if self._end_nodes is None:
            self._end_nodes = [
                n for n in self.nodes if self._graph.node[n].get("end", False)
            ]
        return self._end_nodes

    @property
    def solution_nodes(self):
        if self._solution_nodes is None:
            self._solution_nodes = [
                n for n in self.nodes if self._graph.node[n].get("solution", False)
            ]
        return self._solution_nodes

    @property
    def obstacle_nodes(self):
        if self._obstacle_nodes is None:
            self._obstacle_nodes = [
                n for n in self.nodes if self._graph.node[n].get("obstacle", False)
            ]
        return self._obstacle_nodes

    @property
    def intermediate_solution_nodes(self):
        if self._intermediate_solution_nodes is None:
            self._intermediate_solution_nodes = [
                n for n in self.nodes
                if self._graph.node[n].get("solution", False) and
                   not self._graph.node[n].get("start", False) and
                   not self._graph.node[n].get("end", False)
            ]
        return self._intermediate_solution_nodes

    @property
    def solution_edges(self):
        if self._solution_edges is None:
            self._solution_edges = [
                e for e in self.edges
                if self._graph.get_edge_data(e[0], e[1]).get("solution", False)
            ]
        return self._solution_edges

    @property
    def non_solution_nodes(self):
        if self._non_solution_nodes is None:
            self._non_solution_nodes = [
                n for n in self.nodes
                if not self._graph.node[n].get("solution", False)
            ]
        return self._non_solution_nodes

    @property
    def non_solution_edges(self):
        if self._non_solution_edges is None:
            self._non_solution_edges = [
                e for e in self.edges
                if not self._graph.get_edge_data(e[0], e[1]).get("solution", False)
            ]
        return self._non_solution_edges

    def _make_draw_kwargs(self, **kwargs):
        kwargs.update(self._base_draw_kwargs)
        return kwargs

    def _draw(self, draw_function, zorder=None, **kwargs):
        draw_kwargs = self._make_draw_kwargs(**kwargs)
        collection = draw_function(**draw_kwargs)
        if collection is not None and zorder is not None:
            try:
                # This is for compatibility with older matplotlib.
                collection.set_zorder(zorder)
            except AttributeError:
                # This is for compatibility with newer matplotlib.
                collection[0].set_zorder(zorder)
        return collection

    def draw_nodes(self, **kwargs):
        """Useful kwargs: nodelist, node_size, node_color, linewidths."""
        if ("node_color" in kwargs and
                isinstance(kwargs["node_color"], collections.Sequence) and
                len(kwargs["node_color"]) in {3, 4} and
                not isinstance(kwargs["node_color"][0],
                               (collections.Sequence, np.ndarray))):
            num_nodes = len(kwargs.get("nodelist", self.nodes))
            kwargs["node_color"] = np.tile(
                np.array(kwargs["node_color"])[None], [num_nodes, 1])
        return self._draw(nx.draw_networkx_nodes, **kwargs)

    def draw_edges(self, **kwargs):
        """Useful kwargs: edgelist, width."""
        return self._draw(nx.draw_networkx_edges, **kwargs)

    def draw_graph(self,
                   node_size=200,
                   node_color=(0.4, 0.8, 0.4),
                   node_linewidth=1.0,
                   edge_width=1.0):
        # Plot nodes.
        self.draw_nodes(
            nodelist=self.nodes,
            node_size=node_size,
            node_color=node_color,
            linewidths=node_linewidth,
            zorder=20)
        # Plot edges.
        self.draw_edges(edgelist=self.edges, width=edge_width, arrows=False, zorder=10)

    def draw_graph_with_solution(self,
                                 node_size=200,
                                 node_color=(0.4, 0.8, 0.4),
                                 node_linewidth=1.0,
                                 edge_width=1.0,
                                 start_color="w",
                                 end_color="k",
                                 solution_node_linewidth=3.0,
                                 solution_edge_width=3.0):
        node_border_color = (0.0, 0.0, 0.0, 1.0)
        node_collections = {}
        # Plot start nodes.
        node_collections["start nodes"] = self.draw_nodes(
            nodelist=self.start_nodes,
            node_size=node_size,
            node_color=start_color,
            linewidths=solution_node_linewidth,
            edgecolors=node_border_color,
            zorder=100)
        # Plot end nodes.
        node_collections["end nodes"] = self.draw_nodes(
            nodelist=self.end_nodes,
            node_size=node_size,
            node_color=end_color,
            linewidths=solution_node_linewidth,
            edgecolors=node_border_color,
            zorder=90)
        node_collections["obstacle_nodes"] = self.draw_nodes(
            nodelist=self.obstacle_nodes,
            node_size=node_size,
            node_color="green",
            linewidths=solution_node_linewidth,
            edgecolors=node_border_color,
            zorder=90)
        # Plot intermediate solution nodes.
        if isinstance(node_color, dict):
            c = [node_color[n] for n in self.intermediate_solution_nodes]
        else:
            c = node_color
        node_collections["intermediate solution nodes"] = self.draw_nodes(
            nodelist=self.intermediate_solution_nodes,
            node_size=node_size,
            node_color=c,
            linewidths=solution_node_linewidth,
            edgecolors=node_border_color,
            zorder=80)
        # Plot solution edges.
        node_collections["solution edges"] = self.draw_edges(
            edgelist=self.solution_edges, width=solution_edge_width, arrows=False, zorder=70)

        # Plot non-solution nodes.
        if isinstance(node_color, dict):
            c = [node_color[n] for n in self.non_solution_nodes]
        else:
            c = node_color
        node_collections["non-solution nodes"] = self.draw_nodes(
            nodelist=self.non_solution_nodes,
            node_size=node_size,
            node_color=c,
            linewidths=node_linewidth,
            edgecolors=node_border_color,
            zorder=20)
        # Plot non-solution edges.
        node_collections["non-solution edges"] = self.draw_edges(
            edgelist=self.non_solution_edges, width=edge_width, arrows=False, zorder=10)
        # Set title as solution length.
        self._ax.set_title("Solution length: {}".format(self.solution_length))
        return node_collections

    def draw_graph_value(self,
                         non_end_nodes,
                         node_size=200,
                         node_color=(0.4, 0.8, 0.4),
                         node_linewidth=1.0,
                         edge_width=1.0,
                         start_color="w",
                         end_color="k",
                         solution_node_linewidth=3.0,
                         solution_edge_width=3.0):

        node_border_color = (0.0, 0.0, 0.0, 1.0)
        # Plot start nodes.
        # self.draw_nodes(
        #     nodelist=self.start_nodes,
        #     node_size=node_size,
        #     node_color=start_color,
        #     linewidths=solution_node_linewidth,
        #     edgecolors=node_border_color,
        #     zorder=100)
        # Plot end nodes.
        self.draw_nodes(
            nodelist=self.end_nodes,
            node_size=node_size,
            node_color=end_color,
            linewidths=solution_node_linewidth,
            edgecolors=node_border_color,
            zorder=90)

        if isinstance(node_color, dict):
            c = [node_color[n] for n in non_end_nodes]
        else:
            c = node_color

        self.draw_nodes(
            nodelist=non_end_nodes,
            node_size=node_size,
            node_color=c,
            linewidths=node_linewidth,
            cmap="viridis",
            zorder=80)

        # Plot edges.
        self.draw_edges(edgelist=self.edges, width=edge_width, arrows=False, zorder=10)

        return
