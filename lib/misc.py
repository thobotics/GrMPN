# -*- coding: utf-8 -*-

"""
    misc.py
    
    Created on  : August 11, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

import itertools
import numpy as np
from graph_nets import utils_tf


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def set_diff(seq0, seq1):
    """Return the set difference between 2 sequences as a list."""
    return list(set(seq0) - set(seq1))


def to_one_hot(indices, max_value, axis=-1):
    one_hot = np.eye(max_value)[indices]
    if axis not in (-1, one_hot.ndim):
        one_hot = np.moveaxis(one_hot, -1, axis)
    return one_hot


def get_node_dict(graph, attr):
    """Return a `dict` of node:attribute pairs from a graph."""
    return {k: v[attr] for k, v in graph.node.items()}


def create_feature(attr, fields):
    return np.hstack([np.array(attr[field], dtype=float) for field in fields])


def make_all_runnable_in_session(*args):
    """Lets an iterable of TF graphs be output from a session as NP graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]
