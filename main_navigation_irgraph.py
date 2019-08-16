# -*- coding: utf-8 -*-

"""
    main_navigation_irgraph.py
    
    Created on  : August 11, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import time

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
import numpy as np
from scipy import spatial
import tensorflow as tf
import pickle as pkl

from models.training_gn import TrainingGraphNetwork
from envs.navigation.navigation import Navigation
from envs.navigation.irregular_generator import IrregularGenerator

if __name__ == '__main__':
    tf.reset_default_graph()

    seed = 2
    rand = np.random.RandomState(seed=seed)

    # Model parameters.
    # Number of processing (message-passing) steps.
    num_processing_steps_tr = 15
    num_processing_steps_ge = 15

    # Data / training parameters.
    num_training_iterations = 2000
    batch_size_tr = 10
    batch_size_ge = 10

    mem_refresh = 20
    mem_size_tr = batch_size_tr * mem_refresh
    mem_size_te = batch_size_ge * mem_refresh

    name = "irregular_tr_16_33"

    # Params
    kw_domains = {"end_only": True}

    tr_min_max_nodes = (16, 33)
    te_min_max_nodes = (80, 100)
    train_generator = IrregularGenerator(rand, mem_size=mem_size_tr,
                                         min_max_nodes=tr_min_max_nodes,
                                         batch_size=batch_size_tr)
    valid_generator = IrregularGenerator(rand, mem_size=mem_size_te,
                                         min_max_nodes=te_min_max_nodes,
                                         batch_size=batch_size_ge)
    domain = Navigation(**kw_domains)

    training = TrainingGraphNetwork(name=name, domain=domain,
                                    data_generator=train_generator, mem_refresh=mem_refresh,
                                    n_ptrain=num_processing_steps_tr, n_ptest=num_processing_steps_ge,
                                    kw_domains=kw_domains)
    training.train(train_generator, valid_generator, num_training_iterations)
