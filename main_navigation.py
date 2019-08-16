# -*- coding: utf-8 -*-

"""
    main_navigation.py

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
from envs.navigation.lattice_generator import LatticeGenerator

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('input_train', './data/lattice_navigation/gridworld_12x12.npz', 'Path to training data')
    flags.DEFINE_string('input_test', './data/lattice_navigation/gridworld_12x12.npz', 'Path to testing data')
    flags.DEFINE_string('model', 'GAT', 'Model type: GN, GAT, GATED')
    flags.DEFINE_string('name', './results/grid', 'Saving or Restoring folder name')
    flags.DEFINE_integer('seed', 2, 'Seed')
    flags.DEFINE_boolean('is_train', True, 'Is training, false for testing')
    flags.DEFINE_string('test_file', './results/grid_test.txt', 'Final test results')
    flags.DEFINE_integer('n_ptrain', 15, 'Numbers of processing for training')
    flags.DEFINE_integer('n_ptest', 15, 'Numbers of processing for testing')
    flags.DEFINE_integer('n_train', 2000, 'Training iterations')
    flags.DEFINE_integer('batch_train', 10, 'Training batches (remark: one graph generate many trajectories)')
    flags.DEFINE_integer('batch_test', 10, 'Testing batches (remark: one graph generate many trajectories)')
    flags.DEFINE_integer('mem_refresh', 20, 'Refresh memory each #mem_refresh iterations')
    config = flags.FLAGS

    tf.reset_default_graph()

    name = config.name
    seed = config.seed
    rand = np.random.RandomState(seed=seed)

    # Model parameters.
    # Number of processing (message-passing) steps.
    num_processing_steps_tr = config.n_ptrain
    num_processing_steps_ge = config.n_ptest

    # Data / training parameters.
    num_training_iterations = config.n_train
    batch_size_tr = config.batch_train
    batch_size_ge = config.batch_test

    mem_refresh = config.mem_refresh
    mem_size_tr = batch_size_tr * mem_refresh
    mem_size_te = batch_size_ge * mem_refresh

    # Params
    kw_domains = {"end_only": True}

    tr_grid_file = config.input_train
    te_grid_file = config.input_test
    train_generator = LatticeGenerator(rand, mem_size=mem_size_tr,
                                       mode="online", npz_file=tr_grid_file, test_set=False,
                                       batch_size=batch_size_tr, range=(0, 100))
    valid_generator = LatticeGenerator(rand, mem_size=mem_size_te,
                                       mode="online", npz_file=te_grid_file, test_set=True,
                                       batch_size=batch_size_ge, range=(0, 100))
    domain = Navigation(**kw_domains)

    training = TrainingGraphNetwork(name=name, domain=domain, model_type=config.model,
                                    data_generator=train_generator, mem_refresh=mem_refresh,
                                    n_ptrain=num_processing_steps_tr, n_ptest=num_processing_steps_ge,
                                    kw_domains=kw_domains)

    if config.is_train:
        print("===================== Training =========================")
        training.train(train_generator, valid_generator, num_training_iterations)
    else:
        print("===================== Testing =========================")
        print("Model %s, Name %s, n_ptest %d" % (config.model, config.name, config.n_ptest))
        test_generator = LatticeGenerator(rand, mem_size=0, rand_batch=False,
                                          mode="online", npz_file=te_grid_file, test_set=True,
                                          batch_size=config.batch_test, range=(0, 10))  # (0, 300)

        training.compute_final_accuracy(test_generator, restore_itrs=range(900, 1000, 100), output=config.test_file)
