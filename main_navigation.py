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

import numpy as np
import tensorflow as tf

from models.experiments import Experiments
from envs.navigation.navigation import Navigation
from envs.navigation.lattice_generator import LatticeGenerator

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('input_train', './data/lattice_navigation/gridworld_12x12.npz', 'Path to training data')
    flags.DEFINE_string('input_test', './data/lattice_navigation/gridworld_12x12.npz', 'Path to testing data')
    flags.DEFINE_string('model', 'GAT', 'Model type: GN, GAT')
    flags.DEFINE_integer('n_layers', 2, 'Number of layers')
    flags.DEFINE_integer('n_hidden', 16, 'Hidden units')
    flags.DEFINE_integer('n_heads', 1, 'Number of heads')
    flags.DEFINE_string('name', './results/grid', 'Saving or Restoring folder name')
    flags.DEFINE_string('out_file', './results/grid.txt', 'Final results')
    flags.DEFINE_integer('seed', 2, 'Seed')
    flags.DEFINE_boolean('is_train', True, 'Is training, false for testing')
    flags.DEFINE_integer('n_ptrain', 15, 'Numbers of processing for training')
    flags.DEFINE_integer('n_ptest', 15, 'Numbers of processing for testing')
    flags.DEFINE_integer('n_train', 20, 'Training iterations')
    flags.DEFINE_integer('batch_train', 10, 'Training batches (remark: one graph generate many trajectories)')
    flags.DEFINE_integer('batch_test', 10, 'Testing batches (remark: one graph generate many trajectories)')
    flags.DEFINE_integer('train_max_mem', 0, 'Maximum Memory for training')
    flags.DEFINE_integer('valid_max_mem', 100, 'Maximum Memory for valid')
    flags.DEFINE_integer('mem_size', 200, 'Memory for containing processed data')
    config = flags.FLAGS

    # assert (config.mem_size % config.batch_train == 0)
    # assert (config.mem_size % config.batch_test == 0)

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

    # Params
    kw_opt = {
        "n_layers": config.n_layers,
        "n_hidden": config.n_hidden,
        "n_heads": config.n_heads,
    }
    kw_domains = {
        "domain_type": "lattice",
        "end_only": True,
        "max_path_length": 3.0
    }

    tr_grid_file = config.input_train
    te_grid_file = config.input_test

    domain = Navigation(**kw_domains)

    if config.is_train:
        print("===================== Training =========================")
        train_generator = LatticeGenerator(rand=rand, mem_size=config.mem_size, rand_batch=False,
                                           mode="offline", npz_file=tr_grid_file, test_set=False,
                                           batch_size=batch_size_tr, max_mem=config.train_max_mem)
        valid_generator = LatticeGenerator(rand=rand, mem_size=config.mem_size, rand_batch=False,
                                           mode="offline", npz_file=te_grid_file, test_set=True,
                                           batch_size=batch_size_ge, max_mem=config.valid_max_mem)

        experiments = Experiments(name=name, domain=domain,
                                  model_type=config.model, data_generator=train_generator,
                                  n_ptrain=num_processing_steps_tr, n_ptest=num_processing_steps_ge,
                                  kw_domains=kw_domains, kw_opt=kw_opt)

        experiments.train(train_generator, valid_generator, iteration=num_training_iterations,
                          output=config.out_file, draw_n=5)

        experiments.evaluate_latice(valid_generator, output=config.out_file, draw_n=5)
    else:
        print("===================== Testing =========================")
        print("Model %s, Name %s, n_ptest %d" % (config.model, config.name, config.n_ptest))
        test_generator = LatticeGenerator(rand=rand, mem_size=config.mem_size, rand_batch=False,
                                          mode="offline", npz_file=te_grid_file, test_set=True,
                                          batch_size=config.batch_test, max_mem=config.valid_max_mem)

        experiments = Experiments(name=name, domain=domain,
                                  model_type=config.model, data_generator=test_generator,
                                  n_ptrain=num_processing_steps_tr, n_ptest=num_processing_steps_ge,
                                  kw_domains=kw_domains, kw_opt=kw_opt)

        experiments.evaluate_latice(test_generator, output=config.out_file, draw_n=5)
