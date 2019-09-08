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

import numpy as np
import tensorflow as tf

from models.experiments import Experiments
from envs.navigation.navigation import Navigation
from envs.navigation.irregular_generator import IrregularGenerator
from envs.navigation.irregular_city_generator import IrregularCityGenerator

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('input_train', './data/irregular_graph/irregular_10.pkl', 'Path to training data')
    flags.DEFINE_string('input_test', './data/irregular_graph/irregular_100.pkl', 'Path to testing data')
    flags.DEFINE_string('model', 'GAT', 'Model type: GN, GAT')
    flags.DEFINE_integer('n_layers', 2, 'Number of layers')
    flags.DEFINE_integer('n_hidden', 16, 'Hidden units')
    flags.DEFINE_integer('n_heads', 1, 'Number of heads')
    flags.DEFINE_string('name', './results/irgraph', 'Saving or Restoring folder name')
    flags.DEFINE_string('out_file', './results/irgraph.txt', 'Final results')
    flags.DEFINE_integer('seed', 2, 'Seed')
    flags.DEFINE_boolean('is_train', True, 'Is training, false for testing')
    flags.DEFINE_integer('n_ptrain', 15, 'Numbers of processing for training')
    flags.DEFINE_integer('n_ptest', 15, 'Numbers of processing for testing')
    flags.DEFINE_integer('n_train', 20, 'Training iterations')
    flags.DEFINE_integer('batch_train', 60, 'Training batches (remark: one graph generate many trajectories)')
    flags.DEFINE_integer('batch_test', 30, 'Testing batches (remark: one graph generate many trajectories)')
    flags.DEFINE_integer('train_max_mem', 0, 'Maximum Memory for training')
    flags.DEFINE_integer('valid_max_mem', 100, 'Maximum Memory for valid')
    flags.DEFINE_integer('mem_size', 100, 'Memory for containing processed data')
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

    # Params
    kw_opt = {
        "n_layers": config.n_layers,
        "n_hidden": config.n_hidden,
        "n_heads": config.n_heads,
    }
    kw_domains = {
        "domain_type": "irregular_graph",
        "end_only": True,
        "max_path_length": 0.1
    }

    tr_min_max_nodes = (50, 51)
    te_min_max_nodes = (100, 101)

    domain = Navigation(**kw_domains)

    if config.is_train:
        train_generator = IrregularGenerator(rand=rand, mode="offline", file_name=config.input_train,
                                             mem_size=config.mem_size, rand_batch=False,
                                             # preload_pkl="./data/parsed_graph/train_%s" % os.path.basename(
                                             #     config.input_train),
                                             batch_size=batch_size_tr, test_set=False,
                                             max_mem=config.train_max_mem)
        valid_generator = IrregularGenerator(rand=rand, mode="offline", file_name=config.input_test,
                                             mem_size=config.mem_size, rand_batch=False,
                                             # preload_pkl="./data/parsed_graph/test_%s" % os.path.basename(
                                             #     config.input_test),
                                             batch_size=batch_size_ge, test_set=True,
                                             max_mem=config.valid_max_mem)

        experiments = Experiments(name=name, domain=domain,
                                  model_type=config.model, data_generator=train_generator,
                                  n_ptrain=num_processing_steps_tr, n_ptest=num_processing_steps_ge,
                                  kw_domains=kw_domains, kw_opt=kw_opt)
        print("===================== Training =========================")
        experiments.train(train_generator, valid_generator,
                          iteration=num_training_iterations, output=config.out_file, draw_n=3)

        experiments.evaluate_ir(valid_generator, output=config.out_file, draw_n=5)
    else:
        print("===================== Testing =========================")
        print("Model %s, Name %s, n_ptest %d" % (config.model, config.name, config.n_ptest))

        test_generator = IrregularGenerator(rand=rand, mode="offline", file_name=config.input_test,
                                            mem_size=config.mem_size, test_set=True,
                                            preload_pkl=None,
                                            rand_batch=False, batch_size=batch_size_ge,
                                            max_mem=config.valid_max_mem)

        # test_generator = IrregularGenerator(rand=rand, mode="online",  mem_size=config.mem_size,
        #                                     min_max_nodes=te_min_max_nodes, rand_batch=False,
        #                                     batch_size=batch_size_ge, max_mem=200, theta=250)

        experiments = Experiments(name=name, domain=domain,
                                  model_type=config.model, data_generator=test_generator,
                                  n_ptrain=num_processing_steps_tr, n_ptest=num_processing_steps_ge,
                                  kw_domains=kw_domains, kw_opt=kw_opt)

        experiments.evaluate_ir(test_generator, output=config.out_file, draw_n=5)
