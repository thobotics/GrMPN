# -*- coding: utf-8 -*-

"""
    experiments.py
    
    Created on  : August 11, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

import time
import math
import json
import tensorflow as tf
import numpy as np

from graph_nets import utils_np
from envs.navigation.graph_plotter import draw_sols
from lib.misc import make_all_runnable_in_session, map_equal, convert_to_policy, grid_from_nxgraph
from lib.gppn import mechanism
from lib.gppn.eval import calc_optimal_and_success
import models.gmlp_model as gmlp_model
import models.gat_model as gat_model


class Experiments(object):

    def __init__(self, name, domain, data_generator, model_type="GN",
                 n_ptrain=15, n_ptest=15, seed=0,
                 kw_domains=None, kw_opt=None):

        assert model_type in ["GN", "GAT"]
        kw_domains = kw_domains if kw_domains else {}
        kw_opt = kw_opt if kw_opt else {}

        self.rand = np.random.RandomState(seed=seed)
        self.name = name
        self.domain = domain
        self.n_ptrain = n_ptrain
        self.n_ptest = n_ptest

        """ Construct model """

        # Data.
        # Input and target placeholders.
        # input_ph, target_ph = self.domain.create_placeholders(data_generator)
        input_ph = self.domain.create_placeholders(data_generator)
        self.npossible_ph = tf.placeholder(tf.int32, shape=(None,))
        self.segment_ph = tf.placeholder(tf.int32, shape=(None,))
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))

        # Connect the data to the model.
        # Instantiate the model.

        if model_type == "GN":
            gmlp_model.LATENT_SIZE = kw_opt["n_hidden"]
            gmlp_model.NUM_LAYERS = kw_opt["n_layers"]
            self.model = gmlp_model.EncodeProcessDecode(node_output_size=1)
        elif model_type == "GAT":
            gat_model.LATENT_SIZE = kw_opt["n_hidden"]
            gat_model.NUM_LAYERS = kw_opt["n_layers"]
            gat_model.NUM_HEADS = kw_opt["n_heads"]
            self.model = gat_model.EncodeAttentionDecode(node_output_size=1)

        self.output_ops_tr = self.model(input_ph, n_ptrain)
        self.output_ops_ge = self.model(input_ph, n_ptest)

        """ Construct loss """

        # Training loss.
        loss_ops_tr = self.domain.create_loss_ops(self.output_ops_tr,
                                                  self.npossible_ph, self.segment_ph, self.label_ph)
        # Loss across processing steps.
        self.loss_op_tr = sum(loss_ops_tr) / n_ptrain
        # Test/generalization loss.
        loss_ops_ge = self.domain.create_loss_ops(self.output_ops_ge,
                                                  self.npossible_ph, self.segment_ph, self.label_ph)
        self.loss_op_ge = loss_ops_ge[-1]  # Loss from final processing step.

        # Optimizer.
        learning_rate = 1e-3  # kw_opt["learning_rate"]
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.step_op = optimizer.minimize(self.loss_op_tr)

        # Lets an iterable of TF graphs be output from a session as NP graphs.
        self.input_ph = make_all_runnable_in_session(input_ph)[0]
        # self.input_ph, self.target_ph = make_all_runnable_in_session(input_ph, target_ph)

        """ Construct TF session """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=20)

    def save(self, name, iteration, overwrite_best=False):
        if overwrite_best:
            self.saver.save(self.sess, "%s/model_final.ckpt" % name)
        else:
            self.saver.save(self.sess, "%s/model_i_%d.ckpt" % (name, iteration))

    def restore(self, sess, path):
        self.saver.restore(sess, path)

    def _get_graphs(self, data):
        inputs, graphs, paths, gidxs = data

        """ Find new graph from given trajectories """
        tr_ix = []
        tr_paths = []

        cur_ix = -1
        for i, gid in enumerate(gidxs):
            if not gid == cur_ix:
                tr_ix.append(i)
                cur_ix = gid
        tr_ix.append(len(gidxs))

        for j in range(len(tr_ix) - 1):
            tr_paths.append(paths[tr_ix[j]:tr_ix[j + 1]])

        return inputs, graphs, tr_paths

    def train(self, train_generator, valid_generator, iteration=1000, output="./results.csv", draw_n=0):

        last_iteration = 0
        logged_iterations = []
        losses_tr = []
        corrects_tr = []
        solveds_tr = []
        losses_ge = []
        corrects_ge = []
        solveds_ge = []

        # How much time between logging and printing the current results.
        log_every_seconds = 20

        # Best validation loss
        best_val_loss = 999.0

        print("# (iteration number), T (elapsed seconds), "
              "Ltr (training loss), Lge (test/generalization loss), "
              "Ctr (training fraction nodes/edges labeled correctly), "
              "Str (training fraction examples solved correctly), "
              "Cge (test/generalization fraction nodes/edges labeled correctly), "
              "Sge (test/generalization fraction examples solved correctly)")

        start_time = time.time()
        last_log_time = start_time

        output_file = open(output, "a", buffering=1)
        output_file.write("Epoch, Time, "
                          "TrainingLoss, TestLoss, "
                          "TrainCorrect, TrainSolved, "
                          "TestCorrect, TestSoved\n")

        # for iteration in range(last_iteration, iteration):
        for epoch in range(iteration):

            num_tr_batches = 0
            num_te_batches = 0

            tr_info = {'loss': 0., 'correct': 0., 'solved': 0., 'diff': 0.}
            va_info = {'loss': 0., 'correct': 0., 'solved': 0., 'diff': 0.}
            tr_draws = []
            va_draws = []

            for tr_data in train_generator:

                tr_inputs, tr_graphs, tr_paths = self._get_graphs(tr_data)

                """ Run predicted outputs """
                feed_dict, tr_neigh_idxs, tr_label_idxs = self.domain.fetch_inputs(tr_inputs, tr_paths,
                                                                                   self.input_ph,
                                                                                   self.npossible_ph, self.segment_ph,
                                                                                   self.label_ph)

                train_values = self.sess.run({
                    "step": self.step_op,
                    "loss": self.loss_op_tr,
                    "outputs": self.output_ops_tr,
                }, feed_dict=feed_dict)

                correct, solved, diff = self.domain.compute_accuracy(
                    train_values["outputs"][-1],
                    tr_graphs, tr_paths, tr_neigh_idxs, tr_label_idxs, use_edges=False)

                print(
                    "Iteration {:03d}, Graphs: {:02d}, Correct {:.4f}, Solved {:.4f}, Difference {:.4f}".
                        format(num_tr_batches, len(tr_graphs), correct, solved, diff))

                tr_info["loss"] += train_values["loss"]
                tr_info["correct"] += correct
                tr_info["solved"] += solved
                tr_info["diff"] += diff
                num_tr_batches += 1

                val_idx = 0
                for ix, g in enumerate(tr_graphs):
                    if len(tr_draws) < draw_n:
                        draw_values = [utils_np.graphs_tuple_to_data_dicts(train_values["outputs"][step])[ix]
                                       for step in range(self.n_ptrain)]
                        tr_draws.append([g, draw_values])
                    val_idx += len(tr_paths[ix])

            for va_data in valid_generator:

                te_inputs, te_graphs, te_paths = self._get_graphs(va_data)

                """ Run predicted outputs """
                feed_dict, te_neigh_idxs, te_label_idxs = self.domain.fetch_inputs(te_inputs, te_paths,
                                                                                   self.input_ph,
                                                                                   self.npossible_ph, self.segment_ph,
                                                                                   self.label_ph)

                test_values = self.sess.run({
                    "loss": self.loss_op_ge,
                    "outputs": self.output_ops_ge,
                }, feed_dict=feed_dict)

                correct, solved, diff = self.domain.compute_accuracy(
                    test_values["outputs"][-1],
                    te_graphs, te_paths, te_neigh_idxs, te_label_idxs, use_edges=False)

                print(
                    "Val Iteration {:03d}, Graphs: {:02d}, Correct {:.4f}, Solved {:.4f}, Difference {:.4f}".
                        format(num_te_batches, len(te_graphs), correct, solved, diff))

                va_info["loss"] += test_values["loss"]
                va_info["correct"] += correct
                va_info["solved"] += solved
                va_info["diff"] += diff
                num_te_batches += 1

                val_idx = 0
                for ix, g in enumerate(te_graphs):
                    if len(va_draws) < draw_n:
                        draw_values = [utils_np.graphs_tuple_to_data_dicts(test_values["outputs"][step])[ix]
                                       for step in range(self.n_ptest)]
                        va_draws.append([g, draw_values])
                    val_idx += len(te_paths[ix])

            train_generator.reset_idx(all=True)
            valid_generator.reset_idx(all=True)

            tr_info["loss"] /= num_tr_batches
            tr_info["correct"] /= num_tr_batches
            tr_info["solved"] /= num_tr_batches
            tr_info["diff"] /= num_tr_batches

            va_info["loss"] /= num_te_batches
            va_info["correct"] /= num_te_batches
            va_info["solved"] /= num_te_batches
            va_info["diff"] /= num_te_batches

            elapsed = time.time() - start_time

            losses_tr.append(tr_info["loss"])
            corrects_tr.append(tr_info["correct"])
            solveds_tr.append(tr_info["solved"])
            losses_ge.append(va_info["loss"])
            corrects_ge.append(va_info["correct"])
            solveds_ge.append(va_info["solved"])
            logged_iterations.append(epoch)

            is_best = ""
            self.save(self.name, epoch)
            if va_info["loss"] < best_val_loss:
                best_val_loss = va_info["loss"]
                is_best = "!"
                self.save(self.name, epoch, overwrite_best=True)

            print("# {:05d}, T {:.1f}, Ltr {:.4f}, Lge {:.4f}, Ctr {:.4f}, Str"
                  " {:.4f}, Dtr {:.4f}, Cge {:.4f}, Sge {:.4f}, Dge {:.4f}, Best {:s}".format(
                epoch + 1, elapsed, tr_info["loss"], va_info["loss"],
                tr_info["correct"], tr_info["solved"], tr_info["diff"],
                va_info["correct"], va_info["solved"], va_info["diff"], is_best))

            output_file.write("{:05d}, {:.1f}, {:.4f}, {:.4f}, {:.4f},"
                              " {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:s}\n".format(
                epoch + 1, elapsed, tr_info["loss"], va_info["loss"],
                tr_info["correct"], tr_info["solved"], tr_info["diff"],
                va_info["correct"], va_info["solved"], va_info["diff"], is_best))

            """ Drawing solutions """
            if draw_n > 0:
                tr_draws = tuple(map(list, zip(*tr_draws)))
                draw_sols(tr_draws, self.n_ptrain, data_path=self.name, fig_name="train_sol_%d" % (epoch + 1),
                          max_graphs_to_plot=draw_n, num_steps_to_plot=5)

                va_draws = tuple(map(list, zip(*va_draws)))
                draw_sols(va_draws, self.n_ptest, data_path=self.name, fig_name="test_sol_%d" % (epoch + 1),
                          max_graphs_to_plot=draw_n, num_steps_to_plot=5)

        output_file.close()

    def evaluate_ir(self, data_generator, output, restore_itrs=None, draw_n=0):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        results = {}
        r_itr = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            self.restore(sess, "%s/model_final.ckpt" % self.name)

            correct_all = 0
            solved_all = 0
            diff_all = 0
            n_batches = 0
            draw_maps = []

            for iteration, data in enumerate(data_generator):

                tr_inputs, tr_graphs, tr_paths = self._get_graphs(data)

                """ Run predicted outputs """
                feed_dict, tr_neigh_idxs, tr_label_idxs = self.domain.fetch_inputs(tr_inputs, tr_paths,
                                                                                   self.input_ph,
                                                                                   self.npossible_ph, self.segment_ph,
                                                                                   self.label_ph)

                values = sess.run({
                    "outputs": self.output_ops_ge,
                }, feed_dict=feed_dict)

                correct_ge, solved_ge, diff_ge = self.domain.compute_accuracy(
                    values["outputs"][-1],
                    tr_graphs, tr_paths, tr_neigh_idxs, tr_label_idxs, use_edges=False)

                print(
                    "Iteration {:03d}, Cge {:.4f}, Sge {:.4f}, Dge {:.4f}".format(iteration, correct_ge, solved_ge,
                                                                                  diff_ge))

                val_idx = 0
                for ix, g in enumerate(tr_graphs):
                    if len(draw_maps) < draw_n:
                        draw_values = [utils_np.graphs_tuple_to_data_dicts(values["outputs"][step])[ix]
                                       for step in range(self.n_ptest)]
                        draw_maps.append([g, draw_values])
                    val_idx += len(tr_paths[ix])

                correct_all += correct_ge
                solved_all += solved_ge
                diff_all += diff_ge
                n_batches += 1

            correct_all /= n_batches
            solved_all /= n_batches
            diff_all /= n_batches

            print("=============== Test accuracy ===============")
            print(
                "Model {:04d}: Cge {:.4f}, Sge {:.4f}, Dge {:.4f}".
                    format(r_itr, correct_all, solved_all, diff_all))

            # import pickle
            # with open("./viz_graphs/%s.pkl" % output.split("/")[-1], "wb") as f:
            #     pickle.dump(draw_maps, f)

            draw_maps = tuple(map(list, zip(*draw_maps)))
            draw_sols(draw_maps, self.n_ptest, data_path="./results", fig_name=output.split("/")[-1],
                      max_graphs_to_plot=draw_n, num_steps_to_plot=5)

    def evaluate_latice(self, data_generator, output, restore_itrs=None, draw_n=0):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        results = {}
        output_file = open(output, "a", buffering=1)
        output_file.write("Iteration,Correct,Success,Difference\n")

        best = False
        if restore_itrs is None:
            restore_itrs = [0]
            best = True

        for r_itr in restore_itrs:
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())

                if not best:
                    self.restore(sess, "%s/model_i_%d.ckpt" % (self.name, r_itr))
                else:
                    self.restore(sess, "%s/model_final.ckpt" % self.name)

                correct_all = 0
                solved_all = 0
                diff_all = 0
                n_batches = 0

                grid_maps = []
                draw_maps = []
                mecha = mechanism.Moore()
                num_graphs = 0
                optimal_all = 0.
                successful_all = 0.

                for iteration, data in enumerate(data_generator):

                    tr_inputs, tr_graphs, tr_paths = self._get_graphs(data)

                    """ Run predicted outputs """
                    feed_dict, tr_neigh_idxs, tr_label_idxs = self.domain.fetch_inputs(tr_inputs, tr_paths,
                                                                                       self.input_ph,
                                                                                       self.npossible_ph,
                                                                                       self.segment_ph,
                                                                                       self.label_ph)

                    values = sess.run({
                        "outputs": self.output_ops_ge,
                    }, feed_dict=feed_dict)

                    correct_ge, solved_ge, diff_ge = self.domain.compute_accuracy(
                        values["outputs"][-1],
                        tr_graphs, tr_paths, tr_neigh_idxs, tr_label_idxs, use_edges=False)

                    print(
                        "Iteration {:03d}, Cge {:.4f}, Sge {:.4f}, Dge {:.4f}".format(iteration, correct_ge, solved_ge,
                                                                                      diff_ge))

                    correct_all += correct_ge
                    solved_all += solved_ge
                    diff_all += diff_ge
                    n_batches += 1

                    val_idx = 0
                    for ix, g in enumerate(tr_graphs):

                        cur_grid = grid_from_nxgraph(g, data_generator.grid_dim)
                        cur_opt_value = utils_np.graphs_tuple_to_data_dicts(values["outputs"][-1])[ix]

                        if len(draw_maps) < draw_n:
                            draw_values = [utils_np.graphs_tuple_to_data_dicts(values["outputs"][step])[ix]
                                           for step in range(self.n_ptest)]
                            draw_maps.append([g, draw_values])

                        """ Compute metrics on optimal policy """
                        map_design, goal_map = cur_grid[0], cur_grid[1]
                        goal = np.where(goal_map > 0.)
                        goal = (0, goal[0][0], goal[1][0])

                        pred_policy = convert_to_policy(mecha, g, map_design, cur_opt_value)
                        percent_optimal, percent_successful = calc_optimal_and_success(mecha, map_design, goal,
                                                                                       pred_policy)

                        optimal_all += percent_optimal
                        successful_all += percent_successful
                        num_graphs += 1

                        val_idx += len(tr_paths[ix])

                correct_all /= n_batches
                solved_all /= n_batches
                diff_all /= n_batches

                optimal_all /= num_graphs
                successful_all /= num_graphs

                results[r_itr] = {"correct_all": correct_all, "solved_all": solved_all, "diff_all": diff_all}

                print("=============== Test accuracy ===============")
                print(
                    "Model {:04d}: Cge {:.4f}, Sge {:.4f}, Dge {:.4f}".
                        format(r_itr, correct_all, solved_all, diff_all))
                print("OP Final: {:.4f}, {:.4f}".format(optimal_all, successful_all))

                output_file.write("{:04d},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".
                                  format(r_itr, correct_all, solved_all, diff_all, optimal_all, successful_all))

        print("=============== Summary ===============")
        for k in results:
            print("Model {:04d}: Cge {:.4f}, Sge {:.4f}, Dge {:.4f}".format(k, results[k]["correct_all"],
                                                                            results[k]["solved_all"],
                                                                            results[k]["diff_all"]))
        output_file.close()

        """ Draw Test """
        draw_maps = tuple(map(list, zip(*draw_maps)))
        draw_sols(draw_maps, self.n_ptest, data_path="./results", fig_name=output.split("/")[-1],
                  max_graphs_to_plot=draw_n, num_steps_to_plot=5)

        return results
