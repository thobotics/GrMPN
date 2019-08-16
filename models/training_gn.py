# -*- coding: utf-8 -*-

"""
    training_gn.py
    
    Created on  : August 11, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

import time
import math
import json
import tensorflow as tf
import numpy as np

from envs.navigation.graph_plotter import draw_sols
from lib.misc import make_all_runnable_in_session
from models.gmlp_models import EncodeProcessDecode
from models.gat_models import EncodeAttentionDecode


class TrainingGraphNetwork(object):

    def __init__(self, name, domain, data_generator, model_type="GN",
                 mem_refresh=20, n_ptrain=15, n_ptest=15, seed=0,
                 kw_domains=None, kw_opt=None):

        assert model_type in ["GN", "GAT", "GATED"]
        kw_domains = kw_domains if kw_domains else {}
        kw_opt = kw_opt if kw_opt else {}

        self.rand = np.random.RandomState(seed=seed)
        self.name = name
        self.mem_refresh = mem_refresh
        self.domain = domain
        self.n_ptrain = n_ptrain
        self.n_ptest = n_ptest

        """ Construct model """

        # Data.
        # Input and target placeholders.
        input_ph, target_ph = self.domain.create_placeholders(data_generator)
        self.npossible_ph = tf.placeholder(tf.int32, shape=(None,))
        self.segment_ph = tf.placeholder(tf.int32, shape=(None,))
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))

        # Connect the data to the model.
        # Instantiate the model.

        if model_type == "GN":
            self.model = EncodeProcessDecode(node_output_size=1)
        elif model_type == "GAT":
            self.model = EncodeAttentionDecode(node_output_size=1)

        self.output_ops_tr = self.model(input_ph, n_ptrain)
        self.output_ops_ge = self.model(input_ph, n_ptest)

        """ Construct loss """

        # Training loss.
        loss_ops_tr = self.domain.create_loss_ops(target_ph, self.output_ops_tr,
                                                  self.npossible_ph, self.segment_ph, self.label_ph)
        # Loss across processing steps.
        self.loss_op_tr = sum(loss_ops_tr) / n_ptrain
        # Test/generalization loss.
        loss_ops_ge = self.domain.create_loss_ops(target_ph, self.output_ops_ge,
                                                  self.npossible_ph, self.segment_ph, self.label_ph)
        self.loss_op_ge = loss_ops_ge[-1]  # Loss from final processing step.

        # Optimizer.
        learning_rate = 1e-3  # kw_opt["learning_rate"]
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.step_op = optimizer.minimize(self.loss_op_tr)

        # Lets an iterable of TF graphs be output from a session as NP graphs.
        self.input_ph, self.target_ph = make_all_runnable_in_session(input_ph, target_ph)

        """ Construct TF session """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=20)

    def save(self, name, iteration):
        self.saver.save(self.sess, "%s/model_i_%d.ckpt" % (name, iteration))

    def restore(self, path):
        self.saver.restore(self.sess, path)

    def train(self, train_generator, valid_generator, iteration=1000):

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

        print("# (iteration number), T (elapsed seconds), "
              "Ltr (training loss), Lge (test/generalization loss), "
              "Ctr (training fraction nodes/edges labeled correctly), "
              "Str (training fraction examples solved correctly), "
              "Cge (test/generalization fraction nodes/edges labeled correctly), "
              "Sge (test/generalization fraction examples solved correctly)")

        start_time = time.time()
        last_log_time = start_time

        for iteration in range(last_iteration, iteration):

            if ((iteration + 1) % self.mem_refresh) == 0:
                train_generator.update_memory()
                valid_generator.update_memory()

            feed_dict, graphs, tr_paths, tr_neigh_idxs = self.domain.fetch_data(train_generator,
                                                                                self.input_ph, self.target_ph,
                                                                                self.npossible_ph, self.segment_ph,
                                                                                self.label_ph)

            train_values = self.sess.run({
                "step": self.step_op,
                "target": self.target_ph,
                "loss": self.loss_op_tr,
                "outputs": self.output_ops_tr,
            }, feed_dict=feed_dict)

            the_time = time.time()
            elapsed_since_last_log = the_time - last_log_time
            if elapsed_since_last_log > log_every_seconds:
                last_log_time = the_time

                feed_dict, graphs, te_paths, te_neigh_idxs = self.domain.fetch_data(valid_generator, self.input_ph,
                                                                                    self.target_ph,
                                                                                    self.npossible_ph, self.segment_ph,
                                                                                    self.label_ph)
                test_values = self.sess.run({
                    "target": self.target_ph,
                    "loss": self.loss_op_ge,
                    "outputs": self.output_ops_ge
                }, feed_dict=feed_dict)

                correct_tr, solved_tr, diff_tr = self.domain.compute_accuracy(
                    train_values["target"], train_values["outputs"][-1], tr_paths, tr_neigh_idxs, use_edges=False)
                correct_ge, solved_ge, diff_ge = self.domain.compute_accuracy(
                    test_values["target"], test_values["outputs"][-1], te_paths, te_neigh_idxs, use_edges=False)

                elapsed = time.time() - start_time
                losses_tr.append(train_values["loss"])
                corrects_tr.append(correct_tr)
                solveds_tr.append(solved_tr)
                losses_ge.append(test_values["loss"])
                corrects_ge.append(correct_ge)
                solveds_ge.append(solved_ge)
                logged_iterations.append(iteration)

                print("# {:05d}, T {:.1f}, Ltr {:.4f}, Lge {:.4f}, Ctr {:.4f}, Str"
                      " {:.4f}, Dtr {:.4f}, Cge {:.4f}, Sge {:.4f}, Dge {:.4f}".format(
                    iteration, elapsed, train_values["loss"], test_values["loss"],
                    correct_tr, solved_tr, diff_tr, correct_ge, solved_ge, diff_ge))

            """ Drawing solutions """
            if (iteration % 100) == 0:
                self.save(self.name, iteration)

                """ Draw Train """
                feed_dict, graphs = self.domain.fetch_pred_data(train_generator, self.input_ph, self.target_ph)

                train_values = self.sess.run({
                    "target": self.target_ph,
                    "outputs": self.output_ops_tr,
                }, feed_dict=feed_dict)

                draw_sols(train_values, graphs, self.n_ptrain, data_path=self.name, fig_name="train_sol_%d" % iteration)

                """ Draw Test """
                feed_dict, graphs = self.domain.fetch_pred_data(valid_generator, self.input_ph, self.target_ph)

                test_values = self.sess.run({
                    "target": self.target_ph,
                    "outputs": self.output_ops_ge,
                }, feed_dict=feed_dict)

                draw_sols(test_values, graphs, self.n_ptest, data_path=self.name, fig_name="test_sol_%d" % iteration)

    def compute_final_accuracy(self, data_generator, restore_itrs, output):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        data_generator.update_memory()

        results = {}
        output_file = open(output, "a", buffering=1)
        output_file.write("Iteration,Correct,Success,Difference\n")

        for r_itr in restore_itrs:
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                self.restore("%s/model_i_%d.ckpt" % (self.name, r_itr))

                correct_all = 0
                solved_all = 0
                diff_all = 0
                n_batches = math.ceil(data_generator.max_mem / data_generator.batch_size)

                for iteration in range(0, data_generator.max_mem, data_generator.batch_size):
                    feed_dict, graphs, te_paths, te_neigh_idxs = self.domain.fetch_data(data_generator, self.input_ph,
                                                                                        self.target_ph,
                                                                                        self.npossible_ph,
                                                                                        self.segment_ph,
                                                                                        self.label_ph)
                    values = self.sess.run({
                        "target": self.target_ph,
                        "loss": self.loss_op_ge,
                        "outputs": self.output_ops_ge
                    }, feed_dict=feed_dict)

                    correct_ge, solved_ge, diff_ge = self.domain.compute_accuracy(
                        values["target"], values["outputs"][-1], te_paths, te_neigh_idxs, use_edges=False)

                    print(
                        "Iteration {:03d}, Cge {:.4f}, Sge {:.4f}, Dge {:.4f}".format(iteration, correct_ge, solved_ge,
                                                                                      diff_ge))

                    correct_all += correct_ge / n_batches
                    solved_all += solved_ge / n_batches
                    diff_all += diff_ge / n_batches

                results[r_itr] = {"correct_all": correct_all, "solved_all": solved_all, "diff_all": diff_all}

                print("=============== Test accuracy ===============")
                print(
                    "Model {:04d}: Cge {:.4f}, Sge {:.4f}, Dge {:.4f}".format(r_itr, correct_all, solved_all, diff_all))

                output_file.write("{:04d},{:.4f},{:.4f},{:.4f}\n".format(r_itr, correct_all, solved_all, diff_all))

        print("=============== Summary ===============")
        for k in results:
            print("Model {:04d}: Cge {:.4f}, Sge {:.4f}, Dge {:.4f}".format(k, results[k]["correct_all"],
                                                                            results[k]["solved_all"],
                                                                            results[k]["diff_all"]))
        output_file.close()

        # """ Draw Test """
        # data_generator.batch_size = 1
        # feed_dict, graphs = self.domain.fetch_pred_data(data_generator, self.input_ph, self.target_ph)

        # test_values = self.sess.run({
        #     "target": self.target_ph,
        #     "outputs": self.output_ops_ge,
        # }, feed_dict=feed_dict)

        # draw_sols(test_values, graphs, self.n_ptest, data_path="./results", fig_name=output.split("/")[-1],
        #     max_graphs_to_plot=1, num_steps_to_plot=5)

        return results
