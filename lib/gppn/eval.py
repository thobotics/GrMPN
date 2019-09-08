# -*- coding: utf-8 -*-

"""
    eval.py
    
    Created on  : August 17, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

import numpy as np
from lib.gppn.dijkstra import dijkstra_dist, dijkstra_policy


def calc_optimal_and_success(mechanism, map_design, goal, pred_pol):
    # Get current sample
    md = map_design  # [i][0]
    # gm = goal_map[i]
    # op = opt_policy[i]
    pp = pred_pol  # [i][0]
    # ll = labels[i][0]

    # Extract the goal in 2D coordinates
    # goal = extract_goal(gm)

    # Check how different the predicted policy is from the optimal one
    # in terms of path lengths
    pred_dist = dijkstra_policy(md, mechanism, goal, pp)
    opt_dist = dijkstra_dist(md, mechanism, goal)
    diff_dist = pred_dist - opt_dist

    wall_dist = np.min(pred_dist)  # impossible distance

    for o in range(mechanism.num_orient):
        # Refill the walls in the difference with the impossible distance
        diff_dist[o] += (1 - md) * wall_dist

        # Mask out the walls in the prediction distances
        pred_dist[o] = pred_dist[o] - np.multiply(1 - md, pred_dist[o])

    num_open = md.sum() * mechanism.num_orient  # number of reachable locations
    return (diff_dist == 0).sum() / num_open, 1. - (
            pred_dist == wall_dist).sum() / num_open
