import sys

from lib.gppn import mechanism

sys.path.append('.')
from converter.gridworlds.domains.gridworld import *

sys.path.remove('.')


def extract_action(traj):
    # Given a trajectory, outputs a 1D vector of
    #  actions corresponding to the trajectory.
    n_actions = 8
    action_vecs = np.asarray([[-1., 0.], [1., 0.], [0., 1.], [0., -1.],
                              [-1., 1.], [-1., -1.], [1., 1.], [1., -1.]])
    action_vecs[4:] = 1 / np.sqrt(2) * action_vecs[4:]
    action_vecs = action_vecs.T
    state_diff = np.diff(traj, axis=0)
    norm_state_diff = state_diff * np.tile(
        1 / np.sqrt(np.sum(np.square(state_diff), axis=1)), (2, 1)).T
    prj_state_diff = np.dot(norm_state_diff, action_vecs)
    actions_one_hot = np.abs(prj_state_diff - 1) < 0.00001
    actions = np.dot(actions_one_hot, np.arange(n_actions).T)
    return actions


# def make_data(dom_size, n_domains, max_obs, max_obs_size, n_traj,
#               state_batch_size):
def make_data(mecha, mazes, goals, opt_pols, dom_size, n_traj,
              state_batch_size):
    X_l = []
    S1_l = []
    S2_l = []
    Labels_l = []
    Gs = []

    dom = 0.0
    n_domains = len(mazes)
    # while dom <= n_domains:
    for maze, goal_map, op in zip(mazes, goals, opt_pols):
        # goal = [np.random.randint(dom_size[0]), np.random.randint(dom_size[1])]
        # # Generate obstacle map
        # obs = obstacles([dom_size[0], dom_size[1]], goal, max_obs_size)
        # # Add obstacles to map
        # n_obs = obs.add_n_rand_obs(max_obs)
        # # Add border to map
        # border_res = obs.add_border()
        # # Ensure we have valid map
        # if n_obs == 0 or not border_res:
        #     continue

        # Get final map
        # im = obs.get_final()
        # # Generate gridworld from obstacle map
        # G = gridworld(im, goal[0], goal[1])
        # # Get value prior
        # value_prior = G.t_get_reward_prior()
        # # Sample random trajectories to our goal
        # states_xy, states_one_hot = sample_trajectory(G, n_traj)

        # TODO: DiffDrive mechanism
        goal = list(map(lambda x: x[0][0], zip(np.where(goal_map > 0))))
        goal = goal[1:]  # MOORE
        im = maze
        # Generate gridworld from obstacle map
        G = gridworld(im, goal[0], goal[1])
        # Get value prior
        value_prior = G.t_get_reward_prior()
        states_xy = moore_samples(mecha, maze, goal, op, n_traj)
        Gss = []

        for i in range(n_traj):
            if len(states_xy[i]) > 1:
                # Get optimal actions for each state
                actions = extract_action(states_xy[i])
                ns = states_xy[i].shape[0] - 1
                Gss.append(ns)
                # Invert domain image => 0 = free, 1 = obstacle
                image = 1 - im
                # Resize domain and goal images and concate
                image_data = np.resize(image, (1, 1, dom_size[0], dom_size[1]))
                value_data = np.resize(value_prior,
                                       (1, 1, dom_size[0], dom_size[1]))
                iv_mixed = np.concatenate((image_data, value_data), axis=1)
                X_current = np.tile(iv_mixed, (ns, 1, 1, 1))
                # Resize states
                S1_current = np.expand_dims(states_xy[i][0:ns, 0], axis=1)
                S2_current = np.expand_dims(states_xy[i][0:ns, 1], axis=1)
                # Resize labels
                Labels_current = np.expand_dims(actions, axis=1)
                # Append to output list
                X_l.append(X_current)
                S1_l.append(S1_current)
                S2_l.append(S2_current)
                Labels_l.append(Labels_current)
        dom += 1
        sys.stdout.write("\r" + str(int((dom / n_domains) * 100)) + "%")
        sys.stdout.flush()

        Gs.append(Gss)

    sys.stdout.write("\n")
    # Concat all outputs
    X_f = np.concatenate(X_l)
    S1_f = np.concatenate(S1_l)
    S2_f = np.concatenate(S2_l)
    Labels_f = np.concatenate(Labels_l)
    # Grids = np.concatenate(Gs)
    return X_f, S1_f, S2_f, Labels_f, Gs


def maze_process(filename, dataset_type):
    """
    Data format: list, [train data, test data]
    """
    with np.load(filename) as f:
        dataset2idx = {"train": 0, "valid": 3, "test": 6}
        idx = dataset2idx[dataset_type]
        mazes = f["arr_" + str(idx)]
        goal_maps = f["arr_" + str(idx + 1)]
        opt_policies = f["arr_" + str(idx + 2)]

    # Set proper datatypes
    mazes = mazes.astype(np.float32)
    goal_maps = goal_maps.astype(np.float32)
    opt_policies = opt_policies.astype(np.float32)

    # Print number of samples
    if dataset_type == "train":
        print("Number of Train Samples: {0}".format(mazes.shape[0]))
    elif dataset_type == "valid":
        print("Number of Validation Samples: {0}".format(mazes.shape[0]))
    else:
        print("Number of Test Samples: {0}".format(mazes.shape[0]))
    print("\tSize: {}x{}".format(mazes.shape[1], mazes.shape[2]))
    return mazes, goal_maps, opt_policies


def moore_samples(mecha, map_design, goal, op, n_traj):
    # mecha.print_policy(map_design, [0] + goal, np.argmax(op, axis=0))

    N = map_design.shape[0]
    op = np.argmax(op, axis=0)[0]

    init_states = []
    while len(init_states) < n_traj:
        x = np.random.randint(N)
        y = np.random.randint(N)
        if not [y, x] == goal and map_design[y, x] > 0.:
            init_states.append((y, x))

    trajs = []
    for y, x in init_states:
        traj = [[y, x]]
        while not [y, x] == goal:
            neigh = mecha.neighbors_func(map_design, 1, y, x)
            ns = neigh[op[y][x]][1:]  # TODO: DiffDrive different
            traj.append(ns)
            y, x = ns

        trajs.append(np.array(traj))

    return trajs


# 12x12, 20, 1.0
# 16x16, 40, 1.0
# 28x28, 50, 2.0
# 40x40, 70, 2.0
def main(maze_path="gppn_maze/mazes_40_test.npz",
         n_traj=7,
         state_batch_size=1):
    dataset_types = ["train", "valid", "test"]

    # Read file
    mecha = mechanism.Moore()
    mazes, goal_maps, opt_policies = maze_process(maze_path, dataset_type="train")
    dom_size = mazes.shape[1:]

    # Get path to save dataset
    save_path = "../dataset/gridworld_gppn_test_{0}x{1}".format(dom_size[0], dom_size[1])
    # Get training data
    print("Now making training data...")
    X_out_tr, S1_out_tr, S2_out_tr, Labels_out_tr, Gs_tr = make_data(
        mecha, mazes, goal_maps, opt_policies, dom_size, n_traj, state_batch_size)
    # Get testing data
    print("\nNow making  testing data...")
    mazes, goal_maps, opt_policies = maze_process(maze_path, dataset_type="test")
    X_out_ts, S1_out_ts, S2_out_ts, Labels_out_ts, Gs_ts = make_data(
        mecha, mazes, goal_maps, opt_policies, dom_size, n_traj, state_batch_size)

    next_idx = 0
    for i, p in enumerate(Gs_tr):
        fig = plt.figure()
        plt.imshow(X_out_tr[next_idx][0])
        fig.savefig('grid_%d.png' % i)
        next_idx += sum(p)
        if i > 10:
            break

    # Save dataset
    np.savez_compressed(save_path,
                        X_out_tr, S1_out_tr, S2_out_tr, Labels_out_tr, Gs_tr,
                        X_out_ts, S1_out_ts, S2_out_ts, Labels_out_ts, Gs_ts)


if __name__ == '__main__':
    main()
