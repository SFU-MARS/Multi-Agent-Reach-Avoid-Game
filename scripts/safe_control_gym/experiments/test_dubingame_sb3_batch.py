import os
import math
import time
import argparse
from typing import Callable
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

from odp.Grid import Grid
import matplotlib.pyplot as plt
from safe_control_gym.utils.plotting import animation_dub, current_status_check, record_video,  plot_values_dub, plot_value_1vs1_dub
from safe_control_gym.envs.gym_game.DubinGame import DubinReachAvoidEasierGame


from stable_baselines3 import PPO


# Step 0 initilize the map 
map = {'map': [-1., 1., -1., 1.]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
des = {'goal0': [0.6, 0.8, 0.1, 0.3]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
obstacles = {'obs1': [100, 100, 100, 100]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
# Step 1 load the value function, initilize the grids
value1vs0 = np.load('safe_control_gym/envs/gym_game/values/1vs0Dubin_easier.npy')
value1vs1 = np.load('safe_control_gym/envs/gym_game/values/1vs1Dubin_easier.npy')
grid1vs0 = Grid(np.array([-1.1, -1.1, -math.pi]), np.array([1.1, 1.1, math.pi]), 3, np.array([100, 100, 200]), [2])
grid1vs1 = Grid(np.array([-1.1, -1.1, -math.pi, -1.1, -1.1, -math.pi]), np.array([1.1, 1.1, math.pi, 1.1, 1.1, math.pi]), 
                        6, np.array([28, 28, 28, 28, 28, 28]), [2, 5])


def check_area(state, area):
    """Check if the state is inside the area.

    Parameters:
        state (np.ndarray): the state to check
        area (dict): the area dictionary to be checked.
    
    Returns:
        bool: True if the state is inside the area, False otherwise.
    """
    x, y, theta = state  # Unpack the state assuming it's a 2D coordinate

    for bounds in area.values():
        x_lower, x_upper, y_lower, y_upper = bounds
        if x_lower <= x <= x_upper and y_lower <= y <= y_upper:
            return True

    return False


def getAttackersStatus(attackers, defenders, last_status):
        """Returns the current status of all attackers.

        Returns
            ndarray, shape (num_attackers,)

        """
        num_attackers = attackers.shape[0]
        num_defenders = defenders.shape[0]
        new_status = np.zeros(num_attackers)
        current_attacker_state = attackers
        current_defender_state = defenders

        for num in range(num_attackers):
            if last_status[num]:  # attacker has arrived or been captured
                new_status[num] = last_status[num]
            else: # attacker is free last time
                # check if the attacker arrive at the des this time
                if check_area(current_attacker_state[num], des):
                    new_status[num] = 1
                # # check if the attacker gets stuck in the obstacles this time (it won't usually)
                # elif self._check_area(current_attacker_state[num], self.obstacles):
                #     new_status[num] = -1
                #     break
                else:
                    # check if the attacker is captured
                    for j in range(num_defenders):
                        if np.linalg.norm(current_attacker_state[num][:2] - current_defender_state[j][:2]) <= 0.30:
                            print(f"================ The {num} attacker is captured. ================ \n")
                            new_status[num] = -1
                            break
                        
            return new_status


def check_current_value_dub(attackers, defenders, value_function, grids):
    """ Check the value of the current state of the attackers and defenders.

    Args:
        attackers (np.ndarray): the attackers' states
        defenders (np.ndarray): the defenders' states
        value (np.ndarray): the value function for the game
        grids: the instance of the grid
    
    Returns:
        value (float): the value of the current state of the attackers and defenders
    """
    if len(value_function.shape) == 3:  # 1vs0 game
        joint_slice = grids.get_index(attackers[0])
        # joint_slice = po2slice1vs0_dub(attackers[0], value_function.shape[0])
    elif len(value_function.shape) == 6:  # 1vs1 game
        joint_slice = grids.get_index(np.concatenate((attackers[0], defenders[0])))
        # print(f"The joint slice is {joint_slice}.")
        # joint_slice = po2slice1vs1_dub(attackers[0], defenders[0], value_function.shape[0])
        
    value = value_function[joint_slice]

    return value
        

def test_dubin_sb3(init_type, total_steps):
    # Set up env hyperparameters.
    env_seed = 42  # 2024
    # Setp up algorithm hyperparameters.
    total_timesteps = total_steps
    test_seed = 2024
    T = 15.0  # time for the game
    ctrl_freq = 20 # control frequency

    # Load the trained model
    trained_path = os.path.join('training_results', f"dubin_game/sb3/{init_type}_init/", f'seed_{env_seed}', f'{total_timesteps}steps')
    trained_model = os.path.join(trained_path, 'final_model.zip')
    assert os.path.exists(trained_model), f"[ERROR] The trained model {trained_model} does not exist, please check the loading path or train one first."
    print(f"========== The trained model is loaded from {trained_model}. =========== \n")
    model = PPO.load(trained_model)


    # Generate the attacker positions
    x_range = np.arange(-0.95, 1.0, 0.05)  # from -0.95 to 0.95
    y_range = np.arange(-0.95, 1.0, 0.05)
    # attacker_positions = np.array([(x, y) for x in x_range for y in y_range])
    # attacker_positions = attacker_positions.reshape(len(x_range), len(y_range), 2)
    # Generate the score matrix
    score_matrix = np.zeros((len(x_range), len(y_range)))
    # Record the time
    start_time = time.perf_counter()
    # Loop through each attacker position to run the game
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            initial_attacker = np.array(([[x_range[i], y_range[j], -1.0]]))
            initial_defender = np.array([[0.7, -0.5, 1.00]])
            # print(f"========== The attacker starts at {initial_attacker}. =========== \n")
            envs = DubinReachAvoidEasierGame(random_init=False,
                              seed=test_seed,
                              init_type='random',
                              initial_attacker=initial_attacker, 
                              initial_defender=initial_defender)
            
            step = 0
            attackers_status = []
            attackers_status.append(np.zeros(1))
            attackers_traj, defenders_traj = [], []

            obs = envs._computeObs()  # obs.shape = (6,)
            print(f"========== The initial state is {obs}. ===========")
            # print(f"========== The initial defender is {envs.init_defenders}. ===========")
            # print(f"========== The initial attacker is {envs.init_attackers}. ===========")
            # print(f"========== The initial_attacker is {initial_attacker} and the initial_defender is  {initial_defender}. =========== \n")

            attackers_traj.append(np.array([obs[:3]]))
            defenders_traj.append(np.array([obs[3:]]))

            for sim in range(int(T*ctrl_freq)):
                actions, _ = model.predict(obs, deterministic=True)
                # print(f"Step {step}: the action is {actions}. \n")
                next_obs, reward, terminated, truncated, infos = envs.step(actions)
                step += 1
                # print(f"Step {step}: the reward is {reward}. \n")
                attackers_traj.append(np.array([next_obs[:3]]))
                defenders_traj.append(np.array([next_obs[3:]]))
                # print(f"Step {step}: the relative distance is {np.linalg.norm(next_obs[:, :2] - next_obs[:, 2:])}. \n")
                # print(f"Step {step}: the current position of the attacker is {next_obs[:2]}. \n")
                attackers_status.append(getAttackersStatus(np.array([next_obs[:3]]), np.array([next_obs[3:]]), attackers_status[-1]))

                if terminated or truncated:
                    break
                else:
                    obs = next_obs

            if attackers_status[-1] == -1 : # captured
                # print(f"================ The attacker starts at {initial_attacker} is captured. ================ \n")
                score_matrix[i, j] = +1
            elif attackers_status[-1] == 1: # reached
                # print(f"================ The attacker starts at {initial_attacker} reaches the goal. ================ \n")
                score_matrix[i, j] = -1
            else:
                assert False, "The game is not terminated correctly."

    np.save(f'{trained_path}/score_matrix_{initial_defender[0].tolist()}.npy', score_matrix)
    print(f"========== The score matrix is saved to {trained_path}/score_matrix.npy. =========== \n")
    duration = time.perf_counter() - start_time
    print(f"========== The game is finished. The total time is {duration//60} min {duration%60} seconds. =========== \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--init_type',           default="random",        type=str,           help='The initilaization method (default: random)', metavar='')
    parser.add_argument('--total_steps',         default=1e7,             type=float,         help='The total training steps (default: 2e7)', metavar='')
    
    args = parser.parse_args()
    
    test_dubin_sb3(init_type=args.init_type, total_steps=args.total_steps)

    # python safe_control_gym/experiments/test_dubingame_sb3_batch.py --init_type random --total_steps 1e7
