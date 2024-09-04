import os
import argparse
from typing import Callable
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

from odp.Grid import Grid
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from safe_control_gym.utils.plotting import animation_easier_game, current_status_check, record_video, plot_network_value, plot_values
from safe_control_gym.envs.gym_game.ReachAvoidGame import ReachAvoidEasierGame

from stable_baselines3 import PPO



map = {'map': [-1., 1., -1., 1.]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
des = {'goal0': [0.6, 0.8, 0.1, 0.3]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
obstacles = {'obs1': [100, 100, 100, 100]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]

value1vs1 = np.load(('safe_control_gym/envs/gym_game/values/1vs1Defender_easier.npy'))
grid1vs0 = Grid(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 2, np.array([100, 100])) 
grid1vs1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45]))

def check_area(state, area):
    """Check if the state is inside the area.

    Parameters:
        state (np.ndarray): the state to check
        area (dict): the area dictionary to be checked.
    
    Returns:
        bool: True if the state is inside the area, False otherwise.
    """
    x, y = state  # Unpack the state assuming it's a 2D coordinate

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
                    print(f"================ The {num} attacker has arrived. ================ \n")
                # # check if the attacker gets stuck in the obstacles this time (it won't usually)
                # elif self._check_area(current_attacker_state[num], self.obstacles):
                #     new_status[num] = -1
                #     break
                else:
                    # check if the attacker is captured
                    for j in range(num_defenders):
                        if np.linalg.norm(current_attacker_state[num] - current_defender_state[j]) <= 0.1:
                            print(f"================ The {num} attacker is captured. ================ \n")
                            new_status[num] = -1
                            break

            return new_status


def check_current_value(attackers, defenders, value_function, grids):
    """ Check the value of the current state of the attackers and defenders.

    Args:
        attackers (np.ndarray): the attackers' states
        defenders (np.ndarray): the defenders' states
        value (np.ndarray): the value function for the game
        grid (Grid): the grid for the game

    Returns:
        value (float): the value of the current state of the attackers and defenders
    """
    if len(value_function.shape) == 4:  # 1vs1 game
        # joint_slice = po2slice1vs1(attackers[0], defenders[0], value_function.shape[0])
        joint_slice = grids.get_index(np.concatenate((attackers[0], defenders[0])))
    elif len(value_function.shape) == 6:  # 1vs2 or 2vs1 game
        if attackers.shape[0] == 1:  # 1vs2 game
            # joint_slice = po2slice2vs1(attackers[0], defenders[0], defenders[1], value_function.shape[0])
            joint_slice = grids.get_index(np.concatenate((attackers[0], defenders[0], defenders[1])))
        else:  # 2vs1 game
            # joint_slice = po2slice2vs1(attackers[0], attackers[1], defenders[0], value_function.shape[0])
            joint_slice = grids.get_index(np.concatenate((attackers[0], attackers[1], defenders[0])))

    value = value_function[joint_slice]

    return value
        

def test_sb3(optimality, init_type, total_steps):
    # Set up env hyperparameters.
    n_env = 8
    env_seed = 2024  # 2024
    # Setp up algorithm hyperparameters.
    total_timesteps = total_steps
    batch_size = 64
    n_epochs = 15
    n_steps = 2048
    test_seed = 2024
    target_kl = 0.01

    # Load the trained model
    trained_path = os.path.join('training_results', f"easier_game/sb3/{init_type}/{optimality}/", f'seed_{env_seed}', f'{total_timesteps}steps')
    trained_model = os.path.join('training_results', f"easier_game/sb3/{init_type}/{optimality}/", f'seed_{env_seed}', f'{total_timesteps}steps/', 'final_model.zip')
    assert os.path.exists(trained_model), f"[ERROR] The trained model {trained_model} does not exist, please check the loading path or train one first."
    print(f"========== The trained model is loaded from {trained_model}. =========== \n")
    model = PPO.load(trained_model)
    
    # Create the environment.
    #TODO the defender hits the obs
    # initial_attacker = np.array([[-0.5, 0.8]])
    # initial_defender = np.array([[0.3, -0.3]])
    #TODO the defender hits the obs
    initial_attacker = np.array([[0.0, 0.0]])
    initial_defender = np.array([[-0.5, -0.5]])
    
    # Random test 
    # initial_attacker = np.array([[-0.5, 0.0]])
    # initial_defender = np.array([[0.3, 0.0]])
    
    
    envs = ReachAvoidEasierGame(random_init=False,
                              seed=test_seed,
                              init_type='random',
                              initial_attacker=initial_attacker, 
                              initial_defender=initial_defender)
    print(f"The state space of the env is {envs.observation_space}. \n")  # Box(-1.0, 1.0, (4,)
    print(f"The action space of the env is {envs.action_space}. \n")  # Box(-1.0, 1.0, (2,)
    # value_net = PPO('MlpPolicy',envs, verbose=1)
    # value_net.policy.value_net.load_state_dict(torch.load(trained_value_net))

    step = 0
    attackers_status = []
    attackers_status.append(np.zeros(1))
    attackers_traj, defenders_traj = [], []

    obs, _ = envs.reset()  # obs.shape = (4,)
    initial_obs = obs.copy()
    print(f"========== The initial state is {initial_obs} in the test_game. ========== \n")
    print(f"========== The initial value function is {check_current_value(np.array(obs[:2].reshape(1,2)), np.array(obs[2:].reshape(1,2)), value1vs1, grid1vs1)}. ========== \n")
    # # check the trained value function
    # initial_obs_tensor = torch.tensor(initial_obs, dtype=torch.float32).unsqueeze(0)
    # initial_obs_tensor = initial_obs_tensor.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # features = model.policy.vf_features_extractor(initial_obs_tensor)
    # value_features = model.policy.mlp_extractor.value_net(features)
    # value = model.policy.value_net(value_features)
    # print(f"========== The initial value is {value} in the test_game. ========== \n")

    # plot the value network in the heat map
    fixed_defender_position = np.array([[0.5, 0.0]])
    # fixed_defender_position = np.array([[-0.5, -0.5]])
    plot_values(fixed_defender_position, model, value1vs1, grid1vs1, initial_attacker, trained_path)

    # attackers_traj.append(np.array([obs[:2]]))
    # defenders_traj.append(np.array([obs[2:]]))

    # for sim in range(int(10*200)):
    #     actions, _ = model.predict(obs, deterministic=True)
    #     # print(f"Step {step}: the action is {actions}. \n")
    #     next_obs, reward, terminated, truncated, infos = envs.step(actions)
    #     step += 1
    #     # print(f"Step {step}: the reward is {reward}. \n")
    #     attackers_traj.append(np.array([next_obs[:2]]))
    #     defenders_traj.append(np.array([next_obs[2:]]))
    #     # print(f"Step {step}: the relative distance is {np.linalg.norm(next_obs[:, :2] - next_obs[:, 2:])}. \n")
    #     # print(f"Step {step}: the current position of the attacker is {next_obs[:2]}. \n")
    #     attackers_status.append(getAttackersStatus(np.array([next_obs[:2]]), np.array([next_obs[2:]]), attackers_status[-1]))

    #     if terminated or truncated:
    #         break
    #     else:
    #         obs = next_obs
    # # print(f"================ The {num} game is over at the {step} step ({step / 200} seconds. ================ \n")
    # print(f"================ The game is over at the {step} step ({step / 200} seconds. ================ \n")
    # current_status_check(attackers_status[-1], step)
    # animation_easier_game(attackers_traj, defenders_traj, attackers_status)
    # # # record_video(attackers_traj, defenders_traj, attackers_status, filename=f'1vs1_{datetime.now().strftime("%Y.%m.%d_%H:%M")}.mp4', fps=10)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--optimality',           default="1vs0_1vs1",        type=str,           help='The initilaization method (default: random)', metavar='')
    parser.add_argument('--init_type',           default="random",        type=str,           help='The initilaization method (default: random)', metavar='')
    parser.add_argument('--total_steps',         default=1e7,             type=float,         help='The total training steps (default: 2e7)', metavar='')
    
    args = parser.parse_args()
    
    test_sb3(optimality=args.optimality, init_type=args.init_type, total_steps=args.total_steps)

    # python safe_control_gym/experiments/test_easiergame_sb3.py  --optimality 1vs0_1vs1 --init_type random --total_steps 1e7
