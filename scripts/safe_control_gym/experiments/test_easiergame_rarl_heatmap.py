'''Template training/plotting/testing script.'''

import os
import shutil
from functools import partial

import munch
import yaml
import cv2
import numpy as np
import time
import imageio
import psutil

from odp.Grid import Grid
from safe_control_gym.utils.configuration import ConfigFactoryTest
from safe_control_gym.utils.plotting import plot_from_logs
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_device_from_config, set_seed_from_config
from safe_control_gym.envs.gym_game.ReachAvoidGame import ReachAvoidEasierGame
from safe_control_gym.experiments.test_easiergame_sb3 import check_current_value, getAttackersStatus, current_status_check, animation_easier_game
from safe_control_gym.utils.plotting import plot_values_rarl



map = {'map': [-1., 1., -1., 1.]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
des = {'goal0': [0.6, 0.8, 0.1, 0.3]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
obstacles = {'obs1': [100, 100, 100, 100]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]

value1vs1 = np.load(('safe_control_gym/envs/gym_game/values/1vs1Defender_easier.npy'))
grid1vs0 = Grid(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 2, np.array([100, 100])) 
grid1vs1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45]))


def test():
    '''Training template.
    '''
    # Create the configuration dictionary.
    fac = ConfigFactoryTest()
    config = fac.merge()
    config.algo_config['training'] = False
    config.output_dir = 'training_results'
    total_steps = config.algo_config['max_env_steps']

    # Hanyang: make output_dir
    if config.task == 'cartpole_fixed' or config.task == 'quadrotor_fixed':
        output_dir = os.path.join(config.output_dir, config.task, config.algo, 
                                  f'distb_level{config.test_distb_level}', f'seed_{config.seed}')
    else:
        output_dir = os.path.join(config.output_dir, config.task, config.algo, 
                                  f'seed_{config.seed}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir+'/')
        
    config.output_dir = output_dir
    print(f"==============The saving directory is {config.output_dir}.============== \n")

    set_seed_from_config(config)
    set_device_from_config(config)

    # Define function to create task/env. env_func is the class, not the instance.
    env_func = partial(make,
                       config.task,
                       output_dir=config.output_dir,
                       **config.task_config
                       )
    print(f"==============Env is ready.============== \n")
    
    # Create the controller/control_agent.
    model = make(config.algo,
                env_func,
                checkpoint_path=os.path.join(config.output_dir, 'model_latest.pt'),
                output_dir=config.output_dir,
                use_gpu=config.use_gpu,
                seed=config.seed,  #TODO: seed is not used in the controller.
                **config.algo_config)
    print(f"==============Controller is ready.============== \n")
    
    # Hanyang: load the selected model, the default task (env) for the test is the same as that for training.
    if config.trained_task is None:
        # default: the same task as the training task
        config.trained_task = config.task

    if config.trained_task == 'cartpole_fixed' or config.trained_task == 'quadrotor_fixed':
        model_path = os.path.join(os.path.join('training_results', config.trained_task, config.algo, 
                                               f'distb_level{config.trained_distb_level}', f'seed_{config.seed}', f'{total_steps}steps', 
                                               'model_latest.pt'))
    else:
        # model_path = os.path.join(os.path.join('training_results', config.trained_task, config.algo, 
        #                                        f'seed_{config.seed}', f'{total_steps}steps', 'model_latest.pt'))
        model_path = os.path.join(os.path.join('training_results', config.trained_task, config.algo, 
                                               f'seed_{config.seed}', f'{total_steps}steps', 'model_latest.pt'))
    
    assert os.path.exists(model_path), f"[ERROR] The path '{model_path}' does not exist, please check the loading path or train one first."
    model.load(model_path)
    print(f"==============Model is loaded.============== \n")
    model.agent.eval()
    # model.adversary.eval()
    model.obs_normalizer.set_read_only()
    model.reset()

    # Initilalize the environment
    initial_attacker = np.array([[-0.5, -0.5]])
    initial_defender = np.array([[0.5, 0.0]])
    
    # Random test 
    # initial_attacker = np.array([[-0.5, 0.0]])
    # initial_defender = np.array([[0.3, 0.0]])
    test_seed = 2024

    # plot heatmaps
    # fixed_defender_position = np.array([[-0.5, -0.5]])
    fixed_defender_position = np.array([[0.5, 0.0]])
    plot_values_rarl(config.algo, fixed_defender_position, model, value1vs1, grid1vs1, initial_attacker, config.output_dir)

    # # run a game
    # envs = ReachAvoidEasierGame(random_init=False,
    #                           seed=test_seed,
    #                           init_type='random',
    #                           initial_attacker=initial_attacker, 
    #                           initial_defender=initial_defender)
    # print(f"The state space of the env is {envs.observation_space}. \n")  # Box(-1.0, 1.0, (4,)
    # print(f"The action space of the env is {envs.action_space}. \n")  # Box(-1.0, 1.0, (2,)
    
    # step = 0
    # attackers_status = []
    # attackers_status.append(np.zeros(1))
    # attackers_traj, defenders_traj = [], []

    # obs, info = envs.reset()  # obs.shape = (4,)
    # obs = model.obs_normalizer(obs)
    # initial_obs = obs.copy()
    # print(f"========== The initial state is {initial_obs} in the test_game. ========== \n")
    # print(f"========== The initial value function is {check_current_value(np.array(obs[:2].reshape(1,2)), np.array(obs[2:].reshape(1,2)), value1vs1, grid1vs1)}. ========== \n")
    # attackers_traj.append(np.array([obs[:2]]))
    # defenders_traj.append(np.array([obs[2:]]))

    # for sim in range(int(10*200)):
    #     actions = model.select_action(obs=obs, info=info)
    #     # print(f"Step {step}: the action is {actions}. \n")
    #     next_obs, reward, terminated, truncated, info = envs.step(actions)
    #     step += 1
    #     # print(f"Step {step}: the reward is {reward}. \n")
    #     attackers_traj.append(np.array([next_obs[:2]]))
    #     defenders_traj.append(np.array([next_obs[2:]]))
    #     # print(f"Step {step}: the relative distance is {np.linalg.norm(next_obs[:, :2] - next_obs[:, 2:])}. \n")
    #     print(f"Step {step}: the current position of the attacker is {next_obs[:2]}. \n")
    #     attackers_status.append(getAttackersStatus(np.array([next_obs[:2]]), np.array([next_obs[2:]]), attackers_status[-1]))

    #     if terminated or truncated:
    #         break
    #     else:
    #         obs = model.obs_normalizer(next_obs)

    # # print(f"================ The {num} game is over at the {step} step ({step / 200} seconds. ================ \n")
    # print(f"================ The game is over at the {step} step ({step / 200} seconds. ================ \n")
    # current_status_check(attackers_status[-1], step)
    # animation_easier_game(attackers_traj, defenders_traj, attackers_status)
        
    model.close()


if __name__ == '__main__':
    test()
    # python safe_control_gym/experiments/test_easiergame_rarl_heatmap.py --task rarl_game --algo rarl --use_gpu True --seed 42
    # python safe_control_gym/experiments/test_easiergame_rarl_heatmap.py --task rarl_game --algo rap --use_gpu True --seed 42
    
