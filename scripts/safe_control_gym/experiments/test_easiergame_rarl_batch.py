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
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir+'/')
        
    config.output_dir = output_dir
    print(f"==============The trained directory is {config.output_dir}.============== \n")

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
    model.reset()

    # # Initilalize the environment
    # initial_attacker = np.array([[0.0, 0.0]])
    # initial_defender = np.array([[-0.5, -0.5]])
    
    # Random test 
    # initial_attacker = np.array([[-0.5, 0.0]])
    # initial_defender = np.array([[0.3, 0.0]])
    test_seed = 2024

    # fixed_defender_position = np.array([[-0.5, 0.5]])
    # plot_values_rarl(fixed_defender_position, model, value1vs1, grid1vs1, initial_attacker, config.output_dir)

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
            initial_attacker = np.array(([[x_range[i], y_range[j]]]))
            initial_defender = np.array([[0.5, 0.0]])
            # initial_defender = np.array([[-0.5, -0.5]])
            # print(f"========== The attacker starts at {initial_attacker}. =========== \n")
    
            envs = ReachAvoidEasierGame(random_init=False,
                                    seed=test_seed,
                                    init_type='random',
                                    initial_attacker=initial_attacker, 
                                    initial_defender=initial_defender)

            step = 0
            attackers_status = []
            attackers_status.append(np.zeros(1))
            attackers_traj, defenders_traj = [], []

            obs = envs._computeObs()  # obs.shape = (4,)
            print(f"========== The initial state is {obs}. ===========")
            # print(f"========== The initial defender is {envs.init_defenders}. ===========")
            # print(f"========== The initial attacker is {envs.init_attackers}. ===========")
            # print(f"========== The initial_attacker is {initial_attacker} and the initial_defender is  {initial_defender}. =========== \n")

            attackers_traj.append(np.array([obs[:2]]))
            defenders_traj.append(np.array([obs[2:]]))

            for sim in range(int(15*200)):
                actions = model.select_action(obs=obs)
                # print(f"Step {step}: the action is {actions}. \n")
                next_obs, reward, terminated, truncated, infos = envs.step(actions)
                step += 1
                # print(f"Step {step}: the reward is {reward}. \n")
                attackers_traj.append(np.array([next_obs[:2]]))
                defenders_traj.append(np.array([next_obs[2:]]))
                # print(f"Step {step}: the relative distance is {np.linalg.norm(next_obs[:, :2] - next_obs[:, 2:])}. \n")
                # print(f"Step {step}: the current position of the attacker is {next_obs[:2]}. \n")
                attackers_status.append(getAttackersStatus(np.array([next_obs[:2]]), np.array([next_obs[2:]]), attackers_status[-1]))

                if terminated or truncated:
                    break
                else:
                    obs = next_obs

            if attackers_status[-1] == -1 : # captured
                # print(f"================ The attacker starts at {initial_attacker} is captured. ================ \n")
                score_matrix[i, j] = +1
            elif attackers_status[-1] == 1: # or attackers_status[-1] == 0: # reached the goal
                # print(f"================ The attacker starts at {initial_attacker} reaches the goal. ================ \n")
                score_matrix[i, j] = -1
            else:
                assert False, "The game is not terminated correctly."

    np.save(f'{config.output_dir}/score_matrix_{initial_defender[0].tolist()}.npy', score_matrix)
    print(f"========== The score matrix is saved to {config.output_dir}/score_matrix.npy. =========== \n")
    duration = time.perf_counter() - start_time
    print(f"========== The game is finished. The total time is {duration//60} min {duration%60} seconds. =========== \n")
    
    # # Plot the score matrix
    # fig, ax = plt.subplots()
    # cax = ax.matshow(score_matrix, cmap='coolwarm')
    # fig.colorbar(cax)
    # plt.title(f'The score matrix of the game with {optimality} and {init_type} initialization')
    # plt.xlabel('x-axis')
    # plt.ylabel('y-axis')
    # # plt.savefig(f'{trained_path}/score_matrix.png')
    # plt.show()
    # print(f"========== The score matrix is saved to {trained_path}/score_matrix.png. =========== \n")

        
    model.close()


if __name__ == '__main__':
    test()
    # python safe_control_gym/experiments/test_easiergame_rarl_batch.py --task rarl_game --algo rarl --use_gpu True --seed 42
    # python safe_control_gym/experiments/test_easiergame_rarl_batch.py --task rarl_game --algo rap --use_gpu True --seed 42
    
