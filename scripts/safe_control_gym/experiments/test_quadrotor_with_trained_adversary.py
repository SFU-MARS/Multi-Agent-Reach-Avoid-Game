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

from safe_control_gym.utils.configuration import ConfigFactoryTest
from safe_control_gym.utils.plotting import plot_from_logs
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_device_from_config, set_seed_from_config
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_adversary import QuadrotorAdversary
from safe_control_gym.utils.utils import get_random_state, is_wrapped, set_random_state
from safe_control_gym.envs.env_wrappers.record_episode_statistics import (RecordEpisodeStatistics,
                                                                          VecRecordEpisodeStatistics)


def log_performance(eval_results, config, env_seed):
    output_dir = config.output_dir
    num_episodes = len(eval_results['ep_returns'])
    mean_returns = np.mean(eval_results['ep_returns'])
    std_returns = np.std(eval_results['ep_returns'])
    mean_lengths = np.mean(eval_results['ep_lengths'])
    std_lengths = np.std(eval_results['ep_lengths'])
    if config.trained_task == 'cartpole_fixed' or config.trained_task == 'quadrotor_fixed':
        trained_task = f'{config.trained_task}_distb_level{config.trained_distb_level}'
    else:   
        trained_task = config.trained_task
    
    if config.task == 'cartpole_fixed' or config.task == 'quadrotor_fixed':
        test_task = f'{config.task}_distb_level{config.test_distb_level}'
    else:
        test_task = config.task

    with open(os.path.join(output_dir, f'performance{time.strftime("%m_%d_%H_%M")}.txt'), 'w') as f:
        f.write(f'Test task: {test_task} with env seed {env_seed}.\n')
        f.write(f'Controller: {config.algo} trained in the {trained_task} with {config.seed} seed \n')
        f.write(f'Number of episodes: {num_episodes}\n')
        f.write(f'Performances of returns: {mean_returns: .2f} ± {std_returns: .2f}\n')
        f.write(f'Performances of lengths: {int(mean_lengths)} ± {std_lengths: .2f}\n')
        f.write(f'Original returns: \n {eval_results["ep_returns"]} \n')
        f.write(f'Original lengths: \n {eval_results["ep_lengths"]} \n')

    print((f"****************** The performances are logged.\n ******************"))



def test():
    '''Training template.
    '''
    # Create the configuration dictionary.
    fac = ConfigFactoryTest()
    config = fac.merge()
    config.algo_config['training'] = False
    config.output_dir = 'test_results'
    total_steps = config.algo_config['max_env_steps']
    # print(f"The config.adversary_disturbance is {config.task_config.adversary_disturbance}")

    # Hanyang: make output_dir
    if config.task == 'cartpole_fixed' or config.task == 'quadrotor_fixed':
        output_dir = os.path.join(config.output_dir, config.task, config.algo, 
                                  f'distb_level{config.test_distb_level}', f'seed_{config.seed}', time.strftime("%m_%d_%H_%M"),)
    else:
        output_dir = os.path.join(config.output_dir, config.task, config.algo, 
                                  f'seed_{config.seed}', time.strftime("%m_%d_%H_%M"))
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
    # env_seed = env_func().SEED
    # env_adversary = env_func().adversary_disturbance
    # print(f"============== The test env is {env_func().NAME} with env seed {env_func().SEED} and {env_adversary} adversary_disturbance.============== \n")
    print(f"==============Env is ready.============== \n")
    
    # Create the controller/control_agent.
    ctrl = make(config.algo,
                env_func,
                checkpoint_path=os.path.join(config.output_dir, 'model_latest.pt'),
                output_dir=config.output_dir,
                use_gpu=config.use_gpu,
                seed=2024,  #TODO: seed is not used in the controller.
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
    ctrl.load(model_path)
    print(f"==============Model is loaded.============== \n")
    '''Runs evaluation with current policy.'''
    ctrl.reset()
    ctrl.agent.eval()
    if config.algo == 'rarl':
        ctrl.adversary.eval()
    elif config.algo == 'rap':
        for adv in ctrl.adversaries:
            adv.eval()
    ctrl.obs_normalizer.set_read_only()
    
    # test
    n_episodes = 10
    env_seed = 1000  # useless
    env = QuadrotorAdversary(seed=env_seed)
    if not is_wrapped(env, RecordEpisodeStatistics):
        env = RecordEpisodeStatistics(env, n_episodes)
        # Add episodic stats to be tracked.
        env.add_tracker('constraint_violation', 0, mode='queue')
        env.add_tracker('constraint_values', 0, mode='queue')
        env.add_tracker('mse', 0, mode='queue')
        
    obs, info = env.reset()
    obs = ctrl.obs_normalizer(obs)
    ep_returns, ep_lengths = [], []
    # Hanyang: extend the frames for visualization.
    eval_results = {'frames': []}
    frames = []
    counter = 0
    while len(ep_returns) < n_episodes:
        action = ctrl.select_action(obs=obs, info=info)
        obs, _, done, info = env.step(action)
        counter += 1
        # print(f"The current step is {counter}. \n")
        if done:
            assert 'episode' in info
            ep_returns.append(info['episode']['r'])
            ep_lengths.append(info['episode']['l'])
            counter = 0
            obs, _ = env.reset()
            print(f"The {len(ep_returns)} episode starts to run with the initial state {obs}. \n")
        obs = ctrl.obs_normalizer(obs)
    # Collect evaluation results.
    ep_lengths = np.asarray(ep_lengths)
    ep_returns = np.asarray(ep_returns)
    eval_results['ep_returns'] = ep_returns
    eval_results['ep_lengths'] = ep_lengths
    # eval_results = {'ep_returns': ep_returns, 'ep_lengths': ep_lengths}
    # Other episodic stats from evaluation env.
    if len(env.queued_stats) > 0:
        queued_stats = {k: np.asarray(v) for k, v in env.queued_stats.items()}
        eval_results.update(queued_stats)
    
    ctrl.close()
    log_performance(eval_results, config, env_seed)

    # Save the configuration.
    if config.task == 'cartpole' or config.task == 'cartpole_v0':
        env_func().close()
        with open(os.path.join(config.output_dir,  f'config_{time.strftime("%m_%d_%H_%M")}.yaml'), 'w', encoding='UTF-8') as file:
            config_assemble = munch.unmunchify(config)
            yaml.dump(config_assemble, file, default_flow_style=False)
    else:
        test_distb_type = env_func().distb_type
        test_distb_level = env_func().distb_level
        env_func().close()
        with open(os.path.join(config.output_dir, f'config_{time.strftime("%m_%d_%H_%M")}.yaml'), 'w', encoding='UTF-8') as file:
            config_assemble = munch.unmunchify(config)
            config_assemble['trained_task'] = config.trained_task
            config_assemble['test_distb_type'] = test_distb_type
            config_assemble['test_distb_level'] = test_distb_level
            yaml.dump(config_assemble, file, default_flow_style=False)


if __name__ == '__main__':
    test()
    # python safe_control_gym/experiments/test_quadrotor_with_trained_adversary.py --trained_task quadrotor_boltz --algo ppo --task quadrotor_adversary --seed 42
    # python safe_control_gym/experiments/test_quadrotor_with_trained_adversary.py --trained_task quadrotor_null --algo rarl --task quadrotor_adversary --seed 42
    # python safe_control_gym/experiments/test_quadrotor_with_trained_adversary.py --trained_task quadrotor_null --algo rap --task quadrotor_adversary --seed 42
    # python safe_control_gym/experiments/test_quadrotor_with_trained_adversary.py --trained_task quadrotor_null --algo ppo --task quadrotor_adversary --seed 42
    
    
