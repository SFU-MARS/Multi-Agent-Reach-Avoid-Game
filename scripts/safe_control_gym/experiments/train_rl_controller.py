'''Template training/plotting/testing script.'''

import os
import shutil
import time
from functools import partial

import munch
import yaml

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.plotting import plot_from_logs
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_device_from_config, set_seed_from_config


def train():
    '''Training template.
    '''
    # Create the configuration dictionary.
    fac = ConfigFactory()
    config = fac.merge()
    config.algo_config['training'] = True
    config.algo_config['max_ctrl_steps'] = config.task_config['episode_len_sec'] * config.task_config['ctrl_freq']
    total_steps = config.algo_config['max_env_steps']
    # For take in some attributes to the algorithm
    config.algo_config['render_height'] = config.task_config['render_height']
    config.algo_config['render_width'] = config.task_config['render_width']

    # shutil.rmtree(config.output_dir, ignore_errors=True)
    # Hanyang: create new envs
    if config.task == 'cartpole_fixed' or config.task == 'quadrotor_fixed':
        output_dir = os.path.join(config.output_dir, config.task, config.algo, f'distb_level{config.distb_level}', f'seed_{config.seed}', f'{total_steps}steps')
    else:
        output_dir = os.path.join(config.output_dir, config.task, config.algo, f'seed_{config.seed}', f'{total_steps}steps')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir+'/')
    config.output_dir = output_dir
    print(f"The output directory is {config.output_dir}. \n")

    set_seed_from_config(config)
    set_device_from_config(config)

    # Define function to create task/env.
    env_func = partial(make,
                       config.task,
                       output_dir=config.output_dir,
                       **config.task_config
                       )
    env_disturbances = env_func().disturbances
    env_advserary = env_func().adversary_disturbance
    print(f"==============The env is {env_func().NAME} with disturbances {env_disturbances} and adversary {env_advserary}.============== \n")
    # input("Press Enter to continue...")
    print(f"==============The envs are ready.============== \n")
    

    # Create the controller/control_agent.
    ctrl = make(config.algo,
                env_func,
                checkpoint_path=os.path.join(config.output_dir, 'model_latest.pt'),
                output_dir=config.output_dir,
                use_gpu=config.use_gpu,
                seed=config.seed,
                **config.algo_config)
    ctrl.reset()
    print(f"==============The controller is ready.============== \n")

    # Training.
    print(f"==============Start training.============== \n")
    start_time = time.perf_counter()
    ctrl.learn()
    ctrl.close()
    print(f"==============Training done.============== \n")
    duration = time.perf_counter() - start_time
    print(f"========== The time of training is {duration//3600}hours-{(duration%3600)//60}minutes-{(duration%3600)%60}seconds. ========== \n")


    # Save the configuration.
    if config.task == 'cartpole' or config.task == 'cartpole_v0':
        env_func().close()
        with open(os.path.join(config.output_dir, 'config.yaml'), 'w', encoding='UTF-8') as file:
            config_assemble = munch.unmunchify(config)
            yaml.dump(config_assemble, file, default_flow_style=False)
            
    elif config.task =='rarl_game':
        env_func().close()
        with open(os.path.join(config.output_dir, 'config.yaml'), 'w', encoding='UTF-8') as file:
            config_assemble = munch.unmunchify(config)
            yaml.dump(config_assemble, file, default_flow_style=False)

    elif config.task =='dubin_rarl_game':
        env_func().close()
        with open(os.path.join(config.output_dir, 'config.yaml'), 'w', encoding='UTF-8') as file:
            config_assemble = munch.unmunchify(config)
            yaml.dump(config_assemble, file, default_flow_style=False)

    else:
        env_distb_type = env_func().distb_type
        env_distb_level = env_func().distb_level
        env_func().close()
        with open(os.path.join(config.output_dir, 'config.yaml'), 'w', encoding='UTF-8') as file:
            config_assemble = munch.unmunchify(config)
            config_assemble['env_distb_type'] = env_distb_type
            config_assemble['env_distb_level'] = env_distb_level
            yaml.dump(config_assemble, file, default_flow_style=False)
       

    # make_plots(config)


def make_plots(config):
    '''Produces plots for logged stats during training.
    Usage
        * use with `--func plot` and `--restore {dir_path}` where `dir_path` is
            the experiment folder containing the logs.
        * save figures under `dir_path/plots/`.
    '''
    # Define source and target log locations.
    log_dir = os.path.join(config.output_dir, 'logs')
    plot_dir = os.path.join(config.output_dir, 'plots')
    mkdirs(plot_dir)
    plot_from_logs(log_dir, plot_dir, window=3)
    print('Plotting done.')


if __name__ == '__main__':
    train()
    #TODO remmenber to revise the seed in the corresponding env class
    # python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_rarl --algo rap --use_gpu True --seed 2024
    # python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_rarl --algo rarl --use_gpu True --seed 2024
    # python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_null --algo ppo --use_gpu True --seed 2024