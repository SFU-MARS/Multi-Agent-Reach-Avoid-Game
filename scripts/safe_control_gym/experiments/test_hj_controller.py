'''A LQR and iLQR example.'''

import os
import cv2
import time
import imageio
import pickle
from collections import defaultdict
from functools import partial

import yaml
import matplotlib.pyplot as plt
import munch
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactoryTest
from safe_control_gym.utils.registration import make


def generate_videos(frames, render_width, render_height, output_dir):
    """Hanyang
    Input:
        frames: list, a list contains several lists, each containts a sequence of numpy ndarrays 
        env: the quadrotor and task environment
    """
    # Define the output video parameters
    fps = 24  # Frames per second
    episodes = len(frames)
    
    for episode in range(episodes):
        filename = f'Episode{episode}_{len(frames[episode])}steps_{time.strftime("%m_%d_%H_%M")}.mp4'

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as well (e.g., 'XVID')
        out = cv2.VideoWriter(output_dir+'/'+filename, fourcc, fps, (render_height, render_width))
        # Write frames to the video file
        for frame in frames[episode]:
            frame = np.asarray(frame, dtype=np.uint8)
            out.write(frame)
        # Release the VideoWriter object
        out.release()


def generate_gifs(frames, output_dir):
    """Hanyang
    Input:
        frames: list, a list contains several lists, each containts a sequence of numpy ndarrays 
        env: the quadrotor and task environment
    """
    episodes = len(frames)
    
    for episode in range(episodes):
        images = []
        filename = f'Episode{episode}_{len(frames[episode])}steps_{time.strftime("%m_%d_%H_%M")}.gif'
        for frame in frames[episode]:
            images.append(frame.astype(np.uint8))
        imageio.mimsave(output_dir+'/'+filename, images, duration=20)
        print(f"******************Generate {filename} successfully. \n****************")


def log_performance(eval_results, config):
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
        f.write(f'Test task: {test_task}\n')
        f.write(f'Controller: {config.algo} trained in the {trained_task}\n')
        f.write(f'Number of episodes: {num_episodes}\n')
        f.write(f'Performances of returns: {mean_returns: .2f} ± {std_returns: .2f}\n')
        f.write(f'Performances of lengths: {int(mean_lengths)} ± {std_lengths: .2f}\n')

    print((f"****************** The performances are logged.\n ******************"))


def run(gui=False, n_episodes=1, n_steps=None, save_data=False):
    '''The main function running HJ experiments.

    Args:
        gui (bool): Whether to display the gui and plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
        save_data (bool): Whether to save the collected experiment data.
    '''

    # Create the configuration dictionary.
    CONFIG_FACTORY = ConfigFactoryTest()
    config = CONFIG_FACTORY.merge()

    config.output_dir = 'test_results'

    # Hanyang: make output_dir
    if config.task == 'cartpole_fixed' or config.task == 'quadrotor_fixed':
        output_dir = os.path.join(config.output_dir, config.task, config.algo, f'distb_level{config.test_distb_level}', f'seed_{config.seed}')
    else:
        output_dir = os.path.join(config.output_dir, config.task, config.algo, f'seed_{config.seed}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir+'/')
        
    config.output_dir = output_dir
    print(f"==============The saving directory is {config.output_dir}.============== \n")

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config
                       )
    random_env = env_func(gui=False)

    print(f"==============Env is ready.============== \n")

    # Create controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )
    print(f"==============Controller is ready.============== \n")

    assert config.algo == 'hj', 'This script is only for HJ controller.'
    if config.render:
        eval_results = ctrl.run(render=True, n_episodes=5)
    else:   
        eval_results = ctrl.run(render=False, n_episodes=10)

    ctrl.close()
    # Hanyang: generate videos and gifs
    print("Start to generate videos and gifs.")
    generate_gifs(eval_results['frames'], config.output_dir)
    log_performance(eval_results, config)

    test_distb_type = env_func().distb_type
    test_distb_level = env_func().distb_level
    env_func().close()
    with open(os.path.join(config.output_dir, f'config_{time.strftime("%m_%d_%H_%M")}.yaml'), 'w', encoding='UTF-8') as file:
        config_assemble = munch.unmunchify(config)
        config_assemble['trained_task'] = config.trained_task
        config_assemble['test_distb_type'] = test_distb_type
        config_assemble['test_distb_level'] = test_distb_level
        yaml.dump(config_assemble, file, default_flow_style=False)





    # all_trajs = defaultdict(list)
    # n_episodes = 1 if n_episodes is None else n_episodes

#     # Run the experiment.
#     for _ in range(n_episodes):
#         # Get initial state and create environments
#         init_state, _ = random_env.reset()
#         static_env = env_func(gui=gui, randomized_init=False, init_state=init_state)
#         static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)

#         # Create experiment, train, and run evaluation
#         experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)
#         experiment.launch_training()

#         if n_steps is None:
#             trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1)
#         else:
#             trajs_data, _ = experiment.run_evaluation(training=True, n_steps=n_steps)

#         if gui:
#             post_analysis(trajs_data['obs'][0], trajs_data['action'][0], ctrl.env)

#         # Close environments
#         static_env.close()
#         static_train_env.close()

#         # Merge in new trajectory data
#         for key, value in trajs_data.items():
#             all_trajs[key] += value

#     ctrl.close()
#     random_env.close()
#     metrics = experiment.compute_metrics(all_trajs)
#     all_trajs = dict(all_trajs)

#     if save_data:
#         results = {'trajs_data': all_trajs, 'metrics': metrics}
#         path_dir = os.path.dirname('./temp-data/')
#         os.makedirs(path_dir, exist_ok=True)
#         with open(f'./temp-data/{config.algo}_data_{config.task}_{config.task_config.task}.pkl', 'wb') as file:
#             pickle.dump(results, file)

#     print('FINAL METRICS - ' + ', '.join([f'{key}: {value}' for key, value in metrics.items()]))


# def post_analysis(state_stack, input_stack, env):
#     '''Plots the input and states to determine iLQR's success.

#     Args:
#         state_stack (ndarray): The list of observations of iLQR in the latest run.
#         input_stack (ndarray): The list of inputs of iLQR in the latest run.
#     '''
#     model = env.symbolic
#     stepsize = model.dt

#     plot_length = np.min([np.shape(input_stack)[0], np.shape(state_stack)[0]])
#     times = np.linspace(0, stepsize * plot_length, plot_length)

#     reference = env.X_GOAL
#     if env.TASK == Task.STABILIZATION:
#         reference = np.tile(reference.reshape(1, model.nx), (plot_length, 1))

#     # Plot states
#     fig, axs = plt.subplots(model.nx)
#     for k in range(model.nx):
#         axs[k].plot(times, np.array(state_stack).transpose()[k, 0:plot_length], label='actual')
#         axs[k].plot(times, reference.transpose()[k, 0:plot_length], color='r', label='desired')
#         axs[k].set(ylabel=env.STATE_LABELS[k] + f'\n[{env.STATE_UNITS[k]}]')
#         axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#         if k != model.nx - 1:
#             axs[k].set_xticks([])
#     axs[0].set_title('State Trajectories')
#     axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
#     axs[-1].set(xlabel='time (sec)')

#     # Plot inputs
#     _, axs = plt.subplots(model.nu)
#     if model.nu == 1:
#         axs = [axs]
#     for k in range(model.nu):
#         axs[k].plot(times, np.array(input_stack).transpose()[k, 0:plot_length])
#         axs[k].set(ylabel=f'input {k}')
#         axs[k].set(ylabel=env.ACTION_LABELS[k] + f'\n[{env.ACTION_UNITS[k]}]')
#         axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#     axs[0].set_title('Input Trajectories')
#     axs[-1].set(xlabel='time (sec)')

#     plt.show()


# def wrap2pi_vec(angle_vec):
    # '''Wraps a vector of angles between -pi and pi.

    # Args:
    #     angle_vec (ndarray): A vector of angles.
    # '''
    # for k, angle in enumerate(angle_vec):
    #     while angle > np.pi:
    #         angle -= np.pi
    #     while angle <= -np.pi:
    #         angle += np.pi
    #     angle_vec[k] = angle
    # return angle_vec


if __name__ == '__main__':
    run()
