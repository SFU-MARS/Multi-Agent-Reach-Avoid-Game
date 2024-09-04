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
    print(f"The config.adversary_disturbance is {config.task_config.adversary_disturbance}")
    

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
    env_seed = env_func().SEED
    print(f"============== The test env is {env_func().NAME} with env seed {env_func().SEED}.============== \n")
    print(f"==============Env is ready.============== \n")
    
    # Create the controller/control_agent.
    ctrl = make(config.algo,
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
    ctrl.load(model_path)
    print(f"==============Model is loaded.============== \n")
    ctrl.reset()
    
    if config.render:
        if config.algo == 'ppo':
            # eval_results = ctrl.run(render=False, n_episodes=10) # Hanyang: the maximum number of episodes is 3 if generating videos.
            eval_results = ctrl.run(render=True, n_episodes=3) # Hanyang: the maximum number of episodes is 3 if generating videos.
            
        elif config.algo == 'rarl':
            # eval_results = ctrl.run(render=False, n_episodes=10, use_adv=False) 
            eval_results = ctrl.run(render=True, n_episodes=5, use_adv=False) 
            
        elif config.algo == 'rap':
            # eval_results = ctrl.run(render=False, n_episodes=10, use_adv=False) 
            eval_results = ctrl.run(render=True, n_episodes=5, use_adv=False) 
    else:
        if config.algo == 'ppo':
            eval_results = ctrl.run(render=False, n_episodes=10) # Hanyang: the maximum number of episodes is 3 if generating videos.
            # eval_results = ctrl.run(render=True, n_episodes=3) # Hanyang: the maximum number of episodes is 3 if generating videos.
            
        elif config.algo == 'rarl':
            eval_results = ctrl.run(render=False, n_episodes=10, use_adv=False) 
            # eval_results = ctrl.run(render=True, n_episodes=3, use_adv=False) 
            
        elif config.algo == 'rap':
            eval_results = ctrl.run(render=False, n_episodes=10, use_adv=False) 
            # eval_results = ctrl.run(render=True, n_episodes=3, use_adv=False) 
        
    ctrl.close()
    # Hanyang: generate videos and gifs
    # print("Start to generate videos and gifs.")
    # generate_gifs(eval_results['frames'], config.output_dir)
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
