'''Miscellaneous utility functions.'''

import argparse
import datetime
import json
import os
import random
import subprocess
import sys

import gymnasium as gym
import imageio
import munch
import numpy as np
import torch
import yaml
import math

from odp.Grid import Grid  # Utility functions to initialize the problem


def mkdirs(*paths):
    '''Makes a list of directories.'''

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def eval_token(token):
    '''Converts string token to int, float or str.'''
    if token.isnumeric():
        return int(token)
    try:
        return float(token)
    except TypeError:
        return token


def read_file(file_path, sep=','):
    '''Loads content from a file (json, yaml, csv, txt).

    For json & yaml files returns a dict.
    Ror csv & txt returns list of lines.
    '''
    if len(file_path) < 1 or not os.path.exists(file_path):
        return None
    # load file
    f = open(file_path, 'r')
    if 'json' in file_path:
        data = json.load(f)
    elif 'yaml' in file_path:
        data = yaml.load(f, Loader=yaml.FullLoader)
    else:
        sep = sep if 'csv' in file_path else ' '
        data = []
        for line in f.readlines():
            line_post = [eval_token(t) for t in line.strip().split(sep)]
            # if only sinlge item in line
            if len(line_post) == 1:
                line_post = line_post[0]
            if len(line_post) > 0:
                data.append(line_post)
    f.close()
    return data


def merge_dict(source_dict, update_dict):
    '''Merges updates into source recursively.'''
    for k, v in update_dict.items():
        if k in source_dict and isinstance(source_dict[k], dict) and isinstance(
                v, dict):
            merge_dict(source_dict[k], v)
        else:
            source_dict[k] = v


def get_time():
    '''Gets current timestamp (as string).'''
    start_time = datetime.datetime.now()
    time = str(start_time.strftime('%Y_%m_%d-%X'))
    return time


def get_random_state():
    '''Snapshots the random state at any moment.'''
    return {
        'random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state()
    }


def set_random_state(state_dict):
    '''Resets the random state for experiment restore.'''
    random.setstate(state_dict['random'])
    np.random.set_state(state_dict['numpy'])
    torch.torch.set_rng_state(state_dict['torch'])


def set_seed(seed, cuda=False):
    '''General seeding function for reproducibility.'''
    assert seed is not None, 'Error in set_seed(...), provided seed not valid'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_dir_from_config(config):
    '''Creates a output folder for experiment (and save config files).

    Naming format: {root (e.g. results)}/{tag (exp id)}/{seed}_{timestamp}_{git commit id}
    '''
    # Make run folder (of a seed run for an experiment)
    seed = str(config.seed) if config.seed is not None else '-'
    timestamp = str(datetime.datetime.now().strftime('%b-%d-%H-%M-%S'))
    try:
        commit_id = subprocess.check_output(
            ['git', 'describe', '--tags', '--always']
        ).decode('utf-8').strip()
        commit_id = str(commit_id)
    except BaseException:
        commit_id = '-'
    run_dir = f'seed{seed}_{timestamp}_{commit_id}'
    # Make output folder.
    config.output_dir = os.path.join(config.output_dir, config.tag, run_dir)
    mkdirs(config.output_dir)
    # Save config.
    with open(os.path.join(config.output_dir, 'config.yaml'), 'w') as file:
        yaml.dump(munch.unmunchify(config), file, default_flow_style=False)
    # Save command.
    with open(os.path.join(config.output_dir, 'cmd.txt'), 'a') as file:
        file.write(' '.join(sys.argv) + '\n')


def set_seed_from_config(config):
    '''Sets seed, only set if seed is provided.'''
    seed = config.seed
    if seed is not None:
        set_seed(seed, cuda=config.use_gpu)


def set_device_from_config(config):
    '''Sets device, using GPU is set to `cuda` for now, no specific GPU yet.'''
    use_cuda = config.use_gpu and torch.cuda.is_available()
    config.device = 'cuda' if use_cuda else 'cpu'
    print(f'Using device: {config.device}')


def save_video(name, frames, fps=20):
    '''Convert list of frames (H,W,C) to a video.

    Args:
        name (str): path name to save the video.
        frames (list): frames of the video as list of np.arrays.
        fps (int, optional): frames per second.
    '''
    assert '.gif' in name or '.mp4' in name, 'invalid video name'
    vid_kwargs = {'fps': fps}
    h, w, c = frames[0].shape
    video = np.stack(frames, 0).astype(np.uint8).reshape(-1, h, w, c)
    imageio.mimsave(name, video, **vid_kwargs)


def str2bool(val):
    '''Converts a string into a boolean.

    Args:
        val (str|bool): Input value (possibly string) to interpret as boolean.

    Returns:
        bool: Interpretation of `val` as True or False.
    '''
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('[ERROR] in str2bool(), a Boolean value is expected')


def unwrap_wrapper(env, wrapper_class):
    '''Retrieve a ``VecEnvWrapper`` object by recursively searching.'''
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None


def is_wrapped(env, wrapper_class):
    '''Check if a given environment has been wrapped with a given wrapper.'''
    return unwrap_wrapper(env, wrapper_class) is not None


def Boltzmann(low=0.0, high=2.1, accuracy=0.1):
    energies = np.array(np.arange(low, high, accuracy))  # Example energy levels
    beta = 1.0  # Inverse temperature (1/kT)

    # Calculate Boltzmann weights
    weights = np.exp(-beta * energies)

    # Normalize to get probabilities
    probabilities = weights / np.sum(weights)

    # Generate random samples from the Boltzmann distribution
    random_state = np.around(np.random.choice(energies, p=probabilities), 1)  
    return random_state


def quat2euler(quat):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x, y, z, w = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return [roll_x, pitch_y, yaw_z] # in radians


def distur_gener_quadrotor(states, distb_level):


    def opt_ctrl_non_hcl(uMax, spat_deriv):
        
        uOpt1, uOpt2, uOpt3 = uMax[0],uMax[1], uMax[2]
        uMin = -uMax
   
        if spat_deriv[3] > 0:
            uOpt1 = uMin[0]

        if spat_deriv[4] > 0:
            uOpt2 = uMin[1]
                    
        if spat_deriv[5] > 0:
            uOpt3 = uMin[2]


            
        return (uOpt1, uOpt2, uOpt3)
        
    def spa_deriv(index, V, g, periodic_dims):
            '''
        Calculates the spatial derivatives of V at an index for each dimension

        Args:
            index:
            V:
            g:
            periodic_dims:

        Returns:
            List of left and right spatial derivatives for each dimension

            '''
            spa_derivatives = []

            for dim, idx in enumerate(index):
                if dim == 0:
                    left_index = []
                else:
                    left_index = list(index[:dim])

                if dim == len(index) - 1:
                    right_index = []
                else:
                    right_index = list(index[dim + 1:])

                next_index = tuple(
                    left_index + [index[dim] + 1] + right_index
                )
                prev_index = tuple(
                left_index + [index[dim] - 1] + right_index
                )
                if idx == 0:
                    if dim in periodic_dims:
                        left_periodic_boundary_index = tuple(
                            left_index + [V.shape[dim] - 1] + right_index
                        )
                        left_boundary = V[left_periodic_boundary_index]
                    else:
                        left_boundary = V[index] + np.abs(V[next_index] - V[index]) * np.sign(V[index])
                    left_deriv = (V[index] - left_boundary) / g.dx[dim]
                    right_deriv = (V[next_index] - V[index]) / g.dx[dim]
                elif idx == V.shape[dim] - 1:
                    if dim in periodic_dims:
                        right_periodic_boundary_index = tuple(
                            left_index + [0] + right_index
                        )
                        right_boundary = V[right_periodic_boundary_index]
                    else:
                        right_boundary = V[index] + np.abs(V[index] - V[prev_index]) * np.sign([V[index]])
                    left_deriv = (V[index] - V[prev_index]) / g.dx[dim]
                    right_deriv = (right_boundary - V[index]) / g.dx[dim]
                else:
                    left_deriv = (V[index] - V[prev_index]) / g.dx[dim]
                    right_deriv = (V[next_index] - V[index]) / g.dx[dim]

                spa_derivatives.append((left_deriv + right_deriv) / 2)

            return  spa_derivatives  # np.array(spa_derivatives)  # Hanyang: change the type of the return

            dyn_sys.x = next_state

    def opt_dstb_non_hcl(dMax, spat_deriv):

        dOpt1,dOpt2,dOpt3 = dMax[0],dMax[1],dMax[2]
        dMin = -dMax
        # Joe:
        if spat_deriv[3] > 0:
            dOpt1 = dMin[0]
        if spat_deriv[4] > 0:
            dOpt2 = dMin[1]
        if spat_deriv[5] > 0:
            dOpt3 = dMin[2]

        return (dOpt1, dOpt2, dOpt3)

    def compute_opt_traj(grid: Grid, V, states, umax, dmax): 
            """
        Computes the optimal trajectory, controls and disturbance to a minimal BRT/BRS

        Args:
            grid:
            V:
            current states:
            maximum control
            maximum disturbance


        Returns:
            opt_u: Optimal control at current time step
            opt_d: Optimal disturbance at current time step

            """
            
            gradient = spa_deriv(grid.get_index(states), V, grid, periodic_dims=[0,1,2])
            u = opt_ctrl_non_hcl(umax, gradient)
            d = opt_dstb_non_hcl(dmax, gradient)
                
            return u,d

    
    umax = np.array([5.3*10**-3,  5.3*10**-3,  1.43*10**-4]) 
    # dmax = 0*umax
    assert distb_level <= 3.0  # Hanyang: check the output content
    V = np.load(f'safe_control_gym/hj_distbs/FasTrack_data/quadrotor/quadrotor_{distb_level}_15x15.npy')
    dmax = distb_level * umax

    grid = Grid(np.array([-math.pi/2.4, -math.pi/2.4, -math.pi/2.4, -math.pi, -math.pi, -math.pi]), np.array([math.pi/2.4, math.pi/2.4, math.pi/2.4, math.pi, math.pi, math.pi]), 6, np.array([15,15,15,15,15,15]), [0,1,2])

    [opt_u, opt_d] = compute_opt_traj(grid, V, states, umax, dmax)

    return opt_u, opt_d
  
  
def transfer(distb_level):
    index = int(distb_level * 10)
    allowable_distb_levels = np.arange(0.0, 2.1, 0.1)
    return allowable_distb_levels[index]