'''Hamilton-Jacobi control class for Crazyflies.
Based on work conducted at UTIAS' DSL by SiQi Zhou and James Xu.
'''

import numpy as np

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.hj_distbs.distur_gener import distur_gener_cartpole
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.utils.utils import is_wrapped
from safe_control_gym.envs.env_wrappers.record_episode_statistics import (RecordEpisodeStatistics,
                                                                          VecRecordEpisodeStatistics)


class HJ(BaseController):
    '''Hamilton-Jacobi controller.'''

    def __init__(
            self,
            env_func,
            # Model args.
            distb_level: float = 1.0,  # Hanyang: the disturbance level of the value function used
            **kwargs):
        '''Creates task and controller.

        Args:
            env_func (Callable): Function to instantiate task/environment.
            q_lqr (list): Diagonals of state cost weight.
            r_lqr (list): Diagonals of input/action cost weight.
            discrete_dynamics (bool): If to use discrete or continuous dynamics.
        '''

        super().__init__(env_func, **kwargs)

        self.env = env_func()
        self.env = RecordEpisodeStatistics(self.env)

        # Controller params.
        self.distb_level = distb_level

    def reset(self):
        '''Prepares for evaluation.'''
        self.env.reset()

    def close(self):
        '''Cleans up resources.'''
        self.env.close()

    def select_action(self, obs, info=None):
        '''Determine the action to take at the current timestep.

        Args:
            obs (ndarray): The observation at this timestep.
            info (dict): The info at this timestep.

        Returns:
            action (ndarray): The action chosen by the controller.
        '''

        step = self.extract_step(info)
        
        hj_ctrl_force, _ = distur_gener_cartpole(obs, self.distb_level)
        assert self.env.TASK == Task.STABILIZATION, "The task should be stabilization."

        return hj_ctrl_force
    
    def run(self,
            env=None,
            render=False,
            n_episodes=10,
            verbose=False,
            ):
        '''Runs evaluation with current policy.'''

        if env is None:
            env = self.env
        else:
            if not is_wrapped(env, RecordEpisodeStatistics):
                env = RecordEpisodeStatistics(env, n_episodes)
                # Add episodic stats to be tracked.
                env.add_tracker('constraint_violation', 0, mode='queue')
                env.add_tracker('constraint_values', 0, mode='queue')
                env.add_tracker('mse', 0, mode='queue')

        obs, info = env.reset()
        print(f"Initial observation is {obs}. \n")
        ep_returns, ep_lengths = [], []
        # Hanyang: extend the frames for visualization.
        eval_results = {'frames': []}
        frames = []
        counter = 0
        returns = 0.0
        while len(ep_returns) < n_episodes:
            action = self.select_action(obs=obs, info=info)
            obs, r, done, info = env.step(action)
            counter += 1
            returns += r
            # print(f"The current step is {counter}. \n")
            if render:
                # env.render()
                frames.append(env.render())
            if verbose:
                print(f'obs {obs} | act {action}')
            if done:
                assert 'episode' in info
                ep_returns.append(info['episode']['r'])
                ep_lengths.append(info['episode']['l'])
                # ep_returns.append(counter)
                # ep_lengths.append(returns)
                counter = 0
                if render:
                    eval_results['frames'].append(frames)
                    frames = []
                obs, _ = env.reset()
            # obs = self.obs_normalizer(obs)
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
        return eval_results
