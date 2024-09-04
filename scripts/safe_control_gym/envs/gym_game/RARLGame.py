'''1 vs. 1 reach-avoid game (no obstacles) environment.

'''

import math
import os
import xml.etree.ElementTree as etxml
from copy import deepcopy

import casadi as cs
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from safe_control_gym.envs.benchmark_env import BenchmarkEnv, Cost, Task
from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS, SymmetricStateConstraint
from safe_control_gym.math_and_models.normalization import normalize_angle
from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel


class RARLGameEnv(BenchmarkEnv):
    '''1 vs. 1 reach-avoid game environment for rarl and rap algorithm.

    Including symbolic model, constraints, randomization, adversarial disturbances,
    multiple cost functions, stabilization and trajectory tracking references.

    task_config:
        info_in_reset: True
        randomized_inertial_prop: True
        inertial_prop_randomization_info:
            pole_length:
                distrib: choice
                args: [[1,5,10]]
            pole_mass:
                distrib: uniform
                low: 1
                high: 5
        constraints:
        - constraint_form: bounded_constraint
          constrained_variable: STATE
          active_dims: [2, 3]
          lower_bounds: [-0.2, -0.2]
          upper_bounds: [0.2, 0.2]
          tolerance: [0.05, 0.05, 0.05, 0.05]
        done_on_violation: True
        disturbances:
            observation:
            - disturbance_func: white_noise
              std: 4.0
            action:
            - disturbance_func: white_noise
              std: 4.0
        adversary_disturbance: dynamics
        adversary_disturbance_scale: 0.01
    '''

    NAME = 'rarlgame'

    # URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'cartpole_template.urdf')

    AVAILABLE_CONSTRAINTS = {
        'abs_bound': SymmetricStateConstraint
    }
    AVAILABLE_CONSTRAINTS.update(deepcopy(GENERAL_CONSTRAINTS))

    DISTURBANCE_MODES = {'observation': {'dim': 4}, 'action': {'dim': 2}, 'dynamics': {'dim': 2}}

    def __init__(self,
                 init_state=None,
                 inertial_prop=None,
                 random_init=True,
                 initial_attacker=np.array([-0.5, 0.5]), 
                 initial_defender=np.array([0.0, 0.0]),
                 ctrl_freq=200,
                 # custom args
                 obs_goal_horizon=0,
                 obs_wrap_angle=False,
                 rew_state_weight=1.0,
                 rew_act_weight=0.0001,
                 rew_exponential=True,
                 done_on_out_of_bound=True,
                 seed=42,  # Hanyang: feed the seed
                 # Hanyang: adversary settings
                 adversary_disturbance='action',
                 adversary_disturbance_offset=0.0,
                 adversary_disturbance_scale=1.0,
                 **kwargs
                 ):
        '''Initialize a cartpole environment.

        Args:
            init_state  (ndarray/dict, optional): The initial state of the environment.
                (x, x_dot, theta, theta_dot).
            inertial_prop (dict, optional): The ground truth inertial properties of the environment.
            obs_goal_horizon (int): How many future goal states to append to obervation.
            obs_wrap_angle (bool): If to wrap angle to [-pi, pi] when used in observation.
            rew_state_weight (list/ndarray): Quadratic weights for state in rl reward.
            rew_act_weight (list/ndarray): Quadratic weights for action in rl reward.
            rew_exponential (bool): If to exponentiate negative quadratic cost to positive, bounded [0,1] reward.
            done_on_out_of_bound (bool): If to termiante when state is out of bound.
        '''
        self.obs_goal_horizon = obs_goal_horizon
        self.obs_wrap_angle = obs_wrap_angle
        self.rew_state_weight = np.array(rew_state_weight, ndmin=1, dtype=float)
        self.rew_act_weight = np.array(rew_act_weight, ndmin=1, dtype=float)
        self.rew_exponential = rew_exponential
        self.done_on_out_of_bound = done_on_out_of_bound
        # BenchmarkEnv constructor, called after defining the custom args,
        # since some BenchmarkEnv init setup can be task(custom args)-dependent.
        super().__init__(init_state=init_state, inertial_prop=inertial_prop,
                         adversary_disturbance=adversary_disturbance,
                         adversary_disturbance_offset=adversary_disturbance_offset,
                         adversary_disturbance_scale=adversary_disturbance_scale,
                         **kwargs)
        
        # # Create PyBullet client connection.
        # self.PYB_CLIENT = -1
        # if self.GUI:
        #     self.PYB_CLIENT = p.connect(p.GUI)
        # else:
        #     self.PYB_CLIENT = p.connect(p.DIRECT)
        # # disable urdf caching for randomization via reloading urdf
        # p.setPhysicsEngineParameter(enableFileCaching=0)

        # Set GUI and rendering constants.
        self.RENDER_HEIGHT = int(400)
        self.RENDER_WIDTH = int(640)
        # Game configurations
        self.NUM_ATTACKERS = 1
        self.NUM_DEFENDERS = 1
        self.random_init = random_init
        self.init_state = init_state
        self.frequency = ctrl_freq
        self.initial_players_seed = seed
        self.map={'map': [-1.0, 1.0, -1.0, 1.0]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
        self.des={'goal0': [0.6, 0.8, 0.1, 0.3]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
        self.obstacles={'obs1': [100, 100, 100, 100]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]

        # Set the initial state.
        if init_state is None and self.random_init is False:
            self.init_attackers, self.init_defenders = initial_attacker, initial_defender
        elif init_state is None and self.random_init is True:
            self.init_attackers, self.init_defenders = self.initial_players()
        else:
            raise ValueError('[ERROR] in RARLGame.__init__(), init_state incorrect format.')
        # Save the attacker and defender current states.
        self.current_attacker = self.init_attackers.copy()
        self.current_defender = self.init_defenders.copy()
        self.state = np.concatenate((self.current_attacker.copy(), self.current_defender.copy()))
        #### Initialize/reset counters, players' trajectories and attackers status ###
        self.step_counter = 0
        self.attackers_traj = []
        self.defenders_traj = []
        self.attackers_status = []  # 0 stands for free, -1 stands for captured, 1 stands for arrived 
        self.attackers_actions = []
        self.defenders_actions = []


    def initial_players(self):
        '''Set the initial positions for all players.
        
        Returns:
            attackers (np.ndarray): the initial positions of the attackers
            defenders (np.ndarray): the initial positions of the defenders
        '''
        np.random.seed(self.initial_players_seed)
    
        # Map boundaries
        min_val, max_val = -0.99, 0.99
        
        # # Obstacles and target areas
        # obstacles = [
        #     ([-0.1, 0.1], [-1.0, -0.3]),  # First obstacle
        #     ([-0.1, 0.1], [0.3, 0.6])     # Second obstacle
        # ]
        target = ([0.6, 0.8], [0.1, 0.3])
        
        def is_valid_position(pos):
            x, y = pos
            # Check boundaries
            if not (min_val <= x <= max_val and min_val <= y <= max_val):
                return False
            # # Check obstacles
            # for (ox, oy) in obstacles:
            #     if ox[0] <= x <= ox[1] and oy[0] <= y <= oy[1]:
            #         return False
            # Check target
            if target[0][0] <= x <= target[0][1] and target[1][0] <= y <= target[1][1]:
                return False
            return True
        
        def generate_position(current_seed):
            np.random.seed(current_seed)
            while True:
                pos = np.round(np.random.uniform(min_val, max_val, 2), 1)
                if is_valid_position(pos):
                    return pos
        
        def distance(pos1, pos2):
            return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        
        attacker_seed = self.initial_players_seed
        defender_seed = self.initial_players_seed + 1
        
        while True:
            attacker_pos = generate_position(attacker_seed)
            defender_pos = generate_position(defender_seed)
            
            if distance(attacker_pos, defender_pos) > 1.0:
                break
            defender_seed += 1  # Change the seed for the defender until a valid position is found
        
        self.initial_players_seed += 1
        
        return attacker_pos, defender_pos
    

    def _check_area(self, state, area):
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
    

    def _getAttackersStatus(self):
        """Returns the current status of all attackers.

        Returns
            ndarray, shape (num_attackers,)

        """
        new_status = np.zeros(self.NUM_ATTACKERS)
        if self.step_counter == 0:  # Befire the first step
            return new_status
        else:       
            last_status = self.attackers_status[-1]
            current_attacker_state = self.current_attacker.copy()  # (2,)
            current_defender_state = self.current_defender.copy()

            for num in range(self.NUM_ATTACKERS):
                if last_status[num]:  # attacker has arrived(+1) or been captured(-1)
                    new_status[num] = last_status[num]
                else: # attacker is free last time
                    # check if the attacker arrive at the des this time
                    # print(f"========== The attacker's position is {current_attacker_state} in the _getAttackersStatus() in ReachAvoidGame.py. ========= \n")
                    # print(f"The des is {self.des} in the _getAttackersStatus() in ReachAvoidGame.py. \n")
                    if self._check_area(current_attacker_state, self.des):
                        new_status[num] = 1
                    else:
                        # check if the attacker is captured
                        for j in range(self.NUM_DEFENDERS):
                            if np.linalg.norm(current_attacker_state - current_defender_state) <= 0.1:  # Hanyang: 0.1 is the threshold
                                new_status[num] = -1
                                break

            return new_status


    def reset(self, seed=None):
        '''(Re-)initializes the environment to start an episode.

        Mandatory to call at least once after __init__().

        Args:
            seed (int): An optional seed to reseed the environment.

        Returns:
            obs (ndarray): The initial state of the environment.
            info (dict): A dictionary with information about the dynamics and constraints symbolic models.
        '''

        super().before_reset(seed=seed)
        # Initialize the environment.
        if self.init_state is None and self.random_init is False:
            self.init_attackers, self.init_defenders = self.init_attackers.copy(), self.init_defenders.copy()
        elif self.init_state is None and self.random_init is True:
            self.init_attackers, self.init_defenders = self.initial_players()
        else:
            raise ValueError('[ERROR] in RARLGame.__init__(), init_state incorrect format.')
        
        # Save the attacker and defender current states.
        self.current_attacker = self.init_attackers.copy()
        self.current_defender = self.init_defenders.copy()
        self.state = np.concatenate((self.current_attacker.copy(), self.current_defender.copy()))
        #### Initialize/reset counters, players' trajectories and attackers status ###
        self.step_counter = 0
        self.attackers_traj = []
        self.defenders_traj = []
        self.attackers_status = []  # 0 stands for free, -1 stands for captured, 1 stands for arrived 
        self.attackers_actions = []
        self.defenders_actions = []
        # Log the state and trajectory information
        self.attackers_traj.append(self.current_attacker.copy())
        self.defenders_traj.append(self.current_defender.copy())
        self.attackers_status.append(self._getAttackersStatus().copy())

        obs, info = self._get_observation(), self._get_reset_info()
        obs, info = super().after_reset(obs, info)
        # Return either an observation and dictionary or just the observation.
        if self.INFO_IN_RESET:
            return obs, info
        else:
            return obs
        

    def step(self, action):
        '''Advances the environment by one control step.

        Args:
            action (ndarray): The action applied to the environment for the step.

        Returns:
            obs (ndarray): The state of the environment after the step.
            reward (float): The scalar reward/cost of the step.
            done (bool): Whether the conditions for the end of an episode are met in the step.
            info (dict): A dictionary with information about the constraints evaluations and violations.
        '''

        processed_action = super().before_step(action)
        # Advance the simulation.
        self._advance_simulation(processed_action)
        # Update the state.
        self.state = np.concatenate((self.current_attacker.copy(), self.current_defender.copy()))
        # Log the state and trajectory information
        self.attackers_traj.append(self.current_attacker.copy())
        self.defenders_traj.append(self.current_defender.copy())
        self.attackers_status.append(self._getAttackersStatus().copy())
        self.step_counter += 1
        # Standard Gym return.
        obs = self._get_observation()
        rew = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        obs, rew, done, info = super().after_step(obs, rew, done, info)
        return obs, rew, done, info


    def close(self):
        '''Clean up the environment and PyBullet connection.'''
        print('[INFO] in RARLGame.close(), closing the environment.')
        # if self.PYB_CLIENT >= 0:
        #     p.disconnect(physicsClientId=self.PYB_CLIENT)
        # self.PYB_CLIENT = -1


    def _set_action_space(self):
        # Hanyang: for adversarial agent action space
        defender_lower_bound = np.array([-1.0, -1.0])
        defender_upper_bound = np.array([+1.0, +1.0])
    
        defenders_lower_bound = np.array([defender_lower_bound for i in range(1)])
        defenders_upper_bound = np.array([defender_upper_bound for i in range(1)])
        # Flatten the lower and upper bounds to ensure the action space shape is (2,)
        act_lower_bound = defenders_lower_bound.flatten()
        act_upper_bound = defenders_upper_bound.flatten()

        self.action_scale = 1
        self.physical_action_bounds = (-1 * np.atleast_1d(self.action_scale), np.atleast_1d(self.action_scale))

        self.action_space = spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)


    def _set_observation_space(self):
        # Hanyang: for adversarial agent observation space
        '''Sets the observation space of the environment.'''
        
        attacker_lower_bound = np.array([-1.0, -1.0])
        attacker_upper_bound = np.array([+1.0, +1.0])
        defender_lower_bound = np.array([-1.0, -1.0])
        defender_upper_bound = np.array([+1.0, +1.0])
        
        attackers_lower_bound = np.array([attacker_lower_bound for i in range(1)])
        attackers_upper_bound = np.array([attacker_upper_bound for i in range(1)])
        defenders_lower_bound = np.array([defender_lower_bound for i in range(1)])
        defenders_upper_bound = np.array([defender_upper_bound for i in range(1)])
            
        obs_lower_bound = np.concatenate((attackers_lower_bound, defenders_lower_bound), axis=0)
        obs_upper_bound = np.concatenate((attackers_upper_bound, defenders_upper_bound), axis=0)
        
        # Flatten the lower and upper bounds to ensure the observation space shape is (4,)
        obs_lower_bound = obs_lower_bound.flatten()
        obs_upper_bound = obs_upper_bound.flatten()

        self.state_space = spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)


    def _preprocess_control(self, action):
        '''Converts the raw action input into the one used by .step().

        Args:
            action (ndarray): The raw action input.

        Returns:
            force (float): The scalar, clipped force to apply to the cart.
        '''
        action = self.denormalize_action(action)
        self.current_physical_action = action

        # # Apply disturbances.
        # if 'action' in self.disturbances:  
        #     action = self.disturbances['action'].apply(action, self)
        # if self.adversary_disturbance == 'action':  # Hanyang: default is None in benchmark.py
        #     action = action + self.adv_action
        # self.current_noisy_physical_action = action

        # Save the actual input.
        processed_action = np.clip(action, self.physical_action_bounds[0], self.physical_action_bounds[1])
        self.current_clipped_action = processed_action

        return processed_action


    def normalize_action(self, action):
        '''Converts a physical action into an normalized action if necessary.

        Args:
            action (ndarray): The action to be converted.

        Returns:
            normalized_action (ndarray): The action in the correct action space.
        '''
        if self.NORMALIZED_RL_ACTION_SPACE:
            action = action / self.action_scale

        return action


    def denormalize_action(self, action):
        '''Converts a normalized action into a physical action if necessary.

        Args:
            action (ndarray): The action to be converted.

        Returns:
            physical_action (ndarray): The physical action.
        '''
        if self.NORMALIZED_RL_ACTION_SPACE:
            action = self.action_scale * action

        return action
    

    def _sig_step(self, current_state, action, speed):
        '''Update the state of the game using the single integrator dynamics.

        Args:
            current_state (ndarray, (2,)): The current state of the cartpole.
            action (ndarray, (2,)): The action to apply to the cartpole.
            speed (float): The speed at which to apply the

        Returns:
            x_new (float): The new x position of the cartpole.
            y_new (float): The new y position of the cartpole.
        '''

        def dynamics(current_state, action, speed):
            dx = speed * action[0]
            dy = speed * action[1]
            return (dx, dy)
        # 4th order Runge-Kutta
        dx, dy = dynamics(current_state, action, speed)
        # Compute the k1 terms
        k1_x = 1/self.frequency * dx
        k1_y = 1/self.frequency * dy
        
        # Compute the k2 terms
        k2_x = 1/self.frequency * dx
        k2_y = 1/self.frequency * dy
        
        # Compute the k3 terms
        k3_x = 1/self.frequency * dx
        k3_y = 1/self.frequency * dy
        
        # Compute the k4 terms
        k4_x = 1/self.frequency * dx
        k4_y = 1/self.frequency * dy
        
        # Combine to get the final state
        next_x = current_state[0] + (1/6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        next_y = current_state[1] + (1/6) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)

        return np.array([next_x, next_y]) 


    def _advance_simulation(self, processed_action):
        # Hanyang: add adversarial action forward process here
        '''Apply the processed_action and adversarial actions to the defender and the attacker respectively.

        Args:
            processed_action (float): The control action for the defender.
        '''
        # Apply the defender's action.
        self.current_defender = self._sig_step(self.current_defender.copy(), processed_action.copy(), 1.5)
        # print(f"========== The defender's position is {self.current_defender} in the _advance_simulation() in ReachAvoidGame.py. ========= \n")
        # Apply the attacker's action.
        assert self.adversary_disturbance == 'action'
        assert self.adv_action is not None, 'Adversary action is required.'
        self.current_attacker = self._sig_step(self.current_attacker.copy(), self.adv_action.copy(), 1.0)
        self.adv_action = None
        # print(f"========== The attacker's position is {self.current_attacker} in the _advance_simulation() in ReachAvoidGame.py. ========= \n")


    def _get_observation(self):
        '''Returns the current observation (state) of the environment.

        Returns:
            obs (ndarray): The state (ax, ay, dx, dy) of the 1 vs. 1 reach-avoid game.
        '''
        if not np.array_equal(self.state, np.clip(self.state, self.state_space.low, self.state_space.high)) and self.VERBOSE:
            print('[WARNING]: observation was clipped in RARLGame._get_observation().')
        # Apply observation disturbance.
        obs = deepcopy(self.state)
        if 'observation' in self.disturbances:
            obs = self.disturbances['observation'].apply(obs, self)
        if self.at_reset:
            obs = self.extend_obs(obs, 1)
        else:
            obs = self.extend_obs(obs, self.ctrl_step_counter + 2)
        return obs


    def _get_reward(self):
        """Computes the current reward value.

        Once the attacker is captured: +200
        Once the attacker arrived at the goal: -200
        The defender hits the obstacle: -200
        One step and nothing happens: -current_relative_distance
        In status, 0 stands for free, -1 stands for captured, 1 stands for arrived

        Returns
        -------
        float
            The reward.

        """
        last_attacker_status = self.attackers_status[-2]
        current_attacker_status = self.attackers_status[-1]
        reward = 0.0
        # for num in range(1):
        #     reward += (current_attacker_status[num] - last_attacker_status[num]) * (-200)
        status_change = current_attacker_status[0] - last_attacker_status[0]
        if status_change == 1:  # attacker arrived
            reward += -200
        elif status_change == -1:  # attacker is captured
            reward += 200
        else:  # attacker is free
            reward += 0.0
        # check the defender status
        current_defender_state = self.current_defender.copy()
        reward += -100 if self._check_area(current_defender_state, self.obstacles) else 0.0  # which is 0 when there's no obs
        # check the relative distance difference or relative distance
        current_attacker_state = self.current_attacker.copy()
        current_relative_distance = np.linalg.norm(current_attacker_state - current_defender_state)  # [0.10, 2.82]

        reward += -(current_relative_distance)
        
        return reward


    def _get_done(self):
        """Computes the current done value.
        done = True if all attackers have arrived or been captured.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        # defender hits the obstacle or the attacker is captured or the attacker has arrived or the attacker hits the obstacle
        # check the attacker status
        current_attacker_status = self.attackers_status[-1]
        attacker_done = np.all((current_attacker_status == 1) | (current_attacker_status == -1))
        # if attacker_done:
        #     print(" ========== The attacker is captured or arrived in the _computeTerminated() in ReachAvoidGame.py. ========= \n")
        # check the defender status: hit the obstacle, or the attacker is captured
        current_defender_state = self.current_defender.copy()
        defender_done = self._check_area(current_defender_state, self.obstacles)
        # print(f"========== The defender_done is {defender_done} in ReachAvoidGame.py. ========= \n")
        # if defender_done:
            # print(" ========== The defender hits the obstacle in the _computeTerminated() in ReachAvoidGame.py. ========= \n")
            
        # final done
        done = True if attacker_done or defender_done else False
        
        return done


    def _get_info(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        info = {}
        info['current_steps'] = self.step_counter
        info['current_attackers_status'] = self.attackers_status[-1]
        
        return info 


    def _get_reset_info(self):
        '''Generates the info dictionary returned by every call to .reset().

        Returns:
            info (dict): A dictionary with information about the dynamics and constraints symbolic models.
        '''
        info = {}
        # info['symbolic_model'] = self.symbolic
        # info['physical_parameters'] = {
        #     'pole_effective_length': self.OVERRIDDEN_EFFECTIVE_POLE_LENGTH,
        #     'pole_mass': self.OVERRIDDEN_POLE_MASS,
        #     'cart_mass': self.OVERRIDDEN_CART_MASS
        # }
        # info['x_reference'] = self.X_GOAL
        # info['u_reference'] = self.U_GOAL
        # if self.constraints is not None:
        #     info['symbolic_constraints'] = self.constraints.get_all_symbolic_models()
        #     # NOTE: Cannot evaluate constraints on reset/without inputs.
        #     info['constraint_values'] = self.constraints.get_values(self, only_state=True)  # Fix for input constraints only
        info['init_state'] = self.state
        return info


    def _parse_urdf_parameters(self, file_name):
        '''Parses an URDF file for the robot's properties.

        Args:
            file_name (str, optional): The .urdf file from which the properties should be pased.

        Returns:
            EFFECTIVE_POLE_LENGTH (float): The effective pole length.
            POLE_MASS (float): The pole mass.
            CART_MASS (float): The cart mass.
        '''
        URDF_TREE = (etxml.parse(file_name)).getroot()
        EFFECTIVE_POLE_LENGTH = 0.5 * float(URDF_TREE[3][0][0][0].attrib['size'].split(' ')[-1])  # Note: HALF length of pole.
        POLE_MASS = float(URDF_TREE[3][1][1].attrib['value'])
        CART_MASS = float(URDF_TREE[1][2][0].attrib['value'])
        return EFFECTIVE_POLE_LENGTH, POLE_MASS, CART_MASS


    def _create_urdf(self, file_name, length=None, inertia=None):
        '''For domain randomization.

        Args:
            file_name (str): Path to the base URDF with attributes to modify.
            length (float): Overriden effective pole length.
            inertia (float): Pole inertia (symmetric, Ixx & Iyy).

        Returns:
            tree (obj): xml tree object.
        '''
        tree = etxml.parse(file_name)
        root = tree.getroot()
        # Overwrite pod length.
        if length is not None:
            # Pole visual geometry box.
            out = root[3][0][0][0].attrib['size']
            out = ' '.join(out.split(' ')[:-1] + [str(2 * length)])
            root[3][0][0][0].attrib['size'] = out
            # Pole visual origin.
            out = root[3][0][1].attrib['xyz']
            out = ' '.join(out.split(' ')[:-1] + [str(length)])
            root[3][0][1].attrib['xyz'] = out
            # Pole inertial origin.
            out = root[3][1][0].attrib['xyz']
            out = ' '.join(out.split(' ')[:-1] + [str(length)])
            root[3][1][0].attrib['xyz'] = out
            # Pole inertia.
            root[3][1][2].attrib['ixx'] = str(inertia)
            root[3][1][2].attrib['iyy'] = str(inertia)
            root[3][1][2].attrib['izz'] = str(0.0)
            # Pole collision geometry box.
            out = root[3][2][0][0].attrib['size']
            out = ' '.join(out.split(' ')[:-1] + [str(2 * length)])
            root[3][2][0][0].attrib['size'] = out
            # Pole collision origin.
            out = root[3][2][1].attrib['xyz']
            out = ' '.join(out.split(' ')[:-1] + [str(length)])
            root[3][2][1].attrib['xyz'] = out
        return tree
    
    
    # Hanyang: the dynamics is here, no disturbances are considered
    def _setup_symbolic(self, prior_prop={}, **kwargs):
        '''Creates symbolic (CasADi) models for dynamics, observation, and cost.

        Args:
            prior_prop (dict): specify the prior inertial prop to use in the symbolic model.
        '''
        length = prior_prop.get('pole_length', self.EFFECTIVE_POLE_LENGTH)
        m = prior_prop.get('pole_mass', self.POLE_MASS)
        M = prior_prop.get('cart_mass', self.CART_MASS)
        Mm, ml = m + M, m * length
        g = self.GRAVITY_ACC
        dt = self.CTRL_TIMESTEP
        # Input variables.
        x = cs.MX.sym('x')
        x_dot = cs.MX.sym('x_dot')
        theta = cs.MX.sym('theta')
        theta_dot = cs.MX.sym('theta_dot')
        X = cs.vertcat(x, x_dot, theta, theta_dot)
        U = cs.MX.sym('U')
        nx = 4
        nu = 1
        # Dynamics.
        temp_factor = (U + ml * theta_dot**2 * cs.sin(theta)) / Mm
        theta_dot_dot = ((g * cs.sin(theta) - cs.cos(theta) * temp_factor) / (length * (4.0 / 3.0 - m * cs.cos(theta)**2 / Mm)))
        X_dot = cs.vertcat(x_dot, temp_factor - ml * theta_dot_dot * cs.cos(theta) / Mm, theta_dot, theta_dot_dot)
        # Observation.
        Y = cs.vertcat(x, x_dot, theta, theta_dot)
        # Define cost (quadratic form).
        Q = cs.MX.sym('Q', nx, nx)
        R = cs.MX.sym('R', nu, nu)
        Xr = cs.MX.sym('Xr', nx, 1)
        Ur = cs.MX.sym('Ur', nu, 1)
        cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T @ R @ (U - Ur)
        # Define dynamics and cost dictionaries.
        dynamics = {'dyn_eqn': X_dot, 'obs_eqn': Y, 'vars': {'X': X, 'U': U}}
        cost = {'cost_func': cost_func, 'vars': {'X': X, 'U': U, 'Xr': Xr, 'Ur': Ur, 'Q': Q, 'R': R}}
        # Additional params to cache
        params = {
            # prior inertial properties
            'pole_length': length,
            'pole_mass': m,
            'cart_mass': M,
            # equilibrium point for linearization
            'X_EQ': np.zeros(self.state_dim),
            'U_EQ': np.atleast_2d(self.U_GOAL)[0, :],
        }
        # Setup symbolic model.
        self.symbolic = SymbolicModel(dynamics=dynamics, cost=cost, dt=dt, params=params)


    def render(self):
        print(f"========== render function has not been implemented in the RARLGame.py. ========= \n")
        return None