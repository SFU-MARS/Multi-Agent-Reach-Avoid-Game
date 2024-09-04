'''Base environment class module for the reach-avoid game.

'''

import numpy as np

from odp.Grid import Grid
from safe_control_gym.envs.gym_game.utilities import find_sign_change1vs0, spa_deriv, find_sign_change1vs1
from safe_control_gym.envs.gym_game.BaseRLGame import BaseRLGameEnv
from safe_control_gym.envs.gym_game.BaseGame import Dynamics

from gymnasium import spaces



class ReachAvoidGameEnv(BaseRLGameEnv):
    """Multi-agent reach-avoid games class for SingleIntegrator dynamics."""

    ################################################################################
    
    def __init__(self,
                 num_attackers: int=1,
                 num_defenders: int=1,
                 attackers_dynamics=Dynamics.SIG,  
                 defenders_dynamics=Dynamics.FSIG,
                 initial_attacker: np.ndarray=None,  # shape (num_atackers, state_dim), np.array([[-0.4, -0.8]])
                 initial_defender: np.ndarray=None,  # shape (num_defenders, state_dim), np.array([[0.3, -0.8]])
                 ctrl_freq: int = 200,
                 seed = 42,
                 random_init = True,
                 init_type = 'difficulty_init',
                 uMode="min", 
                 dMode="max",
                 output_folder='results',
                 game_length_sec=10,
                 map={'map': [-1.0, 1.0, -1.0, 1.0]},  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
                 des={'goal0': [0.6, 0.8, 0.1, 0.3]},  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
                 obstacles: dict = {'obs1': [-0.1, 0.1, -1.0, -0.3], 'obs2': [-0.1, 0.1, 0.3, 0.6]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
                 ):
        """Initialization of a generic aviary environment.

        Parameters
        ----------
        num_attackers : int, optional
            The number of attackers in the environment.
        num_defenders : int, optional
            The number of defenders in the environment.
        initial_attacker : np.ndarray, optional
            The initial states of the attackers.
        initial_defender : np.ndarray, optional
            The initial states of the defenders.
        attacker_physics : Physics instance
            A dictionary contains the dynamics of the attackers.
        defender_physics : Physics instance
            A dictionary contains the dynamics of the defenders.
        ctrl_freq : int, optional
            The control frequency of the environment.
        seed : int, optional
        random_init: bool, optional
        init_type: str, optional
        uMode : str, optional
            The mode of the attacker, default is "min".
        dMode : str, optional
            The mode of the defender, default is "max".
        output_folder : str, optional
            The folder where to save logs.
        game_length_sec=20 : int, optional
            The maximum length of the game in seconds.
        map : dict, optional
            The map of the environment, default is rectangle.
        des : dict, optional
            The goal in the environment, default is a rectangle.
        obstacles : dict, optional
            The obstacles in the environment, default is rectangle.

        """
           
        super().__init__(num_attackers=num_attackers, num_defenders=num_defenders, 
                         attackers_dynamics=attackers_dynamics, defenders_dynamics=defenders_dynamics, 
                         initial_attacker=initial_attacker, initial_defender=initial_defender, 
                         ctrl_freq=ctrl_freq, seed=seed, random_init=random_init, init_type=init_type, 
                         output_folder=output_folder
                         )
        
        assert map is not None, "Map must be provided in the game."
        assert des is not None, "Destination must be provided in the game."
        
        self.map = map
        self.des = des
        self.obstacles = obstacles
        self.GAME_LENGTH_SEC = game_length_sec
        self.uMode = uMode
        self.dMode = dMode
        # Load necessary values for the attacker control
        self.grid1vs0 = Grid(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 2, np.array([100, 100])) 
        # self.grid1vs1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45]))
        # self.value1vs1 = np.load('safe_control_gym/envs/gym_game/values/1vs1Attacker.npy')
        self.value1vs0 = np.load('safe_control_gym/envs/gym_game/values/1vs0Attacker.npy')

    
    def step(self, action):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | (dim_action, )
            The input action for the defender.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is over, check the specific implementation of `_computeTerminated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is truncated, check the specific implementation of `_computeTruncated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is trunacted, always false.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        
        #### Step the simulation using the desired physics update ##        
        attackers_action = self._computeAttackerActions()  # ndarray, shape (num_defenders, dim_action)
        clipped_action = np.clip(action.copy(), -1.0, +1.0)  # Hanyang: clip the action to [-1, 1]
        defenders_action = clipped_action.reshape(self.NUM_DEFENDERS, 2)  # ndarray, shape (num_defenders, dim_action)
        self.attackers.step(attackers_action)
        self.defenders.step(defenders_action)
        #### Update and all players' information #####
        self._updateAndLog()
        #### Prepare the return values #############################
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        
        #### Advance the step counter ##############################
        self.step_counter += 1
        #### Log the actions taken by the attackers and defenders ################
        self.attackers_actions.append(attackers_action)
        self.defenders_actions.append(defenders_action)
        
        return obs, reward, terminated, truncated, info
        
    
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
            current_attacker_state = self.attackers._get_state()
            current_defender_state = self.defenders._get_state()

            for num in range(self.NUM_ATTACKERS):
                if last_status[num]:  # attacker has arrived(+1) or been captured(-1)
                    new_status[num] = last_status[num]
                else: # attacker is free last time
                    # check if the attacker arrive at the des this time
                    if self._check_area(current_attacker_state[num], self.des):
                        new_status[num] = 1
                    # # check if the attacker gets stuck in the obstacles this time (it won't usually)
                    # elif self._check_area(current_attacker_state[num], self.obstacles):
                    #     new_status[num] = -1
                    #     break
                    else:
                        # check if the attacker is captured
                        for j in range(self.NUM_DEFENDERS):
                            if np.linalg.norm(current_attacker_state[num] - current_defender_state[j]) <= 0.1:  # Hanyang: 0.1 is the threshold
                                new_status[num] = -1
                                break

            return new_status
        

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
    

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_PLAYERS*dim, ), concatenate the attackers' and defenders' observations.

        """
        obs = self.state.flatten()

        return obs
    
    
    def _computeReward(self):
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
        # for num in range(self.NUM_ATTACKERS):
        #     reward += (current_attacker_status[num] - last_attacker_status[num]) * (-200)
        status_change = current_attacker_status[0] - last_attacker_status[0]
        if status_change == 1:  # attacker arrived
            reward += -200
        elif status_change == -1:  # attacker is captured
            reward += 200
        else:  # attacker is free
            reward += 0.0
        # check the defender status
        current_defender_state = self.defenders._get_state().copy()
        reward += -100 if self._check_area(current_defender_state[0], self.obstacles) else 0.0  # which is 0 when there's no obs
        # check the relative distance difference or relative distance
        current_attacker_state = self.attackers._get_state().copy()  # (num_agents, state_dim)
        current_relative_distance = np.linalg.norm(current_attacker_state[0] - current_defender_state[0])  # [0.10, 2.82]
        # last_relative_distance = np.linalg.norm(self.attackers_traj[-2][0] - self.defenders_traj[-2][0])
        # reward += (current_relative_distance - last_relative_distance) * -1.0 / (2*np.sqrt(2))
        reward += -(current_relative_distance)
        
        return reward

    
    def _computeTerminated(self):
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
        current_defender_state = self.defenders._get_state().copy()
        defender_done = self._check_area(current_defender_state[0], self.obstacles)
        # print(f"========== The defender_done is {defender_done} in ReachAvoidGame.py. ========= \n")
        # if defender_done:
            # print(" ========== The defender hits the obstacle in the _computeTerminated() in ReachAvoidGame.py. ========= \n")
            
        # final done
        done = True if attacker_done or defender_done else False
        
        return done
        
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        if self.step_counter/self.CTRL_FREQ > self.GAME_LENGTH_SEC:
            return True
        else:
            return False

    
    def _computeInfo(self):
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
    

    def _computeAttackerActions(self):
        """Computes the current actions of the attackers.

        """
        control_attackers = np.zeros((self.NUM_ATTACKERS, 2))
        current_attacker_state = self.attackers._get_state().copy()
        for i in range(self.NUM_ATTACKERS):
            neg2pos, pos2neg = find_sign_change1vs0(self.grid1vs0, self.value1vs0, current_attacker_state[i])
            if len(neg2pos):
                control_attackers[i] = self.attacker_control_1vs0(self.grid1vs0, self.value1vs0, current_attacker_state[i], neg2pos)
            else:
                control_attackers[i] = (0.0, 0.0)

        return control_attackers
    

    def attacker_control_1vs0(self, grid1vs0, value1vs0, attacker, neg2pos):
        """Return a list of 2-dimensional control inputs of one defender based on the value function
        
        Args:
        grid1vs0 (class): the corresponding Grid instance
        value1vs0 (ndarray): 1v1 HJ reachability value function with only final slice
        attacker (ndarray, (dim,)): the current state of one attacker
        neg2pos (list): the positions of the value function that change from negative to positive
        """
        current_value = grid1vs0.get_value(value1vs0[..., 0], list(attacker))
        if current_value > 0:
            value1vs0 = value1vs0 - current_value
        v = value1vs0[..., neg2pos] # Minh: v = value1v0[..., neg2pos[0]]
        spat_deriv_vector = spa_deriv(grid1vs0.get_index(attacker), v, grid1vs0)
        opt_a1, opt_a2 = self.optCtrl_1vs0(spat_deriv_vector)

        return (opt_a1, opt_a2)
    

    def attacker_control_1vs1(self, grid1vs1, value1vs1, current_state, neg2pos):
        """Return a list of 2-dimensional control inputs of one defender based on the value function
        
        Args:
        grid1vs1 (class): the corresponding Grid instance
        value1vs1 (ndarray): 1v1 HJ reachability value function with only final slice
        current_state (ndarray, (dim,)): the current state of one attacker + one defender
        neg2pos (list): the positions of the value function that change from negative to positive
        """
        current_value = grid1vs1.get_value(value1vs1[..., 0], list(current_state))
        if current_value > 0:
            value1vs1 = value1vs1 - current_value
        v = value1vs1[..., neg2pos]
        spat_deriv_vector = spa_deriv(grid1vs1.get_index(current_state), v, grid1vs1)
        opt_a1, opt_a2 = self.optCtrl_1vs1(spat_deriv_vector)

        return (opt_a1, opt_a2)
    
    
    def optCtrl_1vs1(self, spat_deriv):
        """Computes the optimal control (disturbance) for the attacker in a 1 vs. 1 game.
        
        Parameters:
            spat_deriv (tuple): spatial derivative in all dimensions
        
        Returns:
            tuple: a tuple of optimal control of the defender (disturbances)
        """
        opt_u1 = self.attackers.uMax
        opt_u2 = self.attackers.uMax
        deriv1 = spat_deriv[0]
        deriv2 = spat_deriv[1]
        crtl_len = np.sqrt(deriv1*deriv1 + deriv2*deriv2)
        if self.uMode == "min":
            if crtl_len == 0:
                opt_u1 = 0.0
                opt_u2 = 0.0
            else:
                opt_u1 = - self.attackers.speed * deriv1 / crtl_len
                opt_u2 = - self.attackers.speed * deriv2 / crtl_len
        else:
            if crtl_len == 0:
                opt_u1 = 0.0
                opt_u2 = 0.0
            else:
                opt_u1 = self.defenders.speed * deriv1 / crtl_len
                opt_u2 = self.defenders.speed * deriv2 / crtl_len

        return (opt_u1, opt_u2)


    def optCtrl_1vs0(self, spat_deriv):
        """Computes the optimal control (disturbance) for the attacker in a 1 vs. 0 game.
        
        Parameters:
            spat_deriv (tuple): spatial derivative in all dimensions
        
        Returns:
            tuple: a tuple of optimal control of the defender (disturbances)
        """
        opt_a1 = self.attackers.uMax
        opt_a2 = self.attackers.uMax
        deriv1 = spat_deriv[0]
        deriv2 = spat_deriv[1]
        ctrl_len = np.sqrt(deriv1*deriv1 + deriv2*deriv2)
        if self.uMode == "min":
            if ctrl_len == 0:
                opt_a1 = 0.0
                opt_a2 = 0.0
            else:
                opt_a1 = - self.attackers.speed * deriv1 / ctrl_len
                opt_a2 = - self.attackers.speed * deriv2 / ctrl_len
        else:
            if ctrl_len == 0:
                opt_a1 = 0.0
                opt_a2 = 0.0
            else:
                opt_a1 = self.attackers.speed * deriv1 / ctrl_len
                opt_a2 = self.attackers.speed * deriv2 / ctrl_len

        return (opt_a1, opt_a2)




class ReachAvoidGameTest(ReachAvoidGameEnv):
    NAME = 'reach_avoid_test'
    def __init__(self, *args,  **kwargs):  # distb_level=1.0, randomization_reset=False,
        # Set disturbance_type to 'fixed' regardless of the input
        # kwargs['random_init'] = False
        # kwargs['initial_attacker'] = np.array([[-0.5, 0.5]])
        # kwargs['initial_defender'] = np.array([[0.3, -0.2]])
        # kwargs['seed'] = 2024
        super().__init__(*args, **kwargs)
    
    
    def initial_players(self):
        '''Set the initial positions for all players.
        
        Returns:
            attackers (np.ndarray): the initial positions of the attackers
            defenders (np.ndarray): the initial positions of the defenders
        '''
        np.random.seed(self.initial_players_seed)
    
        # Map boundaries
        min_val, max_val = -0.99, 0.99
        
        # Obstacles and target areas
        obstacles = [
            ([-0.1, 0.1], [-1.0, -0.3]),  # First obstacle
            ([-0.1, 0.1], [0.3, 0.6])     # Second obstacle
        ]
        target = ([0.6, 0.8], [0.1, 0.3])
        
        def is_valid_position(pos):
            x, y = pos
            # Check boundaries
            if not (min_val <= x <= max_val and min_val <= y <= max_val):
                return False
            # Check obstacles
            for (ox, oy) in obstacles:
                if ox[0] <= x <= ox[1] and oy[0] <= y <= oy[1]:
                    return False
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
        
        return np.array([attacker_pos]), np.array([defender_pos])
    

    def _actionSpace(self):
        """Returns the action space of the environment.
        Formulation: [defenders' action spaces]
        Returns
        -------
        spaces.Box
            A Box of size NUM_DEFENDERS x 2, or 1, depending on the action type.

        """
        
        if self.DEFENDER_PHYSICS == Dynamics.SIG or self.DEFENDER_PHYSICS == Dynamics.FSIG:
            defender_lower_bound = np.array([-1.0, -1.0])
            defender_upper_bound = np.array([+1.0, +1.0])
        elif self.DEFENDER_PHYSICS == Dynamics.DUB3D:
            defender_lower_bound = np.array([-1.0])
            defender_upper_bound = np.array([+1.0])
        else:
            print("[ERROR] in Defender Action Space, BaseRLGameEnv._actionSpace()")
            exit()
        
        # attackers_lower_bound = np.array([attacker_lower_bound for i in range(self.NUM_ATTACKERS)])
        # attackers_upper_bound = np.array([attacker_upper_bound for i in range(self.NUM_ATTACKERS)])

        # if self.NUM_DEFENDERS > 0:
        #     defenders_lower_bound = np.array([defender_lower_bound for i in range(self.NUM_DEFENDERS)])
        #     defenders_upper_bound = np.array([defender_upper_bound for i in range(self.NUM_DEFENDERS)])
            
        #     act_lower_bound = np.concatenate((attackers_lower_bound, defenders_lower_bound), axis=0)
        #     act_upper_bound = np.concatenate((attackers_upper_bound, defenders_upper_bound), axis=0)
        # else:
        #     act_lower_bound = attackers_lower_bound
        #     act_upper_bound = attackers_upper_bound
            
        defenders_lower_bound = np.array([defender_lower_bound for i in range(self.NUM_DEFENDERS)])
        defenders_upper_bound = np.array([defender_upper_bound for i in range(self.NUM_DEFENDERS)])
        # Flatten the lower and upper bounds to ensure the action space shape is (4,)
        act_lower_bound = defenders_lower_bound
        act_upper_bound = defenders_upper_bound

        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)


    def _observationSpace(self):
        """Returns the observation space of the environment.
        Formulation: [attackers' obs spaces, defenders' obs spaces]
        Returns
        -------
        ndarray
            A Box() of shape NUM_PLAYERS x 2, or 3 depending on the observation type.

        """
        
        if self.ATTACKER_PHYSICS == Dynamics.SIG or self.ATTACKER_PHYSICS == Dynamics.FSIG:
            attacker_lower_bound = np.array([-1.0, -1.0])
            attacker_upper_bound = np.array([+1.0, +1.0])
        elif self.ATTACKER_PHYSICS == Dynamics.DUB3D:
            attacker_lower_bound = np.array([-1.0, -1.0, -1.0])
            attacker_upper_bound = np.array([+1.0, +1.0, +1.0])
        else:
            print("[ERROR] Attacker Obs Space in BaseRLGameEnv._observationSpace()")
            exit()
        
        if self.DEFENDER_PHYSICS == Dynamics.SIG or self.DEFENDER_PHYSICS == Dynamics.FSIG:
            defender_lower_bound = np.array([-1.0, -1.0])
            defender_upper_bound = np.array([+1.0, +1.0])
        elif self.DEFENDER_PHYSICS == Dynamics.DUB3D:
            defender_lower_bound = np.array([-1.0, -1.0, -1.0])
            defender_upper_bound = np.array([+1.0, +1.0, +1.0])
        else:
            print("[ERROR] in Defender Obs Space, BaseRLGameEnv._observationSpace()")
            exit()
        
        attackers_lower_bound = np.array([attacker_lower_bound for i in range(self.NUM_ATTACKERS)])
        attackers_upper_bound = np.array([attacker_upper_bound for i in range(self.NUM_ATTACKERS)])

        if self.NUM_DEFENDERS > 0:
            defenders_lower_bound = np.array([defender_lower_bound for i in range(self.NUM_DEFENDERS)])
            defenders_upper_bound = np.array([defender_upper_bound for i in range(self.NUM_DEFENDERS)])
            
            obs_lower_bound = np.concatenate((attackers_lower_bound, defenders_lower_bound), axis=0)
            obs_upper_bound = np.concatenate((attackers_upper_bound, defenders_upper_bound), axis=0)
        else:
            obs_lower_bound = attackers_lower_bound
            obs_upper_bound = attackers_upper_bound
        
        # Flatten the lower and upper bounds to ensure the observation space shape is (4,)
        obs_lower_bound = obs_lower_bound.reshape(1, 4)
        obs_upper_bound = obs_upper_bound.reshape(1, 4)

        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
    


class ReachAvoidEasierGame(ReachAvoidGameEnv):
    NAME = 'reach_avoid_easier'
    def __init__(self, *args,  **kwargs):  
        kwargs['init_type'] = 'random' # 'distance_init'
        kwargs['obstacles'] = {'obs1': [100, 100, 100, 100]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
        super().__init__(*args, **kwargs)
        self.grid1vs0 = Grid(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 2, np.array([100, 100])) 
        self.grid1vs1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45]))
        self.value1vs1_easier = np.load('safe_control_gym/envs/gym_game/values/1vs1Attacker_easier.npy')
        self.value1vs0_easier = np.load('safe_control_gym/envs/gym_game/values/1vs0Attacker_easier.npy')
        self.value1vs1 = np.load('safe_control_gym/envs/gym_game/values/1vs1Defender_easier.npy')
    

    # def _computeAttackerActions(self):
    #     """Computes the sub-optimal control (1 vs. 0 value function only)current actions of the attackers.

    #     """
    #     control_attackers = np.zeros((self.NUM_ATTACKERS, 2))
    #     current_attacker_state = self.attackers._get_state().copy()

    #     # if current_value >= 0:
    #     for i in range(self.NUM_ATTACKERS):
    #         neg2pos, pos2neg = find_sign_change1vs0(self.grid1vs0, self.value1vs0_easier, current_attacker_state[i])
    #         if len(neg2pos):
    #             control_attackers[i] = self.attacker_control_1vs0(self.grid1vs0, self.value1vs0_easier, current_attacker_state[i], neg2pos)
    #         else:
    #             control_attackers[i] = (0.0, 0.0)

    #     return control_attackers

    
    # def _computeAttackerActions(self):
    #     """Computes the the optimal control (1 vs. 1 value functions only) of the attacker.

    #     """
    #     control_attackers = np.zeros((self.NUM_ATTACKERS, 2))
    #     current_attacker_state = self.attackers._get_state().copy()
    #     current_defender_state = self.defenders._get_state().copy()
    #     current_joint_state = np.concatenate((current_attacker_state[0], current_defender_state[0]))
    #     # print(f"========== The current_joint_state is {current_joint_state} in ReachAvoidEasierGame.py. ========= \n")

    #     for i in range(self.NUM_ATTACKERS):
    #         neg2pos, pos2neg = find_sign_change1vs1(self.grid1vs1, self.value1vs1_easier, current_joint_state)
    #         if len(neg2pos):
    #             control_attackers[i] = self.attacker_control_1vs1(self.grid1vs1, self.value1vs1_easier, current_joint_state, neg2pos)
    #         else:
    #             control_attackers[i] = (0.0, 0.0)

    #     return control_attackers


    def _computeAttackerActions(self):
        """Computes the the sub-optimal + optimal control (1 vs. 0 + 1 vs. 1 value functions) of the attacker.

        """
        control_attackers = np.zeros((self.NUM_ATTACKERS, 2))
        current_attacker_state = self.attackers._get_state().copy()
        current_defender_state = self.defenders._get_state().copy()
        current_joint_state = np.concatenate((current_attacker_state[0], current_defender_state[0]))
        # print(f"========== The current_joint_state is {current_joint_state} in ReachAvoidEasierGame.py. ========= \n")
        current_state_slice = self.grid1vs1.get_index(current_joint_state)

        current_value = self.value1vs1[current_state_slice]
        # print(f"========== The current_value is {current_value} in ReachAvoidEasierGame.py. ========= \n")

        if current_value >= 0:
            for i in range(self.NUM_ATTACKERS):
                neg2pos, pos2neg = find_sign_change1vs0(self.grid1vs0, self.value1vs0_easier, current_attacker_state[i])
                if len(neg2pos):
                    control_attackers[i] = self.attacker_control_1vs0(self.grid1vs0, self.value1vs0_easier, current_attacker_state[i], neg2pos)
                else:
                    control_attackers[i] = (0.0, 0.0)
        else:
            for i in range(self.NUM_ATTACKERS):
                neg2pos, pos2neg = find_sign_change1vs1(self.grid1vs1, self.value1vs1_easier, current_joint_state)
                if len(neg2pos):
                    control_attackers[i] = self.attacker_control_1vs1(self.grid1vs1, self.value1vs1_easier, current_joint_state, neg2pos)
                else:
                    control_attackers[i] = (0.0, 0.0)

        return control_attackers
    

    def initial_players(self):
        '''Set the initial positions for all players.
        
        Returns:
            attackers (np.ndarray): the initial positions of the attackers
            defenders (np.ndarray): the initial positions of the defenders
        '''
        # Map boundaries
        map = ([-0.99, 0.99], [-0.99, 0.99])  # The map boundaries
        # # Obstacles and target areas
        # obstacles = [
        #     ([-0.1, 0.1], [-1.0, -0.3]),  # First obstacle
        #     ([-0.1, 0.1], [0.3, 0.6])     # Second obstacle
        # ]
        target = ([0.6, 0.8], [0.1, 0.3])

        def _is_valid_attacker(pos):
            x, y = pos
            # Check map boundaries
            if not (map[0][0] <= x <= map[0][1] and map[1][0] <= y <= map[1][1]):
                return False
            # # Check obstacles
            # for (ox, oy) in obstacles:
            #     if ox[0] <= x <= ox[1] and oy[0] <= y <= oy[1]:
            #         return False
            # Check target
            if target[0][0] <= x <= target[0][1] and target[1][0] <= y <= target[1][1]:
                return False
            return True
        
        def _is_valid_defender(defender_pos, attacker_pos):
            x, y = defender_pos
            # Check map boundaries
            if not (map[0][0] <= x <= map[0][1] and map[1][0] <= y <= map[1][1]):
                return False
            # # Check obstacles
            # for (ox, oy) in obstacles:
            #     if ox[0] <= x <= ox[1] and oy[0] <= y <= oy[1]:
            #         return False
            # Check the relative distance
            if np.linalg.norm(defender_pos - attacker_pos) <= 0.10:
                return False
            return True
        
        def _generate_attacker_pos():
            """Generate a random position for the attacker.
            
            Returns:
                attacker_pos (tuple): the attacker position.
            """
            while True:
                attacker_x = np.random.uniform(map[0][0], map[0][1])
                attacker_y = np.random.uniform(map[1][0], map[1][1])
                attacker_pos = np.round((attacker_x, attacker_y), 1)
                if _is_valid_attacker(attacker_pos):
                    break
            return attacker_pos
        
        def _generate_random_positions(current_seed, init_player_call_counter):
            """Generate a random position for the attacker and defender.

            Args:
                current_seed (int): the random seed.
                self.init_player_call_counter (int): the init_player function call counter.
            
            Returns:
                attacker_pos (tuple): the attacker position.
                defender_pos (tuple): the defender position.
            """
            np.random.seed(current_seed)
            # Generate the attacker position
            attacker_pos = _generate_attacker_pos()
            # Generate the defender position
            while True:
                defender_x = np.random.uniform(map[0][0], map[0][1])
                defender_y = np.random.uniform(map[1][0], map[1][1])
                defender_pos = np.round((defender_x, defender_y), 1)
                if _is_valid_defender(defender_pos, attacker_pos):
                    break
            
            return attacker_pos, defender_pos
        
        attacker_pos, defender_pos = _generate_random_positions(self.initial_players_seed, self.init_player_call_counter)

        print(f"========== attacker_pos: {attacker_pos} in BaseGame.py. ==========")
        print(f"========== defender_pos: {defender_pos} in BaseGame.py. ==========")
        print(f"========== The relative distance is {np.linalg.norm(attacker_pos - defender_pos):.2f} in BaseGame.py. ========== \n ")
        
        self.initial_players_seed += 1  # Increment the random seed
        self.init_player_call_counter += 1  # Increment the call counter
        
        return np.array([attacker_pos]), np.array([defender_pos])
