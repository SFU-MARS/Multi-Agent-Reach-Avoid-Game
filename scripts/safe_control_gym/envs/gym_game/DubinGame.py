'''Env environment class module for the DubinCar3D reach-avoid game.

'''
import math
import numpy as np

from odp.Grid import Grid
from safe_control_gym.envs.gym_game.utilities import find_sign_change1vs0, spa_deriv, find_sign_change1vs1
from safe_control_gym.envs.gym_game.BaseGame import Dynamics
from safe_control_gym.envs.gym_game.ReachAvoidGame import ReachAvoidGameEnv



class DubinReachAvoidEasierGame(ReachAvoidGameEnv):
    NAME = 'reach_avoid_dubin_easier'
    def __init__(self, *args,  **kwargs):  # distb_level=1.0, randomization_reset=False,
        kwargs['attackers_dynamics']=Dynamics.DUB3D
        kwargs['defenders_dynamics']=Dynamics.DUB3D
        kwargs['game_length_sec'] = 15
        kwargs['random_init'] = False
        # kwargs['initial_attacker'] = np.array([[-0.5, 0.5, 1.0]])
        # kwargs['initial_defender'] = np.array([[0.3, -0.2, 1.0]])
        kwargs['ctrl_freq'] = 20
        kwargs['init_type'] = 'random' # 'distance_init'
        kwargs['obstacles'] = {'obs1': [100, 100, 100, 100]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax], no obstacle here
        super().__init__(*args, **kwargs)

        self.grid1vs0 = Grid(np.array([-1.1, -1.1, -math.pi]), np.array([1.1, 1.1, math.pi]), 3, np.array([100, 100, 200]), [2])
        # self.grid1vs1 = Grid(np.array([-1.1, -1.1, -math.pi, -1.1, -1.1, -math.pi]), np.array([1.1, 1.1, math.pi, 1.1, 1.1, math.pi]), 
        #                      6, np.array([28, 28, 28, 28, 28, 28]), [2, 5])
        # self.value1vs1_easier = np.load('safe_control_gym/envs/gym_game/values/1vs1Dubin_easier.npy')
        self.value1vs0_easier = np.load('safe_control_gym/envs/gym_game/values/1vs0Dubin_easier.npy')

        assert self.ATTACKER_PHYSICS == Dynamics.DUB3D, "The attacker physics is not DubinCar3D."
        assert self.DEFENDER_PHYSICS == Dynamics.DUB3D, "The defender physics is not DubinCar3D."
    

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
        defenders_action = clipped_action.reshape(self.NUM_DEFENDERS, 1)  # ndarray, shape (num_defenders, dim_action)
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
        """Returns the current status of all attackers: 0 for free, 1 for arrived, -1 for captured, -2 for stuck in obs.

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
                if last_status[num]:  # attacker has arrived or been captured
                    new_status[num] = last_status[num]
                else: # attacker is free last time
                    # check if the attacker arrive at the des this time
                    if self._check_area(current_attacker_state[num], self.des):
                        new_status[num] = 1
                    # # # check if the attacker gets stuck in the obstacles this time (it won't usually)
                    # elif self._check_area(current_attacker_state[num], self.obstacles):
                    #     new_status[num] = -2
                    #     continue
                    else:
                        # check if the attacker is captured
                        for j in range(self.NUM_DEFENDERS):
                            if np.linalg.norm(current_attacker_state[num][:2] - current_defender_state[j][:2]) <= 0.30:
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
        x, y, theta = state  # Unpack the state assuming it's a 2D coordinate

        for bounds in area.values():
            x_lower, x_upper, y_lower, y_upper = bounds
            if x_lower <= x <= x_upper and y_lower <= y <= y_upper:
                return True

        return False
    
    
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
        current_relative_distance = np.linalg.norm(current_attacker_state[0][:2] - current_defender_state[0][:2])  # [0.10, 2.82]
        reward += -(current_relative_distance)
        
        return reward


    def _computeAttackerActions(self):
        """Computes the the sub-optimal (1 vs. 0 value function) of the attacker.

        """
        control_attackers = np.zeros((self.NUM_ATTACKERS, 1))
        current_attacker_state = self.attackers._get_state().copy()
        for i in range(self.NUM_ATTACKERS):# the attacker is free
            neg2pos, pos2neg = self.find_sign_change1vs0_dub(self.grid1vs0, self.value1vs0_easier, current_attacker_state[i])
            if len(neg2pos):
                control_attackers[i] = self.attacker_control_1vs0_dub(self.grid1vs0, self.value1vs0_easier, current_attacker_state[i], neg2pos)
            else:
                control_attackers[i] = (0.0)
        
        return control_attackers
    

    def find_sign_change1vs0_dub(self, grid1vs0, value1vs0, attacker):
        """Return two positions (neg2pos, pos2neg) of the value function

        Args:
        grid1vs0 (class): the instance of grid
        value1vs0 (ndarray): including all the time slices, shape = [100, 100, 200, len(tau)]
        attacker (ndarray, (dim,)): the current state of one attacker
        """
        current_slices = grid1vs0.get_index(attacker)
        current_value = value1vs0[current_slices[0], current_slices[1], current_slices[2], :]  # current value in all time slices
        neg_values = (current_value<=0).astype(int)  # turn all negative values to 1, and all positive values to 0
        checklist = neg_values - np.append(neg_values[1:], neg_values[-1])
        # neg(True) - pos(False) = 1 --> neg to pos
        # pos(False) - neg(True) = -1 --> pos to neg
        return np.where(checklist==1)[0], np.where(checklist==-1)[0]
    

    def attacker_control_1vs0_dub(self, grid1vs0, value1vs0, attacker, neg2pos):
        """Return a list of 2-dimensional control inputs of one defender based on the value function
        
        Args:
        grid1vs0 (class): the corresponding Grid instance
        value1vs0 (ndarray): 1v0 HJ reachability value function with only final slice
        attacker (ndarray, (dim,)): the current state of one attacker
        neg2pos (list): the positions of the value function that change from negative to positive
        """
        current_value = grid1vs0.get_value(value1vs0[..., 0], list(attacker))
        if current_value > 0:
            value1vs0 = value1vs0 - current_value
        v = value1vs0[..., neg2pos] # Minh: v = value1v0[..., neg2pos[0]]
        spat_deriv_vector = spa_deriv(grid1vs0.get_index(attacker), v, grid1vs0, [2])
        opt_u = self.optCtrl_1vs0_dub(spat_deriv_vector)

        return (opt_u)
    

    def optCtrl_1vs0_dub(self, spat_deriv):
        """Computes the optimal control (disturbance) for the attacker in a 1 vs. 0 game.
        
        Parameters:
            spat_deriv (tuple): spatial derivative in all dimensions
        
        Returns:
            tuple: a tuple of optimal control of the defender (disturbances)
        """
        opt_u = self.attackers.uMax

        if spat_deriv[2] > 0:
            if self.uMode == "min":
                opt_u = - self.attackers.uMax
        else:
            if self.uMode == "max":
                opt_u = - self.attackers.uMax
        
        return opt_u

    

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
            x, y, theta = pos
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
            # Check the angle
            if theta < -np.pi or theta >= np.pi:
                return False
            return True
        
        def _is_valid_defender(defender_state, attacker_state):
            x, y, theta = defender_state
            # Check map boundaries
            if not (map[0][0] <= x <= map[0][1] and map[1][0] <= y <= map[1][1]):
                return False
            # # Check obstacles
            # for (ox, oy) in obstacles:
            #     if ox[0] <= x <= ox[1] and oy[0] <= y <= oy[1]:
            #         return False
            # Check the relative distance
            if np.linalg.norm(defender_state[:2] - attacker_state[:2]) <= 0.30:
                return False
            # Check the angle
            if theta < -np.pi or theta >= np.pi:
                return False
            return True
        
        def _generate_attacker_state():
            """Generate a random state for the attacker.
            
            Returns:
                attacker_state (tuple): the attacker state.
            """
            while True:
                attacker_x = np.random.uniform(map[0][0], map[0][1])
                attacker_y = np.random.uniform(map[1][0], map[1][1])
                attacker_theta = np.random.uniform(-np.pi, np.pi)
                attacker_pos = np.round((attacker_x, attacker_y), 1)
                attacker_state = np.array([attacker_pos[0], attacker_pos[1], attacker_theta])
                if _is_valid_attacker(attacker_state):
                    break
            return attacker_state
        
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
            attacker_state = _generate_attacker_state()
            # Generate the defender position
            while True:
                defender_x = np.random.uniform(map[0][0], map[0][1])
                defender_y = np.random.uniform(map[1][0], map[1][1])
                defender_theta = np.random.uniform(-np.pi, np.pi)
                defender_pos = np.round((defender_x, defender_y), 1)
                defender_state = np.asarray([defender_pos[0], defender_pos[1], defender_theta])
                if _is_valid_defender(defender_state, attacker_state):
                    break
            
            return attacker_state, defender_state
        
        attacker_state, defender_state = _generate_random_positions(self.initial_players_seed, self.init_player_call_counter)

        # print(f"========== attacker_pos: {attacker_state} in DubinGame.py. ==========")
        # print(f"========== defender_pos: {defender_state} in DubinGame.py. ==========")
        # print(f"========== The relative distance is {np.linalg.norm(attacker_state[:2] - defender_state[:2]):.2f} in DubinGame.py. ========== \n ")
        
        self.initial_players_seed += 1  # Increment the random seed
        self.init_player_call_counter += 1  # Increment the call counter
        
        return np.array([attacker_state]), np.array([defender_state])
    

    def _computeTerminated(self):
        """Computes the current done value.
        done = True if all attackers have arrived or been captured.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        # check the status of the attackers
        current_attacker_status = self.attackers_status[-1]
        attacker_done = np.all((current_attacker_status == 1) | (current_attacker_status == -1)) | (current_attacker_status == -2)
        # check the status of the defenders
        current_defender_state = self.defenders._get_state().copy()
        defender_done = self._check_area(current_defender_state[0], self.obstacles)
        # summary
        done = attacker_done or defender_done
        
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
