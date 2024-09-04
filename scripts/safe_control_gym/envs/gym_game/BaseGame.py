'''Base environment class module for the reach-avoid game.

This module also contains enumerations for cost functions, tasks, disturbances, and quadrotor types.
'''

import numpy as np
import gymnasium as gym
from safe_control_gym.envs.gym_game.utilities import make_agents, dubin_inital_check


class Dynamics:
    """Physics implementations enumeration class."""

    SIG = {'id': 'sig', 'action_dim': 2, 'state_dim': 2, 'speed': 1.0}           # Base single integrator dynamics
    FSIG = {'id': 'fsig', 'action_dim': 2, 'state_dim': 2, 'speed': 1.5}         # Faster single integrator dynamics with feedback
    DUB3D = {'id': 'dub3d', 'action_dim': 1, 'state_dim': 3, 'speed': 0.22}       # 3D Dubins car dynamics
    FDUB3D = {'id': 'fdub3d', 'action_dim': 1, 'state_dim': 3, 'speed': 0.22}     # Faster 3D Dubins car dynamics with feedback
    
    
class BaseGameEnv(gym.Env):
    """Base class for the multi-agent reach-avoid game Gym environments."""
    
    def __init__(self,
                 num_attackers: int=1,
                 num_defenders: int=1,
                 attackers_dynamics=Dynamics.SIG,  
                 defenders_dynamics=Dynamics.FSIG,
                 initial_attacker: np.ndarray=None,  # shape (num_atackers, state_dim)
                 initial_defender: np.ndarray=None,  # shape (num_defenders, state_dim)
                 ctrl_freq: int = 200,
                 seed: int = None,
                 random_init: bool = True,
                 init_type: str = 'random',
                 output_folder='results',
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
        random_init : bool, optional
        init_type : str, optional
        output_folder : str, optional
            The folder where to save logs.

        """
        #### Constants #############################################
        self.CTRL_FREQ = ctrl_freq  # normally 200Hz
        self.SIM_TIMESTEP = 1. / self.CTRL_FREQ  # 0.005s
        self.seed = seed
        self.initial_players_seed = seed
        #### Parameters ############################################
        self.NUM_ATTACKERS = num_attackers
        self.NUM_DEFENDERS = num_defenders
        self.NUM_PLAYERS = self.NUM_ATTACKERS + self.NUM_DEFENDERS
        #### Options ###############################################
        self.ATTACKER_PHYSICS = attackers_dynamics
        self.DEFENDER_PHYSICS = defenders_dynamics
        self.OUTPUT_FOLDER = output_folder
        #### Input initial states ####################################
        self.init_attackers = initial_attacker
        self.init_defenders = initial_defender
        #### Housekeeping ##########################################
        self.init_player_call_counter = 0
        self.random_init = random_init
        self.init_type = init_type
        assert self.init_type in ['random', 'distance_init', 'difficulty_init'], "Invalid init_type."
        self._housekeeping()
        #### Update and all players' information #####
        self._updateAndLog()
    

    def _housekeeping(self):
        """Housekeeping function.

        Initialize all loggers, counters, and variables that need to be reset at the beginning of each episode
        in the `reset()` function.

        """
        if self.random_init:
            self.init_attackers, self.init_defenders = self.initial_players()
        else:
            assert self.init_attackers is not None and self.init_defenders is not None, "Need to provide initial positions for all players." 
        #### Set attackers and defenders ##########################
        self.attackers = make_agents(self.ATTACKER_PHYSICS, self.NUM_ATTACKERS, self.init_attackers, self.CTRL_FREQ)
        self.defenders = make_agents(self.DEFENDER_PHYSICS, self.NUM_DEFENDERS, self.init_defenders, self.CTRL_FREQ)
        #### Initialize/reset counters, players' trajectories and attackers status ###
        self.step_counter = 0
        self.attackers_traj = []
        self.defenders_traj = []
        self.attackers_status = []  # 0 stands for free, -1 stands for captured, 1 stands for arrived 
        self.attackers_actions = []
        self.defenders_actions = []
        # self.last_relative_distance = np.zeros((self.NUM_ATTACKERS, self.NUM_DEFENDERS))


    def _updateAndLog(self):
        """Update and log all players' information after inialization, reset(), or step.

        """
        # Update the state
        current_attackers = self.attackers._get_state().copy()
        current_defenders = self.defenders._get_state().copy()
        
        self.state = np.vstack([current_attackers, current_defenders])
        # Log the state and trajectory information
        self.attackers_traj.append(current_attackers)
        self.defenders_traj.append(current_defenders)
        self.attackers_status.append(self._getAttackersStatus().copy())
    
    
    def initial_players(self):
        '''Set the initial positions for all players.
        
        Returns:
            attackers (np.ndarray): the initial positions of the attackers
            defenders (np.ndarray): the initial positions of the defenders
        '''
        # Map boundaries
        map = ([-0.99, 0.99], [-0.99, 0.99])  # The map boundaries
        # Obstacles and target areas
        obstacles = [
            ([-0.1, 0.1], [-1.0, -0.3]),  # First obstacle
            ([-0.1, 0.1], [0.3, 0.6])     # Second obstacle
        ]
        target = ([0.6, 0.8], [0.1, 0.3])

        def _is_valid_attacker(pos):
            x, y = pos
            # Check map boundaries
            if not (map[0][0] <= x <= map[0][1] and map[1][0] <= y <= map[1][1]):
                return False
            # Check obstacles
            for (ox, oy) in obstacles:
                if ox[0] <= x <= ox[1] and oy[0] <= y <= oy[1]:
                    return False
            # Check target
            if target[0][0] <= x <= target[0][1] and target[1][0] <= y <= target[1][1]:
                return False
            return True
        
        def _is_valid_defender(defender_pos, attacker_pos):
            x, y = defender_pos
            # Check map boundaries
            if not (map[0][0] <= x <= map[0][1] and map[1][0] <= y <= map[1][1]):
                return False
            # Check obstacles
            for (ox, oy) in obstacles:
                if ox[0] <= x <= ox[1] and oy[0] <= y <= oy[1]:
                    return False
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

        def _generate_neighborpoint(attacker_pos, distance, radius):
            """
            Generate a random point within a circle whose center is a specified distance away from a given position.

            Parameters:
            attacker_pos (list): The (x, y) coordinates of the initial position.
            distance (float): The distance from the initial position to the center of the circle.
            radius (float): The radius of the circle.

            Returns:
            defender_pos (list): A random (x, y) point whose relative distance between the input position is .
            """

            while True:
                # Randomly choose an angle to place the circle's center
                angle = np.random.uniform(0, 2 * np.pi)
                
                # Determine the center of the circle
                center_x = attacker_pos[0] + distance * np.cos(angle)
                center_y = attacker_pos[1] + distance * np.sin(angle)
                
                # Generate a random point within the circle
                point_angle = np.random.uniform(0, 2 * np.pi)
                point_radius = np.sqrt(np.random.uniform(0, 1)) * radius
                defender_x = center_x + point_radius * np.cos(point_angle)
                defender_y = center_y + point_radius * np.sin(point_angle)

                # In case all generated points are outside of the map
                x_min, x_max, y_min, y_max = map[0][0], map[0][1], map[1][0], map[1][1]
                defender_x = max(min(defender_x, x_max), x_min)
                defender_y = max(min(defender_y, y_max), y_min)
                defender_pos = np.array([defender_x, defender_y])
                
                if _is_valid_defender(defender_pos, attacker_pos):
                    break

            return defender_pos

        def _generate_distance_points(current_seed, init_player_call_counter):
            """Generate the attacker and defender positions based on the relative distance.

            Args:
                current_seed (int): the random seed.
                init_player_call_counter (int): the call counter.

            Returns:
                tuple: the attacker and defender positions.
            """
            np.random.seed(current_seed)
            # Generate the attacker position
            attacker_pos = _generate_attacker_pos()
            
            # Determine the distance based on the call counter
            stage = -1
            if init_player_call_counter < 3000:  # [0.10, 0.20]
                distance_init = 0.15
                r_init = 0.05
                stage = 0
            elif init_player_call_counter < 6000:  # [0.20, 0.50]
                distance_init = 0.35
                r_init = 0.15
                stage = 1
            elif init_player_call_counter < 10000:  # [0.50, 1.00]
                distance_init = 0.75
                r_init = 0.25
                stage = 2
            elif init_player_call_counter < 15000:  # [1.00, 2.00]
                distance_init = 1.50
                r_init = 0.50
                stage = 3
            elif init_player_call_counter < 20000:  # [2.00, 2.80]
                distance_init = 2.40
                r_init = 0.40
                stage = 4
            else:  # [0.10, 2.80]
                distance_init = 1.45
                r_init = 1.35
                stage = 5
            # Generate the defender position
            defender_pos = _generate_neighborpoint(attacker_pos, distance_init, r_init)

            return attacker_pos, defender_pos, stage
        
        def _generate_obstacle_neighborpoints():
            """Generate the attacker and defender positions near the obstacles.
            
            Returns:
                attacker_pos (list): the attacker position.
                defender_pos (list): the defender position
            """
            # Sample y position from the obstacles
            y_positions = [obstacles[0][1], obstacles[1][1]]
            attacker_y = np.random.uniform(*y_positions[np.random.choice(len(y_positions))])
            defender_y = np.random.uniform(*y_positions[np.random.choice(len(y_positions))])
            # Sample x position for attacker and defender
            attacker_x = np.random.uniform(-0.99, -0.15)
            defender_x = np.random.uniform(0.15, 0.99)
            
            attacker_pos = np.array([attacker_x, attacker_y])
            defender_pos = np.array([defender_x, defender_y])
        
            return attacker_pos, defender_pos
        
        def _generate_difficulty_points(current_seed, init_player_call_counter):
            """Generate attacker and defender initial positions based on the difficulty level.
            difficulty_level 0: there is no obstacle between attacker and defender and the relative distance is [0.10, 0.50];
            difficulty_level 1: there is no obstacle between attacker and defender and the relative distance is [0.50, 1.50];
            difficulty_level 2: there is no obstacle between attacker and defender and the relative distance is [1.50, 2.80];
            difficulty_level 3: there is an obstacle between attacker and defender;

            Args:
                difficulty_level (int): the difficulty level of the game, designed by the relative position of obstacles and target areas.
                seed (int): the initialization random seed.

            Returns:
                attacker_pos (list): the initial position of the attacker.
                defender_pos (list): the initial position of the defender.
            """
            np.random.seed(current_seed)
            # Generate the attacker position
            difficulty_level = -1
            if init_player_call_counter < 2000:  # difficulty_level 0, # [0.10, 0.50]
                difficulty_level = 0
                distance = 0.30
                r = 0.20
                attacker_pos = _generate_attacker_pos()
                defender_pos = _generate_neighborpoint(attacker_pos, distance, r)
            elif init_player_call_counter < 5000:  # difficulty_level 1, # [0.50, 1.50]
                difficulty_level = 1
                distance = 1.00
                r = 0.50
                attacker_pos = _generate_attacker_pos()
                defender_pos = _generate_neighborpoint(attacker_pos, distance, r)
            elif init_player_call_counter < 10000:  # difficulty_level 2, # [1.50, 2.80]
                difficulty_level = 2
                distance = 2.15
                r = 0.65
                attacker_pos = _generate_attacker_pos()
                defender_pos = _generate_neighborpoint(attacker_pos, distance, r)
            else:
                difficulty_level = 3
                attacker_pos, defender_pos = _generate_obstacle_neighborpoints()
            
            return attacker_pos, defender_pos, difficulty_level
        
        # Generate the initial positions
        if self.init_type == 'random':
            attacker_pos, defender_pos = _generate_random_positions(self.initial_players_seed, self.init_player_call_counter)
            print(f"========== self.call_counter: {self.init_player_call_counter} in BaseGame.py. ==========")

        elif self.init_type == 'distance_init':
            attacker_pos, defender_pos, stage = _generate_distance_points(self.initial_players_seed, self.init_player_call_counter)
            print(f"========== self.call_counter: {self.init_player_call_counter} and stage: {stage} in BaseGame.py. ==========")

        elif self.init_type == 'difficulty_init':
            attacker_pos, defender_pos, difficulty_level = _generate_difficulty_points(self.initial_players_seed, self.init_player_call_counter)
            print(f"========== self.call_counter: {self.init_player_call_counter} and difficulty_level: {difficulty_level} in BaseGame.py. ==========")
            
        else:
            raise ValueError(f"Invalid init_type: {self.init_type}.")

        print(f"========== attacker_pos: {attacker_pos} in BaseGame.py. ==========")
        print(f"========== defender_pos: {defender_pos} in BaseGame.py. ==========")
        print(f"========== The relative distance is {np.linalg.norm(attacker_pos - defender_pos):.2f} in BaseGame.py. ========== \n ")
        
        self.initial_players_seed += 1  # Increment the random seed
        self.init_player_call_counter += 1  # Increment the call counter
        
        return np.array([attacker_pos]), np.array([defender_pos])


    def reset(self, seed : int = None,
              options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """        
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and all players' information #####
        self._updateAndLog()
        #### Prepare the observation #############################
        obs = self._computeObs()
        info = self._computeInfo()
        
        return obs, info
    

    def _getAttackersStatus(self):
        """Returns the current status of all attackers.
        -------
        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    
    def _computeObs(self):
        """Returns the current observation of the environment.

        """
        obs = self.state.flatten()
        
        return obs
    

    def _computeReward(self):
        """Computes the current reward value(s).

        Must be implemented in a subclass.

        Parameters
        ----------
        clipped_action : ndarray | dict[..]
            The input clipped_action for one or more drones.

        """
        raise NotImplementedError


    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    

    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError


    def _computeInfo(self):
        """Computes the current info dict(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError