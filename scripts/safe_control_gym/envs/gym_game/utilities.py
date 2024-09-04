'''Utility functions for the reach-avoid game.

'''

import math
import time
import numpy as np

from odp.Grid import Grid
from safe_control_gym.envs.gym_game.dynamics.SingleIntegrator import SingleIntegrator
from safe_control_gym.envs.gym_game.dynamics.DubinCar3D import DubinsCar


def make_agents(physics_info, numbers, initials, freqency):
    '''Make the agents with the given physics list, numbers and initials.
    
    Args:
        physics_info (dic): the physics info of the agent
        numbers (int): the number of agents
        initials (np.ndarray): the initial states of all agents
        freqency (int): the frequency of the simulation
    '''
    if physics_info['id'] == 'sig':
        return SingleIntegrator(number=numbers, initials=initials, frequency=freqency, speed=physics_info['speed'])
    elif physics_info['id'] == 'fsig':
        return SingleIntegrator(number=numbers, initials=initials, frequency=freqency, speed=physics_info['speed'])
    elif physics_info['id'] == 'dub3d':
        return DubinsCar(number=numbers, initials=initials, frequency=freqency, speed=physics_info['speed'])
    elif physics_info['id'] == 'fdub3d':
        return DubinsCar(number=numbers, initials=initials, frequency=freqency, speed=physics_info['speed'])
    
    else:
        raise ValueError("Invalid physics info while generating agents.")
    

def dubin_inital_check(initial_attacker, initial_defender):
    """ Make sure the angle is in the range of [-pi, pi), if not, change it.
    
    Args:
        inital_attacker (np.ndarray, (num_attacker, 3)): the initial state of the attacker
        initial_defender (np.ndarray, (num_defender, 3)): the initial state of the defender
    
    Returns:
        initial_attacker (np.ndarray, (num_attacker, 3)): the initial state of the attacker after revision if necessary
        initial_defender (np.ndarray, (num_defender, 3)): the initial state of the defender after revision if necessary
    """
    def normalize_angle(angle):
        while angle >= np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def normalize_states(states):
        if states is not None:
            for state in states:
                state[2] = normalize_angle(state[2])
        return states
    
    initial_attacker = normalize_states(initial_attacker)
    initial_defender = normalize_states(initial_defender)
    
    return initial_attacker, initial_defender


def hj_preparations_sig():
    """ Loads all calculated HJ value functions for the single integrator agents.
    This function needs to be called before any game starts.
    
    Returns:
        value1vs0 (np.ndarray): the value function for 1 vs 0 game with all time slices
        value1vs1 (np.ndarray): the value function for 1 vs 1 game
        grid1vs0 (Grid): the grid for 1 vs 0 game
        grid1vs1 (Grid): the grid for 1 vs 1 game
    """
    start = time.time()
    value1vs0 = np.load('safe_control_gym/envs/gym_game/values/1vs0AttackDefend_g100_speed1.0.npy')
    value1vs1 = np.load('safe_control_gym/envs/gym_game/values/1vs1AttackDefend_g45_dspeed1.5.npy')
    end = time.time()
    print(f"============= HJ value functions loaded Successfully! (Time: {end-start :.4f} seconds) =============")
    grid1vs0 = Grid(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 2, np.array([100, 100])) 
    grid1vs1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45]))
    print(f"============= Grids created Successfully! =============")

    return value1vs0, value1vs1, grid1vs0, grid1vs1


def po2slice1vs1(attacker, defender, grid_size):
    """ Convert the position of the attacker and defender to the slice of the value function for 1 vs 1 game.

    Args:
        attacker (np.ndarray): the attacker's state
        defender (np.ndarray): the defender's state
        grid_size (int): the size of the grid
    
    Returns:
        joint_slice (tuple): the joint slice of the joint state using the grid size

    """
    joint_state = (attacker[0], attacker[1], defender[0], defender[1])  # (xA1, yA1, xD1, yD1)
    joint_slice = []
    grid_points = np.linspace(-1, +1, num=grid_size)
    for i, s in enumerate(joint_state):
        idx = np.searchsorted(grid_points, s)
        if idx > 0 and (
            idx == len(grid_points)
            or math.fabs(s - grid_points[idx - 1])
            < math.fabs(s - grid_points[idx])
        ):
            joint_slice.append(idx - 1)
        else:
            joint_slice.append(idx)

    return tuple(joint_slice)


def po2slice2vs1(attacker_i, attacker_k, defender, grid_size):
    """ Convert the position of the attackers and defender to the slice of the value function for 2 vs 1 game.

    Args:
        attackers (np.ndarray): the attackers' states
        defender (np.ndarray): the defender's state
        grid_size (int): the size of the grid
    
    Returns:
        joint_slice (tuple): the joint slice of the joint state using the grid size

    """
    joint_state = (attacker_i[0], attacker_i[1], attacker_k[0], attacker_k[1], defender[0], defender[1])  # (xA1, yA1, xA2, yA2, xD1, yD1)
    joint_slice = []
    grid_points = np.linspace(-1, +1, num=grid_size)
    for i, s in enumerate(joint_state):
        idx = np.searchsorted(grid_points, s)
        if idx > 0 and (
            idx == len(grid_points)
            or math.fabs(s - grid_points[idx - 1])
            < math.fabs(s - grid_points[idx])
        ):
            joint_slice.append(idx - 1)
        else:
            joint_slice.append(idx)

    return tuple(joint_slice)


def check_1vs1(attacker, defender, value1vs1):
    """ Check if the attacker could escape from the defender in a 1 vs 1 game.

    Args:
        attacker (np.ndarray): the attacker's state
        defender (np.ndarray): the defender's state
        value1vs1 (np.ndarray): the value function for 1 vs 1 game
    
    Returns:
        bool: False, if the attacker could escape (the attacker will win)
    """
    joint_slice = po2slice1vs1(attacker, defender, value1vs1.shape[0])

    return value1vs1[joint_slice] > 0


def judge_1vs1(attackers, defenders, current_attackers_status, value1vs1):
    """ Check the result of the 1 vs 1 game for those free attackers.

    Args:  
        attackers (np.ndarray): the attackers' states
        defenders (np.ndarray): the defenders' states
        attackers_status (np.ndarray): the current moment attackers' status, 0 stands for free, -1 stands for captured, 1 stands for arrived
        value1vs1 (np.ndarray): the value function for 1 vs 1 game
    
    Returns:
        EscapedAttacker1vs1 (a list of lists): the attacker that could escape from the defender in a 1 vs 1 game
    """
    num_attackers, num_defenders = len(attackers), len(defenders)
    EscapedAttacker1vs1 = [[] for _ in range(num_defenders)]

    for j in range(num_defenders):
        for i in range(num_attackers):
            if not current_attackers_status[i]:  # the attcker[i] is free now
                if not check_1vs1(attackers[i], defenders[j], value1vs1):  # the attacker could escape
                    EscapedAttacker1vs1[j].append(i)

    return EscapedAttacker1vs1


def current_status_check(current_attackers_status, step=None):
    """ Check the current status of the attackers.

    Args:
        current_attackers_status (np.ndarray): the current moment attackers' status, 0 stands for free, -1 stands for captured, 1 stands for arrived
        step (int): the current step of the game
    
    Returns:
        status (dic): the current status of the attackers
    """
    num_attackers = len(current_attackers_status)
    num_free, num_arrived, num_captured = 0, 0, 0
    status = {'free': [], 'arrived': [], 'captured': []}
    
    for i in range(num_attackers):
        if current_attackers_status[i] == 0:
            num_free += 1
            status['free'].append(i)
        elif current_attackers_status[i] == 1:
            num_arrived += 1
            status['arrived'].append(i)
        elif current_attackers_status[i] == -1:
            num_captured += 1
            status['captured'].append(i)
        else:
            raise ValueError("Invalid status for the attackers.")
    
    print(f"================= Step {step}: {num_captured}/{num_attackers} attackers are captured \t"
      f"{num_arrived}/{num_attackers} attackers have arrived \t"
      f"{num_free}/{num_attackers} attackers are free =================")

    print(f"================= The current status of the attackers: {status} =================")

    return status


def check_current_value(attackers, defenders, value_function):
    """ Check the value of the current state of the attackers and defenders.

    Args:
        attackers (np.ndarray): the attackers' states
        defenders (np.ndarray): the defenders' states
        value (np.ndarray): the value function for the game
    
    Returns:
        value (float): the value of the current state of the attackers and defenders
    """
    if len(value_function.shape) == 4:  # 1vs1 game
        joint_slice = po2slice1vs1(attackers[0], defenders[0], value_function.shape[0])
    elif len(value_function.shape) == 6:  # 1vs2 or 2vs1 game
        if attackers.shape[0] == 1:  # 1vs2 game
            joint_slice = po2slice2vs1(attackers[0], defenders[0], defenders[1], value_function.shape[0])
        else:  # 2vs1 game
            joint_slice = po2slice2vs1(attackers[0], attackers[1], defenders[0], value_function.shape[0])
    value = value_function[joint_slice]

    return value


def find_sign_change1vs0(grid1vs0, value1vs0, attacker):
    """Return two positions (neg2pos, pos2neg) of the value function

    Args:
    grid1vs0 (class): the instance of grid
    value1vs0 (ndarray): including all the time slices, shape = [100, 100, len(tau)]
    attacker (ndarray, (dim,)): the current state of one attacker
    """
    current_slices = grid1vs0.get_index(attacker)
    current_value = value1vs0[current_slices[0], current_slices[1], :]  # current value in all time slices
    neg_values = (current_value<=0).astype(int)  # turn all negative values to 1, and all positive values to 0
    checklist = neg_values - np.append(neg_values[1:], neg_values[-1])
    # neg(True) - pos(False) = 1 --> neg to pos
    # pos(False) - neg(True) = -1 --> pos to neg
    return np.where(checklist==1)[0], np.where(checklist==-1)[0]


def find_sign_change1vs1(grid1vs1, value1vs1, current_state):
    """Return two positions (neg2pos, pos2neg) of the value function

    Args:
    grid1vs1 (class): the instance of grid
    value1vs1 (ndarray): including all the time slices, shape = [45, 45, 45, 45, len(tau)]
    current_state (ndarray, (dim,)): the current state of one attacker + one defender
    """
    current_slices = grid1vs1.get_index(current_state)
    current_value = value1vs1[current_slices[0], current_slices[1], current_slices[2], current_slices[3], :]  # current value in all time slices
    neg_values = (current_value<=0).astype(int)  # turn all negative values to 1, and all positive values to 0
    checklist = neg_values - np.append(neg_values[1:], neg_values[-1])
    # neg(True) - pos(False) = 1 --> neg to pos
    # pos(False) - neg(True) = -1 --> pos to neg
    return np.where(checklist==1)[0], np.where(checklist==-1)[0]


def spa_deriv(slice_index, value_function, grid, periodic_dims=[]):
    """
    Calculates the spatial derivatives of V at an index for each dimension

    Args:
        slice_index: (a1x, a1y)
        value_function (ndarray): [..., neg2pos] where neg2pos is a list [scalar] or []
        grid (class): the instance of the corresponding Grid
        periodic_dims (list): the corrsponding periodical dimensions []

    Returns:
        List of left and right spatial derivatives for each dimension
    """
    spa_derivatives = []
    for dim, idx in enumerate(slice_index):
        if dim == 0:
            left_index = []
        else:
            left_index = list(slice_index[:dim])

        if dim == len(slice_index) - 1:
            right_index = []
        else:
            right_index = list(slice_index[dim + 1:])

        next_index = tuple(
            left_index + [slice_index[dim] + 1] + right_index
        )
        prev_index = tuple(
            left_index + [slice_index[dim] - 1] + right_index
        )

        if idx == 0:
            if dim in periodic_dims:
                left_periodic_boundary_index = tuple(
                    left_index + [value_function.shape[dim] - 1] + right_index
                )
                left_boundary = value_function[left_periodic_boundary_index]
            else:
                left_boundary = value_function[slice_index] + np.abs(value_function[next_index] - value_function[slice_index]) * np.sign(value_function[slice_index])
            left_deriv = (value_function[slice_index] - left_boundary) / grid.dx[dim]
            right_deriv = (value_function[next_index] - value_function[slice_index]) / grid.dx[dim]
        elif idx == value_function.shape[dim] - 1:
            if dim in periodic_dims:
                right_periodic_boundary_index = tuple(
                    left_index + [0] + right_index
                )
                right_boundary = value_function[right_periodic_boundary_index]
            else:
                right_boundary = value_function[slice_index] + np.abs(value_function[slice_index] - value_function[prev_index]) * np.sign([value_function[slice_index]])
            left_deriv = (value_function[slice_index] - value_function[prev_index]) / grid.dx[dim]
            right_deriv = (right_boundary - value_function[slice_index]) / grid.dx[dim]
        else:
            left_deriv = (value_function[slice_index] - value_function[prev_index]) / grid.dx[dim]
            right_deriv = (value_function[next_index] - value_function[slice_index]) / grid.dx[dim]

        spa_derivatives.append(((left_deriv + right_deriv) / 2)[0])
    return spa_derivatives