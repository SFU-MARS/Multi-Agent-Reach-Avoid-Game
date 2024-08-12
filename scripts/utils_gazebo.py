import numpy as np


def dubin_inital_check(initial_attacker):
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
        for state in states:
            state[2] = normalize_angle(state[2])
        return states
    
    initial_attacker = normalize_states(initial_attacker)
    
    return initial_attacker


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


def optDistb_1vs1(spat_deriv, dMode="max", dMax=1):
    opt_d = dMax
    if spat_deriv[5] > 0:
        if dMode == "min":
            opt_d = - dMax
    else:
        if dMode == "max":
            opt_d = - dMax

    return opt_d


def hj_contoller_defenders_dub_1vs1(attacker_state, defender_state, value1vs1_dub, grid1vs1_dub):
    """Return a tuple of 1-dimensional control inputs of one defender based on the value function
    
    Args:
        attacker_state (ndarray, (dim,)): the current state of the attacker
        defender_state (ndarray, (dim,)): the current state of the defender
        value1vs1_dub (ndarray): 1vs1 HJ reachability value function with only final slice
        grid1vs1_dub (class): the corresponding Grid instance

    Returns:
        opt_d (tuple): the optimal control of the defender
    """
    
    control_defenders = np.zeros((1, 1))  # (num_defenders, control_dim)
    
    a1x, a1y, a1o = attacker_state.copy()
    d1x, d1y, d1o = defender_state.copy()
    jointstate_1vs1 = (a1x, a1y, a1o, d1x, d1y, d1o)

    opt_d = defender_control_1vs1_dub(grid1vs1_dub, value1vs1_dub, jointstate_1vs1)
    control_defenders[0] = (opt_d)

    return control_defenders


def defender_control_1vs1_dub(grid1vs1_dub, value1vs1_dub, jointstate_1vs1):
    """Return a tuple of 2-dimensional control inputs of one defender based on the value function
    
    Args:
        grid1vs1_dub (class): the corresponding Grid instance
        value1v1 (ndarray): 1vs1 HJ reachability value function with only final slice
        joint_states1v1 (tuple): the corresponding positions of (A1, D1)
    
    Returns:
        opt_d (tuple): the optimal control of the defender
    """
    value1vs1s = value1vs1_dub[..., np.newaxis] 
    spat_deriv_vector = spa_deriv(grid1vs1_dub.get_index(jointstate_1vs1), value1vs1s, grid1vs1_dub, [2,5])
    opt_d = optDistb_1vs1(spat_deriv_vector) 

    return (opt_d)


def find_sign_change1vs0_dub(grid1vs0, value1vs0, attacker):
    """Return two positions (neg2pos, pos2neg) of the value function

    Args:
    grid1vs0 (class): the instance of grid
    value1vs0 (ndarray): including all the time slices, shape = [100, 100, 200, len(tau)]
    attacker (ndarray, (dim,)): the current state of one attacker
    """
    current_slices = grid1vs0.get_index(attacker)
    # current_slices = po2slice1vs0_dub(attacker, value1vs0.shape[0])
    current_value = value1vs0[current_slices[0], current_slices[1], current_slices[2], :]  # current value in all time slices
    neg_values = (current_value<=0).astype(int)  # turn all negative values to 1, and all positive values to 0
    checklist = neg_values - np.append(neg_values[1:], neg_values[-1])

    return np.where(checklist==1)[0], np.where(checklist==-1)[0]


def optCtrl_1vs0(spat_deriv, uMode="min"):
    """Return the optimal control input of one defender based on the spatial derivatives
    
    Args:
    spat_deriv (ndarray): the spatial derivatives of the value function
    uMode (str): the mode of the control input, "min" or "max"
    """
    opt_u = 1 # the maximum angular rate of the TurtleBot3

    if spat_deriv[2] > 0:
        if uMode == "min":
            opt_u = -1
    else:
        if uMode == "max":
            opt_u = -1
    return opt_u


def attacker_control_1vs0_dub(grid1vs0, value1vs0, current_state, neg2pos):
    """Return a list of 1-dimensional control inputs of one defender based on the value function
    
    Args:
    grid1vs0 (class): the corresponding Grid instance
    value1vs0 (ndarray): 1vs1 HJ reachability value function with only final slice
    current_state (ndarray, (dim,)): the current state of one agent
    neg2pos (list): the positions of the value function that change from negative to positive
    """
    current_value = grid1vs0.get_value(value1vs0[..., 0], list(current_state))
    if current_value > 0:
        value1vs0 = value1vs0 - current_value
    v = value1vs0[..., neg2pos]
    spat_deriv_vector = spa_deriv(grid1vs0.get_index(current_state), v, grid1vs0)
    opt_u = optCtrl_1vs0(spat_deriv_vector, uMode="min")

    return (opt_u)


def hj_controller_1vs0(value1vs0_dub, grid1vs0_dub, current_state):
    """ This function implements the HJ controller based on the 1 vs. 0 value function.
    Args:
        value1vs0_dub (np.ndarray, (100, 100, 200, slices)): the value function for 1 vs. 0
        grid1vs0_dub (Grid): the grid for the value function
        current_state (np.ndarray): the current state of the agent

    Returns:
        np.ndarray: the control signal
    """
    control_attackers = np.zeros((1, 1))
    neg2pos, pos2neg = find_sign_change1vs0_dub(grid1vs0_dub, value1vs0_dub, current_state)
    if len(neg2pos):
        control_attackers[0] = attacker_control_1vs0_dub(grid1vs0_dub, value1vs0_dub, current_state, neg2pos)
    else:
        control_attackers[0] = (0.0)
    
    return control_attackers

