# import heterocl as hcl
import numpy as np
import math
import random

from odp.computeGraphs.CustomGraphFunctions import my_abs
from odp.Grid import Grid  # Utility functions to initialize the problem
from odp.Shapes import *


def distur_gener_cartpole(states, distb_level, dMax=2.0):
    # Hanyang: generate disturbances in the cartpole and its derivative envs
    mc = 1.0
    mp = 0.1
    l = 0.5

    def opt_ctrl_non_hcl(uMax, state, spat_deriv):
        
        opt_a = uMax[0]
        u_coefficient = spat_deriv[1]/(mc + mp) - (spat_deriv[3]*np.cos(state[2]))/(l*(4/3*(mp+mc) - mp*np.cos(state[2])*np.cos(state[2])))

        if u_coefficient > 0:
            opt_a = -uMax[0]
           
        return opt_a
        
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

    def opt_dstb_non_hcl(dMax, state, spat_deriv):

        opt_d = dMax[0]
        d_coefficient = spat_deriv[1]/(mc + mp) - (spat_deriv[3]*np.cos(state[2]))/(l*(4/3*(mp+mc) - mp*np.cos(state[2])*np.cos(state[2])))

        if d_coefficient < 0:
            opt_d = -dMax[0]

        return opt_d

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
            
            gradient = spa_deriv(grid.get_index(states), V, grid, periodic_dims=[2])
            u = opt_ctrl_non_hcl(umax, states, gradient)
            d = opt_dstb_non_hcl(dmax, states, gradient)
                
            return u,d

    
    uMax = np.array([10])
    # Hanyang: change the umax according to the FasTrack_data
    # dmax = np.array([2.0])
    assert distb_level <= 2.0  # Hanyang: check the output content
    # Hanyang: change the path of the npy file
    V = np.load(f'safe_control_gym/hj_distbs/FasTrack_data/cartpole_{dMax}/cartpole_{distb_level}.npy')
    
    dmax = distb_level * dMax
    
    grid = Grid(np.array([-4.8, -5, -math.pi, -10]), np.array([4.8, 5, math.pi, 10]), 4, np.array([45, 45, 45, 45]), [2])

    [opt_u, opt_d] = compute_opt_traj(grid, V, states, uMax, dmax)

    return opt_u, opt_d

