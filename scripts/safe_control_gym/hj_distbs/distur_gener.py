# import heterocl as hcl
import numpy as np
import math
import random

from odp.computeGraphs.CustomGraphFunctions import my_abs
from odp.Grid import Grid  # Utility functions to initialize the problem
from odp.Shapes import *



def distur_gener(states, distb_level):


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
        # Hanyang: try different calculation
        # if spat_deriv[3] < 0:
        #     dOpt1 = dMin[0]
        # if spat_deriv[4] < 0:
        #     dOpt2 = dMin[1]
        # if spat_deriv[5] < 0:
        #     dOpt3 = dMin[2]

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
    assert distb_level <= 2.0  # Hanyang: check the output content
    V = np.load(f'safe_control_gym/hj_distbs/FasTrack_data/quadrotor/quadrotor_{distb_level}_15x15.npy')
    # V = np.load(f'gym_pybullet_drones/hj_distbs/FasTrack_data/fastrack_{disturbance}_15x15.npy')
    
    dmax = distb_level * umax
    
    grid = Grid(np.array([-math.pi/2.4, -math.pi/2.4, -math.pi/2.4, -math.pi, -math.pi, -math.pi]), np.array([math.pi/2.4, math.pi/2.4, math.pi/2.4, math.pi, math.pi, math.pi]), 6, np.array([15,15,15,15,15,15]), [0,1,2])

    [opt_u, opt_d] = compute_opt_traj(grid, V, states, umax, dmax)

    return opt_u, opt_d
    


def distur_gener_cartpole(states, distb_level, dMax=np.array([2.0])):
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
    V = np.load(f'safe_control_gym/hj_distbs/FasTrack_data/cartpole_{dMax[0]}/cartpole_{distb_level}.npy')
    print(f"The value function load is FasTrack_data/cartpole_{dMax[0]}/cartpole_{distb_level}.npy")
    dmax = distb_level * dMax
    #TODO: Hanyang check the print out of hj disturbance
    grid = Grid(np.array([-4.8, -5, -math.pi, -10]), np.array([4.8, 5, math.pi, 10]), 4, np.array([45, 45, 45, 45]), [2])

    [opt_u, opt_d] = compute_opt_traj(grid, V, states, uMax, dmax)

    return opt_u, opt_d

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

if __name__ == "__main__":
    # Test the function
    # Test quadrotor
    # roll_range = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
    # pitch_range = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
    # roll_rate = [-2.5,-2,-1.5,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.5,2,2.5]
    # pitch_rate = [-2.5,-2,-1.5,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.5,2,2.5]

    # disturbance = 1.5

    # col = 15
    # initial = np.empty((col,6))

    # for i in range(col):
    #     initial[i] = [random.choice(roll_range),random.choice(pitch_range),0,random.choice(roll_rate),random.choice(pitch_rate),0]

    # initial_nodu=np.unique(initial,axis=0)

    # for i in range(len(initial_nodu)):

    #     initial_point = initial_nodu[i]
    
    #     [u,d] = distur_gener(initial_point, disturbance)
    #     print (d)
    
    # Test cartpole
    # Grid(np.array([-4.8, -5, -math.pi, -10]), np.array([4.8, 5, math.pi, 10]), 4, np.array([45, 45, 45, 45]), [2])
    # Define the range and accuracy
    x_limit = 4.8
    x_accuracy = 9.6 / 45
    v_limit = 5
    v_accuracy = 10 / 45
    theta_limit = math.pi
    theta_accuracy = 2 * math.pi / 45
    omega_limit = 10
    omega_accuracy = 20 / 45

    # Generate the list using list comprehension
    x_range = [round(-x_limit + i * x_accuracy, 10) for i in range(int((x_limit + x_limit) / x_accuracy) + 1)]
    v_range = [round(-v_limit + i * v_accuracy, 10) for i in range(int((v_limit + v_limit) / v_accuracy) + 1)]
    theta_range = [round(-theta_limit + i * theta_accuracy, 10) for i in range(int((theta_limit + theta_limit) / theta_accuracy) + 1)]
    omega_range = [round(-omega_limit + i * omega_accuracy, 10) for i in range(int((omega_limit + omega_limit) / omega_accuracy) + 1)]

    distb_level = 0.1

    initial_point = [0, 0, 0, 0]
    [u,d] = distur_gener_cartpole(initial_point, distb_level)
    print(d)






