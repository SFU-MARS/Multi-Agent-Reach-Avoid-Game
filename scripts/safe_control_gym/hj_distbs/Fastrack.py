import heterocl as hcl
import numpy as np
import math
# import imp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.integrate
import random

from odp.computeGraphs.CustomGraphFunctions import my_abs
from odp.Grid import Grid  # Utility functions to initialize the problem
from odp.Shapes import *
from odp.Plots import PlotOptions # Plot options
from odp.solver import HJSolver  # Solver core
from odp.solver import TTRSolver
from scipy.integrate import solve_ivp
import argparse

from dynamics import UAV6D, CartPole4D


class UAVSolution(object):
    def __init__(self):
        # warning: grid bound for each dimension should be close, not be too different. 
        self.grid_num_1 = 15
        self.grid_num_2 = 15
        self.grid = Grid(np.array([-math.pi/2.4, -math.pi/2.4, -math.pi/2.4, -math.pi, -math.pi, -math.pi]), np.array([math.pi/2.4, math.pi/2.4, math.pi/2.4, math.pi, math.pi, math.pi]), 6, np.array([self.grid_num_2,self.grid_num_2,self.grid_num_2,self.grid_num_1,self.grid_num_1,self.grid_num_1]), [0,1,2])
        self.dyn = UAV6D(uMode="min", dMode="max")  # reaching problem
        self.lookback_length = 2  # Look-back length and time step of computation
        self.t_step = 0.001
        self.distb_level = self.dyn.distb_level
        self.result = None
        self.name = "quadrotor"


    def FasTrackTarget(self, grid, ignore_dims, center):
        """
        Customized definition of FasTrack Target, it is similar to Cylinder, but with no radius
        """
        data = np.zeros(grid.pts_each_dim)
        for i in range(grid.dims):
            if i not in ignore_dims:
                # This works because of broadcasting
                data = data + np.power(grid.vs[i] - center[i], 2)
        # data = np.sqrt(data) - radius
        return data


    def get_fastrack(self):
        # Hanyang: add new variable distb_level to generate value function
        self.targ = self.FasTrackTarget(self.grid, [2], np.zeros(6)) # l(x) = V(0, x)
        small_number = 1e-5
        tau = np.arange(start=0, stop=self.lookback_length + small_number, step=self.t_step)
        compMethods = { "TargetSetMode": "maxVWithV0"}  # In this example, we compute based on FasTrack 
        slice = int((self.grid_num_1-1)/2)
        self.po = PlotOptions(do_plot=False, plot_type="3d_plot", plotDims=[0,1,2], slicesCut=[int((self.grid_num_1-1)/2),int((self.grid_num_1-1)/2),int((self.grid_num_1-1)/2)])
        self.result = HJSolver(self.dyn, self.grid, self.targ, tau, compMethods, self.po, saveAllTimeSteps=False)
        np.save("safe_control_gym/hj_distbs/FasTrack_data/quadrotor/quadrotor_{}_{}x{}.npy".format(self.distb_level, self.grid_num_2, self.grid_num_1), self.result)
        print("saving the result ..., done!")

        return self.result, self.grid, slice
    

    def spa_deriv(self,index, V, g, periodic_dims):
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

        return np.array(spa_derivatives)


    def next_state(self, dyn_sys, u, d, delta_t):
        """
        Simulate apply control to dynamical systems for delta_t time

        Args:
            dyn_sys: dynamic system
            u: control
            d: disturbance
            delta_t: duration of control

        Returns:

        """
        init_state = dyn_sys.x
        t_span = [0, delta_t]
        solution = solve_ivp(dyn_sys.dynamics_non_hcl, t_span, init_state, args=[u, d], dense_output=True)
        next_state = solution.y[:, -1]
        if next_state[2] < -np.pi:
            next_state[2] += 2 * np.pi
        elif next_state[2] > np.pi:
            next_state[2] -= 2 * np.pi
        return next_state

    def update_state(self, dyn_sys, next_state):
        dyn_sys.x = next_state


    def compute_opt_traj(self,grid: Grid, V, tau, dyn_sys, initial, subsamples=1, arriveAfter = None, obstVal = None): # default subsample in helperOC is 4
        """
        Computes the optimal trajectory, controls and disturbance to a minimal BRT/BRS

        Args:
            grid:
            V:
            tau:
            dyn_sys:
            subsamples: Number of times opt_u and opt_d are calculated within dt

        Returns:
            traj: State of dyn_sys from time tau[-1] to tau[0]
            opt_u: Optimal control at each time
            opt_d: Optimal disturbance at each time

        """
        

        dt = (tau[1] - tau[0]) / subsamples
        dyn_sys.x = initial

        # first entry is dyn_sys at time tau[-1]
        # second entry is dyn_sys at time tau[-2]...
        traj = np.empty((len(tau), len(dyn_sys.x)))
        # Here it is chcking roll angle. traj[0] is all the roll states
        traj[0] = dyn_sys.x

        # flip the value with respect to the index
        #V = np.flip(V, grid.dims)

        opt_u = []
        opt_d = []
        t = []
        v_log= []
        t_earliest =0 # In helperOC, t_earliest starts at 1


        for iter in range(0,len(tau)):
 
            if iter < t_earliest:
                traj[iter] = np.array(dyn_sys.x)
                t.append(tau[iter])
                v_log.append(grid.get_value(V, dyn_sys.x))
                continue

            # Update trajectory, calculate gradient 
            gradient = self.spa_deriv(grid.get_index(dyn_sys.x), V, grid, periodic_dims=[0,1,2])
            for _ in range(subsamples):
                u = dyn_sys.opt_ctrl_non_hcl(_, dyn_sys.x, gradient)
                d = dyn_sys.opt_dstb_non_hcl(_, dyn_sys.x, gradient)
                #dNone = [0,0,0]
                bestU = u
                bestD = d
                nextState = self.next_state(dyn_sys, bestU, bestD, dt)
                self.update_state(dyn_sys, nextState)
                opt_u.append(u)
                opt_d.append(d)
            
            v_log.append(grid.get_value(V, dyn_sys.x))
            #the agent has entered the target
            if t_earliest == V.shape[-1]:
                traj[iter:] = np.array(dyn_sys.x)
                break

            #if iter != V.shape[-1]:
            traj[iter] = np.array(dyn_sys.x)
            
        return traj, opt_u, opt_d, t, v_log


    def getopt_traj(self,g,V,initial):
        small_number = 1e-5
        tau = np.arange(start=0, stop=self.lookback_length + small_number, step=self.t_step)
        [opt_traj, opt_u, opt_d, t,v_log] = self.compute_opt_traj (g,V,tau,self.dyn,initial)

        return opt_traj, opt_u, opt_d, t, v_log


class CartPoleSolution(object):
    def __init__(self, dMax=10, dMin=-10, distb_level=0.0):
        # warning: grid bound for each dimension should be close, not be too different. 
        self.grid_size = 50
        self.grid = Grid(np.array([-2.4, -10, -math.pi/2, -10]), np.array([2.4, 10, math.pi/2, 10]), 4, np.array([self.grid_size, self.grid_size, self.grid_size, self.grid_size]), [2])
        self.dyn = CartPole4D(x=[0, 0, 0, 0], dMax=dMax, dMin=dMin, uMode="min", dMode="max", distb_level=distb_level)  
        self.lookback_length = 2.0  # Look-back length and time step of computation
        self.t_step = 0.025
        self.distb_level = self.dyn.distb_level
        self.dMax = dMax
        self.result = None
        self.name = "cartpole"


    def FasTrackTarget(self, grid, ignore_dims, center):
        """
        Customized definition of FasTrack Target, it is similar to Cylinder, but with no radius
        """
        radius = 0.5
        data = np.zeros(grid.pts_each_dim)
        for i in range(grid.dims):
            if i not in ignore_dims:
                # This works because of broadcasting
                data = data + np.power(grid.vs[i] - center[i], 2)
        # data = np.sqrt(data) - radius
        return data


    def get_fastrack(self):
        # Hanyang: add new variable distb_level to generate value function
        self.targ = self.FasTrackTarget(self.grid, [0, 1], np.zeros(4)) # l(x) = V(0, x)
        small_number = 1e-5
        tau = np.arange(start=0, stop=self.lookback_length + small_number, step=self.t_step)
        compMethods = { "TargetSetMode": "maxVWithV0"}  # In this example, we compute based on FasTrack 
        # compMethods = { "TargetSetMode": "minVWithV0"}  # BRT
        # compMethods = { "TargetSetMode": None}  # BRS
        # compMethods = { "TargetSetMode": "maxVWithVInit"}  #
        slice = int((self.grid_size-1)/2)  
        self.po = PlotOptions(do_plot=False, plot_type="3d_plot", plotDims=[0,1,2], slicesCut=[int((self.grid_size-1)/2),int((self.grid_size-1)/2),int((self.grid_size-1)/2)])
        self.result = HJSolver(self.dyn, self.grid, self.targ, tau, compMethods, self.po, saveAllTimeSteps=False)
        np.save(f"safe_control_gym/hj_distbs/FasTrack_data/cartpole_{self.dMax}/cartpole_{self.distb_level}.npy", self.result)
        print("saving the result ..., done!")

        return self.result, self.grid, slice
    

    def spa_deriv(self, index, V, g, periodic_dims):
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

        return np.array(spa_derivatives)


    def next_state(self, dyn_sys, u, d, delta_t):
        #TODO: Hanyang: Need to check and modify this function!!!
        """
        Simulate apply control to dynamical systems for delta_t time

        Args:
            dyn_sys: dynamic system
            u: control
            d: disturbance
            delta_t: duration of control

        Returns:

        """
        init_state = dyn_sys.x
        t_span = [0, delta_t]
        solution = solve_ivp(dyn_sys.dynamics_non_hcl, t_span, init_state, args=[u, d], dense_output=True)
        next_state = solution.y[:, -1]
        if next_state[2] < -np.pi:
            next_state[2] += 2 * np.pi
        elif next_state[2] > np.pi:
            next_state[2] -= 2 * np.pi
        return next_state

    def update_state(self, dyn_sys, next_state):
        dyn_sys.x = next_state


    def compute_opt_traj(self,grid: Grid, V, tau, dyn_sys, initial, subsamples=1, arriveAfter = None, obstVal = None): # default subsample in helperOC is 4
        """
        Computes the optimal trajectory, controls and disturbance to a minimal BRT/BRS

        Args:
            grid:
            V:
            tau:
            dyn_sys:
            subsamples: Number of times opt_u and opt_d are calculated within dt

        Returns:
            traj: State of dyn_sys from time tau[-1] to tau[0]
            opt_u: Optimal control at each time
            opt_d: Optimal disturbance at each time

        """
        

        dt = (tau[1] - tau[0]) / subsamples
        dyn_sys.x = initial

        # first entry is dyn_sys at time tau[-1]
        # second entry is dyn_sys at time tau[-2]...
        traj = np.empty((len(tau), len(dyn_sys.x)))
        # Here it is chcking roll angle. traj[0] is all the roll states
        traj[0] = dyn_sys.x

        # flip the value with respect to the index
        #V = np.flip(V, grid.dims)

        opt_u = []
        opt_d = []
        t = []
        v_log= []
        t_earliest =0 # In helperOC, t_earliest starts at 1


        for iter in range(0,len(tau)):
 
            if iter < t_earliest:
                traj[iter] = np.array(dyn_sys.x)
                t.append(tau[iter])
                v_log.append(grid.get_value(V, dyn_sys.x))
                continue

            # Update trajectory, calculate gradient 
            gradient = self.spa_deriv(grid.get_index(dyn_sys.x), V, grid, periodic_dims=[0,1,2])
            for _ in range(subsamples):
                u = dyn_sys.opt_ctrl_non_hcl(_, dyn_sys.x, gradient)
                d = dyn_sys.opt_dstb_non_hcl(_, dyn_sys.x, gradient)
                #dNone = [0,0,0]
                bestU = u
                bestD = d
                nextState = self.next_state(dyn_sys, bestU, bestD, dt)
                self.update_state(dyn_sys, nextState)
                opt_u.append(u)
                opt_d.append(d)
            
            v_log.append(grid.get_value(V, dyn_sys.x))
            #the agent has entered the target
            if t_earliest == V.shape[-1]:
                traj[iter:] = np.array(dyn_sys.x)
                break

            #if iter != V.shape[-1]:
            traj[iter] = np.array(dyn_sys.x)
            
        return traj, opt_u, opt_d, t, v_log


    def getopt_traj(self,g,V,initial):
        small_number = 1e-5
        tau = np.arange(start=0, stop=self.lookback_length + small_number, step=self.t_step)
        [opt_traj, opt_u, opt_d, t,v_log] = self.compute_opt_traj (g,V,tau,self.dyn,initial)

        return opt_traj, opt_u, opt_d, t, v_log



def plot_2d(grid, V, index, slicecut):#

    dims_plot = index
    dim1, dim2= dims_plot[0], dims_plot[1]
    V_2D = V[:,:,slicecut,slicecut,slicecut,slicecut]
    complex_x = complex(0, grid.pts_each_dim[dim1])
    complex_y = complex(0, grid.pts_each_dim[dim2])
    mg_X, mg_Y= np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x, grid.min[dim2]:grid.max[dim2]: complex_y]

    fig = go.Figure(data=go.Contour(
        x=mg_X.flatten(),
        y=mg_Y.flatten(),
        z=V_2D.flatten(),
    ))
    fig.show()


def plot_optimal(opt_traj,v_log,opt_u,opt_d,initial):


    # plotting functions for traj 
    opt_traj_ = np.empty((6,len(opt_traj)))
    opt_u_d = np.empty((6,len(opt_traj)))
    
    for index in range(0,len(opt_traj)): 
        opt_traj_[0][index]=opt_traj[index][0]
        opt_traj_[1][index]=opt_traj[index][1]
        opt_traj_[2][index]=opt_traj[index][2]
        opt_traj_[3][index]=opt_traj[index][3]
        opt_traj_[4][index]=opt_traj[index][4]
        opt_traj_[5][index]=opt_traj[index][5]

        opt_u_d[0][index]=opt_u[index][0]
        opt_u_d[1][index]=opt_u[index][1]
        opt_u_d[2][index]=opt_u[index][2]
        opt_u_d[3][index]=opt_d[index][0]
        opt_u_d[4][index]=opt_d[index][1]
        opt_u_d[5][index]=opt_d[index][2]

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=180/np.pi*opt_traj_[0],name="Roll Angle",mode = "lines"))
    fig.add_trace(go.Scatter(y=180/np.pi*opt_traj_[1],name="Pitch Angle",mode = "lines"))
    fig.add_trace(go.Scatter(y=180/np.pi*opt_traj_[2],name="Yaw Angle",mode = "lines"))

    fig.update_layout(
    title="Angle (States) Over Timestep with disturbance with initial value{}".format(initial), 
    xaxis_title="Time Step", yaxis_title="Degrees"
    )
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=180/np.pi*opt_traj_[3],name="Roll Rate",mode = "lines"))
    fig.add_trace(go.Scatter(y=180/np.pi*opt_traj_[4],name="Pitch Rate",mode = "lines"))
    fig.add_trace(go.Scatter(y=180/np.pi*opt_traj_[5],name="Yaw Rate",mode = "lines"))

    fig.update_layout(
    title="Angle Rate (States) Over Timestep (0.001s step, 2s time) with disturbance", xaxis_title="Time Step", yaxis_title="Degrees/sec"
    )
    fig.show()


    fig = go.Figure(data=go.Scatter(
            y=v_log, line=dict(color="crimson")),
            layout_title_text="Value Function with initial value{}".format(initial),
            layout_yaxis_range=[0,2]

        )
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=opt_u_d[0],name="tau_x",mode = "lines"))
    fig.add_trace(go.Scatter(y=opt_u_d[1],name="tau_y",mode = "lines"))
    fig.add_trace(go.Scatter(y=opt_u_d[2],name="tau_z",mode = "lines"))

    fig.update_layout(
    title="Optimal Control Over Timestep (0.001s step, 2s time) with disturbance", xaxis_title="Time Step", yaxis_title="Torques"
    )
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=opt_u_d[3],name="tau_wx",mode = "lines"))
    fig.add_trace(go.Scatter(y=opt_u_d[4],name="tau_wy",mode = "lines"))
    fig.add_trace(go.Scatter(y=opt_u_d[5],name="tau_wz",mode = "lines"))

    fig.update_layout(
    title="Optimal Disturbance Over Timestep (0.001s step, 2s time)", xaxis_title="Time Step", yaxis_title="Torques"
    )
    fig.show()

def maxDiff(a):
    vmin = a[0]
    dmax = 0
    for i in range(len(a)):
        if (a[i] < vmin):
            vmin = a[i]
        elif (a[i] - vmin > dmax):
            dmax = a[i] - vmin
    return dmax


def evaluation (iteration, threshold):


    roll_range = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
    pitch_range = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
    roll_rate = [-2.5,-2,-1.5,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.5,2,2.5]
    pitch_rate = [-2.5,-2,-1.5,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.5,2,2.5]

    initial = np.empty((iteration,6))

    for i in range(iteration):
        initial[i] = [random.choice(roll_range),random.choice(pitch_range),0,random.choice(roll_rate),random.choice(pitch_rate),0]
    initial_nodu=np.unique(initial,axis=0)

    for i in range(len(initial_nodu)):
        [opt_traj, opt_u, opt_d, t, v_log] = uavsol.getopt_traj(grid,V,initial_nodu[i])
        # plot optimal control
        if maxDiff(v_log) > threshold:
            print ("These initial values lead to an unstable system under disturbance")
            print (initial_nodu[i])
            plot_optimal(opt_traj,v_log,opt_u,opt_d,initial_nodu[i])
        else: 
            print (initial_nodu[i],"gives a stable system")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description ='Calculate Value Function using HJ')
    
    parser.add_argument('--dynamics', default="quadrotor", type=str, help='which dynamics to use for data')
    parser.add_argument('--load_data', action='store_true', help="whether to load precollected data")
    parser.add_argument('--evaluate', action='store_true', help= "whether to plot evaluation")
    parser.add_argument('--dMax', default=10.0, type=float, help= "for cartpole dynamics, the maximum disturbance")
    parser.add_argument('--distb_level', default=0.0, type=float, help= "for cartpole dynamics, the distb level")

    args = parser.parse_args()
    
    if args.dynamics == "quadrotor":
        # python Fastrack.py --dynamics cartpole
        uavsol = UAVSolution()
        distb_level = uavsol.distb_level
        if args.load_data: 
            print("Loading the value function.")
            slicecut = 7  #for 15*15
            V = np.load(f'safe_control_gym/hj_distbs/FasTrack_data/quadrotor/quadrotor_{distb_level}_15x15.npy')
            grid = Grid(np.array([-math.pi/2.4, -math.pi/2.4, -math.pi/2.4, -math.pi, -math.pi, -math.pi]), np.array([math.pi/2.4, math.pi/2.4, math.pi/2.4, math.pi, math.pi, math.pi]), 6, np.array([15,15,15,15,15,15]), [0,1,2])
            
        else: 
            print("Calculate new value functioning.")
            [V, grid, slicecut] = uavsol.get_fastrack()

    elif args.dynamics == "cartpole":
        # python safe_control_gym/hj_distbs/Fastrack.py --dynamics cartpole --dMax 2.0 --distb_level 0.0
        # Hanyang: need to change the dMax when calculate input
        dMax = args.dMax
        dMin = -args.dMax
        distb_level = args.distb_level

        cartpolesol = CartPoleSolution(dMax=dMax, dMin=dMin, distb_level=distb_level)  
        distb_level = cartpolesol.distb_level
        if args.load_data: 
            print("Loading the value function.")
            slicecut = 7  #for 15*15
            # V = np.load(f'safe_control_gym/hj_distbs/FasTrack_data/cartpole/cartpole_{distb_level}.npy')
            V = np.load(f'safe_control_gym/hj_distbs/FasTrack_data/cartpole_test/cartpole_{distb_level}.npy')
            grid = Grid(np.array([-4.8, -5, -math.pi, -10]), np.array([4.8, 5, math.pi, 10]), 4, np.array([45, 45, 45, 45]), [2])
            
        else: 
            print("Calculate new value functioning.")
            [V, grid, slicecut] = cartpolesol.get_fastrack()

    ## if you want to see the 2D value function plot with prelaoding please uncomment the code below
    # you should comment out the line where we calculated the [V,grid,slicecut] if you want to plot pre-calculated data

    # '''    
    # slicecut = 7  #for 15*15
    # V = np.load('fastrack_0.5_15x15.npy')
    # grid = Grid(np.array([-math.pi/2.4, -math.pi/2.4, -math.pi/2.4, -math.pi, -math.pi, -math.pi]), np.array([math.pi/2.4, math.pi/2.4, math.pi/2.4, math.pi, math.pi, math.pi]), 6, np.array([15,15,15,15,15,15]), [0,1,2])
    # '''

    # plot_2d(grid, V, [1,2], slicecut) # we are plotting first two dimensions
    
    # if args.evaluate:
    # ## if you want to evaluate your calculated control/disturbance, please uncomment the following code
    # # The code randomly sampled states from the pool, and check whether the control/disturbance gives a stable system
    # # It will generate the plots for unstable systems in your broswer
    # # The creteria is "threshold" -> The difference between maxvalue and minvalue in value functions for all time step

    #     iteration = 50 # how many samples you want to generate
    #     threshold = 5 # max diff of values
    #     evaluation (iteration, threshold) 