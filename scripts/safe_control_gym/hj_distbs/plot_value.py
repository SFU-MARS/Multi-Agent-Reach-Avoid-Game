import math
import time
import numpy as np

from odp.Grid import Grid
from utility import plot_value
from plot_options import PlotOptions
from plotting_utilities import plot_isosurface

# Load the value function and the grid
value_cartpole = np.load('safe_control_gym/hj_distbs/FasTrack_data/cartpole_2.0/cartpole_1.0.npy')

# grid_cartpole = Grid(np.array([-4.8, -5, -math.pi, -10]), np.array([4.8, 5, math.pi, 10]), 4, np.array([50, 50, 50, 50]), [2])
grid_cartpole = Grid(np.array([-2.4, -10, -math.pi/2, -10]), np.array([2.4, 10, math.pi/2, 10]), 4, np.array([50, 50, 50, 50]), [2])

# grid_quadrotor = Grid(np.array([-math.pi/2.4, -math.pi/2.4, -math.pi/2.4, -math.pi, -math.pi, -math.pi]), 
#                       np.array([math.pi/2.4, math.pi/2.4, math.pi/2.4, math.pi, math.pi, math.pi]), 6, 
#                       np.array([15,15,15,15,15,15]), 
#                       [0,1,2])

# Plotting positions
# plot_state = np.array([[0, 0, 0, 0]])  # [x, v, theta, w]
# print(plot_state.shape)

# plot_value(plot_state, value_cartpole, grid_cartpole, plot_dim=[0, 1])

# center = np.zeros(4)
# target_set = np.zeros(grid_cartpole.pts_each_dim)
# for i in range(grid_cartpole.dims):
#     target_set = target_set + np.power(grid_cartpole.vs[i] - center[i], 2) 

# target_set = np.sqrt(target_set) - 0.5
print(f"The value function at the origin is {value_cartpole[tuple([25, 25, 25, 25])]}")
# po = PlotOptions(do_plot=True, plot_type="set", plotDims=[0, 1, 2], slicesCut=[25])
po = PlotOptions(do_plot=True, plot_type="set", plotDims=[2, 3], slicesCut=[25, 25])

plot_isosurface(grid_cartpole, value_cartpole, po)
# plot_isosurface(grid_cartpole, target_set, po)
