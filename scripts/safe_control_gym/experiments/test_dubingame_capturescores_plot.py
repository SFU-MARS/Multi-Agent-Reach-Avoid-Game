import math
import numpy as np
import matplotlib.pyplot as plt
from odp.Grid import Grid
from safe_control_gym.utils.plotting import po2slice1vs1
    

fixed_defender_position = np.array([[0.7, -0.5, 1.00]])
# initial_attacker = np.array([[-0.7, 0.5, -1.0]])


x_range = np.arange(-0.95, 1.0, 0.05)  # from -0.95 to 0.95
y_range = np.arange(-0.95, 1.0, 0.05)

# dubin car scores
loaded_scores = np.load('training_results/dubin_game/sb3/random_init/seed_42/10000000.0steps/score_matrix_[0.7, -0.5, 1.00].npy')

# Process
loaded_scores = loaded_scores.T

# Ensure that x_range and y_range cover from -1.0 to 1.0 for plotting purposes
extended_x_range = np.linspace(-1.0, 1.0, len(x_range))
extended_y_range = np.linspace(-1.0, 1.0, len(y_range))

# Plot the HJ value function
# Step 1 load the value function, initilize the grids
# value1vs0 = np.load('safe_control_gym/envs/gym_game/values/1vs0Dubin_easier.npy')
value1vs1 = np.load('safe_control_gym/envs/gym_game/values/1vs1Dubin_easier.npy')
# grid1vs0 = Grid(np.array([-1.1, -1.1, -math.pi]), np.array([1.1, 1.1, math.pi]), 3, np.array([100, 100, 200]), [2])
grid1vs1 = Grid(np.array([-1.1, -1.1, -math.pi, -1.1, -1.1, -math.pi]), np.array([1.1, 1.1, math.pi, 1.1, 1.1, math.pi]), 
                        6, np.array([28, 28, 28, 28, 28, 28]), [2, 5])

initial_attacker = np.array([[-0.7, 0.5, -1.0]])
a1x_slice, a1y_slice, a1o_slice, d1x_slice, d1y_slice, d1o_slice = grid1vs1.get_index(np.concatenate((initial_attacker[0], fixed_defender_position[0])))
value_function1vs1 = value1vs1[:, :, a1o_slice, d1x_slice, d1y_slice, d1o_slice].squeeze()
value_function1vs1 = np.swapaxes(value_function1vs1, 0, 1)

# a1x_slice, a1y_slice, d1x_slice, d1y_slice = po2slice1vs1(initial_attacker[0], fixed_defender_position[0], value1vs1.shape[0])
# value_function1vs1 = value1vs1[:, :, d1x_slice, d1y_slice].squeeze()
# value_function1vs1 = np.swapaxes(value_function1vs1, 0, 1)
# print(f"The shape of the value_function1vs1 is {value_function1vs1.shape}")
dims_plot = [0, 1]
dim1, dim2 = dims_plot[0], dims_plot[1]
x_hj = np.linspace(-1.1, 1.1, value_function1vs1.shape[dim1])
y_hj = np.linspace(-1.1, 1.1, value_function1vs1.shape[dim2])

# Create the 2D plot
plt.figure(figsize=(8, 8))
plt.imshow(loaded_scores, extent=[-1, 1, -1, 1], origin='lower', aspect='auto', cmap='Pastel1')  # cmap='viridis', Pastel1,Pastel2

# Add color bar to indicate the score values
# plt.colorbar(label='Scores')
plt.scatter(fixed_defender_position[0][0], fixed_defender_position[0][1], color='magenta', marker='*', s=100, label='Fixed Defender')
contour = plt.contour(x_hj, y_hj, value_function1vs1, levels=0, colors='#4B0082', linewidths=3.0, linestyles='dashed')  # colors='magenta'

# Set the x and y axis labels
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')

# Set the title of the plot
# plt.title('2D Plot of Scores')
# plt.savefig(f'training_results/easier_game/sb3/random/1vs0_1vs1/seed_2024/10000000.0steps/score_matrix_{fixed_defender_position[0]}.png')
# plt.savefig(f'training_results/rarl_game/rarl/seed_42/score_matrix_{fixed_defender_position[0]}.png')
plt.savefig(f'training_results/dubin_game/sb3/random_init/seed_42/10000000.0steps/score_matrix_{fixed_defender_position[0]}.png')
# Show the plot
plt.show()
