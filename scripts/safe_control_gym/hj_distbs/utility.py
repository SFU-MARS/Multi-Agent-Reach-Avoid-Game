import math
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.graph_objects import Layout



def po2slice1vs1(plot_state, grid_size):
    """ Convert the position to the slice of the value functio.

    Args:
        attacker (np.ndarray): the attacker's state
        grid_size (int): the size of the grid
    
    Returns:
        joint_slice (tuple): the joint slice of the joint state using the grid size

    """
    joint_state = (plot_state[0][0], plot_state[0][1], plot_state[0][2], plot_state[0][3])  # (xA1, yA1, xD1, yD1)
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


def plot_value(plot_state, value, grid, plot_dim=[0, 1]):
    """Plot the value function of the game.

    Args:
        attackers (np.ndarray): The attackers' states.
        defenders (np.ndarray): The defenders' states.
        plot_attacker (int): The attacker to plot the value function, other attackers are ignored.
        plot_defender (int): The defender to plot the value function, other defenders are ignored.
        fix_agent (int): The agent to fix (1 for defender, 0 for attacker).
        value1vs1 (np.ndarray): The value function of the 1 vs. 1 game.
        grid1vs1 (Grid): The grid of the 1 vs. 1 game.

    Returns:
        None
    """
    position_slices = po2slice1vs1(plot_state, value.shape[0]) # x_slice, v_slice, theta_slice, w_slice
    plot_slices = [slice(None)] * value.ndim
    plot_slices[plot_dim[0]] = position_slices[plot_dim[0]]
    plot_slices[plot_dim[1]] = position_slices[plot_dim[1]]
    dim1, dim2  = plot_dim
    # Convert the list of slices to a tuple and apply to the array
    value_function = value[tuple(plot_slices)]   

    complex_x = complex(0, grid.pts_each_dim[dim1])
    complex_y = complex(0, grid.pts_each_dim[dim2])
    mg_X, mg_Y = np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x, grid.min[dim2]:grid.max[dim2]: complex_y]
    plot_x = plot_state[:, 0]
    plot_y = plot_state[:, 1]

    print("Plotting beautiful 2D plots. Please wait\n")
    fig = go.Figure(data=go.Contour(
        x=mg_X.flatten(),
        y=mg_Y.flatten(),
        z=value_function.flatten(),
        zmin=0.0,
        ncontours=1,
        contours_coloring = 'none', # former: lines 
        name= "Zero-Level", # zero level
        line_width = 1.5,
        line_color = 'magenta',
        zmax=0.0,
    ), layout=Layout(plot_bgcolor='rgba(0,0,0,0)')) #,paper_bgcolor='rgba(0,0,0,0)'
    # plot target
    fig.add_shape(type='rect', x0=0.6, y0=0.1, x1=0.8, y1=0.3, line=dict(color='purple', width=3.0), name="Target")
    fig.add_trace(go.Scatter(x=[0.6, 0.8], y=[0.1, 0.1], mode='lines', name='Target', line=dict(color='purple')))
    # plot obstacles
    fig.add_shape(type='rect', x0=-0.1, y0=0.3, x1=0.1, y1=0.6, line=dict(color='black', width=3.0))
    fig.add_shape(type='rect', x0=-0.1, y0=-1.0, x1=0.1, y1=-0.3, line=dict(color='black', width=3.0))
    fig.add_trace(go.Scatter(x=[-0.1, 0.1], y=[0.3, 0.3], mode='lines', name='Obstacle', line=dict(color='black')))
    # plot 
    fig.add_trace(go.Scatter(x=plot_x, y=plot_y, mode="markers", name='Attacker', marker=dict(symbol="triangle-up", size=10, color='red')))
    # figure settings
    fig.update_layout(title={'text': f"<b>1 vs. 1 value function<b>", 'y':0.85, 'x':0.4, 'xanchor': 'center','yanchor': 'top', 'font_size': 20})
    fig.update_layout(autosize=False, width=580, height=500, margin=dict(l=50, r=50, b=100, t=100, pad=0), paper_bgcolor="White", xaxis_range=[-1, 1], yaxis_range=[-1, 1], font=dict(size=20)) # $\mathcal{R} \mathcal{A}_{\infty}^{21}$
    fig.update_xaxes(showline = True, linecolor = 'black', linewidth = 2.0, griddash = 'dot', zeroline=False, gridcolor = 'Lightgrey', mirror=True, ticks='outside') # showgrid=False
    fig.update_yaxes(showline = True, linecolor = 'black', linewidth = 2.0, griddash = 'dot', zeroline=False, gridcolor = 'Lightgrey', mirror=True, ticks='outside') # showgrid=False,
    fig.show()
    print("Please check the plot on your browser.")
    