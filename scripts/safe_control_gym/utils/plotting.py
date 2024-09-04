'''Plotting utilities.'''

import os
import os.path as osp
import re
import cv2
import math
import torch
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Layout
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DIV_LINE_WIDTH = 50


COLORS = [
    'blue',
    'green',
    'red',
    'black',
    'cyan',
    'magenta',
    'yellow',
    'brown',
    'purple',
    'pink',
    'orange',
    'teal',
    'coral',
    'lightblue',
    'lime',
    'lavender',
    'turquoise',
    'darkgreen',
    'tan',
    'salmon',
    'gold',
    'lightpurple',
    'darkred',
    'darkblue',
]


LINE_STYLES = [
    ('solid', 'solid'),
    ('dotted', 'dotted'),
    ('dashed', 'dashed'),
    ('dashdot', 'dashdot'),
]


LINE_STYLES2 = [('loosely dotted', (0, (1, 10))), ('dotted', (0, (1, 1))),
                ('densely dotted', (0, (1, 1))),
                ('loosely dashed', (0, (5, 10))), ('dashed', (0, (5, 5))),
                ('densely dashed', (0, (5, 1))),
                ('loosely dashdotted', (0, (3, 10, 1, 10))),
                ('dashdotted', (0, (3, 5, 1, 5))),
                ('densely dashdotted', (0, (3, 1, 1, 1))),
                ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
                ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
                ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
                ]


def rolling_window(a, window):
    '''Window data.'''
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def window_func(x, y, window, func):
    '''Evaluate a function on windowed data.'''
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window - 1:], yw_func


def filter_log_dirs(pattern, negative_pattern=' ', root='./log', **kwargs):
    '''Gets list of experiment folders as specified.'''
    dirs = [item[0] for item in os.walk(root)]
    leaf_dirs = []
    for i in range(len(dirs)):
        if i + 1 < len(dirs) and dirs[i + 1].startswith(dirs[i]):
            continue
        leaf_dirs.append(dirs[i])
    names = []
    p = re.compile(pattern)
    neg_p = re.compile(negative_pattern)
    for leaf_dir in leaf_dirs:
        if p.match(leaf_dir) and not neg_p.match(leaf_dir):
            names.append(leaf_dir)
            print(leaf_dir)
    print('')
    return sorted(names)


def align_runs(xy_list, x_num_max=None):
    '''Aligns the max of the x data across runs.'''
    x_max = float('inf')
    for x, y in xy_list:
        # Align length of x data (get min across all runs).
        x_max = min(x_max, len(x))
    if x_num_max:
        x_max = min(x_max, x_num_max)
    aligned_list = [[x[:x_max], y[:x_max]] for x, y in xy_list]
    return aligned_list


def smooth_runs(xy_list, window=10):
    '''Smooth the data curves by mean filtering.'''
    smoothed_list = [
        window_func(np.asarray(x), np.asarray(y), window, np.mean)
        for x, y in xy_list
    ]
    return smoothed_list


def select_runs(xy_list, criterion, top_k=0):
    '''Pickes the top k runs based on a criterion.'''
    perf = [criterion(y) for _, y in xy_list]
    top_k_runs = np.argsort(perf)[-top_k:]
    selected_list = []
    for r, (x, y) in enumerate(xy_list):
        if r in top_k_runs:
            selected_list.append((x, y))
    return selected_list


def interpolate_runs(xy_list, interp_interval=100):
    '''Uses the same x data by interpolation across runs.'''
    x_right = float('inf')
    for x, y in xy_list:
        x_right = min(x_right, x[-1])
    # Shape: (data_len,).
    x = np.arange(0, x_right, interp_interval)
    y = []
    for x_, y_ in xy_list:
        y.append(np.interp(x, x_, y_))
    # Shape: (num_runs, data_len).
    y = np.asarray(y)
    return x, y


def load_from_log_file(path):
    '''Return x, y sequence data from the stat csv.'''
    with open(path, 'r') as f:
        lines = f.readlines()
    # Labels.
    xk, yk = [k.strip() for k in lines[0].strip().split(',')]
    # Values.
    x, y = [], []
    for line in lines[1:]:
        data = line.strip().split(',')
        x.append(float(data[0].strip()))
        y.append(float(data[1].strip()))
    x = np.array(x)
    y = np.array(y)
    return xk, x, yk, y


def load_from_logs(log_dir):
    '''Return dict of stats under log_dir folder (`exp_dir/logs/`).'''
    log_files = []
    # Fetch all log files.
    for r, _, f in os.walk(log_dir):
        for file in f:
            if '.log' in file:
                log_files.append(os.path.join(r, file))
    # Fetch all stats from log files.
    data = {}
    for path in log_files:
        name = path.split(log_dir)[-1].replace('.log', '')
        xk, x, yk, y = load_from_log_file(path)
        data[name] = (xk, x, yk, y)
    return data


def plot_from_logs(src_dir, out_dir, window=None, keys=None):
    '''Generate a plot for each stat in an experiment `logs` folder.

    Args:
        src_dir (str): folder to read logs.
        out_dir (str): folder to save figures.
        window (int): window size for smoothing.
        keys (list): specify name of stats to plot, None means plot all.
    '''
    # Find all logs.
    log_files = []
    for r, _, f in os.walk(src_dir):
        for file in f:
            if '.log' in file:
                log_files.append(os.path.join(r, file))
    # Make a figure for each log file.
    stats = {}
    for path in log_files:
        name = path.split(src_dir)[-1].replace('.log', '')
        if keys:
            if name not in keys:
                continue
        xk, x, yk, y = load_from_log_file(path)
        stats[name] = (xk, x, yk, y)
        if window:
            x, y = window_func(x, y, window, np.mean)
        plt.clf()
        plt.plot(x, y)
        plt.title(name)
        plt.xlabel(xk)
        plt.ylabel(yk)
        plt.savefig(os.path.join(out_dir, name.replace('/', '-') + '.jpg'))
    return stats


def plot_from_tensorboard_log(src_dir,
                              out_dir,
                              window=None,
                              keys=None,
                              xlabel='step'):
    '''Generates a plot for each stat from tfb log file in source folder.'''
    event_acc = EventAccumulator(src_dir)
    event_acc.Reload()
    if not keys:
        keys = event_acc.Tags()['scalars']
    stats = {}
    for k in keys:
        _, x, y = zip(*event_acc.Scalars(k))
        x, y = np.asarray(x), np.asarray(y)
        stats[k] = (x, y)
        if window:
            x, y = window_func(x, y, window, np.mean)
        plt.clf()
        plt.plot(x, y)
        plt.title(k)
        plt.xlabel(xlabel)
        plt.ylabel(k)
        # Use '-' instead of '/' to connect group and stat name.
        out_path = os.path.join(out_dir, k.replace('/', '-') + '.jpg')
        plt.savefig(out_path)
    return stats


def plot_from_experiments(legend_dir_specs,
                          out_path='temp.jpg',
                          scalar_name=None,
                          title='Traing Curves',
                          xlabel='Epochs',
                          ylabel='Loss',
                          window=None,
                          x_num_max=None,
                          num_std=1,
                          use_tb_log=True
                          ):
    '''Generates plot among algos, each with several seed runs.

    Example:
        make a plot on average reward for gnn and mlp:

        > plot_from_experiments(
            {
                'gnn': [
                    'results/algo1/seed0',
                    'results/algo1/seed1',
                    'results/algo1/seed2'
                ],
                'mlp': [
                    'results/algo2/seed6',
                    'results/algo2/seed1',
                    'results/algo2/seed9',
                    'results/algo2/seed3'
                ],
            },
            out_path='avg_reward.jpg',
            scalar_name='loss_eval/total_rewards',
            title='Average Reward',
            xlabel='Epochs',
            ylabel='Reward',
            window=10
        )
    '''
    assert scalar_name is not None, 'Must provide a scalar name to plot'
    # Get all stats.
    stats = defaultdict(list)
    for stat, dirs in legend_dir_specs.items():
        for d in dirs:
            # Pick from either log source (tensorboard or log text files).
            if use_tb_log:
                event_acc = EventAccumulator(d)
                event_acc.Reload()
                _, x, y = zip(*event_acc.Scalars(scalar_name))
                del event_acc
            else:
                path = os.path.join(d, 'logs', scalar_name + '.log')
                _, x, _, y = load_from_log_file(path)
            # Smoothing.
            x, y = np.asarray(x), np.asarray(y)
            if window:
                x, y = window_func(x, y, window, np.mean)
            stats[stat].append([x, y])
    # Post-processing.
    x_max = float('inf')
    for _, runs in stats.items():
        for x, y in runs:
            # Align length of x data (get min across all runs & all algos).
            x_max = min(x_max, len(x))
    if x_num_max:
        x_max = min(x_max, x_num_max)
    processed_stats = {}
    for name, runs in stats.items():
        # Use same x for all runs to an algorithm.
        x = np.array([x[:x_max] for x, _ in runs])[0]
        # Different y for different runs.
        y = np.stack([y[:x_max] for _, y in runs])
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        processed_stats[name] = [x, y_mean, y_std]
    # Actual plot.
    plt.clf()
    for i, name in enumerate(processed_stats.keys()):
        color = COLORS[i]
        x, y_mean, y_std = processed_stats[name]
        plt.plot(x, y_mean, label=name, color=color)
        plt.fill_between(x,
                         y_mean + num_std * y_std,
                         y_mean - num_std * y_std,
                         alpha=0.3,
                         color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    return stats, processed_stats


def get_log_dirs(all_logdirs,
                 select=None,
                 exclude=None
                 ):
    '''Find all folders for plotting.

    All 3 arguments can be exposed as list args from command line.

    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           pull data from it;
        2) if not, check to see if the entry is a prefix for a
           real directory, and pull data from that.
    '''
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1] == os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)

            def fulldir(x):
                return osp.join(basedir, x)

            prefix = logdir.split(os.sep)[-1]
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])
    # Enforce selection rules, which check logdirs for certain substrings. Makes it easier to look
    # at graphs from particular ablations, if you launch many jobs at once with similar names.
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [
            log for log in logdirs if all(not (x in log) for x in exclude)
        ]
    # Verify logdirs.
    print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '=' * DIV_LINE_WIDTH)
    return logdirs


def animation(attackers_traj, defenders_traj, attackers_status):
    """Animate the game.

    Args:
        attackers_traj (list): List of attackers' trajectories.
        defenders_traj (list): List of defenders' trajectories.
        attackers_status (list): List of attackers' status.

    Returns:
        None
    """
    # Determine the number of steps
    num_steps = len(attackers_traj)
    num_attackers = attackers_traj[0].shape[0]
    num_defenders = defenders_traj[0].shape[0]

    # Create frames for animation
    frames = []
    for step in range(num_steps):
        attackers = attackers_traj[step]
        defenders = defenders_traj[step]
        status = attackers_status[step]

        x_list = []
        y_list = []
        symbol_list = []
        color_list = []

        # Go through list defenders
        for j in range(num_defenders):
            x_list.append(defenders[j][0])
            y_list.append(defenders[j][1])
            symbol_list += ["square"]
            color_list += ["blue"]
        
        # Go through list of attackers
        for i in range(num_attackers):
            x_list.append(attackers[i][0])
            y_list.append(attackers[i][1])
            if status[i] == -1:  # attacker is captured
                symbol_list += ["cross-open"]
            elif status[i] == 1:  # attacker has arrived
                symbol_list += ["circle"]
            else:  # attacker is free
                symbol_list += ["triangle-up"]
            color_list += ["red"]

        # Generate a frame based on the characteristic of each agent
        frames.append(go.Frame(data=go.Scatter(x=x_list, y=y_list, mode="markers", name="Agents trajectory",
                                               marker=dict(symbol=symbol_list, size=5, color=color_list), showlegend=False)))

    
    # Static object - obstacles, goal region, grid
    fig = go.Figure(data = go.Scatter(x=[0.6, 0.8], y=[0.1, 0.1], mode='lines', name='Target', line=dict(color='purple')),
                    layout=Layout(plot_bgcolor='rgba(0,0,0,0)', updatemenus=[dict(type="buttons",
                                                                            buttons=[dict(label="Play", method="animate",
                                                                            args=[None, {"frame": {"duration": 30, "redraw": True},
                                                                            "fromcurrent": True, "transition": {"duration": 0}}])])]), frames=frames) # for the legend

    # plot target
    fig.add_shape(type='rect', x0=0.6, y0=0.1, x1=0.8, y1=0.3, line=dict(color='purple', width=2.0), name="Target")
    # plot obstacles
    fig.add_shape(type='rect', x0=-0.1, y0=0.3, x1=0.1, y1=0.6, line=dict(color='black', width=2.0), name="Obstacle")
    fig.add_shape(type='rect', x0=-0.1, y0=-1.0, x1=0.1, y1=-0.3, line=dict(color='black', width=2.0))
    fig.add_trace(go.Scatter(x=[-0.1, 0.1], y=[0.3, 0.3], mode='lines', name='Obstacle', line=dict(color='black')))

    # figure settings
    # fig.update_layout(showlegend=False)  # to display the legends or not
    fig.update_layout(autosize=False, width=560, height=500, margin=dict(l=50, r=50, b=100, t=100, pad=0),
                      title={'text': "<b>Our method, t={}s<b>".format(num_steps/200), 'y':0.85, 'x':0.4, 'xanchor': 'center','yanchor': 'top', 'font_size': 20}, paper_bgcolor="White", xaxis_range=[-1, 1], yaxis_range=[-1, 1], font=dict(size=20)) # LightSteelBlue
    fig.update_xaxes(showline = True, linecolor = 'black', linewidth = 2.0, griddash = 'dot', zeroline=False, gridcolor = 'Lightgrey', mirror=True, ticks='outside') # showgrid=False
    fig.update_yaxes(showline = True, linecolor = 'black', linewidth = 2.0, griddash = 'dot', zeroline=False, gridcolor = 'Lightgrey', mirror=True, ticks='outside') # showgrid=False,
    fig.show()


def animation_easier_game(attackers_traj, defenders_traj, attackers_status):
    """Animate the game.

    Args:
        attackers_traj (list): List of attackers' trajectories.
        defenders_traj (list): List of defenders' trajectories.
        attackers_status (list): List of attackers' status.

    Returns:
        None
    """
    # Determine the number of steps
    num_steps = len(attackers_traj)
    num_attackers = attackers_traj[0].shape[0]
    num_defenders = defenders_traj[0].shape[0]

    # Create frames for animation
    frames = []
    for step in range(num_steps):
        attackers = attackers_traj[step]
        defenders = defenders_traj[step]
        status = attackers_status[step]

        x_list = []
        y_list = []
        symbol_list = []
        color_list = []

        # Go through list defenders
        for j in range(num_defenders):
            x_list.append(defenders[j][0])
            y_list.append(defenders[j][1])
            symbol_list += ["square"]
            color_list += ["blue"]
        
        # Go through list of attackers
        for i in range(num_attackers):
            x_list.append(attackers[i][0])
            y_list.append(attackers[i][1])
            if status[i] == -1:  # attacker is captured
                symbol_list += ["cross-open"]
            elif status[i] == 1:  # attacker has arrived
                symbol_list += ["circle"]
            else:  # attacker is free
                symbol_list += ["triangle-up"]
            color_list += ["red"]

        # Generate a frame based on the characteristic of each agent
        frames.append(go.Frame(data=go.Scatter(x=x_list, y=y_list, mode="markers", name="Agents trajectory",
                                               marker=dict(symbol=symbol_list, size=5, color=color_list), showlegend=False)))

    
    # Static object - obstacles, goal region, grid
    fig = go.Figure(data = go.Scatter(x=[0.6, 0.8], y=[0.1, 0.1], mode='lines', name='Target', line=dict(color='purple')),
                    layout=Layout(plot_bgcolor='rgba(0,0,0,0)', updatemenus=[dict(type="buttons",
                                                                            buttons=[dict(label="Play", method="animate",
                                                                            args=[None, {"frame": {"duration": 30, "redraw": True},
                                                                            "fromcurrent": True, "transition": {"duration": 0}}])])]), frames=frames) # for the legend

    # plot target
    fig.add_shape(type='rect', x0=0.6, y0=0.1, x1=0.8, y1=0.3, line=dict(color='purple', width=2.0), name="Target")
    # # plot obstacles
    # fig.add_shape(type='rect', x0=-0.1, y0=0.3, x1=0.1, y1=0.6, line=dict(color='black', width=2.0), name="Obstacle")
    # fig.add_shape(type='rect', x0=-0.1, y0=-1.0, x1=0.1, y1=-0.3, line=dict(color='black', width=2.0))
    # fig.add_trace(go.Scatter(x=[-0.1, 0.1], y=[0.3, 0.3], mode='lines', name='Obstacle', line=dict(color='black')))

    # figure settings
    # fig.update_layout(showlegend=False)  # to display the legends or not
    fig.update_layout(autosize=False, width=560, height=500, margin=dict(l=50, r=150, b=100, t=100, pad=0),
                      title={'text': "<b>Our method, t={}s<b>".format(num_steps/200), 'y':0.85, 'x':0.4, 'xanchor': 'center','yanchor': 'top', 'font_size': 20}, paper_bgcolor="White", xaxis_range=[-1, 1], yaxis_range=[-1, 1], font=dict(size=20)) # LightSteelBlue
    fig.update_xaxes(showline = True, linecolor = 'black', linewidth = 2.0, griddash = 'dot', zeroline=False, gridcolor = 'Lightgrey', mirror=True, ticks='outside') # showgrid=False
    fig.update_yaxes(showline = True, linecolor = 'black', linewidth = 2.0, griddash = 'dot', zeroline=False, gridcolor = 'Lightgrey', mirror=True, ticks='outside') # showgrid=False,
    fig.show()
    

def animation_dub(attackers_traj, defenders_traj, attackers_status):
    """Animate the game.

    Args:
        attackers_traj (list): List of attackers' trajectories.
        defenders_traj (list): List of defenders' trajectories.
        attackers_status (list): List of attackers' status.

    Returns:
        None
    """
    # Determine the number of steps
    num_steps = len(attackers_traj)
    num_attackers = attackers_traj[0].shape[0]
    if defenders_traj is not None:
        num_defenders = defenders_traj[0].shape[0]

    # Create frames for animation
    frames = []
    arrow_length = 0.1  # Length of the arrow indicating direction

    def calculate_arrow_end(x, y, heading):
        end_x = x + arrow_length * np.cos(heading)
        end_y = y + arrow_length * np.sin(heading)
        return end_x, end_y

    for step in range(num_steps):
        attackers = attackers_traj[step]
        if defenders_traj is not None:
            defenders = defenders_traj[step]
        status = attackers_status[step]

        attacker_x_list = []
        attcker_y_list = []
        attacker_symbol_list = []
        attacker_color_list = []
        defender_x_list = []
        defender_y_list = []
        defender_symbol_list = []
        defender_color_list = []

        # Go through list of defenders
        if defenders_traj is not None:
            for j in range(num_defenders):
                defender_x_list.append(defenders[j][0])
                defender_y_list.append(defenders[j][1])
                defender_symbol_list += ["square"]
                defender_color_list += ["blue"]

                # Calculate defender arrow end point
                defender_end_x, defender_end_y = calculate_arrow_end(defenders[j][0], defenders[j][1], defenders[j][2])
                defender_x_list.append(defender_end_x)
                defender_y_list.append(defender_end_y)
                defender_symbol_list += ["line-ns"]

        # Go through list of attackers
        for i in range(num_attackers):
            attacker_x_list.append(attackers[i][0])
            attcker_y_list.append(attackers[i][1])
            if status[i] == -1:  # attacker is captured
                attacker_symbol_list += ["cross-open"]
            elif status[i] == 1:  # attacker has arrived
                attacker_symbol_list += ["circle"]
            else:  # attacker is free
                attacker_symbol_list += ["triangle-up"]
            attacker_color_list += ["red"]

            # Calculate attacker arrow end point
            attacker_end_x, attacker_end_y = calculate_arrow_end(attackers[i][0], attackers[i][1], attackers[i][2])
            attacker_x_list.append(attacker_end_x)
            attcker_y_list.append(attacker_end_y)
            attacker_symbol_list += ["line-ns"]
        
        frames.append(go.Frame(data=[go.Scatter(x=attacker_x_list, y=attcker_y_list, mode="markers+lines", line=dict(color="red"), name="Attacker trajectory", marker=dict(symbol=attacker_symbol_list, size=10, color=attacker_color_list), showlegend=False), 
                                     go.Scatter(x=defender_x_list, y=defender_y_list, mode="markers+lines", line=dict(color="blue"), name="Defender trajectory", marker=dict(symbol=defender_symbol_list, size=10, color=defender_color_list), showlegend=False)], 
                               traces=[0, 1]))

    # Static object - obstacles, goal region, grid
    fig = go.Figure(data=go.Scatter(x=[0.6, 0.8], y=[0.1, 0.1], 
                                    mode='lines', name='Target', 
                                    line=dict(color='purple')), 
                    layout=Layout(plot_bgcolor='rgba(0,0,0,0)', 
                                  updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 30, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}])])]))
    fig.update(frames=frames)
    # plot target
    fig.add_shape(type='rect', x0=0.6, y0=0.1, x1=0.8, y1=0.3, line=dict(color='purple', width=3.0), name="Target")
    # plot obstacles
    # fig.add_shape(type='rect', x0=-0.1, y0=0.3, x1=0.1, y1=0.6, line=dict(color='white', width=3.0), name="Obstacle")
    # fig.add_shape(type='rect', x0=-0.1, y0=-1.0, x1=0.1, y1=-0.3, line=dict(color='white', width=3.0))
    fig.add_trace(go.Scatter(x=[-0.1, 0.1], y=[-1.0, +1.0], mode='lines', name='Obstacle', line=dict(color='white')))

    # figure settings
    fig.update_layout(autosize=False, width=560, height=500, margin=dict(l=50, r=150, b=100, t=100, pad=0),
                      title={'text': "<b>Game recording, t={}s<b>".format(num_steps / 20), 'y': 0.85, 'x': 0.4, 'xanchor': 'center', 'yanchor': 'top', 'font_size': 20}, paper_bgcolor="White", xaxis_range=[-1.0, 1.0], yaxis_range=[-1.0, 1.0], font=dict(size=20))
    fig.update_xaxes(showline=True, linecolor='black', linewidth=2.0, griddash='dot', zeroline=False, gridcolor='Lightgrey', mirror=True, ticks='outside')  # showgrid=False
    fig.update_yaxes(showline=True, linecolor='black', linewidth=2.0, griddash='dot', zeroline=False, gridcolor='Lightgrey', mirror=True, ticks='outside')  # showgrid=False,
    fig.show()


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


def record_video(attackers_traj, defenders_traj, attackers_status, filename='animation.mp4', fps=10):
    # Ensure the save directory exists
    save_dir = os.path.join('test_results/', 'game')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir+'/')
    # Full path for the video file
    video_path = os.path.join(save_dir, filename)

    # VideoWriter setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (800, 800)
    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    # Set up the plot
    fig, ax = plt.subplots()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    
    # Plot the obstacles
    obstacles = [
        [(-0.1, -1.0), (0.1, -1.0), (0.1, -0.3), (-0.1, -0.3), (-0.1, -1.0)], 
        [(-0.1, 0.3), (0.1, 0.3), (0.1, 0.6), (-0.1, 0.6), (-0.1, 0.3)]
    ]
    for obstacle in obstacles:
        x, y = zip(*obstacle)
        ax.plot(x, y, "k-")

    # Plot the target
    target = [(0.6, 0.1), (0.8, 0.1), (0.8, 0.3), (0.6, 0.3), (0.6, 0.1)]
    x, y = zip(*target)
    ax.plot(x, y, "g-")

    for i, (attackers, defenders, status) in enumerate(zip(attackers_traj, defenders_traj, attackers_status)):
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])

        # Re-plot the obstacles and target on each frame
        for obstacle in obstacles:
            x, y = zip(*obstacle)
            ax.plot(x, y, color="black", linestyle="-")
        x, y = zip(*target)
        ax.plot(x, y, color="purple", linestyle="-")

        # Plot attackers
        free_attackers = attackers[status == 0]
        captured_attackers = attackers[status == -1]
        arrived_attackers = attackers[status == 1]

        if free_attackers.size > 0:
            ax.scatter(free_attackers[:, 0], free_attackers[:, 1], c='red', marker='^', label='Free Attackers')
        if captured_attackers.size > 0:
            ax.scatter(captured_attackers[:, 0], captured_attackers[:, 1], c='red', marker='p', label='Captured Attackers')
        if arrived_attackers.size > 0:
            ax.scatter(arrived_attackers[:, 0], arrived_attackers[:, 1], c='green', marker='^', label='Arrived Attackers')

        # Plot defenders
        if defenders.size > 0:
            ax.scatter(defenders[:, 0], defenders[:, 1], c='blue', marker='s', label='Defenders')

        # Convert Matplotlib plot to a frame for OpenCV
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Resize and convert to BGR for OpenCV
        img = cv2.resize(img, frame_size)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Write frame to video
        out.write(img)
    print(f"========== Animation saved at {video_path}. ==========")
    # Release the video writer
    out.release()


def plot_network_value(fixed_defender_position, model):
    # Define the fixed values for the last two dimensions
    fixed_values = fixed_defender_position[0].tolist()  # list like [0.0, 0.0]

    # Generate a grid of (x, y) values
    x_values = np.linspace(-1, 1, 100)
    y_values = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x_values, y_values)

    # Initialize an empty array to store the value function
    Z = np.zeros_like(X)

    # Iterate over the grid and calculate the value for each (x, y) pair
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            obs = [X[i, j], Y[i, j]] + fixed_values
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
            
            # Extract features using the vf_features_extractor
            features = model.policy.vf_features_extractor(obs_tensor)
            
            # Pass the features through the mlp_extractor value_net
            value_features = model.policy.mlp_extractor.value_net(features)
            
            # Pass the value features through the final value_net layer
            value = model.policy.value_net(value_features)
            
            # Store the value in the Z array
            Z[i, j] = value.item()

    # Plot the value function as a heatmap
    plt.figure(figsize=(8, 6))
    # plt.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis', aspect='auto')
    # plt.colorbar(label='Value')
    # contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    # plt.colorbar(contour, label='Value')
    contourf = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    # contour = plt.contour(X, Y, Z, levels=50, colors='black', linewidths=0.5)
    plt.colorbar(contourf, label='Value')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Value function heatmap with the fixed defender at {fixed_defender_position}')
    plt.show()


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


def plot_values(fixed_defender_position, model, value1vs1, grid1vs1, attacker, save_dir=None):
    # Plot the hj value and the trained value network value in one figure
    # Define the fixed values for the last two dimensions
    fixed_values = fixed_defender_position[0].tolist()  # list like [0.0, 0.0]

    # Generate a grid of (x, y) values
    x_values = np.linspace(-1, 1, 100)
    y_values = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x_values, y_values)

    # Initialize an empty array to store the value function
    Z = np.zeros_like(X)

    # Iterate over the grid and calculate the value for each (x, y) pair
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            obs = [X[i, j], Y[i, j]] + fixed_values
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
            
            # Extract features using the vf_features_extractor
            features = model.policy.vf_features_extractor(obs_tensor)
            
            # Pass the features through the mlp_extractor value_net
            value_features = model.policy.mlp_extractor.value_net(features)
            
            # Pass the value features through the final value_net layer
            value = model.policy.value_net(value_features)
            
            # Store the value in the Z array
            Z[i, j] = value.item()

    # Prepare the hj value
    a1x_slice, a1y_slice, d1x_slice, d1y_slice = po2slice1vs1(attacker[0], fixed_defender_position[0], value1vs1.shape[0])
    value_function1vs1 = value1vs1[:, :, d1x_slice, d1y_slice].squeeze()
    value_function1vs1 = np.swapaxes(value_function1vs1, 0, 1)
    # print(f"The shape of the value_function1vs1 is {value_function1vs1.shape}")
    dims_plot = [0, 1]
    dim1, dim2 = dims_plot[0], dims_plot[1]
    x_hj = np.linspace(-1, 1, value_function1vs1.shape[dim1])
    y_hj = np.linspace(-1, 1, value_function1vs1.shape[dim2])

    # if save_dir is not None:
    #     scores = np.load(f'{save_dir}/scores_matrix_{fixed_defender_position[0].tolist()}.npy')
    # Plot the value function as a heatmap
    plt.figure(figsize=(8, 8))
    # plt.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis', aspect='auto')
    # plt.colorbar(label='Value')
    # plt.colorbar(contour, label='Value')
    contourf = plt.contourf(X, Y, Z, levels=50, cmap='coolwarm')  # viridis, cividis, coolwarm
    # contour = plt.contour(X, Y, Z, levels=50, colors='black', linewidths=0.5)
    contour = plt.contour(x_hj, y_hj, value_function1vs1, levels=0, colors='#4B0082', linewidths=3.0, linestyles='dashed') # colors='magenta', colors='#4B0082'
    plt.scatter(fixed_defender_position[0][0], fixed_defender_position[0][1], color='magenta', marker='*', s=300, label='Fixed Defender')

    # plt.colorbar(contourf, label='Value')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title(f'Value function heatmap with the fixed defender at {fixed_defender_position[0]}')
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'ours_network_values{fixed_defender_position[0]}.png'))
    plt.show()


def plot_values_rarl(algo, fixed_defender_position, model, value1vs1, grid1vs1, attacker, save_dir=None):
    # Plot the hj value and the trained value network value in one figure
    # Define the fixed values for the last two dimensions
    fixed_values = fixed_defender_position[0].tolist()  # list like [0.0, 0.0]

    # Generate a grid of (x, y) values
    x_values = np.linspace(-1, 1, 100)
    y_values = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x_values, y_values)

    # Initialize an empty array to store the value function
    Z = np.zeros_like(X)

    # Iterate over the grid and calculate the value for each (x, y) pair
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            obs = [X[i, j], Y[i, j]] + fixed_values
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
            
            # Extract features using the vf_features_extractor
            value = model.agent.ac.critic(obs_tensor)
            # Store the value in the Z array
            Z[i, j] = value.item()

    # Prepare the hj value
    a1x_slice, a1y_slice, d1x_slice, d1y_slice = po2slice1vs1(attacker[0], fixed_defender_position[0], value1vs1.shape[0])
    value_function1vs1 = value1vs1[:, :, d1x_slice, d1y_slice].squeeze()
    value_function1vs1 = np.swapaxes(value_function1vs1, 0, 1)
    # print(f"The shape of the value_function1vs1 is {value_function1vs1.shape}")
    dims_plot = [0, 1]
    dim1, dim2 = dims_plot[0], dims_plot[1]
    x_hj = np.linspace(-1, 1, value_function1vs1.shape[dim1])
    y_hj = np.linspace(-1, 1, value_function1vs1.shape[dim2])
    # if save_dir is not None:
    #     scores = np.load(f'{save_dir}/scores_matrix_{fixed_defender_position[0].tolist()}.npy')
    # Plot the value function as a heatmap
    plt.figure(figsize=(8, 8))
    # plt.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis', aspect='auto')
    # plt.colorbar(label='Value')
    # plt.colorbar(contour, label='Value')
    contourf = plt.contourf(X, Y, Z, levels=50, cmap='coolwarm')  # viridis
    # contour = plt.contour(X, Y, Z, levels=50, colors='black', linewidths=0.5)
    contour = plt.contour(x_hj, y_hj, value_function1vs1, levels=0, colors='#4B0082', linewidths=3.0, linestyles='dashed') # colors='magenta', colors='#4B0082'
    plt.scatter(fixed_defender_position[0][0], fixed_defender_position[0][1], color='magenta', marker='*', s=300, label='Fixed Defender')

    # plt.colorbar(contourf, label='Value')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title(f'{algo} value network with the fixed defender at {fixed_defender_position[0]}')
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{algo}_network_values{fixed_defender_position[0]}.png'))
    plt.show()


def plot_values_dub(fixed_defender_position, model, value1vs1, grid1vs1, attacker, save_dir=None):
    # Plot the hj value and the trained value network value in one figure
    # Define the fixed values for the last two dimensions
    fixed_values = fixed_defender_position[0].tolist()  # list like [0.0, 0.0, 0.0]

    # Generate a grid of (x, y) values
    x_values = np.linspace(-1.1, 1.1, 100)
    y_values = np.linspace(-1.1, 1.1, 100)
    X, Y = np.meshgrid(x_values, y_values)

    # Initialize an empty array to store the value function
    Z = np.zeros_like(X)

    # Iterate over the grid and calculate the value for each (x, y) pair
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            obs = [X[i, j], Y[i, j], fixed_values[2]] + fixed_values
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
            
            # Extract features using the vf_features_extractor
            features = model.policy.vf_features_extractor(obs_tensor)
            
            # Pass the features through the mlp_extractor value_net
            value_features = model.policy.mlp_extractor.value_net(features)
            
            # Pass the value features through the final value_net layer
            value = model.policy.value_net(value_features)
            
            # Store the value in the Z array
            Z[i, j] = value.item()

    # Prepare the hj value
    _, _, a1o_slice, d1x_slice, d1y_slice, d1o_slice = grid1vs1.get_index(np.concatenate((attacker[0], fixed_defender_position[0])))
    value_function1vs1 = value1vs1[:, :, a1o_slice, d1x_slice, d1y_slice, d1o_slice].squeeze()
    value_function1vs1 = np.swapaxes(value_function1vs1, 0, 1)
    # print(f"The shape of the value_function1vs1 is {value_function1vs1.shape}")
    dims_plot = [0, 1]
    dim1, dim2 = dims_plot[0], dims_plot[1]
    x_hj = np.linspace(-1, 1, value_function1vs1.shape[dim1])
    y_hj = np.linspace(-1, 1, value_function1vs1.shape[dim2])

    # if save_dir is not None:
    #     scores = np.load(f'{save_dir}/scores_matrix_{fixed_defender_position[0].tolist()}.npy')
    # Plot the value function as a heatmap
    plt.figure(figsize=(8, 8))
    # plt.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis', aspect='auto')
    # plt.colorbar(label='Value')
    # plt.colorbar(contour, label='Value')
    contourf = plt.contourf(X, Y, Z, levels=50, cmap='coolwarm')  # viridis, cividis, coolwarm
    # contour = plt.contour(X, Y, Z, levels=50, colors='black', linewidths=0.5)
    contour = plt.contour(x_hj, y_hj, value_function1vs1, levels=0, colors='#4B0082', linewidths=3.0, linestyles='dashed') # colors='magenta', colors='#4B0082'
    plt.scatter(fixed_defender_position[0][0], fixed_defender_position[0][1], color='magenta', marker='*', s=100, label='Fixed Defender')

    # plt.colorbar(contourf, label='Value')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title(f'Value function heatmap with the fixed defender at {fixed_defender_position[0]}')
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'hj_network_values{fixed_defender_position[0]}.png'))
    plt.show()


# For dubin car reach-avoid games
def plot_value_1vs1_dub(attackers, defenders, plot_attacker, plot_defender, fix_agent, value1vs1_dub, grid1vs1_dub):
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
    # a1x_slice, a1y_slice, a1o_slice, d1x_slice, d1y_slice, d1o_slice = po2slice1vs1_dub(attackers[plot_attacker], defenders[plot_defender], value1vs1_dub.shape[0])
    a1x_slice, a1y_slice, a1o_slice, d1x_slice, d1y_slice, d1o_slice = grid1vs1_dub.get_index(np.concatenate((attackers[plot_attacker], defenders[plot_defender])))

    if fix_agent == 1:  # fix the defender
        value_function1vs1 = value1vs1_dub[:, :, a1o_slice, d1x_slice, d1y_slice, d1o_slice]
        dims_plot = [0, 1]
        dim1, dim2 = dims_plot[0], dims_plot[1]
    else:
        value_function1vs1 = value1vs1_dub[a1x_slice, a1y_slice, a1o_slice, :, :, d1o_slice]
        dims_plot = [3, 4]
        dim1, dim2 = dims_plot[0], dims_plot[1]

    complex_x = complex(0, grid1vs1_dub.pts_each_dim[dim1])
    complex_y = complex(0, grid1vs1_dub.pts_each_dim[dim2])
    mg_X, mg_Y = np.mgrid[grid1vs1_dub.min[dim1]:grid1vs1_dub.max[dim1]: complex_x, grid1vs1_dub.min[dim2]:grid1vs1_dub.max[dim2]: complex_y]
    x_attackers = attackers[:, 0]
    y_attackers = attackers[:, 1]
    x_defenders = defenders[:, 0]
    y_defenders = defenders[:, 1]

    heading_attackers = attackers[:, 2]
    heading_defenders = defenders[:, 2]

    arrow_length = 0.1  # Length of the arrow indicating direction

    def calculate_arrow_end(x, y, heading):
        end_x = x + arrow_length * np.cos(heading)
        end_y = y + arrow_length * np.sin(heading)
        return end_x, end_y

    arrow_attackers = [calculate_arrow_end(x, y, heading) for x, y, heading in zip(x_attackers, y_attackers, heading_attackers)]
    arrow_defenders = [calculate_arrow_end(x, y, heading) for x, y, heading in zip(x_defenders, y_defenders, heading_defenders)]


    print("Plotting beautiful 2D plots. Please wait\n")
    fig = go.Figure(data=go.Contour(
        x=mg_X.flatten(),
        y=mg_Y.flatten(),
        z=value_function1vs1.flatten(),
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
    # # plot obstacles
    # fig.add_shape(type='rect', x0=-0.1, y0=0.3, x1=0.1, y1=0.6, line=dict(color='black', width=3.0))
    # fig.add_shape(type='rect', x0=-0.1, y0=-1.0, x1=0.1, y1=-0.3, line=dict(color='black', width=3.0))
    # fig.add_trace(go.Scatter(x=[-0.1, 0.1], y=[0.3, 0.3], mode='lines', name='Obstacle', line=dict(color='black')))
    # # plot attackers
    fig.add_trace(go.Scatter(x=x_attackers, y=y_attackers, mode="markers", name='Attacker', marker=dict(symbol="triangle-up", size=10, color='red')))
    # plot defenders
    fig.add_trace(go.Scatter(x=x_defenders, y=y_defenders, mode="markers", name='Fixed Defender', marker=dict(symbol="square", size=10, color='green')))
    
    # plot attacker arrows
    for (x_start, y_start), (x_end, y_end) in zip(zip(x_attackers, y_attackers), arrow_attackers):
        fig.add_trace(go.Scatter(x=[x_start, x_end], y=[y_start, y_end], mode='lines', line=dict(color='red'), showlegend=False))
    # plot defender arrows
    for (x_start, y_start), (x_end, y_end) in zip(zip(x_defenders, y_defenders), arrow_defenders):
        fig.add_trace(go.Scatter(x=[x_start, x_end], y=[y_start, y_end], mode='lines', line=dict(color='green'), showlegend=False))


    # figure settings
    fig.update_layout(title={'text': f"<b>1 vs. 1 value function<b>", 'y':0.85, 'x':0.4, 'xanchor': 'center','yanchor': 'top', 'font_size': 20})
    fig.update_layout(autosize=False, width=580, height=500, margin=dict(l=50, r=50, b=100, t=100, pad=0), paper_bgcolor="White", xaxis_range=[-1.0, 1.0], yaxis_range=[-1.0, 1.0], font=dict(size=20)) # $\mathcal{R} \mathcal{A}_{\infty}^{21}$
    fig.update_xaxes(showline = True, linecolor = 'black', linewidth = 2.0, griddash = 'dot', zeroline=False, gridcolor = 'Lightgrey', mirror=True, ticks='outside') # showgrid=False
    fig.update_yaxes(showline = True, linecolor = 'black', linewidth = 2.0, griddash = 'dot', zeroline=False, gridcolor = 'Lightgrey', mirror=True, ticks='outside') # showgrid=False,
    fig.show()
    print("Please check the plot on your browser.")


def plot_value_1vs0_dub(attackers, value1vs0_dub, grid1vs0_dub, slice=0):
    """Plot the value function of the game.

    Args:
        attackers (np.ndarray): The attackers' states.
        plot_attacker (int): The attacker to plot the value function, other attackers are ignored.
        plot_defender (int): The defender to plot the value function, other defenders are ignored.
        fix_agent (int): The agent to fix (1 for defender, 0 for attacker).
        value1vs1 (np.ndarray): The value function of the 1 vs. 1 game.
        grid1vs1 (Grid): The grid of the 1 vs. 1 game.

    Returns:
        None
    """
    a1x_slice, a1y_slice, a1o_slice = grid1vs0_dub.get_index(attackers[0])

    value_function1vs1 = value1vs0_dub[:, :, a1o_slice, slice]
    dims_plot = [0, 1]
    dim1, dim2 = dims_plot[0], dims_plot[1]
    
    complex_x = complex(0, grid1vs0_dub.pts_each_dim[dim1])
    complex_y = complex(0, grid1vs0_dub.pts_each_dim[dim2])
    mg_X, mg_Y = np.mgrid[grid1vs0_dub.min[dim1]:grid1vs0_dub.max[dim1]: complex_x, grid1vs0_dub.min[dim2]:grid1vs0_dub.max[dim2]: complex_y]
    x_attackers = attackers[:, 0]
    y_attackers = attackers[:, 1]

    heading_attackers = attackers[:, 2]

    arrow_length = 0.1  # Length of the arrow indicating direction

    def calculate_arrow_end(x, y, heading):
        end_x = x + arrow_length * np.cos(heading)
        end_y = y + arrow_length * np.sin(heading)
        return end_x, end_y

    arrow_attackers = [calculate_arrow_end(x, y, heading) for x, y, heading in zip(x_attackers, y_attackers, heading_attackers)]
    
    fig = go.Figure(data=go.Contour(
        x=mg_X.flatten(),
        y=mg_Y.flatten(),
        z=value_function1vs1.flatten(),
        zmin=0.0,
        ncontours=1,
        contours_coloring = 'none', # former: lines 
        name= "Zero-Level", # zero level
        line_width = 1.5,
        line_color = 'magenta',
        zmax=0.0,
    ), layout=Layout(plot_bgcolor='rgba(0,0,0,0)')) #,paper_bgcolor='rgba(0,0,0,0)'
    # plot target
    fig.add_shape(type='rect', x0=0.6, y0=0.1, x1=0.8, y1=0.30, line=dict(color='purple', width=3.0), name="Target")
    fig.add_trace(go.Scatter(x=[0.6, 0.8], y=[0.1, 0.1], mode='lines', name='Target', line=dict(color='purple')))
    # plot obstacles
    fig.add_shape(type='rect', x0=-0.1, y0=0.3, x1=0.1, y1=0.6, line=dict(color='black', width=3.0))
    fig.add_shape(type='rect', x0=-0.1, y0=-1.0, x1=0.1, y1=-0.30, line=dict(color='black', width=3.0))
    fig.add_trace(go.Scatter(x=[-0.1, 0.1], y=[0.3, 0.3], mode='lines', name='Obstacle', line=dict(color='black')))
    # plot attackers
    fig.add_trace(go.Scatter(x=x_attackers, y=y_attackers, mode="markers", name='Attacker', marker=dict(symbol="triangle-up", size=10, color='red')))
    
    # plot attacker arrows
    for (x_start, y_start), (x_end, y_end) in zip(zip(x_attackers, y_attackers), arrow_attackers):
        fig.add_trace(go.Scatter(x=[x_start, x_end], y=[y_start, y_end], mode='lines', line=dict(color='red'), showlegend=False))
    
    # figure settings
    fig.update_layout(title={'text': f"<b>1 vs. 0 value function<b>", 'y':0.85, 'x':0.4, 'xanchor': 'center','yanchor': 'top', 'font_size': 20})
    fig.update_layout(autosize=False, width=580, height=500, margin=dict(l=50, r=205, b=100, t=100, pad=0), paper_bgcolor="White", xaxis_range=[-1.0, 1.0], yaxis_range=[-1.0, 1.0], font=dict(size=20)) # $\mathcal{R} \mathcal{A}_{\infty}^{21}$
    fig.update_xaxes(showline = True, linecolor = 'black', linewidth = 2.0, griddash = 'dot', zeroline=False, gridcolor = 'Lightgrey', mirror=True, ticks='outside') # showgrid=False
    fig.update_yaxes(showline = True, linecolor = 'black', linewidth = 2.0, griddash = 'dot', zeroline=False, gridcolor = 'Lightgrey', mirror=True, ticks='outside') # showgrid=False,
    fig.show()
    print("Please check the plot on your browser.")