#!/usr/bin/env python3

import math, time
from GridProcessing import Grid
import rospy
from geometry_msgs.msg import Twist, TransformStamped
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer

import rospy
import numpy as np

#### Game Settings ####
grid_size = 100
grid_size_theta = 200
boundary = 2.0
angularv = 0.4
ctrl_freq = 20

start = time.time()
# New value function based on the 4x4 map
value1vs0_dub = np.load('/home/marslab/catkin_ws/src/turtlebot3_controller/scripts/values/DubinCar1vs0_grid100_medium_0.4angularv_20hz.npy')
grid1vs0_dub = Grid(np.array([-boundary, -boundary, -math.pi]), np.array([boundary, boundary, math.pi]), 3, np.array([grid_size, grid_size, grid_size_theta]), [2])

# Original value function based on the 2x2 map
# value1vs0_dub = np.load('/home/marslab/catkin_ws/src/turtlebot3_controller/scripts/values/DubinCar1vs0_grid100_medium_1.0angularv.npy')
# grid1vs0_dub: Grid = Grid(np.array([-1.0, -1.0, -math.pi]), np.array([1.0, 1.0, math.pi]), 3, 
#                     np.array([100, 100, 200]), [2])
end = time.time()
print(f"============= HJ value functions loaded Successfully! (Time: {end-start :.4f} seconds) =============")
print(f"========== The shape of value1vs0_dub is {value1vs0_dub.shape}. ========== \n")
rospy.loginfo("Value function has been loaded.")


def check_heading(heading):
    while heading >= np.pi:
        heading -= 2*np.pi
    while heading < -np.pi:
        heading += 2*np.pi

    return heading


def filter(raw_data):
    """ This function filters the raw data from the sensors, the filtering method has not been implemented yet.
    Args:
        raw_data (np.ndarray): the raw data from the sensors

    Returns:
        np.ndarray: the filtered data
    """

    # rotation = raw_data.transform.rotation

    x = raw_data.position.x
    y = raw_data.position.y
    heading = raw_data.orientation.z
    # heading = check_heading(raw_data.transform.rotation.z)

    filtered_data = [x, y, heading]

    return filtered_data


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
    opt_u = 0.4 # the maximum angular rate of the TurtleBot3

    if spat_deriv[2] > 0:
        if uMode == "min":
            opt_u = -0.4
    else:
        if uMode == "max":
            opt_u = -0.4
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
    spat_deriv_vector = spa_deriv(grid1vs0.get_index(current_state), v, grid1vs0, [2])
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


# vicon_data: TransformStamped = TransformStamped()

def gazebo_data_callback(gazebo_d: Odometry):   
    rospy.loginfo("========== The 1vs0 game starts! ==========")
    # rospy.loginfo(vicon_d.pose[1].position)
    # rospy.loginfo(gazebo_d.pose.pose)
    # Initialize the Publisher 
    # Filter the raw data
    filtered_data = filter(gazebo_d.pose.pose)
    rospy.loginfo(f"filtered Gazebo data: {filtered_data}\n")  # The info here is not consistent with the robot state in Gazebo
    current_state = np.array(filtered_data)
    joint_slice = grid1vs0_dub.get_index(current_state)
    value1vs0_dubs = value1vs0_dub[..., 0]
    rospy.loginfo(f"The current value function is {value1vs0_dubs[joint_slice]}.")
    # Call the hj_controller_1vs0 function to get the control signal
    control_signal = hj_controller_1vs0(value1vs0_dub, grid1vs0_dub, filtered_data)
    rospy.loginfo(f"The control signal is {control_signal[0][0]}.")


    # rate = rospy.Rate(20)

    cmd = Twist()

    robot_x = filtered_data[0]
    robot_y = filtered_data[1]
    
    if robot_x >= 0.6 and robot_x <= 0.8 and robot_y >= 0.1 and robot_y <= 0.3:
        cmd.linear.x = 0
        cmd.angular.z = 0
        hj_pub.publish(cmd)
        rospy.signal_shutdown("Reached Goal State")
    else:
        cmd.linear.x = 0.22
        # cmd.linear.y = 0.707
        cmd.angular.z = control_signal[0][0]
        # cmd.linear.x = 0.707
        # cmd.linear.y = 0.707
        # cmd.angular.z = control_signal[0][0]

    # # rospy.loginfo(f"Publishing: {cmd.linear} and {cmd.angular}")
    hj_pub.publish(cmd)
    # rate.sleep()
    
def cleanUp():
    cmd = Twist()
    cmd.linear.x = 0
    cmd.angular.z = 0
    hj_pub.publish(cmd)

if __name__ == "__main__":
    rospy.init_node("hj_controller")

    rospy.loginfo("Initialize the hj controller.")

    rospy.on_shutdown(cleanUp)

    hj_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=100)
   
    gazebo_sub = Subscriber("/odom", Odometry, queue_size=1)
    ts = TimeSynchronizer([gazebo_sub], queue_size=1)
    # ts = ApproximateTimeSynchronizer([attacker_sub, defender_sub], queue_size=60)
    ts.registerCallback(gazebo_data_callback)
    rospy.spin()