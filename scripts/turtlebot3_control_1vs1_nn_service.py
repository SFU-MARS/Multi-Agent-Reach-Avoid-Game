#!/usr/bin/env python3

import math
from GridProcessing import Grid
import rospy
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped
from vicon_bridge.srv import viconGrabPose
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
import numpy as np
from utils import *
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from stable_baselines3 import PPO
import torch
import torch.nn as nn

value1vs0_dub = np.load('/home/marslab/catkin_ws/src/turtlebot3_controller/scripts/values/1vs0Dubin_easier.npy')
grid1vs0_dub: Grid = Grid(np.array([-1.1, -1.1, -math.pi]), np.array([1.1, 1.1, math.pi]), 3, np.array([100, 100, 200]), [2])

value1vs1_dub = np.load('/home/marslab/catkin_ws/src/turtlebot3_controller/scripts/values/1vs1Dubin_easier.npy')
grid1vs1_dub: Grid = Grid(np.array([-1.1, -1.1, -math.pi, -1.1, -1.1, -math.pi]), np.array([1.1, 1.1, math.pi, 1.1, 1.1, math.pi]), 
                        6, np.array([28, 28, 28, 28, 28, 28]), [2, 5])

# load the trained model
trained_model = "/home/marslab/catkin_ws/src/turtlebot3_controller/scripts/dubin_HARLT.zip"
model = PPO.load(trained_model)
print("========== The model has been loaded. ==========")

rospy.loginfo("Value functions has been loaded.")


def check_heading(heading):
    while heading >= np.pi:
        heading -= 2*np.pi
    while heading < -np.pi:
        heading += 2*np.pi

    return heading


def filter(raw_data: PoseStamped):
    """ This opt_ufunction filters the raw data from the sensors, the filtering method has not been implemented yet.
    Args:
        raw_data (np.ndarray): the raw data from the sensors

    Returns:
        np.ndarray: the filtered data
    """

    # rotation = raw_data.transform.rotation

    # rospy.loginfo(raw_data)

    x = raw_data.pose.position.x
    y = raw_data.pose.position.y
    # heading = raw_data.transform.rotation.z
    angles = euler_from_quaternion([raw_data.pose.orientation.x, raw_data.pose.orientation.y, raw_data.pose.orientation.z, raw_data.pose.orientation.w])
    heading = check_heading(angles[2])
    # heading = angles[2]
    # heading = check_heading(raw_data.transform.rotation.z)

    filtered_data = [x, y, heading]

    return filtered_data


def vicon_data_callback():
    rospy.loginfo("Get the initial position of the attacker and the defender.")
    # rate = rospy.Rate(200)
    while not rospy.is_shutdown():
        attacker_data: PoseStamped = agentLocation("turtlebot3_2", "turtlebot3_2", 1)
        defender_data: PoseStamped = agentLocation("turtlebot3_3", "turtlebot3_3", 1)

        # Filter the raw data
        filterd_attacker = filter(attacker_data.pose)  # (attacker_state, defender_state)
        filterd_defender = filter(defender_data.pose)
        rospy.loginfo(f"Attacker data: {filterd_attacker} \n")
        rospy.loginfo(f"Defender data: {filterd_defender}. \n")

        current_attacker_state = np.array(filterd_attacker)
        current_defender_state = np.array(filterd_defender)

        joint_1vs1 = (current_attacker_state[0], current_attacker_state[1], current_attacker_state[2],
                    current_defender_state[0], current_defender_state[1], current_defender_state[2])
        joint_1vs1_slice = grid1vs1_dub.get_index(joint_1vs1)
        rospy.loginfo(f"The current value function for Defender is {value1vs1_dub[joint_1vs1_slice]}.\n")
        
        # attacker control
        # attacker_state_slice = grid1vs0_dub.get_index(current_attacker_state)
        # value1vs0_dubs = value1vs0_dub[..., 0]
        control_attacker = hj_controller_1vs0(value1vs0_dub, grid1vs0_dub, current_attacker_state)
        # rospy.loginfo(f"The control command of the attacker is {control_attacker}. \n")

        # defender control
        obs = np.concatenate((filterd_attacker, filterd_defender))
        control_defender, _ = model.predict(obs, deterministic=True)
        # control_defender = hj_contoller_defenders_dub_1vs1(current_attacker_state, current_defender_state, value1vs1_dub, grid1vs1_dub)
        rospy.loginfo(f"The control command of the defender is {control_defender}. \n")


        attacker_cmd = Twist()
        defender_cmd = Twist()

        attacker_x = current_attacker_state[0]
        attacker_y = current_attacker_state[1]
        defender_x = current_defender_state[0]
        defender_y = current_defender_state[1]

        if attacker_x >= 0.6 and attacker_x <= 0.8 and attacker_y >= 0.1 and attacker_y <= 0.30:
            rospy.loginfo("The attacker has reached Goal State")
            rospy.signal_shutdown("The attacker has reached Goal State")
        
        elif np.linalg.norm(current_defender_state[:2] - current_attacker_state[:2]) <= 0.30:
            rospy.loginfo("The attacker has been captured by the defender")
            rospy.signal_shutdown("The attacker has been captured by the defender.")
        else:
            attacker_cmd.linear.x = 0.22  # 1.0
            attacker_cmd.angular.z = control_attacker[0][0]
            defender_cmd.linear.x = 0.22
            defender_cmd.angular.z = control_defender[0][0]

        attacker_pub.publish(attacker_cmd)
        defender_pub.publish(defender_cmd)

        rospy.loginfo("-----------")

        # rate.sleep()


if __name__ == "__main__":
    rospy.init_node("hj_controller_1vs1")

    rospy.loginfo("Initialize the hj controller for 1vs1.")
    # rate = rospy.Rate(200)

    agentLocation = rospy.ServiceProxy('/vicon/grab_vicon_pose/', viconGrabPose)
    agentLocation.wait_for_service()

    # Initialize the Publisher 
    attacker_pub = rospy.Publisher("/turtlebot2/cmd_vel", Twist, queue_size=30)
    defender_pub = rospy.Publisher("/turtlebot3/cmd_vel", Twist, queue_size=30)

    vicon_data_callback()

    rospy.spin()

