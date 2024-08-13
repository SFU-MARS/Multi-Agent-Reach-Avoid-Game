#!/usr/bin/env python3

import math, time
from GridProcessing import Grid
import rospy
from geometry_msgs.msg import Twist, TransformStamped
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import GetModelState
from nav_msgs.msg import Odometry
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import rospy
import numpy as np

from utils_gazebo import hj_controller_1vs0, hj_contoller_defenders_dub_1vs1

#### Game Settings ####
grid_size = 100
grid_size_theta = 200
boundary = 2.0
angularv = 0.4
ctrl_freq = 20

start = time.time()
# New value function based on the 2x2 map
value1vs0_dub = np.load('/home/marslab/catkin_ws/src/turtlebot3_controller/scripts/values/DubinCar1vs0_grid100_medium_0.4angularv_20hz_1.0map.npy')

# value1vs0_dub = np.load('/home/marslab/catkin_ws/src/turtlebot3_controller/scripts/values/DubinCar1vs0_grid100_medium_0.4angularv_20hz.npy')
# value1vs1_dub = np.load('/home/marslab/catkin_ws/src/turtlebot3_controller/scripts/values/DubinCar1vs1_grid28_medium_0.4angularv_ctrl20hz_2.0map.npy')
# # grid1vs0_dub = Grid(np.array([-boundary, -boundary, -math.pi]), np.array([boundary, boundary, math.pi]), 3, np.array([grid_size, grid_size, grid_size_theta]), [2])
# grid1vs1_dub = Grid(np.array([-boundary, -boundary, -math.pi, -boundary, -boundary, -math.pi]), np.array([boundary, boundary, math.pi, boundary, boundary, math.pi]),
#              6, np.array([28, 28, 28, 28, 28, 28]), [2, 5])

# Original value function based on the 2x2 map
value1vs1_dub = np.load('/home/marslab/catkin_ws/src/turtlebot3_controller/scripts/values/DubinCar1vs1_grid28_medium_1.0angularv_ctrl20hz_1.0map.npy')
grid1vs1_dub: Grid = Grid(np.array([-1.0, -1.0, -math.pi, -1.0, -1.0, -math.pi]), 
                          np.array([1.0, 1.0, math.pi, 1.0, 1.0, math.pi]), 
                          6, np.array([28, 28, 28, 28, 28, 28]), [2,5])
# value1vs0_dub = np.load('/home/marslab/catkin_ws/src/turtlebot3_controller/scripts/values/DubinCar1vs0_grid100_medium_1.0angularv.npy')
grid1vs0_dub: Grid = Grid(np.array([-1.0, -1.0, -math.pi]), np.array([1.0, 1.0, math.pi]), 3, np.array([100, 100, 200]), [2])
end = time.time()
print(f"============= HJ value functions loaded Successfully! (Time: {end-start :.4f} seconds) =============")
# print(f"========== The shape of value1vs0_dub is {value1vs0_dub.shape}. ========== \n")
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
    angles = euler_from_quaternion([raw_data.orientation.x, raw_data.orientation.y, raw_data.orientation.z, raw_data.orientation.w])
    rospy.loginfo(f"The angles are {angles}. \n")
    heading = angles[2]
    # heading = check_heading(raw_data.orientation.z)

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

# vicon_data: TransformStamped = TransformStamped()

# def gazebo_data_callback(attacker_data: Odometry, defender_data: Odometry):   
#     # print(attacker_data.pose[1].position.x)
#     # print(defender_data.pose[2].position.x)
#     # rate = rospy.Rate(50)
#     rospy.loginfo("========== The 1vs1 game starts! ==========")
#     # Filter the raw data
#     filterd_attacker = filter(attacker_data.pose.pose)  # (attacker_state, defender_state)
#     filterd_defender = filter(defender_data.pose.pose)
#     rospy.loginfo(f"Attacker data: {filterd_attacker} \n")
#     rospy.loginfo(f"Defender data: {filterd_defender}. \n")

#     current_attacker_state = np.array(filterd_attacker)
#     current_defender_state = np.array(filterd_defender)

#     joint_1vs1 = (current_attacker_state[0], current_attacker_state[1], current_attacker_state[2],
#                   current_defender_state[0], current_defender_state[1], current_defender_state[2])
#     joint_1vs1_slice = grid1vs1_dub.get_index(joint_1vs1)
#     rospy.loginfo(f"The current value function is {value1vs1_dub[joint_1vs1_slice]}.\n")
#     # rospy.loginfo(f"The current value function for Defender is {value1vs1_dub_20hz[joint_1vs1_slice]}.\n")

    

#     # attacker control
#     # attacker_state_slice = grid1vs0_dub.get_index(current_attacker_state)
#     # value1vs0_dubs = value1vs0_dub[..., 0]
#     control_attacker = hj_controller_1vs0(value1vs0_dub, grid1vs0_dub, current_attacker_state)
#     rospy.loginfo(f"The control command of the attacker is {control_attacker}. \n")

#     # defender control
#     control_defender = hj_contoller_defenders_dub_1vs1(current_attacker_state, current_defender_state, value1vs1_dub, grid1vs1_dub)
#     # control_defender = hj_contoller_defenders_dub_1vs1(current_attacker_state, current_defender_state, value1vs1_dub_20hz, grid1vs1_dub)
#     # TODO: Hanyang debug whether it's the calculation that makes the Publisher slow
#     # control_defender = 1.0
#     rospy.loginfo(f"The control command of the defender is {control_defender}. \n")


#     attacker_cmd = Twist()
#     defender_cmd = Twist()

#     attacker_x = current_attacker_state[0]
#     attacker_y = current_attacker_state[1]
#     defender_x = current_defender_state[0]
#     defender_y = current_defender_state[1]

#     if attacker_x >= 0.6 and attacker_x <= 0.8 and attacker_y >= 0.1 and attacker_y <= 0.30:
#         rospy.loginfo("The attacker has reached Goal State")
#         rospy.signal_shutdown("The attacker has reached Goal State")    
#     elif np.linalg.norm(current_defender_state[:2] - current_attacker_state[:2]) <= 0.30:
#         rospy.loginfo("The attacker has been captured by the defender")
#         rospy.signal_shutdown("The attacker has been captured by the defender.")
#     else:
#         attacker_cmd.linear.x = 0.22 # 1.0
#         attacker_cmd.angular.z = control_attacker[0][0]
#         # attacker_cmd.linear.x = 0.0 # 1.0
#         # attacker_cmd.angular.z = 0.0
#         defender_cmd.linear.x = 0.22
#         defender_cmd.angular.z = control_defender[0][0]

#     attacker_pub.publish(attacker_cmd)
#     defender_pub.publish(defender_cmd)
    
#     # rate.sleep()

# def cleanUp():
#     cmd = Twist()
#     cmd.linear.x = 0
#     cmd.angular.z = 0
#     attacker_pub.publish(cmd)
#     defender_pub.publish(cmd)

def getControlls():
    rospy.loginfo("Get the initial position of the attacker and the defender.")
    # Run until shutdown
    while not rospy.is_shutdown():
        resp_attacker: Odometry = modelLocation("tb3_0", "world")
        resp_defender: Odometry = modelLocation("tb3_1", "world")

        filtered_attacker = filter(resp_attacker.pose)
        filtered_defender = filter(resp_defender.pose)

        rospy.loginfo(f"Attacker's position: {filtered_attacker}")
        rospy.loginfo(f"Defender's position: {filtered_defender}")


        current_attacker_state = np.array(filtered_attacker)
        current_defender_state = np.array(filtered_defender)

        joint_1vs1 = (current_attacker_state[0], current_attacker_state[1], current_attacker_state[2],
                    current_defender_state[0], current_defender_state[1], current_defender_state[2])
        rospy.loginfo(f"The joint_1vs1 is {joint_1vs1}. \n")
        joint_1vs1_slice = grid1vs1_dub.get_index(joint_1vs1)
        rospy.loginfo(f"The joint_1vs1_slice is {joint_1vs1_slice}. \n")
        rospy.loginfo(f"The current value function is {value1vs1_dub[joint_1vs1_slice]}.\n")
        # rospy.loginfo(f"The current value function for Defender is {value1vs1_dub_20hz[joint_1vs1_slice]}.\n")

        control_attacker = hj_controller_1vs0(value1vs0_dub, grid1vs0_dub, current_attacker_state)
        rospy.loginfo(f"The control command of the attacker is {control_attacker}. \n")
        
        control_defender = hj_contoller_defenders_dub_1vs1(current_attacker_state, current_defender_state, value1vs1_dub, grid1vs1_dub)
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
            attacker_cmd.linear.x = 0.22 # 1.0
            attacker_cmd.angular.z = control_attacker[0][0]
            # attacker_cmd.linear.x = 0.0 # 1.0
            # attacker_cmd.angular.z = 0.0
            defender_cmd.linear.x = 0.22
            defender_cmd.angular.z = control_defender[0][0]

        attacker_pub.publish(attacker_cmd)
        defender_pub.publish(defender_cmd)
        rospy.loginfo("------------------------")

        
def cleanUp():
    cmd = Twist()
    cmd.linear.x = 0
    cmd.angular.z = 0
    attacker_pub.publish(cmd)
    defender_pub.publish(cmd)

if __name__ == "__main__":
    rospy.init_node("hj_controller")

    rospy.loginfo("Initialize the hj controller.")

    rospy.on_shutdown(cleanUp)

    modelLocation = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState, )
    modelLocation.wait_for_service()

    # Initialize the Publisher 
    attacker_pub = rospy.Publisher("/tb3_0/cmd_vel", Twist, queue_size=20)
    defender_pub = rospy.Publisher("/tb3_1/cmd_vel", Twist, queue_size=20)
    
    getControlls()

    # gazebo_sub_attacker = Subscriber("/tb3_0/odom", Odometry, queue_size=None)  #TODO: Hanyang: This matters!!! 2024.8.12
    # gazebo_sub_defender = Subscriber("/tb3_1/odom", Odometry, queue_size=None)  #TODO: Hanyang: This matters!!! 2024.8.12

    # ts = TimeSynchronizer([gazebo_sub_attacker, gazebo_sub_defender], queue_size=2) #TODO: Hanyang: This matters!!! 2024.8.12
    # ts = TimeSynchronizer([gazebo_sub_defender, gazebo_sub_attacker], queue_size=2) #TODO: Hanyang: This matters!!! 2024.8.12

    # ts = ApproximateTimeSynchronizer([gazebo_sub_attacker, gazebo_sub_defender], queue_size=4, slop=0.1)
    # ts.registerCallback(gazebo_data_callback)
    rospy.spin()