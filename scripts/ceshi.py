#!/usr/bin/env python3

import rospy
import math
from GridProcessing import Grid
import rospy
from geometry_msgs.msg import Twist, TransformStamped
from message_filters import Subscriber, TimeSynchronizer
import numpy as np
from utils import *
from safe_control_gym.controllers.rarl import rarl

# value1vs1_dub = np.load('/home/marslab/catkin_ws/src/turtlebot3_controller/scripts/values/DubinCar1vs1_grid28_medium_1.0angularv.npy')
# grid1vs1_dub: Grid = Grid(np.array([-1.0, -1.0, -math.pi, -1.0, -1.0, -math.pi]), 
#                           np.array([1.0, 1.0, math.pi, 1.0, 1.0, math.pi]), 
#                           6, np.array([28, 28, 28, 28, 28, 28]), [2,5])

if __name__ == "__main__":
    print("Hello world~")
    # rospy.init_node("test_node")
    # rospy.loginfo("Test node has been started")

    # rate = rospy.Rate(10)

    # while not rospy.is_shutdown():
    #     rospy.loginfo("Test node is running")
    #     rate.sleep()

    # print(value1vs1_dub.shape)