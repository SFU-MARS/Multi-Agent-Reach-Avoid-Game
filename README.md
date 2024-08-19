# Multi-Agent-Reach-Avoid-Game
The repository for the multi-agent reach-avoid game using HJ reachability.


## Start-Up Sequence

1. Change the ip-address of `ROS_MASTER_URI` and `ROS_HOSTNAME` to match the master laptops ip (using `ifconfig`) in the .bashrc file using `nano ~/.bashrc` command and then run `source ~/.bashrc` to update the terminal with the changes made

2. Run the ros core in the terminal using the `roscore` command


### Running the Gazebo Simulation

#### 1vs0 Game
1. Run the 1vs0 Gazebo Simulation using the `roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch` command. This launches a empty world in gazebo with a single turltebot.
2. To change the position of the turtlebot within gazebo, you can either change directly within Gazebo, or for repeated tests, change within the launch file found here: `/home/marslab/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/launch/turtlebot3_empty_world.launch`
3. Once Gazebo is running, you can run the custom control scripts by running, `rosrun turtlebot3_controller {name of python script}`.

#### 1vs1 Game
1. Run the 1vs1 Gazebo Simulation using the `roslaunch turtlebot3_gazebo multi_turtlebot3.launch` command. This launches a empty world in gazebo with a two turtlebots turltebots, the attacker and defender.

2. To change the location of the turtlebot within gazebo, you can either change directly within Gazebo, or for repeated tests, change within the launch file found here: `/home/marslab/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/launch/multi_turtlebot3.launch`
3. Once Gazebo is running, you can run the custom control scripts by running, `rosrun turtlebot3_controller {name of python script}`.

### Running Real World Experiments
1. Start Vicon System, turn on Vicon PC and connect to the TP-LINK wifi on both the PC and the master computer
2. Change the ip-address of `ROS_MASTER_URI` and `ROS_HOSTNAME` to match the master laptops ip (using `ifconfig`) in the .bashrc file using `nano ~/.bashrc` command.
3. Once `roscore` is running, launch the bridge between the vicon system and the master computer by using the `roslaunch vicon_bridge vicon.launch`. This will start publishing data about the markers on the field.
4. Turn on Turtlebot3(s) and connect to them using `ssh ubuntu@{ipaddress_of_turtlebot}` which can be obtained by connecting the pi on the turtlebot to a monitor, logging in using the credentials (username: `ubuntu`, password: `turtlebot`) and running `ipconfig`. 
5. Once connected through ssh, run the bringup file for the Turtlebot3 using `roslaunch turtlebot3_bringup turtlebot3_robot.launch`

#### 1vs0 Game
1. Once the Turtlebot3 is up, bringup is running and Turtlebot3 is placed on the field. Simply run the specific script using, `rosrun turtlebot3_controller {name of python script}`. It is recommended to restart the vicon_bridge roslaunch just so that it is sending up-to-date information about the Turtlebots

#### 1vs1 Game
1. Once both Turtlebot3 is up, bringup is running via `ssh` on both bots, and they are placed on the field. Simply run the specific script using, `rosrun turtlebot3_controller {name of python script}`. It is recommended to restart the vicon_bridge roslaunch just so that it is sending up-to-date information about the Turtlebots


## Things to keep in mind
1. Both Gazebo and Vicon System returns the orientation of the Turtlebot(s) in Quaternion, while the value function have been generated using Euler angles. Currently the scripts created are converting the incoming orientation data to euler using, `euler_from_quaternion([raw_data.orientation.x, raw_data.orientation.y, raw_data.orientation.z, raw_data.orientation.w])` within the `filter` function.
2. The forward and turning velocity of the turtlebot can change whether the experiment succeeds and the values may change between the Gazebo simulation and the real world experiments.


