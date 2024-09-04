import os
import time
from copy import deepcopy
from datetime import datetime

import casadi as cs
import numpy as np
import pybullet_data
import pybullet as p
import torch
from gymnasium import spaces
import xml.etree.ElementTree as etxml

from safe_control_gym.envs.benchmark_env import BenchmarkEnv
from safe_control_gym.envs.benchmark_env import Cost, Task
from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS
from safe_control_gym.envs.gym_pybullet_drones.base_distb_aviary import DroneModel, Physics
from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel
from safe_control_gym.math_and_models.transformations import csRotXYZ
from safe_control_gym.utils.utils import Boltzmann, quat2euler, distur_gener_quadrotor, transfer
from safe_control_gym.utils.configuration import ConfigFactoryTestAdversary
from safe_control_gym.utils.registration import make
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_distb import QuadrotorNullDistb, QuadrotorRARLDistb


class QuadrotorAdversary(BenchmarkEnv):
    '''6D quadrotor environment with trained adversary networks for testing quadrotor with trained adversary.

    Including symbolic model, constraints, randomization, adversarial disturbances,
    multiple cost functions, stabilization and trajectory tracking references.
    '''

    NAME = 'quadrotor_adversary'
    URDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
    AVAILABLE_CONSTRAINTS = deepcopy(GENERAL_CONSTRAINTS)

    DISTURBANCE_MODES = {  # Set at runtime by QUAD_TYPE
        'observation': {
            'dim': -1
        },
        'action': {
            'dim': -1
        },
        'dynamics': {
            'dim': -1
        }
    }

    INERTIAL_PROP_RAND_INFO = {
        'M': {  # Nominal: 0.027
            'distrib': 'uniform',
            'low': 0.022,
            'high': 0.032
        },
        'Ixx': {  # Nominal: 1.4e-5
            'distrib': 'uniform',
            'low': 1.3e-5,
            'high': 1.5e-5
        },
        'Iyy': {  # Nominal: 1.4e-5
            'distrib': 'uniform',
            'low': 1.3e-5,
            'high': 1.5e-5
        },
        'Izz': {  # Nominal: 2.17e-5
            'distrib': 'uniform',
            'low': 2.07e-5,
            'high': 2.27e-5
        }
    }

    INIT_STATE_RAND_INFO = {
        'init_x': {
            'distrib': 'uniform',
            'low': -0.5,
            'high': 0.5
        },
        'init_x_dot': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_y': {
            'distrib': 'uniform',
            'low': -0.5,
            'high': 0.5
        },
        'init_y_dot': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_z': {
            'distrib': 'uniform',
            'low': 0.1,
            'high': 1.5
        },
        'init_z_dot': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_phi': {
            'distrib': 'uniform',
            'low': -0.3,
            'high': 0.3
        },
        'init_theta': {
            'distrib': 'uniform',
            'low': -0.3,
            'high': 0.3
        },
        'init_psi': {
            'distrib': 'uniform',
            'low': -0.3,
            'high': 0.3
        },
        'init_p': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_theta_dot': {  # TODO: replace with q.
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_q': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_r': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        }
    }

    TASK_INFO = {
        'stabilization_goal': [0, 1],
        'stabilization_goal_tolerance': 0.05,
        'trajectory_type': 'circle',
        'num_cycles': 1,
        'trajectory_plane': 'zx',
        'trajectory_position_offset': [0.5, 0],
        'trajectory_scale': -0.5,
        'proj_point': [0, 0, 0.5],
        'proj_normal': [0, 1, 1],
    }

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 physics: Physics = Physics.PYB,
                 record=False,
                 gui=False,
                 verbose=False,
                 info_in_reset=True,
                 # Hanyang: derive the following parameters from the BenchmarkEnv
                 pyb_freq: int = 200,
                 ctrl_freq: int = 100,
                 episode_len_sec: int = 10,
                 init_state=None,
                 init_xyzs=np.array([[0, 0, 1]], dtype=np.float32),
                 init_rpys=np.array([[0, 0, 0]], dtype=np.float32),
                 inertial_prop=None,
                 # custom args
                 norm_act_scale=0.1,
                 obs_goal_horizon=0,
                 # Hanyang: initialize some important attributes disturbances parameters 
                 randomized_init: bool = True,
                 distb_type = 'adversary', 
                 distb_level: float=0.0,
                 seed=None,
                 disturbances=None,
                 adversary_disturbance='action',
                 adversary_disturbance_offset=0.0,
                 adversary_disturbance_scale=2.0,
                 **kwargs
                 ):
        '''Initialize a quadrotor with hj distb environment.

        Args:
            num_drones (int, optional): The desired number of drones in the aviary.
            record (bool, optional): Whether to save a video of the simulation in folder`files/videos/`.
            init_state (ndarray, optional): The initial state of the environment, (z, z_dot) or (x, x_dot, z, z_dot theta, theta_dot).
            init_xyzs (ndarray | None, optional, (NUM_DRONES, 3)): The shaped array containing the initial XYZ position of the drones.
            init_rpys (ndarray | None, optional, (NUM_DRONES, 3)): The shaped array containing the initial orientations of the drones (in radians).
            inertial_prop (ndarray, optional): The inertial properties of the environment (M, Ixx, Iyy, Izz).
            quad_type (QuadType, optional): The choice of motion type (1D along z, 2D in the x-z plane, or 3D).
            norm_act_scale (float): Scaling the [-1,1] action space around hover thrust when `normalized_action_space` is True.
            obs_goal_horizon (int): How many future goal states to append to obervation.
            rew_state_weight (list/ndarray): Quadratic weights for state in rl reward.
            rew_act_weight (list/ndarray): Quadratic weights for action in rl reward.
            rew_exponential (bool): If to exponentiate negative quadratic cost to positive, bounded [0,1] reward.
            done_on_out_of_bound (bool): If to termiante when state is out of bound.
            info_mse_metric_state_weight (list/ndarray): Quadratic weights for state in mse calculation for info dict.
            episode_len_sec (int, optional): Maximum episode duration in seconds.
            randomized_init (bool, optional): Whether to randomize the initial state.
            disturbance_type (str, optional): The type of disturbance to be applied to the drones [None, 'fixed', 'boltzmann', 'random', 'rarl', 'rarl-population'].
            distb_level (float, optional): The level of disturbance to be applied to the drones.
            seed (int, optional): Seed for the random number generator.
            adversary_disturbance (str, optional): If to use adversary/external disturbance.
        '''
        self.norm_act_scale = norm_act_scale
        self.obs_goal_horizon = obs_goal_horizon

        # Constants.
        self.GRAVITY_ACC = 9.8
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180
        # Parameters.
        self.DRONE_MODEL = DroneModel(drone_model)
        self.URDF_PATH = os.path.join(self.URDF_DIR, self.DRONE_MODEL.value + '.urdf')
        self.NUM_DRONES = num_drones
        self.PHYSICS = Physics(physics)
        self.RECORD = record
        # Load the drone properties from the .urdf file.
        self.MASS, \
            self.L, \
            self.THRUST2WEIGHT_RATIO, \
            self.J, \
            self.J_INV, \
            self.KF, \
            self.KM, \
            self.COLLISION_H, \
            self.COLLISION_R, \
            self.COLLISION_Z_OFFSET, \
            self.MAX_SPEED_KMH, \
            self.GND_EFF_COEFF, \
            self.PROP_RADIUS, \
            self.DRAG_COEFF, \
            self.DW_COEFF_1, \
            self.DW_COEFF_2, \
            self.DW_COEFF_3, \
            self.PWM2RPM_SCALE, \
            self.PWM2RPM_CONST, \
            self.MIN_PWM, \
            self.MAX_PWM = self._parse_urdf_parameters(self.URDF_PATH)
        self.GROUND_PLANE_Z = -0.05
        if verbose:
            print(
                '[INFO] QuadrotorAdversary.__init__() loaded parameters from the drone\'s .urdf: \
                \n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f}, \
                \n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f}, \
                \n[INFO] gnd_eff_coeff {:f}, prop_radius {:f}, \
                \n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f}, \
                \n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f} \
                \n[INFO] pwm2rpm_scale {:f}, pwm2rpm_const {:f}, min_pwm {:f}, max_pwm {:f}'
                .format(self.MASS, self.L, self.J[0, 0], self.J[1, 1], self.J[2, 2],
                        self.KF, self.KM, self.THRUST2WEIGHT_RATIO,
                        self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS,
                        self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1,
                        self.DW_COEFF_2, self.DW_COEFF_3, self.PWM2RPM_SCALE,
                        self.PWM2RPM_CONST, self.MIN_PWM, self.MAX_PWM))
        # Compute constants.
        self.GRAVITY = self.GRAVITY_ACC * self.MASS
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4 * self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO * self.GRAVITY) / (4 * self.KF))
        # self.MAX_THRUST = (4 * self.KF * self.MAX_RPM**2)
        self.MAX_THRUST = self.GRAVITY * self.THRUST2WEIGHT_RATIO / 4
        self.MAX_XY_TORQUE = (self.L * self.KF * self.MAX_RPM**2)
        self.MAX_Z_TORQUE = (2 * self.KM * self.MAX_RPM**2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt(
            (15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        # Hanyang: initialize the disturbance parameters and the initial state randomization parameters here.
        self.distb_type = distb_type
        self.distb_level = distb_level
        assert self.distb_type in ['fixed', 'boltzmann', 'random_hj', 'random', 'wind', 'adversary', None], f"[ERROR] The disturbance type '{self.distb_type}' is not supported now. \n"
        self.init_xy_lim = 0.25
        self.init_z_lim = 0.1
        self.init_rp_lim = np.pi/6
        self.init_y_lim = 2*np.pi
        self.init_vel_lim = 0.1
        self.init_rp_vel_lim = 200 * self.DEG2RAD
        self.init_y_vel_lim = 20 * self.DEG2RAD
       
        super().__init__(gui=gui, verbose=verbose, pyb_freq=pyb_freq, ctrl_freq=ctrl_freq, info_in_reset=info_in_reset,
                         episode_len_sec=episode_len_sec, init_state=init_state, randomized_init=randomized_init,
                         seed=seed, disturbances=disturbances, adversary_disturbance=adversary_disturbance, 
                         adversary_disturbance_offset=adversary_disturbance_offset, adversary_disturbance_scale=adversary_disturbance_scale,
                         **kwargs)

        # Hanyang: Create X_GOAL and U_GOAL references for the assigned task.
        self.U_GOAL = np.ones(self.action_dim) * self.MASS * self.GRAVITY_ACC / self.action_dim
        if self.TASK == Task.STABILIZATION:
            self.X_GOAL = np.hstack([np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) for _ in range(self.NUM_DRONES)])  # x = [x, y, z, r, p, y]
            self.TARGET_POS = np.array([0,0,1])
        elif self.TASK == Task.TRAJ_TRACKING:
            POS_REF, VEL_REF, _ = self._generate_trajectory(traj_type=self.TASK_INFO['trajectory_type'],
                                                    traj_length=self.EPISODE_LEN_SEC,
                                                    num_cycles=self.TASK_INFO['num_cycles'],
                                                    traj_plane=self.TASK_INFO['trajectory_plane'],
                                                    position_offset=self.TASK_INFO['trajectory_position_offset'],
                                                    scaling=self.TASK_INFO['trajectory_scale'],
                                                    sample_time=self.CTRL_TIMESTEP
                                                    )  # Each of the 3 returned values is of shape (Ctrl timesteps, 3)
      
        # Hanyang: Add some randomization parameters to initial conditions
        self.init_xy_lim = 0.25
        self.init_z_lim = 0.1
        self.init_rp_lim = np.pi/6
        self.init_y_lim = 2*np.pi
        self.init_vel_lim = 0.1
        self.init_rp_vel_lim = 200 * self.DEG2RAD
        self.init_y_vel_lim = 20 * self.DEG2RAD
        
        # Hanyang: Set the limits for termination (get_done)
        self.rp_limit = 75 * self.DEG2RAD  # rad
        self.rpy_dot_limit = 1000 * self.DEG2RAD  # rad/s
        self.z_lim = 0.1  # m
        
        # Hanyang: Set the penalties for rewards (get_reward)
        self.penalty_action =1e-4
        self.penalty_angle = 1e-2
        self.penalty_angle_rate = 1e-3
        self.penalty_terminal = 100
        
        # Set prior/symbolic info.
        self._setup_symbolic()

        # Connect to PyBullet.
        self.PYB_CLIENT = -1
        if gui:
            # With debug GUI.
            self.PYB_CLIENT = p.connect(p.GUI)  # p.connect(p.GUI, options='--opengl2')
            p.resetDebugVisualizerCamera(cameraDistance=3,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 0],
                                         physicsClientId=self.PYB_CLIENT)
            ret = p.getDebugVisualizerCamera(physicsClientId=self.PYB_CLIENT)
            if verbose:
                print('viewMatrix', ret[2])
                print('projectionMatrix', ret[3])
        else:
            # Without debug GUI.
            self.PYB_CLIENT = p.connect(p.DIRECT)
            # Uncomment the following line to use EGL Render Plugin #
            # Instead of TinyRender (CPU-based) in PYB's Direct mode
            # if platform == 'linux':
            #     p.setAdditionalSearchPath(pybullet_data.getDataPath())
            #     plugin = p.loadPlugin(egl.get_filename(), '_eglRendererPlugin')
            #     print('plugin=', plugin)
        self.RENDER_WIDTH = int(1920)
        self.RENDER_HEIGHT = int(1440)
        self.FRAME_PER_SEC = 24
        self.CAPTURE_FREQ = int(self.PYB_FREQ / self.FRAME_PER_SEC)
        self.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(
            distance=3,
            yaw=-30,
            pitch=-30,
            roll=0,
            cameraTargetPosition=[0, 0, 0],
            upAxisIndex=2,
            physicsClientId=self.PYB_CLIENT)
        self.CAM_PRO = p.computeProjectionMatrixFOV(fov=60.0,
                                                    aspect=self.RENDER_WIDTH / self.RENDER_HEIGHT,
                                                    nearVal=0.1,
                                                    farVal=1000.0)
        # Set default initial poses when loading drone's urdf model.
        # can be overriden later for specific tasks (as sub-classes) in reset()
        # Hanyang: add a if else statement to set the initial position of the drones
        if init_xyzs is None:
            self.INIT_XYZS = np.vstack([np.array([x * 4 * self.L for x in range(self.NUM_DRONES)]),
                                        np.array([y * 4 * self.L for y in range(self.NUM_DRONES)]),
                                        np.ones(self.NUM_DRONES) * (self.COLLISION_H / 2 - self.COLLISION_Z_OFFSET+.1)
                                        ]).transpose().reshape(self.NUM_DRONES, 3)
        elif np.array(init_xyzs).shape == (self.NUM_DRONES,3):
            self.INIT_XYZS = init_xyzs
        else:
            print("[ERROR] invalid initial_xyzs in QuadrotorAdversary.__init__(), try init_xyzs.reshape(NUM_DRONES,3)")
        if init_rpys is None:
            self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        elif np.array(init_rpys).shape == (self.NUM_DRONES,3):
            self.INIT_RPYS = init_rpys
        else:
            print("[ERROR] invalid initial_rpys in QuadrotorAdversary.__init__(), try init_rpys.reshape(NUM_DRONES,3)")
        # Hanyang: load the trained adversarial model 
        fac = ConfigFactoryTestAdversary()
        config = fac.merge()
        config.algo_config['training'] = False
        config.output_dir = 'test_results/quadrotor_adversary'
        total_steps = config.algo_config['max_env_steps']
        # print(f"The config is {config}")
        trained_model = 'training_results/quadrotor_null/rarl/seed_42/10000000steps/model_latest.pt'
        env_func = QuadrotorNullDistb
        self.rarl = make(config.algo,
                    env_func,
                    checkpoint_path=os.path.join(config.output_dir, 'model_latest.pt'),
                    output_dir=config.output_dir,
                    use_gpu=config.use_gpu,
                    seed=config.seed,  
                    **config.algo_config)
        self.rarl.load(trained_model)
        self.rarl.reset()
        self.rarl.agent.eval()
        self.rarl.adversary.eval()
        self.rarl.obs_normalizer.set_read_only()


    def close(self):
        '''Terminates the environment.'''
        if self.RECORD and self.GUI:
            p.stopStateLogging(self.VIDEO_ID, physicsClientId=self.PYB_CLIENT)
        if self.PYB_CLIENT >= 0:
            p.disconnect(physicsClientId=self.PYB_CLIENT)
        self.PYB_CLIENT = -1 

    
    def _reset_simulation(self):
        '''Housekeeping function.

        Allocation and zero-ing of the variables, PyBullet's parameters/objects, 
        and disturbances parameters in the `reset()` function.
        '''
        # Initialize/reset counters and zero-valued variables.
        self.RESET_TIME = time.time()
        self.first_render_call = True
        self.X_AX = -1 * np.ones(self.NUM_DRONES)
        self.Y_AX = -1 * np.ones(self.NUM_DRONES)
        self.Z_AX = -1 * np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1 * np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM = False
        self.last_input_switch = 0
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        # Hanyang: add self.last_action and self.current_action
        self.last_action = np.zeros((self.NUM_DRONES, 4)) # the last action executed just now
        # Initialize the drones kinematic information.
        # Hanyang: change the initial position of the drones to [0, 0, 1]
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.pos[:, 2] = 1
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))
            
        # Hanyang: set the disturbances parameters
        if self.distb_type == 'boltzmann':
            self.distb_level = Boltzmann(low=0.0, high=2.1, accuracy=0.1)
        elif self.distb_type == 'random_hj':
            self.distb_level = np.round(np.random.uniform(0.0, 2.1), 1)
        # Check the validity of the disturbance level
        if self.distb_type == None:
                assert self.distb_level == 0.0, "distb_level should be 0.0 when distb_type is None"
        elif self.distb_type == 'fixed':
            assert transfer(self.distb_level) in np.arange(0.0, 2.1, 0.1), "distb_level should be in np.arange(0.0, 2.1, 0.1)"
        
        # Set PyBullet's parameters.
        p.resetSimulation(physicsClientId=self.PYB_CLIENT)
        p.setGravity(0, 0, -self.GRAVITY_ACC, physicsClientId=self.PYB_CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.PYB_CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.PYB_CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.PYB_CLIENT)
        # Load ground plane, drone and obstacles models.
        # self.PLANE_ID = p.loadURDF('plane.urdf', [0, 0, self.GROUND_PLANE_Z], physicsClientId=self.PYB_CLIENT)
        self.PLANE_ID = p.loadURDF('plane.urdf', physicsClientId=self.PYB_CLIENT)
        
        self.DRONE_IDS = np.array([
            p.loadURDF(self.URDF_PATH,
                       self.INIT_XYZS[i, :],
                       p.getQuaternionFromEuler(self.INIT_RPYS[i, :]),
                       flags=p.URDF_USE_INERTIA_FROM_FILE,
                       physicsClientId=self.PYB_CLIENT)
            for i in range(self.NUM_DRONES)
        ])
        for i in range(self.NUM_DRONES):
            p.changeDynamics(self.DRONE_IDS[i], -1, linearDamping=0, angularDamping=0)
        # Update and store the drones kinematic information.
        # self._update_and_store_kinematic_information()  # Hanyang: comment out this line since it's called in the reset function
        # Start video recording.
        self._start_video_recording()
        # # Show frame of references of drones, will severly slow down the GUI.
        # for i in range(self.NUM_DRONES):
        # if gui:
        #     self._show_drone_local_axes(i)


    def reset(self, seed=None):
        '''(Re-)initializes the environment to start an episode.

        Mandatory to call at least once after __init__().

        Args:
            seed (int): An optional seed to reseed the environment.

        Returns:
            obs (ndarray): The initial state of the environment.
            info (dict): A dictionary with information about the dynamics and constraints symbolic models.
        '''

        super().before_reset(seed=seed)
        # PyBullet simulation reset.
        self._reset_simulation()

        # Hanyang: Add some randomization to initial conditions
        if self.RANDOMIZED_INIT:
            for id in range(self.NUM_DRONES):
                self.pos[id] += np.random.uniform(-self.init_xy_lim, self.init_xy_lim, 3)

                self.rpy[id] += np.random.uniform(-self.init_rp_lim, self.init_rp_lim, 3)
                self.rpy[id][2] = np.random.uniform(-self.init_y_lim, self.init_y_lim)
                self.quat[id] = p.getQuaternionFromEuler(self.rpy[id])

                self.vel[id] += np.random.uniform(-self.init_vel_lim, self.init_vel_lim, 3)

                self.ang_v[id] += np.random.uniform(-self.init_rp_vel_lim, self.init_rp_vel_lim, 3)
                self.ang_v[id][2] = np.random.uniform(-self.init_y_vel_lim, self.init_y_vel_lim)
            # Hanyang: Connect to PyBullet ###################################
            for id in range(self.NUM_DRONES):
                p.resetBasePositionAndOrientation(self.DRONE_IDS[id], posObj=self.pos[id], ornObj=self.quat[id], physicsClientId=self.PYB_CLIENT)
                R = np.array(p.getMatrixFromQuaternion(self.quat[id])).reshape(3, 3)
                p.resetBaseVelocity(self.DRONE_IDS[id], linearVelocity=self.vel[id], angularVelocity=R.T@self.ang_v[id], physicsClientId=self.PYB_CLIENT)
        
        # Update BaseAviary internal variables before calling self._get_observation().
        self._update_and_store_kinematic_information()
        obs, info = self._get_observation(), self._get_reset_info()
        obs, info = super().after_reset(obs, info)

        # Return either an observation and dictionary or just the observation.
        if self.INFO_IN_RESET:
            return obs, info
        else:
            return obs


    def step(self, action):
        '''Advances the environment by one control step.

        Pass the commanded RPMs and the adversarial force to the superclass .step().
        The PyBullet simulation is stepped PYB_FREQ/CTRL_FREQ times in BaseAviary.

        Args:
            action (ndarray): The action returned by the controller, in the shape of (4,)

        Returns:
            obs (ndarray): The state of the environment after the step.
            reward (float): The scalar reward/cost of the step.
            done (bool): Whether the conditions for the end of an episode are met in the step.
            info (dict): A dictionary with information about the constraints evaluations and violations.
        '''

        # Get the preprocessed pwm for each motor
        pwm = super().before_step(action)
        #Hanyang: generate adversary disturbance with the trained NN here deepcopy(self.state)
        disturbance_force = None
        # Advance the simulation.
        self._advance_simulation(pwm, disturbance_force)
        # Standard Gym return.
        # Hanyang: revise the following code to get the obs, rew, done, info
        obs = self._get_observation()
        rew = self._get_reward(action)
        done = self._get_done()
        info = self._get_info()
        obs, rew, done, info = super().after_step(obs, rew, done, info)
        # Hanyang: log the action generated by the policy network
        self.last_action = action.reshape((self.NUM_DRONES, 4)).copy()
        return obs, rew, done, info


    def _advance_simulation(self, clipped_action, disturbance_force):
        '''Advances the environment by one simulation step.

        Args:
            clipped_action (ndarray): The input action for one or more drone,
                                         as PWMs by the specific implementation of
                                         `_preprocess_control()` in each subclass.
            disturbance_force (ndarray, optional): Disturbance force, applied to all drones.
        '''
        clipped_action = np.reshape(clipped_action, (self.NUM_DRONES, 4))
        # Repeat for as many as the aggregate physics steps.
        for _ in range(self.PYB_STEPS_PER_CTRL):
            # Update and store the drones kinematic info for certain
            # Between aggregate steps for certain types of update.
            if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [
                    Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG,
                    Physics.PYB_DW, Physics.PYB_GND_DRAG_DW
            ]:
                self._update_and_store_kinematic_information()
            # Step the simulation using the desired physics update.
            for i in range(self.NUM_DRONES):
                # Hanayng: calculate the HJ disturbances or randomized disturbances
                if self.distb_type == 'random':
                    # # Original random ranges
                    # low = np.array([-5.3e-3, -5.3e-3, -1.43e-4])
                    # high = np.array([5.3e-3, 5.3e-3, 1.43e-4])
                    # test random ranges: 1.2 times the original ranges
                    low = np.array([-5.3e-3*1.2, -5.3e-3*1.2, -1.43e-4*1.2])
                    high = np.array([5.3e-3*1.2, 5.3e-3*1.2, 1.43e-4*1.2])
                    # Generate a random sample
                    hj_distbs = np.random.uniform(low, high)
                elif self.distb_type == 'wind':  # contant wind disturbances
                    # hj_distbs = np.array([0.0, 0.004, 0.0])
                    # low = np.array([-5.3e-3, -5.3e-3, -1.43e-4])
                    # high = np.array([5.3e-3, 5.3e-3, 1.43e-4])
                    # Generate a random sample
                    # hj_distbs = np.random.uniform(low, high)
                    # hj_distbs = np.array([0.0, hj_distbs[1], 0.0])
                    # hj_distbs = np.array([-0.00424, -0.00424, 0.0])
                    # hj_distbs = (0.00424, 0.0, 0.0)
                    hj_distbs = (0.0, 0.00424, 0.0)
                    # hj_distbs = (0.00424, 0.00424, 0.0)
                    print(f"[INFO] The disturbance in the wind distb is {hj_distbs}. \n")
                elif self.distb_type == 'adversary':  # adversary disturbances
                    hj_distbs = (0.0, 0.0, 0.0)
                else: # fixed-hj, null, random_hj or boltzmann disturbances
                    current_angles = quat2euler(self._get_drone_state_vector(i)[3:7])  # convert quaternion to eulers
                    current_angle_rates = self._get_drone_state_vector(i)[13:16]
                    current_state = np.concatenate((current_angles, current_angle_rates), axis=0)
                    _, hj_distbs = distur_gener_quadrotor(current_state, self.distb_level)
                    # print(f"[INFO] The type-{self.distb_type} with {self.distb_level}-level is {hj_distbs}. \n")
                
                if self.PHYSICS == Physics.PYB:
                    # self._physics(clipped_action[i, :], i)
                    # Hanyang: add the disturbances to the physics
                    self._physics_pwm(clipped_action[i, :], hj_distbs, i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._ground_effect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._ground_effect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)
                # # Apply disturbance
                # if disturbance_force is not None:
                #     pos = self._get_drone_state_vector(i)[:3]
                #     p.applyExternalForce(
                #         self.DRONE_IDS[i],
                #         linkIndex=4,  # Link attached to the quadrotor's center of mass.
                #         forceObj=disturbance_force,
                #         posObj=pos,
                #         flags=p.WORLD_FRAME,
                #         physicsClientId=self.PYB_CLIENT)
            # PyBullet computes the new state, unless Physics.DYN.
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.PYB_CLIENT)
            # Save the last applied action (e.g. to compute drag).
            self.last_clipped_action = clipped_action
        # Update and store the drones kinematic information.
        self._update_and_store_kinematic_information()


    def _physics_pwm(self, pwm, hj_distbs, nth_drone):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        pwm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        disturbances : ndarray
            (3)-shaped array of floats containin 
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        thrust_normed = pwm / 60000  # cast into [0, 1]
        # Hanyang: the former one used is motor model, in agents.py, disturbance-CrazyFlie-simulation
        forces = self.MAX_THRUST * np.clip(thrust_normed, 0, 1)  # self.angvel2thrust(n)
        torques = 5.96e-3 * forces + 1.56e-5  # # Parameters from Julian Förster
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])

        # if self.DRONE_MODEL == DroneModel.RACE:
        #     torques = -torques
        # Hanyang: debug
        # print(f"The forces are {forces}")
        # print(f"The z_torques are {z_torque}")
        for i in range(4):
            p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.PYB_CLIENT
                                 )
        p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                              4,
                              torqueObj=[0, 0, z_torque],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.PYB_CLIENT
                              )
        # # Hanyang: Apply disturbances (torques) 
        # p.applyExternalTorque(self.DRONE_IDS[nth_drone],
        #                       4,
        #                       torqueObj=[hj_distbs[0], 0, 0],
        #                       flags=p.LINK_FRAME,
        #                       physicsClientId=self.PYB_CLIENT
        #                       )
        # p.applyExternalTorque(self.DRONE_IDS[nth_drone],
        #                       4,
        #                       torqueObj=[0, hj_distbs[1], 0],
        #                       flags=p.LINK_FRAME,
        #                       physicsClientId=self.PYB_CLIENT
        #                       )


    def render(self, mode='human'):
        '''Retrieves a frame from PyBullet rendering.

        Args:
            mode (str): Unused.

        Returns:
            frame (ndarray): A multidimensional array with the RGB frame captured by PyBullet's camera.
        '''

        [w, h, rgb, _, _] = p.getCameraImage(width=self.RENDER_WIDTH,
                                             height=self.RENDER_HEIGHT,
                                             shadow=1,
                                             viewMatrix=self.CAM_VIEW,
                                             projectionMatrix=self.CAM_PRO,
                                             renderer=p.ER_TINY_RENDERER,
                                             flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                             physicsClientId=self.PYB_CLIENT)

        # Hanyang: resize the frame
        rgb_array = np.array(rgb)
        rgb_array = rgb_array[:, :, :3]
        
        return rgb_array


    def _setup_symbolic(self, prior_prop={}, **kwargs):
        #TODO: Hanyang: not implemented 12D dynamics yet
        '''Creates symbolic (CasADi) models for dynamics, observation, and cost.

        Args:
            prior_prop (dict): specify the prior inertial prop to use in the symbolic model.
        '''
        m = prior_prop.get('M', self.MASS)
        Iyy = prior_prop.get('Iyy', self.J[1, 1])
        g, length = self.GRAVITY_ACC, self.L
        dt = self.CTRL_TIMESTEP
        # Define states.
        z = cs.MX.sym('z')
        z_dot = cs.MX.sym('z_dot')
        u_eq = m * g
        # if self.QUAD_TYPE == QuadType.ONE_D:
        #     nx, nu = 2, 1
        #     # Define states.
        #     X = cs.vertcat(z, z_dot)
        #     # Define input thrust.
        #     T = cs.MX.sym('T')
        #     U = cs.vertcat(T)
        #     # Define dynamics equations.
        #     X_dot = cs.vertcat(z_dot, T / m - g)
        #     # Define observation equation.
        #     Y = cs.vertcat(z, z_dot)
        # elif self.QUAD_TYPE == QuadType.TWO_D:
        #     nx, nu = 6, 2
        #     # Define states.
        #     x = cs.MX.sym('x')
        #     x_dot = cs.MX.sym('x_dot')
        #     theta = cs.MX.sym('theta')
        #     theta_dot = cs.MX.sym('theta_dot')
        #     X = cs.vertcat(x, x_dot, z, z_dot, theta, theta_dot)
        #     # Define input thrusts.
        #     T1 = cs.MX.sym('T1')
        #     T2 = cs.MX.sym('T2')
        #     U = cs.vertcat(T1, T2)
        #     # Define dynamics equations.
        #     X_dot = cs.vertcat(x_dot,
        #                        cs.sin(theta) * (T1 + T2) / m, z_dot,
        #                        cs.cos(theta) * (T1 + T2) / m - g, theta_dot,
        #                        length * (T2 - T1) / Iyy / np.sqrt(2))
        #     # Define observation.
        #     Y = cs.vertcat(x, x_dot, z, z_dot, theta, theta_dot)
        # elif self.QUAD_TYPE == QuadType.THREE_D:
        nx, nu = 12, 4
        Ixx = prior_prop.get('Ixx', self.J[0, 0])
        Izz = prior_prop.get('Izz', self.J[2, 2])
        J = cs.blockcat([[Ixx, 0.0, 0.0],
                            [0.0, Iyy, 0.0],
                            [0.0, 0.0, Izz]])
        Jinv = cs.blockcat([[1.0 / Ixx, 0.0, 0.0],
                            [0.0, 1.0 / Iyy, 0.0],
                            [0.0, 0.0, 1.0 / Izz]])
        gamma = self.KM / self.KF
        x = cs.MX.sym('x')
        y = cs.MX.sym('y')
        phi = cs.MX.sym('phi')  # Roll
        theta = cs.MX.sym('theta')  # Pitch
        psi = cs.MX.sym('psi')  # Yaw
        x_dot = cs.MX.sym('x_dot')
        y_dot = cs.MX.sym('y_dot')
        p_body = cs.MX.sym('p')  # Body frame roll rate
        q_body = cs.MX.sym('q')  # body frame pith rate
        r_body = cs.MX.sym('r')  # body frame yaw rate
        # PyBullet Euler angles use the SDFormat for rotation matrices.
        Rob = csRotXYZ(phi, theta, psi)  # rotation matrix transforming a vector in the body frame to the world frame.

        # Define state variables.
        X = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body)

        # Define inputs.
        f1 = cs.MX.sym('f1')
        f2 = cs.MX.sym('f2')
        f3 = cs.MX.sym('f3')
        f4 = cs.MX.sym('f4')
        U = cs.vertcat(f1, f2, f3, f4)

        # From Ch. 2 of Luis, Carlos, and Jérôme Le Ny. 'Design of a trajectory tracking controller for a
        # nanoquadcopter.' arXiv preprint arXiv:1608.05786 (2016).

        # Defining the dynamics function.
        # We are using the velocity of the base wrt to the world frame expressed in the world frame.
        # Note that the reference expresses this in the body frame.
        oVdot_cg_o = Rob @ cs.vertcat(0, 0, f1 + f2 + f3 + f4) / m - cs.vertcat(0, 0, g)
        pos_ddot = oVdot_cg_o
        pos_dot = cs.vertcat(x_dot, y_dot, z_dot)
        Mb = cs.vertcat(length / cs.sqrt(2.0) * (f1 + f2 - f3 - f4),
                        length / cs.sqrt(2.0) * (-f1 + f2 + f3 - f4),
                        gamma * (-f1 + f2 - f3 + f4))
        rate_dot = Jinv @ (Mb - (cs.skew(cs.vertcat(p_body, q_body, r_body)) @ J @ cs.vertcat(p_body, q_body, r_body)))
        ang_dot = cs.blockcat([[1, cs.sin(phi) * cs.tan(theta), cs.cos(phi) * cs.tan(theta)],
                                [0, cs.cos(phi), -cs.sin(phi)],
                                [0, cs.sin(phi) / cs.cos(theta), cs.cos(phi) / cs.cos(theta)]]) @ cs.vertcat(p_body, q_body, r_body)
        X_dot = cs.vertcat(pos_dot[0], pos_ddot[0], pos_dot[1], pos_ddot[1], pos_dot[2], pos_ddot[2], ang_dot, rate_dot)

        Y = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body)
        # Set the equilibrium values for linearizations.
        X_EQ = np.zeros(self.state_dim)
        U_EQ = np.ones(self.action_dim) * u_eq / self.action_dim
        # Define cost (quadratic form).
        Q = cs.MX.sym('Q', nx, nx)
        R = cs.MX.sym('R', nu, nu)
        Xr = cs.MX.sym('Xr', nx, 1)
        Ur = cs.MX.sym('Ur', nu, 1)
        cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T @ R @ (U - Ur)
        # Define dynamics and cost dictionaries.
        dynamics = {'dyn_eqn': X_dot, 'obs_eqn': Y, 'vars': {'X': X, 'U': U}}
        cost = {
            'cost_func': cost_func,
            'vars': {
                'X': X,
                'U': U,
                'Xr': Xr,
                'Ur': Ur,
                'Q': Q,
                'R': R
            }
        }
        # Additional params to cache
        params = {
            # prior inertial properties
            'quad_mass': m,
            'quad_Iyy': Iyy,
            'quad_Ixx': Ixx if 'Ixx' in locals() else None,
            'quad_Izz': Izz if 'Izz' in locals() else None,
            # equilibrium point for linearization
            'X_EQ': X_EQ,
            'U_EQ': U_EQ,
        }
        # Setup symbolic model.
        self.symbolic = SymbolicModel(dynamics=dynamics, cost=cost, dt=dt, params=params)


    def _set_action_space(self):
        '''Sets the action space of the environment.'''
        action_dim = 4
        act_lower_bound = np.array(-1*np.ones(action_dim))
        act_upper_bound = np.array(+1*np.ones(action_dim))
        # Hanyang: define the action space for 6D quadrotor
        self.action_space = spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)


    def _set_observation_space(self):
        #### OBS SPACE OF SIZE 17
        #### Observation vector ### pos, quat, vel, ang_v, last_clipped_action
        lo = -np.inf
        hi = np.inf

        obs_lower_bound = np.array([lo,lo,0, lo,lo,lo,lo, lo,lo,lo, lo,lo,lo] )
        obs_upper_bound = np.array([hi,hi,hi, hi,hi,hi,hi, hi,hi,hi, hi,hi,hi] )
        #### Add action buffer to observation space ################
        act_lo = -1
        act_hi = +1
        obs_lower_bound = np.hstack([obs_lower_bound, np.array([act_lo,act_lo,act_lo,act_lo])])
        obs_upper_bound = np.hstack([obs_upper_bound, np.array([act_hi,act_hi,act_hi,act_hi])])
        # Define the state space for the dynamics.
        self.state_space = spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)


    def _setup_disturbances(self):
        '''Sets up the disturbances.'''
        # Custom disturbance info.
        self.DISTURBANCE_MODES['observation']['dim'] = self.obs_dim
        self.DISTURBANCE_MODES['action']['dim'] = self.action_dim
        # self.DISTURBANCE_MODES['dynamics']['dim'] = int(self.QUAD_TYPE)
        self.DISTURBANCE_MODES['dynamics']['dim'] = 4  # Hanyang: revise this from 6 to 4
        # Hanyang: add adversary observation and action space for rarl
        self.adversary_observation_space = self.observation_space
        self.adversary_action_space = self.action_space
        
        super()._setup_disturbances()


    def _preprocess_control(self, action):
        '''Converts the action passed to .step() into motors' PWMs (ndarray of shape (4,)).

        Args:
            action (ndarray): The raw action input, of shape (4,).

        Returns:
            action (ndarray): The motors PWMs to apply to the quadrotor.
        '''
        action = self.denormalize_action(action)  # Hanayng: this line doesn't work actually
        self.current_physical_action = action
        # # Apply disturbances.
        # if 'action' in self.disturbances:  # Hanyang: self.disturbances = {}
        #     action = self.disturbances['action'].apply(action, self)
        # if self.adversary_disturbance == 'action':  # Hanyang: default is None in benchmark.py
        #     action = action + self.adv_action
        # Hanyang: apply the adversarial action generated by the trained adversary
        # Hanyang: calculate and apply the adversary action
        assert self.distb_type == 'adversary', print("The distb_type of this env should be adversary. \n")
        precess_current_obs = self.rarl.obs_normalizer(deepcopy(self.state))
        with torch.no_grad():
            action_adv = self.rarl.adversary.ac.act(torch.from_numpy(precess_current_obs).float())
        clipped_adv_action = np.clip(action_adv, self.adversary_action_space.low, self.adversary_action_space.high)
        self.adv_action = clipped_adv_action * self.adversary_disturbance_scale + self.adversary_disturbance_offset
        # print(f"[INFO] The adversary action is {self.adv_action}.")
        # print(f"[INFO] The original action is {action}. \n")
        action = action + self.adv_action
        
        self.current_noisy_physical_action = action
        
        pwm = 30000 + np.clip(action, -1, +1) * 30000

        return pwm


    def normalize_action(self, action):
        '''Converts a physical action into an normalized action if necessary.

        Args:
            action (ndarray): The action to be converted.

        Returns:
            normalized_action (ndarray): The action in the correct action space.
        '''
        if self.NORMALIZED_RL_ACTION_SPACE:  # Hanyang: NORMALIZED_RL_ACTION_SPACE is set to False default
            action = (action / self.hover_thrust - 1) / self.norm_act_scale

        return action


    def denormalize_action(self, action):
        '''Converts a normalized action into a physical action if necessary.

        Args:
            action (ndarray): The action to be converted.

        Returns:
            physical_action (ndarray): The physical action.
        '''
        if self.NORMALIZED_RL_ACTION_SPACE:
            action = (1 + self.norm_act_scale * action) * self.hover_thrust

        return action


    def _get_observation(self):
        '''Returns the current observation (state) of the environment.

        Returns:
            obs (ndarray): The state of the quadrotor, of size 17 depending on QUAD_TYPE.
        '''
        # Hanyang: it seems now it only supports 1 drone due to the observation disturbance
        full_state = self._get_drone_state_vector(0)  
        self.state = np.hstack([full_state[0:3], full_state[3:7], full_state[10:13], full_state[13:16], full_state[16:20]]).reshape(17,)
        # Apply observation disturbance.
        full_state = deepcopy(self.state)
        # if 'observation' in self.disturbances:
        #     full_state = self.disturbances['observation'].apply(full_state, self)

        return full_state


    def _get_reward(self, action):
        '''Computes the current step's reward value.
        
        Args:
            action (ndarray): The action generated by the controller.

        Returns:
            reward (float): The evaluated reward/cost.
        '''
        # RL cost.
        assert self.COST == Cost.RL_REWARD, print("Now we only support RL_REWARD. \n")
        state = self.state  # (17,)
        normed_clipped_a = 0.5 * (np.clip(action, -1, 1) + 1)
        penalty_action = self.penalty_action * np.linalg.norm(normed_clipped_a)  # Hanyang: self.penalty_action = 1e-4
        # penalty_rpy = self.penalty_angle * np.linalg.norm(state[7:10])
        penalty_rpy_dot = self.penalty_angle_rate * np.linalg.norm(state[13:16])  # Hanyang: self.penalty_angle_rate = 1e-3
        penalty_terminal = self.penalty_terminal if self._get_done() else 0.  # Hanyang: self.penalty_terminal = 100

        # penalties = np.sum([penalty_action, penalty_rpy, penalty_rpy_dot, penalty_terminal])
        penalties = np.sum([penalty_action, penalty_rpy_dot, penalty_terminal])

        dist = np.linalg.norm(state[0:3] - self.TARGET_POS)
        reward = -dist - penalties
    
        return reward


    def _get_done(self):
        '''Computes the conditions for termination of an episode, do not consider hte max control steps here

        Returns:
            done (bool): Whether an episode is over.
        '''
        state = self._get_drone_state_vector(0) # (20,)
        # state = self.state  # (17,)
        rp = state[7:9]  # rad
        rp_limit = rp[np.abs(rp) > self.rp_limit].any()
        
        rpy_dot = state[13:16]  # rad/s
        rpy_dot_limit = rpy_dot[np.abs(rpy_dot) > self.rpy_dot_limit].any()
        
        z = state[2]  # m
        z_limit = z < self.z_lim

        # done = True if position_limit or rp_limit or rpy_dot_limit or z_limit else False
        done = True if rp_limit or rpy_dot_limit or z_limit else False
        
        # if done:
        #     self.out_of_bounds = True
        
        return done


    def _get_info(self):
        '''Generates the info dictionary returned by every call to .step().

        Returns:
            info (dict): A dictionary with information about the constraints evaluations and violations.
        '''
        info = {}
        if self.TASK == Task.STABILIZATION and self.COST == Cost.QUADRATIC:
            info['goal_reached'] = self.goal_reached  # Add boolean flag for the goal being reached.
        # # if self.done_on_out_of_bound:
        # info['out_of_bounds'] = self.out_of_bounds
        # Add MSE.
        state = deepcopy(self.state)
        if self.TASK == Task.STABILIZATION:
            xyz_rpy = np.concatenate([state[0:3], state[7:10]])
            state_error = xyz_rpy - self.X_GOAL[0]
        elif self.TASK == Task.TRAJ_TRACKING:
            # state[4] = normalize_angle(state[4])
            wp_idx = min(self.ctrl_step_counter + 1, self.X_GOAL.shape[0] - 1)  # +1 so that state is being compared with proper reference state.
            state_error = state - self.X_GOAL[wp_idx]
        # Filter only relevant dimensions.
        state_error = state_error 
        info['mse'] = np.sum(state_error ** 2)
        # Hanyang: add more info
        info['current_episode_distb_level'] = self.distb_level
        # if self.constraints is not None:
        #     info['constraint_values'] = self.constraints.get_values(self)
        #     info['constraint_violations'] = self.constraints.get_violations(self)
        return info


    def _get_reset_info(self):
        '''Generates the info dictionary returned by every call to .reset().

        Returns:
            info (dict): A dictionary with information about the dynamics and constraints symbolic models.
        '''
        info = {}
        state = deepcopy(self.state)
        if self.TASK == Task.STABILIZATION:
            xyz_rpy = np.concatenate([state[0:3], state[7:10]])
            state_error = xyz_rpy - self.X_GOAL[0]
        elif self.TASK == Task.TRAJ_TRACKING:
            # state[4] = normalize_angle(state[4])
            wp_idx = min(self.ctrl_step_counter + 1, self.X_GOAL.shape[0] - 1)  # +1 so that state is being compared with proper reference state.
            state_error = state - self.X_GOAL[wp_idx]
        # Filter only relevant dimensions.
        state_error = state_error 
        info['mse'] = np.sum(state_error ** 2)
        # Hanyang: add more info
        info['current_episode_distb_level'] = self.distb_level
        info['initial_state'] = state
        info['initial_position'] = state[0:3]
        info['initial_rpy'] = state[7:10]
        info['initial_velocity'] = state[10:13]
        # if self.constraints is not None:
        #     info['symbolic_constraints'] = self.constraints.get_all_symbolic_models()
            
        return info 
    

    def _start_video_recording(self):
        '''Starts the recording of a video output.

        The format of the video output is .mp4, if GUI is True, or .png, otherwise.
        The video is saved under folder `files/videos`.
        '''
        if self.RECORD and self.GUI:
            self.VIDEO_ID = p.startStateLogging(
                loggingType=p.STATE_LOGGING_VIDEO_MP4,
                fileName=os.path.join(self.output_dir, 'videos/video-{}.mp4'.format(datetime.now().strftime('%m.%d.%Y_%H.%M.%S'))),
                physicsClientId=self.PYB_CLIENT)
        if self.RECORD and not self.GUI:
            self.FRAME_NUM = 0
            self.IMG_PATH = os.path.join(self.output_dir, 'quadrotor_videos/video-{}/'.format(datetime.now().strftime('%m.%d.%Y_%H.%M.%S')))
            os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)
    

    def _update_and_store_kinematic_information(self):
        '''Updates and stores the drones kinematic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).
        '''
        for i in range(self.NUM_DRONES):
            self.pos[i], self.quat[i] = p.getBasePositionAndOrientation(
                self.DRONE_IDS[i], physicsClientId=self.PYB_CLIENT)
            self.rpy[i] = p.getEulerFromQuaternion(self.quat[i])
            self.vel[i], self.ang_v[i] = p.getBaseVelocity(
                self.DRONE_IDS[i], physicsClientId=self.PYB_CLIENT)
    

    def _get_drone_state_vector(self, nth_drone):
        '''Returns the state vector of the n-th drone.

        Args:
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns:
            ndarray. (20,)-shaped array of floats containing the state vector of the n-th drone.
                     Check the only line in this method and `_update_and_store_kinematic_information()`
                     to understand its format.
        '''
        # Hanyang: use self.last_action (the output of the controller) instead of self.last_clipped_action (the rmp or pwm signals)
        state = np.hstack([
            self.pos[nth_drone, :], self.quat[nth_drone, :],
            self.rpy[nth_drone, :], self.vel[nth_drone, :],
            self.ang_v[nth_drone, :], self.last_action[nth_drone, :]
        ])
        return state.reshape(20,)
    

    def _parse_urdf_parameters(self, file_name):
        '''Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.
        '''
        URDF_TREE = etxml.parse(file_name).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [
            float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')
        ]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        PWM2RPM_SCALE = float(URDF_TREE[0].attrib['pwm2rpm_scale'])
        PWM2RPM_CONST = float(URDF_TREE[0].attrib['pwm2rpm_const'])
        MIN_PWM = float(URDF_TREE[0].attrib['pwm_min'])
        MAX_PWM = float(URDF_TREE[0].attrib['pwm_max'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
            GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3, \
            PWM2RPM_SCALE, PWM2RPM_CONST, MIN_PWM, MAX_PWM
