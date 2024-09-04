'''BaseDistbAviary module.

This module contains the superclass of the Quadrotor environment, BaseDistbAviary.
BaseDistbAviary implements most of the integration with PyBullet.

The module also contains enumerations for drone models, PyBullet physics updates
image types captured by PyBullet's camera rendering.
'''

import os
import pkgutil
import time
import xml.etree.ElementTree as etxml
from datetime import datetime
from enum import Enum

import numpy as np
import pybullet as p
import pybullet_data

from safe_control_gym.envs.benchmark_env import BenchmarkEnv
from safe_control_gym.utils.utils import Boltzmann, quat2euler, distur_gener_quadrotor, transfer

egl = pkgutil.get_loader('eglRenderer')


class DroneModel(str, Enum):
    '''Drone models enumeration class.'''

    CF2X = 'cf2x'  # Bitcraze Craziflie 2.0 in the X configuration.


class Physics(str, Enum):
    '''Physics implementations enumeration class.'''

    PYB = 'pyb'  # Base PyBullet physics update.
    DYN = 'dyn'  # Update with an explicit model of the dynamics.
    PYB_GND = 'pyb_gnd'  # PyBullet physics update with ground effect.
    PYB_DRAG = 'pyb_drag'  # PyBullet physics update with drag.
    PYB_DW = 'pyb_dw'  # PyBullet physics update with downwash.
    PYB_GND_DRAG_DW = 'pyb_gnd_drag_dw'  # PyBullet physics update with ground effect, drag, and downwash.


class ImageType(int, Enum):
    '''Camera capture image type enumeration class.'''

    RGB = 0  # Red, green, blue (and alpha).
    DEP = 1  # Depth.
    SEG = 2  # Segmentation by object id.
    BW = 3  # Black and white.


class BaseDistbAviary(BenchmarkEnv):
    '''Base class for 'drone aviary' Gym environments.'''
    NAME = 'base_distb_aviary'
    URDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 physics: Physics = Physics.PYB,
                 record=False,
                 gui=False,
                 verbose=False,
                 # Hanyang: derive the following parameters from the BenchmarkEnv
                 pyb_freq: int = 200,
                 ctrl_freq: int = 100,
                 episode_len_sec: int = 10,
                 init_state=None,
                 init_xyzs=None,
                 init_rpys=None,
                 randomized_init: bool = True,
                 distb_type = 'fixed', 
                 distb_level: float=0.0,
                 seed=None,
                 adversary_disturbance=None,
                 **kwargs):
        '''Initialization of a generic aviary environment.

        Args:
            drone_model (DroneModel, optional): The desired drone type (detailed in an .urdf file
                                                in folder `assets`).
            num_drones (int, optional): The desired number of drones in the aviary.
            physics (Physics, optional): The desired implementation of PyBullet physics/custom
                                         dynamics.
            record (bool, optional): Whether to save a video of the simulation in folder
                                     `files/videos/`.
            gui (bool, optional): Whether to use PyBullet's GUI.
            verbose (bool, optional): If to suppress environment print statetments.
            pyb_freq (int, optional): The frequency at which PyBullet steps (a multiple of ctrl_freq).
            ctrl_freq (int, optional): The frequency at which the environment steps.
            episode_len_sec (int, optional): Maximum episode duration in seconds.
            init_state (ndarray/dict, optional): The initial state of the environment
            init_xyzs (ndarray | None, optional, (NUM_DRONES, 3)): The shaped array containing the initial XYZ position of the drones.
            init_rpys (ndarray | None, optional, (NUM_DRONES, 3)): The shaped array containing the initial orientations of the drones (in radians).
            randomized_init (bool, optional): Whether to randomize the initial state.
            disturbance_type (str, optional): The type of disturbance to be applied to the drones [None, 'fixed', 'boltzmann', 'random', 'rarl', 'rarl-population'].
            distb_level (float, optional): The level of disturbance to be applied to the drones.
            seed (int, optional): Seed for the random number generator.
            adversary_disturbance (str, optional): If to use adversary/external disturbance.
        '''
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
                '[INFO] BaseDistbAviary.__init__() loaded parameters from the drone\'s .urdf: \
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
        # BenchmarkEnv constructor.
        super().__init__(gui=gui, verbose=verbose, pyb_freq=pyb_freq, ctrl_freq=ctrl_freq, 
                         episode_len_sec=episode_len_sec, init_state=init_state, randomized_init=randomized_init,
                         seed=seed, adversary_disturbance=adversary_disturbance, **kwargs)
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
            print("[ERROR] invalid initial_xyzs in BaseDistbAviary.__init__(), try init_xyzs.reshape(NUM_DRONES,3)")
        if init_rpys is None:
            self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        elif np.array(init_rpys).shape == (self.NUM_DRONES,3):
            self.INIT_RPYS = init_rpys
        else:
            print("[ERROR] invalid initial_rpys in BaseDistbAviary.__init__(), try init_rpys.reshape(NUM_DRONES,3)")
            
            
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
        
    
    def before_step(self, action):
        '''Pre-processing before calling `.step()`.

        Args:
            action (ndarray): The raw action returned by the controller.

        Returns:
            action (ndarray): The processed action to be executed in the shape of shape (4,).
        '''
        # Sanity check (reset at least once).
        self._check_initial_reset()
        # Save the raw input action.
        action = np.atleast_1d(np.squeeze(action))

        if action.ndim != 1:
            raise ValueError('[ERROR]: The action returned by the controller must be 1 dimensional.')

        self.current_raw_action = action
        # Pre-process/clip the action
        processed_action = self._preprocess_control(action)
        return processed_action
    

    def _advance_simulation(self, clipped_action, disturbance_force=None):
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
                    hj_distbs = (0.003, 0.003, 0.0)  # seems good!
                    # hj_distbs = (0.002, 0.002, 0.0)  # seems good!
                    # hj_distbs = (0.0, 0.00424, 0.0)
                    # hj_distbs = (-0.00424, 0.00424, 0.0)  # disaster
                # elif self.distb_type == 'adversary':
                #     hj_distbs = (0.0, 0.0, 0.0)
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

    def render(self, mode='human', close=False):
        '''Prints a textual output of the environment.

        Args:
            mode (str, optional): Unused.
            close (bool, optional): Unused.
        '''
        if self.first_render_call and not self.GUI:
            print(
                '[WARNING] BaseDistbAviary.render() is implemented as text-only, re-initialize the environment using Aviary(gui=True) to use PyBullet\'s graphical interface'
            )
            self.first_render_call = False
        if self.VERBOSE:
            print(
                '\n[INFO] BaseDistbAviary.render() ——— it {:04d}'.format(
                    self.pyb_step_counter),
                '——— wall-clock time {:.1f}s,'.format(time.time() - self.RESET_TIME),
                'simulation time {:.1f}s@{:d}Hz ({:.2f}x)'.format(
                    self.pyb_step_counter * self.TIMESTEP, self.SIM_FREQ,
                    (self.pyb_step_counter * self.TIMESTEP) / (time.time() - self.RESET_TIME)))
            for i in range(self.NUM_DRONES):
                print(
                    '[INFO] BaseDistbAviary.render() ——— drone {:d}'.format(i),
                    '——— x {:+06.2f}, y {:+06.2f}, z {:+06.2f}'.format(
                        self.pos[i, 0], self.pos[i, 1], self.pos[i, 2]),
                    '——— velocity {:+06.2f}, {:+06.2f}, {:+06.2f}'.format(
                        self.vel[i, 0], self.vel[i, 1], self.vel[i, 2]),
                    '——— roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}'.format(
                        self.rpy[i, 0] * self.RAD2DEG,
                        self.rpy[i, 1] * self.RAD2DEG,
                        self.rpy[i, 2] * self.RAD2DEG),
                    '——— angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ——— '.
                    format(self.ang_v[i, 0], self.ang_v[i, 1], self.ang_v[i,
                                                                          2]))

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

    def _physics(self, rpm, nth_drone):
        '''Base PyBullet physics implementation.

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
        '''
        forces = np.array(rpm**2) * self.KF
        torques = np.array(rpm**2) * self.KM
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.PYB_CLIENT)
        p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                              4,
                              torqueObj=[0, 0, z_torque],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.PYB_CLIENT)
    
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
        # Hanyang: Apply disturbances (torques) 
        p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                              4,
                              torqueObj=[hj_distbs[0], 0, 0],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.PYB_CLIENT
                              )
        p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                              4,
                              torqueObj=[0, hj_distbs[1], 0],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.PYB_CLIENT
                              )
        # p.applyExternalTorque(self.DRONE_IDS[nth_drone],
        #                       4,
        #                       torqueObj=[0, 0, disturbances[2]],
        #                       flags=p.LINK_FRAME,
        #                       physicsClientId=self.PYB_CLIENT
        #                       )

    def _ground_effect(self, rpm, nth_drone):
        '''PyBullet implementation of a ground effect model.

        Inspired by the analytical model used for comparison in (Shi et al., 2019).

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
        '''
        # Kin. info of all links (propellers and center of mass)
        link_states = np.array(
            p.getLinkStates(self.DRONE_IDS[nth_drone],
                            linkIndices=[0, 1, 2, 3, 4],
                            computeLinkVelocity=1,
                            computeForwardKinematics=1,
                            physicsClientId=self.PYB_CLIENT))
        # Simple, per-propeller ground effects.
        prop_heights = np.array([
            link_states[0, 0][2], link_states[1, 0][2], link_states[2, 0][2],
            link_states[3, 0][2]
        ])
        prop_heights = np.clip(prop_heights, self.GND_EFF_H_CLIP, np.inf)
        gnd_effects = np.array(rpm**2) * self.KF * self.GND_EFF_COEFF \
            * (self.PROP_RADIUS / (4 * prop_heights))**2
        if np.abs(self.rpy[nth_drone, 0]) < np.pi / 2 and np.abs(
                self.rpy[nth_drone, 1]) < np.pi / 2:
            for i in range(4):
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     i,
                                     forceObj=[0, 0, gnd_effects[i]],
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.PYB_CLIENT)
        # TODO: a more realistic model accounting for the drone's
        # Attitude and its z-axis velocity in the world frame.

    def _drag(self, rpm, nth_drone):
        '''PyBullet implementation of a drag model.

        Based on the the system identification in (Forster, 2015).

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
        '''
        # Rotation matrix of the base.
        base_rot = np.array(p.getMatrixFromQuaternion(
            self.quat[nth_drone, :])).reshape(3, 3)
        # Simple draft model applied to the base/center of mass #
        drag_factors = -1 * self.DRAG_COEFF * np.sum(
            np.array(2 * np.pi * rpm / 60))
        drag = np.dot(base_rot, drag_factors * np.array(self.vel[nth_drone, :]))
        p.applyExternalForce(self.DRONE_IDS[nth_drone],
                             4,
                             forceObj=drag,
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.PYB_CLIENT)

    def _downwash(self, nth_drone):
        '''PyBullet implementation of a ground effect model.

        Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.

        Args:
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
        '''
        for i in range(self.NUM_DRONES):
            delta_z = self.pos[i, 2] - self.pos[nth_drone, 2]
            delta_xy = np.linalg.norm(
                np.array(self.pos[i, 0:2]) - np.array(self.pos[nth_drone, 0:2]))
            if delta_z > 0 and delta_xy < 10:  # Ignore drones more than 10 meters away
                alpha = self.DW_COEFF_1 * (self.PROP_RADIUS / (4 * delta_z))**2
                beta = self.DW_COEFF_2 * delta_z + self.DW_COEFF_3
                downwash = [0, 0, -alpha * np.exp(-.5 * (delta_xy / beta)**2)]
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     4,
                                     forceObj=downwash,
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.PYB_CLIENT)

    def _dynamics(self, rpm, nth_drone):
        '''Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
        '''
        # Current state.
        pos = self.pos[nth_drone, :]
        quat = self.quat[nth_drone, :]
        rpy = self.rpy[nth_drone, :]
        vel = self.vel[nth_drone, :]
        rpy_rates = self.rpy_rates[nth_drone, :]
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        # Compute forces and torques.
        forces = np.array(rpm**2) * self.KF
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY])
        z_torques = np.array(rpm**2) * self.KM
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        if self.DRONE_MODEL == DroneModel.CF2X:
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L / np.sqrt(2))
            y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * (self.L / np.sqrt(2))
        elif self.DRONE_MODEL == DroneModel.CF2P:
            x_torque = (forces[1] - forces[3]) * self.L
            y_torque = (-forces[0] + forces[2]) * self.L
        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))
        rpy_rates_deriv = np.dot(self.J_INV, torques)
        no_pybullet_dyn_accs = force_world_frame / self.MASS
        # Update state.
        vel = vel + self.TIMESTEP * no_pybullet_dyn_accs
        rpy_rates = rpy_rates + self.TIMESTEP * rpy_rates_deriv
        pos = pos + self.TIMESTEP * vel
        rpy = rpy + self.TIMESTEP * rpy_rates
        # Set PyBullet's state.
        p.resetBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
                                          pos,
                                          p.getQuaternionFromEuler(rpy),
                                          physicsClientId=self.PYB_CLIENT)
        # Note: the base's velocity only stored and not used #
        p.resetBaseVelocity(
            self.DRONE_IDS[nth_drone],
            vel,
            rpy_rates,  # ang_vel not computed by DYN
            physicsClientId=self.PYB_CLIENT)
        # Store the roll, pitch, yaw rates for the next step #
        self.rpy_rates[nth_drone, :] = rpy_rates

    def _show_drone_local_axes(self, nth_drone):
        '''Draws the local frame of the n-th drone in PyBullet's GUI.

        Args:
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
        '''
        if self.GUI:
            AXIS_LENGTH = 2 * self.L
            self.X_AX[nth_drone] = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[AXIS_LENGTH, 0, 0],
                lineColorRGB=[1, 0, 0],
                parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.X_AX[nth_drone]),
                physicsClientId=self.PYB_CLIENT)
            self.Y_AX[nth_drone] = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, AXIS_LENGTH, 0],
                lineColorRGB=[0, 1, 0],
                parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.Y_AX[nth_drone]),
                physicsClientId=self.PYB_CLIENT)
            self.Z_AX[nth_drone] = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, 0, AXIS_LENGTH],
                lineColorRGB=[0, 0, 1],
                parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.Z_AX[nth_drone]),
                physicsClientId=self.PYB_CLIENT)

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
