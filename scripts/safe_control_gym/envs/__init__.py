'''Register environments.'''

from safe_control_gym.utils.registration import register

register(idx='cartpole',
         entry_point='safe_control_gym.envs.gym_control.cartpole:CartPole',
         config_entry_point='safe_control_gym.envs.gym_control:cartpole.yaml')

register(idx='quadrotor',
         entry_point='safe_control_gym.envs.gym_pybullet_drones.quadrotor:Quadrotor',
         config_entry_point='safe_control_gym.envs.gym_pybullet_drones:quadrotor.yaml')

# Hanyang
register(idx='cartpole_v0',
         entry_point='safe_control_gym.envs.gym_control.cartpole_v0:CartPole',
         config_entry_point='safe_control_gym.envs.gym_control:cartpole_v0.yaml')

register(idx='cartpole_distb',
         entry_point='safe_control_gym.envs.gym_control.cartpole_distb:CartPoleDistbEnv',
         config_entry_point='safe_control_gym.envs.gym_control:cartpole_distb.yaml')

register(idx='cartpole_fixed',
         entry_point='safe_control_gym.envs.gym_control.cartpole_distb:CartPoleFixedDistb',
         config_entry_point='safe_control_gym.envs.gym_control:cartpole_distb.yaml')

register(idx='cartpole_boltz',
         entry_point='safe_control_gym.envs.gym_control.cartpole_distb:CartPoleBoltzDistb',
         config_entry_point='safe_control_gym.envs.gym_control:cartpole_distb.yaml')

register(idx='cartpole_null',
         entry_point='safe_control_gym.envs.gym_control.cartpole_distb:CartPoleNullDistb',
         config_entry_point='safe_control_gym.envs.gym_control:cartpole_distb.yaml')

register(idx='cartpole_randomhj',
         entry_point='safe_control_gym.envs.gym_control.cartpole_distb:CartPoleRandomHJDistb',
         config_entry_point='safe_control_gym.envs.gym_control:cartpole_distb.yaml')

register(idx='cartpole_random',
         entry_point='safe_control_gym.envs.gym_control.cartpole_distb:CartPoleRandomDistb',
         config_entry_point='safe_control_gym.envs.gym_control:cartpole_distb.yaml')

register(idx='quadrotor_distb',
         entry_point='safe_control_gym.envs.gym_pybullet_drones.quadrotor_distb:QuadrotorDistb',
         config_entry_point='safe_control_gym.envs.gym_pybullet_drones:quadrotor_distb.yaml')

register(idx='quadrotor_fixed',
         entry_point='safe_control_gym.envs.gym_pybullet_drones.quadrotor_distb:QuadrotorFixedDistb',
         config_entry_point='safe_control_gym.envs.gym_pybullet_drones:quadrotor_distb.yaml')

register(idx='quadrotor_boltz',
         entry_point='safe_control_gym.envs.gym_pybullet_drones.quadrotor_distb:QuadrotorBoltzDistb',
         config_entry_point='safe_control_gym.envs.gym_pybullet_drones:quadrotor_distb.yaml')

register(idx='quadrotor_null',
         entry_point='safe_control_gym.envs.gym_pybullet_drones.quadrotor_distb:QuadrotorNullDistb',
         config_entry_point='safe_control_gym.envs.gym_pybullet_drones:quadrotor_distb.yaml')

register(idx='quadrotor_rarl',
         entry_point='safe_control_gym.envs.gym_pybullet_drones.quadrotor_distb:QuadrotorRARLDistb',
         config_entry_point='safe_control_gym.envs.gym_pybullet_drones:quadrotor_adversary.yaml')

register(idx='quadrotor_random',
         entry_point='safe_control_gym.envs.gym_pybullet_drones.quadrotor_distb:QuadrotorRandomDistb',
         config_entry_point='safe_control_gym.envs.gym_pybullet_drones:quadrotor_distb.yaml')

register(idx='quadrotor_randomhj',
         entry_point='safe_control_gym.envs.gym_pybullet_drones.quadrotor_distb:QuadrotorRandomHJDistb',
         config_entry_point='safe_control_gym.envs.gym_pybullet_drones:quadrotor_distb.yaml')

register(idx='quadrotor_wind',
         entry_point='safe_control_gym.envs.gym_pybullet_drones.quadrotor_distb:QuadrotorWindDistb',
         config_entry_point='safe_control_gym.envs.gym_pybullet_drones:quadrotor_distb.yaml')

register(idx='quadrotor_adversary',
         entry_point='safe_control_gym.envs.gym_pybullet_drones.quadrotor_adversary:QuadrotorAdversary',
         config_entry_point='safe_control_gym.envs.gym_pybullet_drones:quadrotor_adversary.yaml')

register(idx='reach_avoid',
         entry_point='safe_control_gym.envs.gym_game.ReachAvoidGame:ReachAvoidGameEnv',
         config_entry_point='safe_control_gym.envs.gym_game:game.yaml')

register(idx='reach_avoid_test',
         entry_point='safe_control_gym.envs.gym_game.ReachAvoidGame:ReachAvoidGameTest')

register(idx='rarl_game',
         entry_point='safe_control_gym.envs.gym_game.RARLGame:RARLGameEnv',
         config_entry_point='safe_control_gym.envs.gym_game:game.yaml')

register(idx='dubin_rarl_game',
         entry_point='safe_control_gym.envs.gym_game.DubinRARLGame:DubinRARLGameEnv',
         config_entry_point='safe_control_gym.envs.gym_game:dubin_game.yaml')
