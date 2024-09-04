'''Register controllers.'''

from safe_control_gym.utils.registration import register

register(idx='lqr',
         entry_point='safe_control_gym.controllers.lqr.lqr:LQR',
         config_entry_point='safe_control_gym.controllers.lqr:lqr.yaml')

register(idx='ilqr',
         entry_point='safe_control_gym.controllers.lqr.ilqr:iLQR',
         config_entry_point='safe_control_gym.controllers.lqr:ilqr.yaml')

register(idx='mpc',
         entry_point='safe_control_gym.controllers.mpc.mpc:MPC',
         config_entry_point='safe_control_gym.controllers.mpc:mpc.yaml')

register(idx='linear_mpc',
         entry_point='safe_control_gym.controllers.mpc.linear_mpc:LinearMPC',
         config_entry_point='safe_control_gym.controllers.mpc:linear_mpc.yaml')

register(idx='gp_mpc',
         entry_point='safe_control_gym.controllers.mpc.gp_mpc:GPMPC',
         config_entry_point='safe_control_gym.controllers.mpc:gp_mpc.yaml')

register(idx='pid',
         entry_point='safe_control_gym.controllers.pid.pid:PID',
         config_entry_point='safe_control_gym.controllers.pid:pid.yaml')

register(idx='ppo',
         entry_point='safe_control_gym.controllers.ppo.ppo:PPO',
         config_entry_point='safe_control_gym.controllers.ppo:ppo.yaml')

register(idx='sac',
         entry_point='safe_control_gym.controllers.sac.sac:SAC',
         config_entry_point='safe_control_gym.controllers.sac:sac.yaml')

register(idx='ddpg',
         entry_point='safe_control_gym.controllers.ddpg.ddpg:DDPG',
         config_entry_point='safe_control_gym.controllers.ddpg:ddpg.yaml')

register(idx='safe_explorer_ppo',
         entry_point='safe_control_gym.controllers.safe_explorer.safe_ppo:SafeExplorerPPO',
         config_entry_point='safe_control_gym.controllers.safe_explorer:safe_ppo.yaml')

register(idx='rarl',
         entry_point='safe_control_gym.controllers.rarl.rarl:RARL',
         config_entry_point='safe_control_gym.controllers.rarl:rarl.yaml')

register(idx='rap',
         entry_point='safe_control_gym.controllers.rarl.rap:RAP',
         config_entry_point='safe_control_gym.controllers.rarl:rap.yaml')

# Hanyang
register(idx='rarlgame',
         entry_point='safe_control_gym.controllers.rarl.rarl:RARL',
         config_entry_point='safe_control_gym.controllers.rarl:rarl_easiergame.yaml')

register(idx='rapgame',
         entry_point='safe_control_gym.controllers.rarl.rap:RAP',
         config_entry_point='safe_control_gym.controllers.rarl:rap_easiergame.yaml')
register(idx='hj',
         entry_point='safe_control_gym.controllers.hj.hj:HJ',
         config_entry_point='safe_control_gym.controllers.hj:hj.yaml')

register(idx='ppo_v0',
         entry_point='safe_control_gym.controllers.ppo_v0.ppo:PPO',
         config_entry_point='safe_control_gym.controllers.ppo_v0:ppo.yaml')