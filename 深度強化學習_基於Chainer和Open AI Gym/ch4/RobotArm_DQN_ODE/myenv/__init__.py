from gym.envs.registration import register

register(
    id='RobotArmODE-v0',
    entry_point='myenv.env:RobotArmODEEnv',
)

