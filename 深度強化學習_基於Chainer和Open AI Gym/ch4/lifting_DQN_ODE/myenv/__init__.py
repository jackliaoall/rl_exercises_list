from gym.envs.registration import register

register(
    id='LiftingODE-v0',
    entry_point='myenv.env:LiftingODEEnv',
)
