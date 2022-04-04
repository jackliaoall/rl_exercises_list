from gym.envs.registration import register

register(
    id='Lifting-v0',
    entry_point='myenv.env:LiftingEnv',
)
