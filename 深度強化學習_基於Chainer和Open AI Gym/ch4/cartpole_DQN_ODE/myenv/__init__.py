from gym.envs.registration import register

register(
    id='CartPoleODE-v0',
    entry_point='myenv.env:CartPoleODEEnv',
)
