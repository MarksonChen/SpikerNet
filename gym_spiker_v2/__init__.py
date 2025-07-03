from gymnasium.envs.registration import register

register(
    id='spiker-v2',
    entry_point='gym_spiker_v2:BGEnv',  # Entry point for BGTC model
)
