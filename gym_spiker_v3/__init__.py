from gymnasium.envs.registration import register

register(
    id='spiker-v3',
    entry_point='gym_spiker_v3:BGEnv',  # Entry point for BGTC model
)

