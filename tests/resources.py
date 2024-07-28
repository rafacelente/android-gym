MOCK_AGENT_DATA = {
    "name": "test_agent",
        "num_observations": 10,
        "num_actions": 2,
        "init_state": {
            "pos": [0.0, 0.0, 0.0],
            "rot": [0.0, 0.0, 0.0, 0.0],
            "linear_velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
            "default_joint_angles": {
                "joint1": 0.0,
                "joint2": 0.0,
            }
        },
        "commands": {
            "curriculum": False,
            "max_commands": 1.0,
            "num_commands": 2,
            "ranges": {
                "command1": [0.0, 1.0],
                "command2": [0.0, 1.0],
            }
        },
        "noise": {
            "add_noise": False
        },
        "normalization": {
            "observation_clip": 100.0,
            "action_clip": 1.0,
            "observation_scales": {
                "observation1": 1.0,
                "observation2": 1.0,
            }
        },
        "rewards": {
            "only_positive_rewards": False,
            "reward_map": {
                "reward1": 0.0,
                "reward2": 0.0,
            }
        },
        "controls": {
            "stiffness": 0.1,
            "damping": 0.1,
            "action_scale": 0.5,
            "decimation": 1,
        },
        "asset": {
            "file": "/path/to/agent.urdf",
        }
}

MOCK_ENV_DATA = {
    "env_name": "test_env",
    "num_envs": 1,
    "episode_length_seconds": 50,
    "agents": MOCK_AGENT_DATA
}