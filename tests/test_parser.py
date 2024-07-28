import pytest
from .resources import MOCK_AGENT_DATA, MOCK_ENV_DATA

def test_none_priveleged_obs():
    from android_gym.parser import Agent
    mock_data = MOCK_AGENT_DATA.copy()
    agent = Agent(**mock_data)
    assert agent.num_privileged_observations == None
    mock_data["num_privileged_observations"] = None
    agent = Agent(**mock_data)
    assert agent.num_privileged_observations == None

def test_agent_parser():
    from android_gym.parser import Agent
    agent = Agent(**MOCK_AGENT_DATA)
    assert True

def test_env_parser():
    from android_gym.parser import EnvConfig
    env = EnvConfig(**MOCK_ENV_DATA)
    assert env.env_name == "test_env"
    assert env.num_envs == 1
    assert env.viewer.ref_env == 0
    assert env.agents.name == "test_agent"

    