import os
from .legged_robot_config import LeggedRobotCfgPPO
from .chape.chape import Chape


from android_gym.utils.task_registry import task_registry
from android_gym.parser import EnvConfig, SimConfig
from android_gym import ANDROID_GYM_ENVS_DIR

task_registry.register(
    name="chape",
    task_class=Chape,
    env_cfg=EnvConfig.from_yaml(os.path.join(ANDROID_GYM_ENVS_DIR, "chape/chape_env.yaml")),
    sim_cfg=SimConfig.from_yaml(os.path.join(ANDROID_GYM_ENVS_DIR, "chape/chape_sim.yaml")),
    train_cfg=LeggedRobotCfgPPO(),
)