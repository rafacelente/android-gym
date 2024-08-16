from .parser import Agent

import os

ANDROID_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ANDROID_GYM_ENVS_DIR = os.path.join(ANDROID_GYM_ROOT_DIR, 'android_gym', 'envs')