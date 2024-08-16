from typing import Literal
import sys
from isaacgym import gymapi
from isaacgym import gymutil
import torch
from ..parser import EnvConfig, SimConfig

class BaseEnv:
    def __init__(
            self,
            cfg: EnvConfig,
            sim_params: SimConfig,
            sim_device: Literal["cuda", "gpu"] = "gpu",
            headless: bool = True,
            profile: bool = False,
            physics_engine: gymapi.SimType = gymapi.SIM_PHYSX,
    ):
        self.gym = gymapi.acquire_gym()
        self.cfg = cfg
        self.headless = headless
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        assert "cuda" in sim_device or "gpu" in sim_device, f"Only CUDA is supported, got {sim_device}"
        self.sim_device = "cuda" # TODO: add support for CPU pipelines 
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(
            self.sim_device)
        
        self.device = self.sim_device # TODO: cpu spport
        
        # env specific configurations
        self.num_envs = cfg.num_envs
        self.agents = cfg.agents
        # self.agents = [self.agents] if not isinstance(self.agents, list) else self.agents
        #self.num_agents = len(self.agents)
        #self.multi_agent = self.num_agents > 1
        # TODO: Fix this to work with multiple agents,
        # cant be bothered doing it right now
        # self.num_obs = [agent.num_observations for agent in self.agents]
        # self.num_actions = [agent.num_actions for agent in self.agents]
        # self.num_priveleged_obs = [agent.num_privileged_observations for agent in self.agents]
        self.num_obs = self.agents.num_observations
        self.num_actions = self.agents.num_actions
        self.num_priveleged_obs = self.agents.num_privileged_observations

        if not self.headless:
            self.graphics_device_id = self.sim_device_id
        else:
            self.graphics_device_id = -1

        if not profile:
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)
        
        # buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), dtype=torch.float32
        ).to(self.device)# .squeeze() # in case of single agent, squeeze
        self.rew_buf = torch.zeros(
            (self.num_envs), dtype=torch.float32
        ).to(self.device) # .squeeze() # in case of single agent, squeeze
        self.reset_buf = torch.zeros(
            self.num_envs, dtype=torch.long
        ).to(self.device)
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long
        ).to(self.device)
        self.time_out_buf = torch.zeros(
            self.num_envs, dtype=torch.long
        ).to(self.device)
        # TODO: theoretically, we could have certain agents
        # with privileged and others without, but for now
        # we assume all agents have the same number of privileged obs
        if self.num_priveleged_obs is not None:
            if all(t is not None for t in self.num_priveleged_obs):
                self.privileged_obs_buf = torch.zeros(
                    (self.num_envs, sum(self.num_priveleged_obs)), dtype=torch.float32
                ).squeeze()
            else:
                self.privileged_obs_buf = None
        
        # extra buffers that don't go into the policy update
        # use these for logging, debugging, etc.
        self.extras = {}

        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # FIXME: what is this used for?
        self.enable_viewer_sync = True
        self.viewer = None

        if not self.headless:
            self.viewer = self.gym.create_viewer(
                self.sim,
                gymapi.CameraProperties()
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer,
                gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer,
                gymapi.KEY_V, "toggle_viewer_sync"
            )
            camera_properties = gymapi.CameraProperties()
            camera_properties.width = 720
            camera_properties.height = 480
            camera_handle = self.gym.create_camera_sensor(
                self.envs[0], camera_properties)
            self.camera_handle = camera_handle
        else:
            # pass
            camera_properties = gymapi.CameraProperties()
            camera_properties.width = 720
            camera_properties.height = 480
            camera_handle = self.gym.create_camera_sensor(
                self.envs[0], camera_properties)
            self.camera_handle = camera_handle

    def create_sim(self):
        self.up_axis_idx = 2 #self.sim_params.up_axis
        self.sim = self.gym.create_sim(
            compute_device=self.sim_device_id,
            graphics_device=self.graphics_device_id,
            type=self.physics_engine,
            params=self.sim_params,
        )
        self.create_terrain_meshes()
        self.create_envs()

    def get_observations(self):
        return self.obs_buf, {"observations": {}}
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def create_terrain_meshes(self):
        raise NotImplementedError

    def create_envs(self):
        raise NotImplementedError
    
    def reset_idx(self, env_ids):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(
                (
                    self.num_envs,
                    self.num_actions,
                ), 
                device=self.device,
            ).squeeze()
        )
        return obs, privileged_obs
    
    def render(self, sync_frame_time=True):
        if self.viewer:
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
    

        



            
        


        
