from typing import Literal, Dict, List, Union, Any, Optional
from typing_extensions import Self
from pydantic import (
    BaseModel,
    field_validator,
    model_validator
)
from .agent import Agent

class Terrain(BaseModel):
    mesh_type: Literal["none", "plane", "trimesh", "heightfield"] = 'plane'
    horizontal_scale: float = 0.1 # [m]
    vertical_scale: float = 0.005 # [m]
    border_size: int = 25 # [m]
    curriculum: bool = True
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0

    # rough terrain only:
    measure_heights: bool = True
    measured_points_x: List[float] = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
    measured_points_y: List[float] = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
    selected: bool = False # select a unique terrain type and pass all arguments
    terrain_kwargs: Dict[str, Any] | None = None # Dict of arguments for selected terrain
    max_init_terrain_level: int = 5 # starting curriculum state
    terrain_length: float = 8.0
    terrain_width: float = 8.0
    num_rows: int= 10 # number of terrain rows (levels)
    num_cols: int = 20 # number of terrain cols (types)
    # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
    terrain_proportions: List[float] = [0.1, 0.1, 0.35, 0.25, 0.2]
    # trimesh only:
    slope_treshold: float = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    @model_validator(mode="after")
    def trimesh_requires_slope_threshold(self) -> Self:
        if self.mesh_type == 'trimesh' and not hasattr(self, 'slope_treshold'):
            raise ValueError('slope_treshold is required for trimesh terrain')
        return self
    
    @model_validator(mode="after")
    def validate_curriculum(self) -> Self:
        if self.mesh_type not in ["heightfield", "trimesh"]:
            self.curriculum = False
        return self

class Viewer(BaseModel):
    ref_env: int = 0
    pos: List[float] = [3.0, 0.0, 2.0]
    lookat: List[float] = [0.0, 0.0, 0.5]

class EnvConfig(BaseModel):
    env_name: str
    num_envs: int = 1
    env_spacing: float = 3.0
    episode_length_seconds: int
    viewer: Viewer = Viewer()
    terrain: Terrain = Terrain()
    agents: Union[List[Agent], Agent]
    # TODO: props

    @field_validator("num_envs")
    @classmethod
    def num_envs_must_be_at_least_1(cls, v) -> int:
        if v < 1:
            raise ValueError(F'num_envs must be positive, got {v}')
        return v
    
    @field_validator("episode_length_seconds")
    @classmethod
    def parse_episode_length_seconds(cls, v) -> int:
        if v < 0:
            v = 50
        return v

    @field_validator("agents")
    @classmethod
    def rename_agents(cls, v) -> Union[List[Agent], Agent]:
        if isinstance(v, list):
            for i, agent in enumerate(v):
                agent.name = f'[{i}]{agent.name}'
        return v

    @field_validator('env_name')
    @classmethod
    def name_must_not_have_spaces(cls, v) -> str:
        if ' ' in v:
            raise ValueError('env_name must not have spaces')
        return v