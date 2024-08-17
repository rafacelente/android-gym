from typing import List, Dict, Union, Literal, Optional
from typing_extensions import Self
from pydantic import (
    BaseModel, 
    field_validator,
    model_validator
)

# FIXME: The ValueErrors should be replaced with pydantic.ValidationErrors

class InitState(BaseModel):
    pos: List[float]
    rot: List[float]
    linear_velocity: List[float]
    angular_velocity: List[float]
    default_joint_angles: Dict[str, float]

    @field_validator('pos')
    @classmethod
    def pos_must_be_3d(cls, v) -> List[float]:
        if len(v) != 3:
            raise ValueError('pos must be 3D')
        return v

    @field_validator('rot')
    @classmethod
    def rot_must_be_4d(cls, v) -> List[float]:
        if len(v) != 4:
            raise ValueError('rot must be 4D')
        return v
    
    @field_validator('linear_velocity')
    @classmethod
    def linear_velocity_must_be_3d(cls, v) -> List[float]:
        if len(v) != 3:
            raise ValueError('linear_velocity must be 3D')
        return v
    
    @field_validator('angular_velocity')
    @classmethod
    def angular_velocity_must_be_3d(cls, v) -> List[float]:
        if len(v) != 3:
            raise ValueError('angular_velocity must be 3D')
        return v

class Commands(BaseModel):
    curriculum: bool = False
    max_commands: float = 1.0
    num_commands: int
    command_ranges: Dict[str, List[float]]
    resampling_time: float = 10.

    @field_validator('command_ranges')
    @classmethod
    def ranges_must_be_2d(cls, v) -> Dict[str, List[float]]:
        for k, r in v.items():
            if len(r) != 2:
                raise ValueError('ranges must be 2D')
        return v
    
    @model_validator(mode="after")
    def num_commands_validator(self) -> Self:
        if self.num_commands != len(self.command_ranges):
            raise ValueError('num_commands must match number of ranges')
        return self
    
class Noise(BaseModel):
    add_noise: bool = False
    noise_map: Optional[Dict[str, float]] = None

    @model_validator(mode="after")
    def noise_map_validator(self) -> Self:
        if self.add_noise and self.noise_map is None:
            raise ValueError('noise_map must be provided when add_noise is True')
        return self

class Normalization(BaseModel):
    observation_clip: float = 100.0
    action_clip: float = 3.14
    observation_scales: Dict[str, float]
    command_scales: Optional[Dict[str, float]] = None
    
class Rewards(BaseModel):
    only_positive_rewards: bool = False
    reward_map: Dict[str, float]
    penalize_contacts_on: List[str] = []
    terminate_after_contacts_on: List[str] = []
    custom_terminations: List[str] = []

    @field_validator('reward_map')
    @classmethod
    def reward_name_must_not_have_spaces(cls, v) -> Dict[str, float]:
        for k in v.keys():
            if ' ' in k:
                raise ValueError(f'Reward name {k} must not have spaces')
        return v


class SafetyScales(BaseModel):
    pos_scale: float = 1.0
    vel_scale: float = 1.0
    torque_scale: float = 1.0

class Controls(BaseModel):
    stiffness: Union[Dict[str, float], float]
    damping: Union[Dict[str, float], float]
    action_scale: float = 0.5
    decimation: int = 1
    safety_scales: SafetyScales = SafetyScales()

"""
TODO: 
This should take into consideration each DOF.
Right now it applies the same properties for all
DOFs.
"""
class DOFProperties(BaseModel):
    density: float = 0.001
    angular_damping: float = 0.0
    linear_damping: float = 0.0
    max_angular_velocity: float = 1000.0
    max_linear_velocity: float = 1000.0
    armature: float = 0.0
    thickness: float = 0.01

class Asset(BaseModel):
    file: str
    foot_name: Optional[str] = None
    knee_name: Optional[str] = None
    collapse_fixed_joints: bool = True
    fix_base_link: bool = False
    default_dof_drive_mode: Union[int, Literal["none", "pos", "vel", "effort"]] = 1
    allow_self_collision: bool = False
    replace_cylinder_with_capsule: bool = True
    flip_visual_attachments: bool = True
    dof_properties: Optional[DOFProperties] = DOFProperties()

    @field_validator('default_dof_drive_mode')
    @classmethod
    def parse_dof_drive_mode(cls, v) -> int:
        drive_map = {
            "none": 0,
            "pos": 1,
            "vel": 2,
            "effort": 3
        }
        if isinstance(v, str):
            if v not in drive_map.keys():
                raise ValueError(f'Invalid drive mode {v}')
            return drive_map[v]
        else:
            if v not in drive_map.values():
                raise ValueError(f'Invalid drive mode {v}. Must be one of {list(drive_map.values())}')
            return v

class Agent(BaseModel):
    name: str
    num_observations: int
    num_privileged_observations: Optional[int] = None
    num_actions: int
    init_state: InitState
    disable_gravity: bool = False
    commands: Commands
    noise: Noise = Noise()
    normalization: Normalization
    rewards: Rewards
    controls: Controls
    asset: Asset
    # TODO(MAJOR): DOMAIN RANDOMIZATION - SEE legged_robots.py _process_rigid_shape_props and _process_rigid_body_props
    
    # TODO: there may be cases where the number of actions
    # is not equal to the number of joints (inactive joints),
    # but skipping for now
    @model_validator(mode="after")
    def num_actions_equals_num_joints(self) -> Self:
        if self.num_actions != len(self.init_state.default_joint_angles.keys()):
            raise ValueError(
                f'num_actions must match number of joints but found:',
                f'num_actions: {self.num_actions}',
                f'num_joints: {len(self.init_state.default_joint_angles.keys())}')
        return self
    
    @model_validator(mode="after")
    def maybe_match_commands_and_normalizations(self) -> Self:
        if self.normalization.command_scales is None:
            self.normalization.command_scales = {}
            for command in self.commands.command_ranges.keys():
                if command not in self.normalization.observation_scales.keys():
                    self.normalization.command_scales[command] = 1.0
                else:
                    self.normalization.command_scales[command] = self.normalization.observation_scales[command]
        assert len(self.normalization.command_scales) == len(self.commands.command_ranges) == self.commands.num_commands
        return self
            
    

