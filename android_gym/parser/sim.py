from typing import Literal, List, Union, Dict
from typing_extensions import Self
from pydantic import (
    BaseModel,
    field_validator,
)

class PhysxParams(BaseModel):
    num_threads: int = 0
    solver_type: Literal[0,1] = 0 # 0: pgs, 1: tgs
    num_position_iterations: int = 4
    num_velocity_iterations: int = 1
    contact_offset: float = 0.01
    rest_offset: float = 0.0
    bounce_threshold_velocity: float = 0.5
    max_depenetration_velocity: float = 1.0
    max_gpu_contact_pairs: int = 8388608 # 2**23
    default_buffer_size_multiplier: float = 5.0
    contact_collection: int = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
    
    @field_validator('solver_type')
    @classmethod
    def parse_solver_type(cls, v) -> Literal[0,1]:
        if isinstance(v, str):
            solver_map = {
                "pgs": 0,
                "tgs": 1
            }
            return solver_map[v]
        return v


class SimConfig(BaseModel):
    dt: float = 0.01
    substeps: int = 1
    gravity: List[float] = [0.0, 0.0, -9.81]
    up_axis: Literal[0, 1, 2] = 1
    physx: PhysxParams = PhysxParams()

    @classmethod
    def from_dict(cls, data: Dict) -> Self:
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, path: str) -> Self:
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @field_validator('up_axis')
    @classmethod
    def parse_up_axis(cls, v) -> Literal[0, 1, 2]:
        axis_map = {
            "x": 0,
            "y": 1,
            "z": 2
        }
        if isinstance(v, str):
            if v not in axis_map:
                raise ValueError(f"Invalid up_axis: {v}") 
            return axis_map[v]
        return v