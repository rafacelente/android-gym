from typing import Literal, List, Union, Optional
from pydantic import (
    BaseModel,
    field_validator,
)

class PhysxParams(BaseModel):
    num_threads: int = 0
    solver_type: Union[Literal[0,1], Literal["pgs", "tgs"]] = "pgs"
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
    def parse_solver_type(cls, v) -> Union[Literal[0,1], Literal["pgs", "tgs"]]:
        solver_map = {
            "pgs": 0,
            "tgs": 1
        }
        if isinstance(v, str):
            return solver_map[v]
        return v


class SimConfig(BaseModel):
    dt: float = 0.01
    substeps: int = 1
    gravity: List[float] = [0.0, 0.0, -9.81]
    up_axis: Union[Literal[0, 1, 2], Literal["x", "y", "z"]] = 1
    physx: PhysxParams = PhysxParams()

    @field_validator('up_axis')
    @classmethod
    def parse_up_axis(cls, v) -> Union[Literal[0, 1, 2], Literal["x", "y", "z"]]:
        axis_map = {
            "x": 0,
            "y": 1,
            "z": 2
        }
        if isinstance(v, str):
            return axis_map[v]
        return v