from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    n_filters: int = 3
    kernel_size: int = 3
    padding: Literal["valid", "same"] = "valid"
    n_blocks: int = 2
    