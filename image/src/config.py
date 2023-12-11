from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    n_filters: int = 3
    kernel_size: int = 3
    padding: Literal["valid", "same"] = "valid"
    n_blocks: int = 2
    # SVHN-only
    dropout_rate: float = 0.2
    final_dropout_rate: float = 0.4
    n_units: int = 128
    
