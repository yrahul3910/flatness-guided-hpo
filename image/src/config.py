from typing import Literal, TypedDict

type HpoOption = int | str | float
type HpoOptionRange = tuple[int, int] | tuple[float, float] | list[str] | list[int]


class Config(TypedDict):
    n_filters: int
    kernel_size: int
    padding: Literal["valid", "same"]
    n_blocks: int
    dropout_rate: float | None
    final_dropout_rate: float | None
    n_units: int
    learning_rate: float
    weight_decay: float


class HpoSpace(TypedDict):
    n_filters: tuple[int, int]
    kernel_size: tuple[int, int]
    padding: list[Literal["valid", "same"]]
    n_blocks: tuple[int, int]
    dropout_rate: tuple[float, float] | None
    final_dropout_rate: tuple[float, float] | None
    n_units: list[int]
    learning_rate: tuple[float, float]
    weight_decay: tuple[float, float]


hpo_space: HpoSpace = {
    "n_filters": (32, 128),
    "kernel_size": (2, 6),
    "padding": ["valid", "same"],
    "n_blocks": (2, 5),
    "dropout_rate": (0.2, 0.6),
    "final_dropout_rate": (0.3, 0.7),
    "n_units": [32, 64, 128, 256, 512],
    "learning_rate": (1e-4, 1e-2),
    "weight_decay": (1e-5, 1e-1),
}
