"""Run a unit-coordinate BOHB adapter with validation selection and one held-out test evaluation.

Numeric KDE coordinates stay in [0, 1]. Padding and dense-width coordinates retain BOHB's ordered-discrete treatment,
using unit-bin centers for the shared conditional/logarithmic CIFAR decoder. This is a changed search-space adapter.
"""

import json
import math
import traceback
from collections.abc import Mapping
from importlib.metadata import version
from itertools import count
from typing import Literal, TypedDict

import bohb.configspace as bohb_space
import numpy as np
from bohb import BOHB

from image.src.config import Config
from image.src.experiment import SearchRun
from image.src.matched_contract import N_COORDINATES, decode

CATEGORY_CENTERS = {2: [0.25, 0.75], 4: [(index + 0.5) / 5 for index in range(5)]}


class ProposalRecord(TypedDict):
    event: Literal["start", "end"]
    call_id: int
    latent: list[float | str]
    config: Config | None
    raw_budget: float
    epochs: int
    selection_eligible: bool
    loss: float | str | None
    error: str | None


def unit_space(seed: int) -> bohb_space.ConfigurationSpace:
    """Keep numeric KDE units normalized and categorical values at decoder bin centers.

    Parameter creation order is coordinate order: the installed Configuration sorts by each parameter's creation ID.
    """
    parameters = []

    for index in range(N_COORDINATES):
        name = f"x{index}"

        if index in CATEGORY_CENTERS:
            parameters.append(bohb_space.CategoricalHyperparameter(name, CATEGORY_CENTERS[index]))
        else:
            parameters.append(bohb_space.UniformHyperparameter(name, 0.0, 1.0))

    return bohb_space.ConfigurationSpace(parameters, seed=seed)


if __name__ == "__main__":
    with SearchRun("bohb-unit-coordinate") as run:
        max_budget = 100
        call_ids = count(1)
        proposal_path = run.directory / "bohb-proposals.jsonl"
        metadata = {
            "adapter": "bohb-unit-coordinate",
            "decoder": "image.src.matched_contract.decode",
            "coordinate_names": [f"x{i}" for i in range(N_COORDINATES)],
            "continuous_bounds": [0.0, 1.0],
            "categorical_centers": CATEGORY_CENTERS,
            "categorical_kde_type": "installed ordered-discrete; to_list uses category indices",
            "fit_policy": "fresh model for every objective call, including repeated configurations and rungs",
            "packages": {name: version(name) for name in ("bohb-hpo", "numpy", "scipy", "statsmodels")},
        }
        (run.directory / "adapter.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

        def evaluate_bohb(config: Mapping[str, float], budget: float = 100) -> float:
            """Record latent/decoded proposals and preserve the optimizer RNG across cold model fits.

            Recoverable model errors retain loss 100. Every other exception is recorded and propagated.
            """
            # BOHB truncates some nominal 100-epoch rungs to 99 before invoking this callback.
            epochs = max_budget if budget >= max_budget - 1 else round(budget)
            coordinates = [float(config[f"x{i}"]) for i in range(N_COORDINATES)]
            record = ProposalRecord(
                event="start",
                call_id=next(call_ids),
                latent=[value if math.isfinite(value) else str(value) for value in coordinates],
                config=None,
                raw_budget=budget,
                epochs=epochs,
                selection_eligible=epochs >= max_budget,
                loss=None,
                error=None,
            )
            # BOHB and SciPy use the legacy global stream; SearchRun reseeds it for each model fit.
            optimizer_rng_state = np.random.get_state()  # noqa: NPY002 - preserve the installed sampler's RNG.

            try:
                decoded = decode(coordinates)
                record["config"] = decoded

                with proposal_path.open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")

                loss = 1.0 - run.evaluate(decoded, epochs=epochs, select=epochs >= max_budget)
                record["loss"] = loss if math.isfinite(loss) else str(loss)
            except BaseException as error:
                record["error"] = repr(error)

                if isinstance(error, (ValueError, RuntimeError)):
                    traceback.print_exc()
                    record["loss"] = 100.0

                    return 100.0

                raise
            else:
                return loss
            finally:
                np.random.set_state(optimizer_rng_state)  # noqa: NPY002 - restore the installed sampler's RNG.
                record["event"] = "end"

                with proposal_path.open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")

        opt = BOHB(
            configspace=unit_space(run.seed),
            evaluate=evaluate_bohb,
            min_budget=1,
            max_budget=max_budget,
        )
        opt.optimize()
        run.finish()
