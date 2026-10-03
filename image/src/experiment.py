"""Record seeded image searches and evaluate only the validation-selected checkpoint on the test set."""

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from types import TracebackType

import numpy as np
from keras import backend
from keras.models import Model, load_model
from keras.utils import set_random_seed
from sklearn.metrics import accuracy_score

from image.src.config import Config, hpo_space
from image.src.util import get_data, run_experiment


class SearchRun:
    """Keep a run's source, seeds, trials, and selected model together in a fresh directory.

    `evaluate` returns validation accuracy and checkpoints each improvement. `finish` tests the incumbent once;
    it cannot be called twice. Every trial attempt, including an exception, is appended to `events.jsonl`.
    An exception exits the context as a failed run and leaves all existing evidence intact.
    """

    def __init__(self, method: str) -> None:
        """Create a new evidence directory and load the requested dataset.

        Existing output directories are rejected so a repeated command cannot overwrite an earlier run.
        Source snapshots include local modifications that the Git revision alone does not describe.
        """
        self.dataset = next(
            (os.environ[key] for key in ("DATASET", "CLOUD_EXP_DATASET", "SWEEP_DATASET") if os.environ.get(key)),
            "cifar10",
        )
        self.n_classes = 100 if self.dataset == "cifar100" else 10
        self.seed = int(os.getenv("FGHPO_SEED", "0"))
        self.method = method
        self.trial = 0
        self.probe = 0
        self.best_accuracy = -math.inf
        self.finished = False
        self.started = time.perf_counter()

        run_id = f"{method}-{self.dataset}-seed{self.seed}-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}"
        root = Path(os.getenv("SWEEP_RESULTS_DIR", "sweep-results"))
        self.directory = Path(os.getenv("FGHPO_OUTPUT_DIR", str(root / run_id)))
        self.directory.mkdir(parents=True, exist_ok=False)
        self.events = self.directory / "events.jsonl"
        self.checkpoint = self.directory / "incumbent.keras"

        source_root = Path(__file__).resolve().parents[2]
        sources = [*sorted(source_root.glob("image/**/*.py")), source_root / "pyproject.toml", source_root / "uv.lock"]
        hashes = {}

        for source in sources:
            relative = source.relative_to(source_root)
            target = self.directory / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()

        if (git := shutil.which("git")) is None:
            msg = "Git is required to record the experiment revision."
            raise RuntimeError(msg)

        # The executable is resolved locally and all arguments are fixed; no shell or external command is accepted.
        git_head = subprocess.run(  # noqa: S603
            [git, "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        metadata = {
            "method": method,
            "dataset": self.dataset,
            "seed": self.seed,
            "command": sys.argv,
            "git_head": git_head,
            "source_sha256": hashes,
            "search_space": hpo_space,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: version(name) for name in ("keras", "tensorflow", "numpy", "scikit-learn")},
            "selection_metric": "validation_accuracy",
            "checkpoint_metric": "validation_loss",
            "validation_split": "last 20 percent of official training examples; proxy uses first 80 percent",
            "test_policy": "evaluate validation incumbent exactly once after search",
            "evidence_status": os.getenv("FGHPO_EVIDENCE_STATUS", "exploratory"),
        }
        (self.directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        set_random_seed(self.seed)
        self.data = get_data(self.dataset)

    def __enter__(self) -> "SearchRun":
        """Use this run's recorder for the enclosing search."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Record completion or failure and release backend state without suppressing errors."""
        self._record(
            {
                "event": "run_end",
                "finished": self.finished,
                "error": repr(exception) if exception else None,
                "seconds": time.perf_counter() - self.started,
            }
        )
        backend.clear_session()

    def _record(self, event: Mapping[str, str | int | float | bool | Config | None]) -> None:
        with self.events.open("a") as stream:
            stream.write(json.dumps(event, default=lambda value: value.item()) + "\n")

    def evaluate(self, config: Config, epochs: int = 100, *, select: bool = True) -> float:
        """Train a candidate with a recorded seed and retain an improved validation incumbent.

        Each attempt gets seed `run seed + trial number`. Errors are logged and re-raised so the optimizer can apply
        its documented failure policy. Set `select=False` for lower-fidelity trials that may inform an optimizer
        but must not become its final model. Every candidate uses the same validation split.
        """
        if self.finished:
            msg = "The test set has already been evaluated; this search is closed."
            raise RuntimeError(msg)

        self.trial += 1
        trial_seed = self.seed + self.trial
        set_random_seed(trial_seed)
        started = time.perf_counter()
        candidate = self.directory / "candidate.keras"
        record: dict[str, str | int | float | bool | Config | None] = {
            "event": "trial",
            "trial": self.trial,
            "seed": trial_seed,
            "config": config,
            "epochs": epochs,
            "selection_eligible": select,
        }

        try:
            score = run_experiment(self.data, config, self.n_classes, self.dataset, epochs, checkpoint=candidate)

            record["validation_accuracy"] = score

            if select and score > self.best_accuracy:
                candidate.replace(self.checkpoint)
                self.best_accuracy = score
                record["incumbent"] = True
        except BaseException as error:
            record["error"] = repr(error)
            raise
        finally:
            record["seconds"] = time.perf_counter() - started

            self._record(record)
            candidate.unlink(missing_ok=True)
            backend.clear_session()

        return score

    def record_proxy(
        self,
        config: Config,
        convexity: float,
        *,
        seed: int | None = None,
        error: str | None = None,
    ) -> None:
        """Keep finite and failed proxy attempts, then release their backend state."""
        self.probe += 1
        self._record(
            {
                "event": "proxy",
                "attempt": self.probe,
                "seed": seed,
                "config": config,
                "convexity": float(convexity) if math.isfinite(convexity) else None,
                "error": error,
                "seconds_since_start": time.perf_counter() - self.started,
            }
        )
        backend.clear_session()

    def finish(self) -> float:
        """Evaluate the saved validation incumbent once and return test accuracy.

        A test attempt closes the search even if evaluation fails. Calling this before a successful eligible trial,
        or after a previous test attempt, raises `RuntimeError` without reading the test data.
        """
        if self.finished or not self.checkpoint.exists():
            msg = "Final evaluation needs a successful, unfinished search."
            raise RuntimeError(msg)

        self.finished = True
        model = load_model(self.checkpoint)

        if not isinstance(model, Model):
            msg = "Saved incumbent is not a Keras model."
            raise TypeError(msg)

        prediction = np.argmax(model.predict(self.data.x_test), axis=-1)
        truth = self.data.y_test

        if truth.ndim > 1:
            truth = np.argmax(truth, axis=1)

        accuracy = float(accuracy_score(truth, prediction))

        self._record(
            {
                "event": "final_test",
                "validation_accuracy": self.best_accuracy,
                "test_accuracy": accuracy,
                "seconds": time.perf_counter() - self.started,
            }
        )
        sys.stdout.write(f"Selected validation accuracy: {self.best_accuracy}; final test accuracy: {accuracy}\n")
        sys.stdout.write(f"Run evidence: {self.directory}\n")

        return accuracy
