"""Profile nine frozen CIFAR candidates without retaining calibration quality metrics.

Run from the repository root with PROTOCOL EXPECTED_SHA256 INPUT_DIRECTORY NEW_OUTPUT_DIRECTORY. The reviewed
protocol must identify all source and input hashes before execution; checkpoint state is calibration-only.
"""

import hashlib
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from image.src.config import Config
from image.src.matched_training import FitSettings, TrainingArrays, _metrics, _probe_score, _validate, weights_hash
from image.src.util import get_cifar10_model

BATCH_SIZE = 64
WARMUP_BATCHES = 2
TIMED_BATCHES = 16
FIT_ROWS = 1500
VALIDATION_ROWS = 1024
CASES = 9
COOPERATIVE_SECONDS = 600
EXPECTED_VERSIONS = {"python": "3.12", "keras": "3.13.2", "tensorflow": "2.16.2", "numpy": "1.26.4"}


@dataclass(frozen=True)
class Case:
    dataset: str
    repetition_id: int
    candidate_id: str
    purpose: str
    config: Config
    initialization: int
    shuffle: int
    augmentation_root: int
    translation: int
    flip: int
    forward_mac_per_image: int


@dataclass
class Record:
    case: Case
    status: str = "incomplete"
    timings: dict[str, float] = field(default_factory=dict)
    parameters: int = 0
    trainable_parameters: int = 0
    initial_weights_sha256: str = ""
    final_weights_sha256: str = ""
    checkpoint_sha256: str = ""
    checkpoint_bytes: int = 0
    optimizer_updates: int | None = None
    optimizer_counter_error: str | None = None
    gpu_peak_bytes: int | None = None
    memory_error: str | None = None
    error: str | None = None


def write_json(path: Path, value: object) -> None:
    _ = path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def timed[T](record: Record, name: str, deadline: float, action: Callable[[], T]) -> T:
    if time.perf_counter() >= deadline:
        msg = "Calibration cooperative deadline reached."
        raise TimeoutError(msg)

    started = time.perf_counter()

    try:
        value = action()
    finally:
        record.timings[name] = time.perf_counter() - started

    if time.perf_counter() >= deadline:
        msg = "Calibration cooperative deadline reached after an indivisible operation."
        raise TimeoutError(msg)

    return value


def load_arrays(path: Path, classes: int) -> TrainingArrays:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"fit_x", "fit_y", "validation_x", "validation_y"}:
            msg = "Calibration archives must contain exactly the four training-only array keys."
            raise ValueError(msg)

        data = TrainingArrays(**{name: archive[name] for name in archive.files})

    _validate(data, FitSettings(), "probe")

    if data.fit_x.shape != (FIT_ROWS, 32, 32, 3) or data.validation_x.shape != (VALIDATION_ROWS, 32, 32, 3):
        msg = "Calibration requires exactly 1500 fit and 1024 validation rows."
        raise ValueError(msg)

    if data.fit_y.shape[1] != classes:
        msg = "Calibration class count differs from the frozen candidate."
        raise ValueError(msg)

    return data


def build(case: Case) -> keras.Model:
    keras.backend.clear_session()
    keras.utils.set_random_seed(case.initialization)
    model = get_cifar10_model(
        case.config,
        n_class=10 if case.dataset == "cifar10" else 100,
        decay_steps=62500,
        augmentation_seeds=(case.translation, case.flip),
    )
    model.build((None, 32, 32, 3))

    return model


def profile(case: Case, archive: Path, directory: Path, deadline: float) -> None:
    """Retain each completed stage and abort the calibration if any case cannot finish.

    Metrics and scores are computed only by the existing backend helpers and immediately discarded. No calibration
    checkpoint initializes a research run. The deadline is checked between indivisible backend operations.
    """
    record = Record(case)
    checkpoint = directory / "calibration-only.keras"
    model: keras.Model | None = None

    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
        data = timed(
            record,
            "load_validate_arrays",
            deadline,
            lambda: load_arrays(archive, 10 if case.dataset == "cifar10" else 100),
        )
        model = timed(record, "build_initialize", deadline, lambda: build(case))
        record.parameters = model.count_params()
        record.trainable_parameters = sum(math.prod(weight.shape) for weight in model.trainable_weights)
        record.initial_weights_sha256 = timed(record, "initial_weight_hash", deadline, lambda: weights_hash(model))
        order = np.random.default_rng(case.shuffle).permutation(FIT_ROWS)

        for batch in range(WARMUP_BATCHES + TIMED_BATCHES):
            rows = order[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
            stage = "warmup" if batch < WARMUP_BATCHES else "timed"
            _ = timed(
                record,
                f"train_{stage}_{batch:02d}",
                deadline,
                lambda rows=rows: model.train_on_batch(data.fit_x[rows], data.fit_y[rows]),
            )

        for index in range(3):
            stage = "warmup" if index == 0 else "timed"
            _ = timed(
                record,
                f"inference_{stage}_{index}",
                deadline,
                lambda: _metrics(model, data.validation_x, data.validation_y, BATCH_SIZE),
            )

        _ = timed(record, "probe_score_1500", deadline, lambda: _probe_score(model, data.fit_x, BATCH_SIZE))
        _ = timed(record, "checkpoint_save", deadline, lambda: model.save(checkpoint))
        record.checkpoint_bytes = checkpoint.stat().st_size
        record.checkpoint_sha256 = timed(
            record, "checkpoint_hash", deadline, lambda: hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        )
        record.final_weights_sha256 = timed(record, "final_weight_hash", deadline, lambda: weights_hash(model))
        record.status = "complete"
    except Exception as error:
        record.status = "incomplete" if isinstance(error, TimeoutError) else "failed"
        record.error = f"{type(error).__name__}: {error}"
        raise
    finally:
        if model is not None:
            try:
                record.optimizer_updates = int(model.optimizer.iterations.numpy())
            except (ValueError, RuntimeError, tf.errors.OpError) as error:
                record.optimizer_counter_error = f"{type(error).__name__}: {error}"

        try:
            record.gpu_peak_bytes = int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
        except (ValueError, tf.errors.OpError) as error:
            record.memory_error = f"{type(error).__name__}: {error}"

        write_json(directory / "timing.json", asdict(record))


def main() -> None:
    """Require frozen inputs and GPU execution, then profile all nine cases in their declared order.

    Output directories are created exclusively. Errors stop the cohort without replacement cases; prior timing
    records and calibration-only checkpoints remain available. The external launcher must enforce a 900-second cap.
    """
    deadline = time.perf_counter() + COOPERATIVE_SECONDS
    protocol_path, expected, inputs, output = sys.argv[1:]
    raw = Path(protocol_path).read_bytes()

    if hashlib.sha256(raw).hexdigest() != expected:
        msg = "Protocol hash differs from the externally frozen calibration protocol."
        raise ValueError(msg)

    protocol = json.loads(raw)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "protocol.json", protocol)

    for name, digest in protocol["source_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            msg = f"Calibration source hash differs: {name}."
            raise ValueError(msg)

    if set(protocol["input_sha256"]) != {"cifar10.npz", "cifar100.npz"} or len(protocol["cases"]) != CASES:
        msg = "Freeze both training-only input archives and all nine cases before execution."
        raise ValueError(msg)

    for name, digest in protocol["input_sha256"].items():
        if hashlib.sha256((Path(inputs) / name).read_bytes()).hexdigest() != digest:
            msg = f"Calibration input hash differs: {name}."
            raise ValueError(msg)

    versions = {
        "python": ".".join(map(str, sys.version_info[:2])),
        "keras": keras.__version__,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
    }
    write_json(directory / "versions.json", versions)

    if versions != EXPECTED_VERSIONS:
        msg = f"Calibration runtime versions differ from the frozen environment: {versions!r}."
        raise RuntimeError(msg)

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    keras.mixed_precision.set_global_policy("float32")
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.set_synchronous_execution(True)
    devices = tf.config.list_physical_devices("GPU")

    if len(devices) != 1:
        msg = "Calibration requires exactly one GPU; CPU fallback is prohibited."
        raise RuntimeError(msg)

    tf.config.experimental.set_memory_growth(devices[0], enable=True)
    details = tf.config.experimental.get_device_details(devices[0])
    write_json(
        directory / "runtime.json",
        {"tensorflow": tf.__version__, "keras": keras.__version__, "gpu": details, "protocol_sha256": expected},
    )

    for index, raw_case in enumerate(protocol["cases"]):
        case = Case(**raw_case)
        case_directory = directory / f"case-{index:02d}-{case.dataset}-{case.candidate_id}"
        case_directory.mkdir(exist_ok=False)
        profile(case, Path(inputs) / f"{case.dataset}.npz", case_directory, deadline)

    write_json(directory / "completion.json", {"complete": True, "cases": CASES, "calibration_only": True})


if __name__ == "__main__":
    main()
