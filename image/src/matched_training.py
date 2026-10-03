"""Train paired CIFAR phases from explicit fit and validation arrays.

Callers own dataset provenance, subset selection, candidate identity and resource authorization. This backend has no
loader or test-data API. Completed final phases restore the actual Keras validation-loss callback's best weights;
interrupted phases retain current weights and optimizer state without claiming a selected checkpoint.
"""

import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple

import keras
import numpy as np
from numpy.typing import NDArray

from image.src import util
from image.src.config import Config

type FloatArray = NDArray[np.float32]
type JsonValue = None | bool | int | float | str | Config | list[JsonValue] | dict[str, JsonValue]
LABEL_DIMENSIONS = 2
PROBABILITY_TOLERANCE = 1e-5
LEGACY_THRESHOLD = 0.1


@dataclass(frozen=True)
class TrainingArrays:
    fit_x: FloatArray
    fit_y: FloatArray
    validation_x: FloatArray
    validation_y: FloatArray


@dataclass(frozen=True)
class PhaseSeeds:
    initialization: int
    shuffle: int
    augmentation: tuple[int, int]


@dataclass(frozen=True)
class FitSettings:
    max_epochs: int = 100
    patience: int = 25
    batch_size: int = 64
    time_limit_seconds: float | None = None


@dataclass
class PhaseResult:
    status: Literal["complete", "incomplete", "failed"] = "failed"
    rms: float | None = None
    head_trace: float | None = None
    score_status: str = "not_measured"
    validation_ce: float | None = None
    validation_accuracy: float | None = None
    epochs: int = 0
    optimizer_steps: int = 0
    best_epoch: int | None = None
    stop_epoch: int | None = None
    checkpoint: Path | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


def array_hash[DType: np.generic](array: NDArray[DType]) -> str:
    """Hash array shape, dtype and C-order bytes so different layouts cannot alias silently."""
    digest = hashlib.sha256(f"{array.shape}:{array.dtype}".encode())
    digest.update(array.tobytes(order="C"))

    return digest.hexdigest()


def weights_hash(model: keras.Model) -> str:
    """Identify ordered model weights, including BatchNorm statistics, independently of ZIP timestamps."""
    return hashlib.sha256("".join(array_hash(weight) for weight in model.get_weights()).encode()).hexdigest()


def _write_json(path: Path, value: object, *, append: bool = False) -> None:
    with path.open("a" if append else "w") as stream:
        _ = stream.write(json.dumps(value, allow_nan=False, default=str) + "\n")


def _metrics(model: keras.Model, images: FloatArray, labels: FloatArray, batch_size: int) -> tuple[float, float]:
    """Evaluate compiled Keras metrics with fresh sample-weighted accumulators, including the tail."""
    model.reset_metrics()
    logs = {}

    for start in range(0, len(images), batch_size):
        logs = model.test_on_batch(
            images[start : start + batch_size], labels[start : start + batch_size], return_dict=True
        )

    loss, accuracy = float(logs["loss"]), float(logs["accuracy"])

    if not math.isfinite(loss) or not math.isfinite(accuracy) or loss < 0 or not 0 <= accuracy <= 1:
        msg = "Compiled inference metrics are invalid."
        raise ValueError(msg)

    return loss, accuracy


class HeadTrace(NamedTuple):
    terms: NDArray[np.float64]
    hidden_squared: NDArray[np.float64]
    probability_squared: NDArray[np.float64]
    disagreement: float


def head_trace_terms(probability: FloatArray, hidden: FloatArray, kernel: FloatArray, bias: FloatArray) -> HeadTrace:
    """Measure kernel-only Euclidean CE curvature with frozen features and bias.

    Float64 affine-head reconstruction must agree with model probabilities. The returned per-example terms retain
    sample weighting across arbitrary batches. Compensating feature/kernel rescaling changes this raw trace.
    """
    logits = hidden.astype(np.float64) @ kernel.astype(np.float64) + bias.astype(np.float64)
    exponential = np.exp(logits - logits.max(axis=1, keepdims=True))
    reconstructed = exponential / exponential.sum(axis=1, keepdims=True)
    disagreement = float(np.max(np.abs(reconstructed - probability)))
    hidden_squared = np.sum(hidden.astype(np.float64) ** 2, axis=1)
    probability_squared = np.sum(reconstructed**2, axis=1)
    traces = (1 - probability_squared) * hidden_squared

    if (
        not math.isfinite(disagreement)
        or disagreement > PROBABILITY_TOLERANCE
        or not np.isfinite(traces).all()
        or np.any(traces < 0)
    ):
        msg = "Head trace or affine probability reconstruction is invalid."
        raise ValueError(msg)

    return HeadTrace(traces, hidden_squared, probability_squared, disagreement)


def _probe_score(
    model: keras.Model, images: FloatArray, batch_size: int
) -> tuple[float | None, float, list[JsonValue]]:
    """Keep the original float32 batch maximum and a sample-mean exact head trace.

    Image order is the caller's saved probe-subset order, independent of training shuffles. The original infinity
    fallback is returned as `None`; explicit component flags make strict JSON retain nonfinite outcomes honestly.
    """
    inference = keras.Model(model.inputs, [model.layers[-1].output, model.layers[-2].output])
    kernel, bias = model.layers[-1].get_weights()
    denominator = np.sqrt(np.mean(kernel**2))
    best = -math.inf
    trace_sum = 0.0
    batches: list[JsonValue] = []

    for start in range(0, len(images), batch_size):
        probability, hidden = inference(images[start : start + batch_size], training=False)
        trace = head_trace_terms(probability.numpy(), hidden.numpy(), kernel, bias)
        trace_sum += float(trace.terms.sum())
        p_rms, h_rms = np.sqrt(np.mean(probability.numpy() ** 2)), np.sqrt(np.mean(hidden.numpy() ** 2))

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = float(p_rms * h_rms / denominator)

        raw = {"p_rms": float(p_rms), "h_rms": float(h_rms), "weight_rms": float(denominator), "ratio": ratio}
        batches.append(
            {
                "start": start,
                "count": min(batch_size, len(images) - start),
                "components": {key: value if math.isfinite(value) else None for key, value in raw.items()},
                "nonfinite": {key: repr(value) for key, value in raw.items() if not math.isfinite(value)},
                "head_trace": {
                    "hidden_squared_norm": trace.hidden_squared.tolist(),
                    "probability_squared_norm": trace.probability_squared.tolist(),
                    "terms": trace.terms.tolist(),
                    "probability_reconstruction_max_error": trace.disagreement,
                },
            }
        )
        if math.isfinite(ratio):
            best = max(best, ratio)

    return (best if best > 0 else None), trace_sum / len(images), batches


def _augmentation_state(model: keras.Model) -> list[JsonValue]:
    return [layer.generator.state.numpy().tolist() for layer in model.layers[:2]]


def _validate(data: TrainingArrays, settings: FitSettings, phase: str) -> None:
    """Reject malformed phase inputs before allocating a model.

    Empty/nonfinite arrays, invalid one-hot labels or impossible settings raise `ValueError`. The caller remains
    responsible for fit/validation provenance and for keeping official test arrays outside this API.
    """
    if phase not in {"probe", "final"}:
        msg = "Expected probe or final phase."
        raise ValueError(msg)

    if settings.max_epochs < 1 or settings.batch_size < 1 or settings.patience < 0:
        msg = "Epochs/batch size must be positive and patience nonnegative."
        raise ValueError(msg)

    if settings.time_limit_seconds is not None and (
        not math.isfinite(settings.time_limit_seconds) or settings.time_limit_seconds < 0
    ):
        msg = "Time limit must be finite and nonnegative."
        raise ValueError(msg)

    for images, labels in [(data.fit_x, data.fit_y), (data.validation_x, data.validation_y)]:
        if (
            len(images) < 1
            or images.shape[1:] != (32, 32, 3)
            or images.dtype != np.float32
            or labels.dtype != np.float32
            or labels.ndim != LABEL_DIMENSIONS
            or labels.shape[1] not in {10, 100}
            or len(labels) != len(images)
            or labels.shape[1] != data.fit_y.shape[1]
            or not np.isfinite(images).all()
            or not np.isfinite(labels).all()
            or not np.isin(labels, (0, 1)).all()
            or not np.all(labels.sum(axis=1) == 1)
        ):
            msg = "Expected finite nonempty 32x32 RGB images and matching one-hot class matrices."
            raise ValueError(msg)


def _train_epochs(
    model: keras.Model,
    data: TrainingArrays,
    phase: str,
    settings: FitSettings,
    rng: np.random.Generator,
    directory: Path,
    result: PhaseResult,
    started: float,
) -> keras.callbacks.EarlyStopping:
    """Run paired permutations and the actual callback, retaining each completed epoch before continuing.

    The time bound is approximate because individual batches and inference are not interrupted. Partial epochs leave
    their permutation and current optimizer state available, but never receive a completed epoch record.
    """
    maximum = 1 if phase == "probe" else settings.max_epochs
    callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=settings.patience, restore_best_weights=True)
    callback.set_model(model)
    callback.set_params({"epochs": maximum, "steps": math.ceil(len(data.fit_x) / settings.batch_size), "verbose": 0})
    callback.on_train_begin()
    model.stop_training = False
    training_seconds, inference_seconds = 0.0, 0.0
    result.status = "incomplete"

    for epoch in range(maximum):
        permutation = rng.permutation(len(data.fit_x))
        np.save(directory / f"shuffle-{epoch + 1}.npy", permutation)
        model.reset_metrics()
        training_started = time.perf_counter()

        for start in range(0, len(permutation), settings.batch_size):
            if settings.time_limit_seconds is not None and time.perf_counter() - started >= settings.time_limit_seconds:
                result.metadata["training_seconds"] = training_seconds + time.perf_counter() - training_started
                result.metadata["inference_seconds"] = inference_seconds

                return callback

            batch = permutation[start : start + settings.batch_size]
            _ = model.train_on_batch(data.fit_x[batch], data.fit_y[batch])

        training_seconds += time.perf_counter() - training_started

        if not all(np.isfinite(weight).all() for weight in model.get_weights()):
            msg = "Model weights became nonfinite."
            raise ValueError(msg)

        inference_started = time.perf_counter()
        train_ce, train_accuracy = _metrics(model, data.fit_x, data.fit_y, settings.batch_size)
        val_ce, val_accuracy = _metrics(model, data.validation_x, data.validation_y, settings.batch_size)
        inference_seconds += time.perf_counter() - inference_started
        result.epochs = epoch + 1

        if phase == "final":
            callback.on_epoch_end(epoch, {"val_loss": val_ce})
            result.best_epoch = callback.best_epoch + 1

        _write_json(
            directory / "history.jsonl",
            {
                "epoch": result.epochs,
                "train_ce": train_ce,
                "train_accuracy": train_accuracy,
                "validation_ce": val_ce,
                "validation_accuracy": val_accuracy,
                "optimizer_steps": int(model.optimizer.iterations.numpy()),
                "learning_rate": float(model.optimizer.learning_rate.numpy()),
                "shuffle_sha256": array_hash(permutation),
                "best_epoch": result.best_epoch,
                "callback_wait": callback.wait,
                "callback_stop": bool(model.stop_training),
                "augmentation_state": _augmentation_state(model),
                "elapsed_seconds": time.perf_counter() - started,
            },
            append=True,
        )

        if model.stop_training:
            result.stop_epoch = result.epochs
            break

    result.status = "complete"
    result.metadata.update({"training_seconds": training_seconds, "inference_seconds": inference_seconds})

    return callback


def _save_result(
    model: keras.Model | None,
    result: PhaseResult,
    rng: np.random.Generator,
    directory: Path,
    started: float,
    primary_error: BaseException | None,
) -> None:
    """Save terminal evidence even when checkpointing fails, without masking a training exception.

    Serialization failures propagate on a successful path. During an active error, stderr retains secondary write
    failures and the original exception remains the one observed by the caller.
    """
    save_error = None
    checkpoint_started = time.perf_counter()

    try:
        if model is not None:
            result.optimizer_steps = int(model.optimizer.iterations.numpy())
            result.metadata["augmentation_state"] = _augmentation_state(model)
            result.metadata["shuffle_rng_state"] = dict(rng.bit_generator.state)
            result.checkpoint = directory / ("model.keras" if result.status == "complete" else "partial.keras")
            model.save(result.checkpoint)
            result.metadata["checkpoint_sha256"] = hashlib.sha256(result.checkpoint.read_bytes()).hexdigest()
            result.metadata["weights_sha256"] = weights_hash(model)
    except BaseException as error:
        save_error = error
        result.status = "failed"
        result.metadata["checkpoint_error"] = repr(error)
        if primary_error is None:
            raise
    finally:
        result.metadata["checkpoint_seconds"] = time.perf_counter() - checkpoint_started
        result.metadata["elapsed_seconds"] = time.perf_counter() - started

        try:
            _write_json(directory / "result.json", asdict(result))
        except (OSError, ValueError, TypeError) as error:
            if primary_error is None and save_error is None:
                raise
            _ = sys.stderr.write(f"Secondary result-write failure: {error!r}\n")
        finally:
            keras.backend.clear_session()


def fit_phase(
    data: TrainingArrays,
    config: Config,
    phase: Literal["probe", "final"],
    seeds: PhaseSeeds,
    settings: FitSettings,
    directory: Path,
) -> PhaseResult:
    """Train one fresh phase and retain its complete evidence in a new directory.

    Probe phases train one constant-rate epoch; final phases use `max_epochs` for their cosine horizon and the installed
    EarlyStopping callback. Supplied array order determines score batches; an independent stream shuffles training.
    Approximate limits are checked before batches. Incomplete phases save current state without restoration. Exceptions
    preserve partial evidence and propagate. Existing directories cannot be overwritten. Completed final checkpoints
    contain restored model weights and terminal optimizer state, matching Keras restore-best-weights semantics.
    """
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = PhaseResult()
    model = None
    primary_error = None
    rng = np.random.default_rng(seeds.shuffle)
    result.metadata.update({"phase": phase, "seeds": asdict(seeds), "settings": asdict(settings)})
    result.metadata["config"] = config

    try:
        _validate(data, settings, phase)
        keras.backend.clear_session()
        keras.utils.set_random_seed(seeds.initialization)
        horizon = settings.max_epochs * math.ceil(len(data.fit_x) / settings.batch_size) if phase == "final" else None
        model = util.get_cifar10_model(
            config, data.fit_y.shape[1], decay_steps=horizon, augmentation_seeds=seeds.augmentation
        )
        model.build((None, 32, 32, 3))

        result.metadata.update(
            {
                "initial_weights_sha256": weights_hash(model),
                "initial_augmentation_state": _augmentation_state(model),
                "fit_x_sha256": array_hash(data.fit_x),
                "fit_y_sha256": array_hash(data.fit_y),
                "validation_x_sha256": array_hash(data.validation_x),
                "validation_y_sha256": array_hash(data.validation_y),
                "parameter_count": model.count_params(),
                "initial_optimizer_steps": int(model.optimizer.iterations.numpy()),
                "decay_steps": horizon,
                "backend_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "util_source_sha256": hashlib.sha256(Path(util.__file__).read_bytes()).hexdigest(),
                "util_import_path": str(Path(util.__file__).resolve()),
            }
        )
        _write_json(directory / "phase.json", result.metadata)
        callback = _train_epochs(model, data, phase, settings, rng, directory, result, started)

        if result.status == "complete":
            if phase == "final":
                callback.on_train_end()
                result.metadata["best_monitored_ce"] = float(callback.best)

            inference_started = time.perf_counter()
            result.validation_ce, result.validation_accuracy = _metrics(
                model, data.validation_x, data.validation_y, settings.batch_size
            )

            if phase == "probe":
                result.rms, result.head_trace, batches = _probe_score(model, data.fit_x, settings.batch_size)
                result.score_status = "finite_positive" if result.rms is not None else "positive_infinity_fallback"
                result.metadata["score_batches"] = batches
                result.metadata["legacy_threshold_eligible"] = result.rms is not None and result.rms > LEGACY_THRESHOLD

            result.metadata["final_inference_seconds"] = time.perf_counter() - inference_started
    except BaseException as error:
        primary_error = error
        result.status = "failed"
        result.metadata["error"] = repr(error)
        raise
    finally:
        _save_result(model, result, rng, directory, started, primary_error)

    return result
