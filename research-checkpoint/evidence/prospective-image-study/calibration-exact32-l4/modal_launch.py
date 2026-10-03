"""Run one frozen calibration in an ephemeral Modal app with bounded local supervision.

Invoke with the new local output directory. The child mode is internal; the parent reserves 30 seconds of its
900-second budget for stopping the app. No cloud action occurs on import. Outputs contain only JSON and logs.
"""

import hashlib
import io
import json
import logging
import os
import re
import signal
import site
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

import modal

PAYLOAD = Path(".cloudexp/research-audit-20261002/prospective-image-study/exact32-l4-calibration/payload.tar.gz")
PAYLOAD_SHA = "ef2c246211741aa0c0ff60026baa5fcbbe80ccc850673cbfbd16df4edf9fd13c"
PROTOCOL_SHA = "113b8af3c5e1596a39ea7395d3a08de94fb8458b65c157b66c388e9cf5ea51a9"
STUDY = "sweep-results/research-audit-20261002/prospective-image-study"
RETURN_LIMIT = 100 * 1024 * 1024
WORKER_SECONDS = 870
APP_ID = re.compile(r"\bap-[A-Za-z0-9]{22}\b")


def remote_calibration(deadline: float) -> tuple[bytes, str, int]:
    """Return JSON/log evidence, including partial output from a failed calibration.

    The absolute deadline survives provider preemption retries. Only the byte-verified frozen payload is extracted;
    checkpoint files remain ephemeral and cannot be returned. The subprocess includes import time in its limit.
    """
    attempt = {
        "event": "fghpo-calibration-attempt",
        "attempt_id": str(uuid.uuid4()),
        "started": time.time(),
        "task_id": os.environ.get("MODAL_TASK_ID", "unknown"),
    }
    _ = sys.stdout.write(json.dumps(attempt) + "\n")
    sys.stdout.flush()

    if time.time() >= deadline - 20:
        message = "The shared calibration deadline has expired."
        raise TimeoutError(message)

    if hashlib.sha256(Path("/payload.tar.gz").read_bytes()).hexdigest() != PAYLOAD_SHA:
        message = "Remote payload differs from the frozen archive."
        raise ValueError(message)

    with tempfile.TemporaryDirectory(prefix="fghpo-") as temporary:
        project = Path(temporary)

        with tarfile.open("/payload.tar.gz", "r:gz") as archive:
            archive.extractall(project, filter="data")

        output = project / "results"
        command = [
            sys.executable,
            f"{STUDY}/calibration-exact32-l4/calibrate.py",
            f"{STUDY}/calibration-exact32-l4/timing-protocol.json",
            PROTOCOL_SHA,
            f"{STUDY}/calibration-inputs",
            str(output),
        ]
        libraries = [str(path) for root in site.getsitepackages() for path in Path(root).glob("nvidia/*/lib")]
        binaries = [str(path) for root in site.getsitepackages() for path in Path(root).glob("nvidia/cuda_nvcc/bin")]
        environment = {
            **os.environ,
            "PYTHONPATH": str(project),
            "LD_LIBRARY_PATH": os.pathsep.join([*libraries, os.environ.get("LD_LIBRARY_PATH", "")]),
            "PATH": os.pathsep.join([*binaries, os.environ.get("PATH", "")]),
        }
        inventory: dict[str, str | list[str]] = {"nvidia_library_paths": libraries, "nvidia_binary_paths": binaries}

        for name, arguments in (
            ("gpu", ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"]),
            ("packages", [sys.executable, "-m", "pip", "list", "--format=json"]),
        ):
            inventory[name] = subprocess.run(  # noqa: S603 - Two fixed inventory commands; no shell.
                arguments, capture_output=True, text=True, timeout=10, check=True
            ).stdout

        started = time.time()
        status = {
            "started": started,
            "deadline": deadline,
            "payload_sha256": PAYLOAD_SHA,
            "timed_out": False,
            "attempt": attempt,
        }

        with (project / "stdout.log").open("wb") as stdout, (project / "stderr.log").open("wb") as stderr:
            try:
                result = subprocess.run(  # noqa: S603 - Fixed executable/argument list; no shell.
                    command,
                    cwd=project,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=min(600, max(1, deadline - time.time() - 20)),
                    check=False,
                )
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
                status["timed_out"] = True

        status["finished"] = time.time()
        status["returncode"] = returncode
        files = [(path.relative_to(project).as_posix(), path.read_bytes()) for path in sorted(output.rglob("*.json"))]

        for name in ("stdout.log", "stderr.log"):
            path = project / name
            status[f"{name}_bytes"] = path.stat().st_size

            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 16 * 1024 * 1024))
                files.append((name, stream.read()))

        files.append(("inventory.json", json.dumps(inventory).encode()))
        files.append(("remote-status.json", json.dumps(status, sort_keys=True).encode()))
        files.append(
            (
                "file-sha256.json",
                json.dumps({name: hashlib.sha256(content).hexdigest() for name, content in files}).encode(),
            )
        )
        buffer = io.BytesIO()

        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name, content in files:
                member = tarfile.TarInfo(name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))

        payload = buffer.getvalue()

        if len(payload) > RETURN_LIMIT or sum(len(content) for _, content in files) > RETURN_LIMIT:
            message = "JSON/log return exceeds 100 MiB."
            raise ValueError(message)

        return payload, hashlib.sha256(payload).hexdigest(), returncode


def verify_evidence(payload: bytes, digest: str) -> str:
    """Check sizes, member paths, types and per-file hashes without extracting remote files."""
    if len(payload) > RETURN_LIMIT or hashlib.sha256(payload).hexdigest() != digest:
        message = "Returned evidence failed its size/hash check."
        raise ValueError(message)

    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]

        if len(names) != len(set(names)) or sum(member.size for member in members) > RETURN_LIMIT:
            message = "Duplicate evidence members or excessive uncompressed size."
            raise ValueError(message)

        for member in members:
            parts = Path(member.name).parts
            allowed = member.name in {
                "stdout.log",
                "stderr.log",
                "remote-status.json",
                "file-sha256.json",
                "inventory.json",
            }
            allowed = allowed or (bool(parts) and parts[0] == "results" and member.name.endswith(".json"))

            if not allowed or not member.isfile() or member.name.startswith("/") or ".." in parts:
                message = "Unexpected evidence member path or type."
                raise ValueError(message)

        contents: dict[str, bytes] = {}

        for member in members:
            stream = archive.extractfile(member)

            if stream is None:
                message = "Evidence member has no regular file stream."
                raise ValueError(message)

            with stream:
                contents[member.name] = stream.read()

        hashes = json.loads(contents.pop("file-sha256.json"))

        if {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()} != hashes:
            message = "Evidence file hashes differ from the returned manifest."
            raise ValueError(message)

        return json.loads(contents["remote-status.json"])["attempt"]["attempt_id"]


def child(output: Path, deadline: float) -> None:
    """Make exactly one remote call; cancel its containers if result retrieval fails."""
    if modal.__version__ != "1.5.5" or sys.version_info[:2] != (3, 12):
        message = "The reviewed launcher requires Python 3.12 and Modal 1.5.5."
        raise RuntimeError(message)

    class AppIdHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()

            if message.startswith("Created new app with id ") and (match := APP_ID.search(message)):
                with (output / "app-id.txt").open("x") as stream:
                    stream.write(match.group())
                    stream.flush()
                    os.fsync(stream.fileno())

    logger = logging.getLogger("modal-client")

    for handler in logger.handlers:
        handler.setLevel(logger.level)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(AppIdHandler())
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .run_commands(
            "timeout 300 python -m pip install 'tensorflow[and-cuda]==2.16.2' keras==3.13.2 numpy==1.26.4 "
            "scipy==1.12.0 scikit-learn==1.8.0"
        )
        .add_local_file(PAYLOAD.resolve(), "/payload.tar.gz")
    )
    app = modal.App("fghpo-cal-exact32-l4-20261003")
    function = app.function(
        image=image,
        gpu="L4",
        cpu=(2, 2),
        memory=(8192, 16384),
        min_containers=0,
        max_containers=1,
        buffer_containers=0,
        scaledown_window=2,
        retries=0,
        startup_timeout=120,
        timeout=720,
        serialized=True,
        include_source=False,
    )(remote_calibration)

    with modal.enable_output(), app.run():
        (output / "app-id-after-build.txt").write_text(str(app.app_id))
        call = function.spawn(deadline)
        (output / "call-id.txt").write_text(call.object_id)

        try:
            payload, digest, returncode = call.get(timeout=max(1, deadline - time.time() + 20))

            attempt_id = verify_evidence(payload, digest)

            with (output / "results.tar.gz").open("xb") as stream:
                stream.write(payload)

            (output / "result.json").write_text(
                json.dumps(
                    {"sha256": digest, "bytes": len(payload), "returncode": returncode, "attempt_id": attempt_id}
                )
            )
        except BaseException:
            call.cancel(terminate_containers=True)
            raise

        if returncode != 0:
            message = "Calibration failed; partial JSON/log evidence was retained."
            raise RuntimeError(message)


def confirmed_stopped(app_id: str | None, output: Path) -> bool:
    """Confirm a stopped app and an empty exact-app container query within one shared ten-second budget.

    App-list task counts can lag behind shutdown. Both raw responses are retained, and a nonempty container query
    leaves cleanup unconfirmed even when the app lifecycle reports stopped.
    """
    deadline = time.monotonic() + 10
    status = subprocess.run(  # noqa: S603 - Fixed executable/argument list; no shell.
        [sys.executable, "-m", "modal", "app", "list", "--json"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    (output / "app-status.json").write_text(status.stdout)
    matching = [row for row in json.loads(status.stdout) if row["app_id"] == app_id]

    if len(matching) != 1 or matching[0]["state"] != "stopped":
        return False

    remaining = deadline - time.monotonic()

    if remaining <= 0:
        raise subprocess.TimeoutExpired(status.args, 10, output=status.stdout)

    containers = subprocess.run(  # noqa: S603 - Exact app ID and fixed CLI argument list; no shell.
        [sys.executable, "-m", "modal", "container", "list", "--app-id", str(app_id), "--json"],
        capture_output=True,
        text=True,
        timeout=remaining,
        check=True,
    )
    (output / "container-status.json").write_text(containers.stdout)

    return json.loads(containers.stdout) == []


def supervise(output: Path) -> None:
    """Record the app ID during builds and stop the app on success, failure, or the local deadline.

    The worker has 870 seconds including image construction. Cleanup gets bounded subprocess waits within the
    remaining 30 seconds. A failed stop is retained for manual follow-up; it is never treated as confirmed cleanup.
    """
    started = time.monotonic()
    deadline = time.time() + 840

    if hashlib.sha256(PAYLOAD.read_bytes()).hexdigest() != PAYLOAD_SHA:
        message = "Local payload differs from the reviewed archive."
        raise ValueError(message)

    output.mkdir(parents=True, exist_ok=False)
    ledger: dict[str, str | float | bool | None | list[str]] = {
        "started": time.time(),
        "deadline": deadline,
        "payload_sha256": PAYLOAD_SHA,
        "app_id": None,
    }
    log_path = output / "modal.log"

    with log_path.open("wb") as log:
        worker = subprocess.Popen(  # noqa: S603 - This script and the current interpreter; no shell.
            [sys.executable, str(Path(__file__).resolve()), "--child", str(output.resolve()), str(deadline)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        try:
            worker.wait(timeout=max(1, WORKER_SECONDS - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            ledger["supervisor_timeout"] = True
        finally:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGINT)

                try:
                    worker.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait(timeout=2)

            captured = (
                (output / "app-id.txt").read_text()
                if (output / "app-id.txt").exists()
                else log_path.read_text(errors="replace")
            )
            match = APP_ID.search(captured)
            app_id = match.group() if match else None

            ledger.update({"app_id": app_id, "worker_returncode": worker.returncode})

            if app_id is not None:
                with (output / "stop.log").open("wb") as stop_log:
                    try:
                        stopped = subprocess.run(  # noqa: S603 - Fixed executable/argument list; no shell.
                            [sys.executable, "-m", "modal", "app", "stop", app_id, "--yes"],
                            stdout=stop_log,
                            stderr=subprocess.STDOUT,
                            timeout=10,
                            check=False,
                        )
                        ledger["stop_returncode"] = stopped.returncode
                    except subprocess.TimeoutExpired:
                        ledger["stop_error"] = "App stop exceeded its 10-second limit; cleanup is unconfirmed."

            try:
                ledger["cleanup_verified"] = confirmed_stopped(app_id, output)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as error:
                ledger["cleanup_verified"] = False
                ledger["status_error"] = str(error)

            attempts = set(re.findall(r'"attempt_id": "([0-9a-f-]{36})"', log_path.read_text(errors="replace")))
            returned = json.loads((output / "result.json").read_text()) if (output / "result.json").exists() else {}
            ledger["observed_attempts"] = sorted(attempts)
            ledger["cost_gate_eligible"] = (
                attempts == {returned.get("attempt_id")} and worker.returncode == 0 and ledger["cleanup_verified"]
            )
            ledger["lost_partial_work"] = "unknown if interrupted before evidence return"
            ledger["elapsed_seconds"] = time.monotonic() - started
            (output / "ledger.json").write_text(json.dumps(ledger, indent=2))

    if worker.returncode != 0 or not ledger["cleanup_verified"] or not ledger["cost_gate_eligible"]:
        message = "Calibration launcher or cleanup failed; inspect the retained ledger and logs."
        raise RuntimeError(message)


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(Path(sys.argv[2]), float(sys.argv[3]))
    else:
        supervise(Path(sys.argv[1]))
