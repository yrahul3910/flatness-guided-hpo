"""Stream a frozen scientific block once; persist each phase before permitting the next.

No cloud action occurs on import. Invoke JOB_JSON AUDIT_DIR --block DATASET-REPETITION.
Each independent controller includes image build and verified app shutdown in its fixed deadline.
"""
import argparse
import hashlib
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

CHUNK = 8 * 1024 * 1024
APP_ID = re.compile(r"\bap-[A-Za-z0-9]{22}\b")
HERE = "sweep-results/research-audit-20261002/prospective-image-study/scientific-execution"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def durable(path, value):
    path = Path(path)
    with path.with_suffix(path.suffix + ".tmp").open("xb") as stream:
        stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(stream.name, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def frame_bytes(frame):
    return len(frame.get("data", b"")) + len(encoded({k: v for k, v in frame.items() if k != "data"}))


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=2)


def claim_start(queue, permit):
    """An absent or mismatched permit is fatal; this function never replaces it."""
    if queue.get(partition="start", block=False) != permit or time.time() >= permit["deadline"]:
        raise RuntimeError("No valid one-use START permit remains.")


def remote_block(job, block, deadline, queue, permit):
    claim_start(queue, permit)  # Before imports, subprocesses or any TensorFlow/model allocation.
    attempt = str(uuid.uuid4())
    print(json.dumps({"attempt_id": attempt, "admitted": True}), flush=True)
    sent, charged = {}, 0
    with tempfile.TemporaryDirectory(prefix="fghpo-science-") as temporary:
        project = Path(temporary)
        if sha("/payload.tar.gz") != job["payload_sha256"]:
            raise ValueError("Remote payload hash mismatch.")
        with tarfile.open("/payload.tar.gz", "r:gz") as archive:
            archive.extractall(project, filter="data")
        control = project / "control"; control.mkdir()
        durable(control / "block.json", block)
        libraries = [str(p) for root in site.getsitepackages() for p in Path(root).glob("nvidia/*/lib")]
        binaries = [str(p) for root in site.getsitepackages() for p in Path(root).glob("nvidia/cuda_nvcc/bin")]
        environment = {**os.environ, "PYTHONPATH": str(project), "LD_LIBRARY_PATH": os.pathsep.join(libraries),
                       "PATH": os.pathsep.join([*binaries, os.environ.get("PATH", "")])}
        command = [sys.executable, f"{HERE}/run_block.py", str(control / "block.json"),
                   job["training_assignment_path"], str(deadline), str(control)]
        with (control / "stdout.log").open("wb") as stdout, (control / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(command, cwd=project, env=environment, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            sequence = 0
            try:
                while time.time() < deadline - 5:
                    request_path = control / "request.json"
                    request = json.loads(request_path.read_text()) if request_path.exists() else None
                    if not request or request["sequence"] != sequence:
                        if process.poll() is not None:
                            raise RuntimeError("Training process exited without a verified terminal bundle.")
                        time.sleep(.1); continue
                    root = project / block["output"]
                    root.mkdir(parents=True, exist_ok=True)
                    # The child is blocked. Snapshot its closed/append-only logs into the output tree.
                    for name in ("stdout.log", "stderr.log"):
                        (root / name).write_bytes((control / name).read_bytes())
                    files = []
                    for path in sorted(root.rglob("*")):
                        if path.is_symlink():
                            raise ValueError("Symlink in scientific artifacts.")
                        if not path.is_file():
                            continue
                        name = path.relative_to(project).as_posix()
                        size, stamp = path.stat().st_size, path.stat().st_mtime_ns
                        previous = sent.get(name, (0, None, None))
                        if (size, stamp) == (previous[0], previous[2]):
                            continue
                        offset = previous[0]
                        if previous[1] and (path.suffix not in {".jsonl", ".log"} or size < offset):
                            raise ValueError("A transferred immutable artifact changed.")
                        with path.open("rb") as stream:
                            prefix = hashlib.sha256(stream.read(offset)).hexdigest()
                            if previous[1] and prefix != previous[1]:
                                raise ValueError("An append-only artifact prefix changed.")
                            delta = hashlib.file_digest(stream, "sha256").hexdigest()
                        files.append({"path": name, "offset": offset, "size": size - offset,
                                      "sha256": delta, "base_sha256": previous[1], "final_sha256": sha(path),
                                      "mtime_ns": stamp})
                    manifest = {**request, "attempt_id": attempt, "files": files}
                    digest = hashlib.sha256(encoded(manifest)).hexdigest()
                    frames = [{"kind": "begin", "manifest": manifest}]
                    projected = sum(f["size"] for f in files) + frame_bytes(frames[0])
                    if charged + projected > block["max_return_bytes"]:
                        raise ValueError("Declared phase transfer exceeds the frozen byte limit.")
                    for index in range(len(files) + 2):
                        if index == 0:
                            outgoing = iter(frames)
                        elif index == len(files) + 1:
                            outgoing = iter([{"kind": "end", "digest": digest}])
                        else:
                            item = files[index - 1]
                            def chunks(item=item):
                                with (project / item["path"]).open("rb") as stream:
                                    stream.seek(item["offset"])
                                    for position in range(0, item["size"], CHUNK):
                                        yield {"kind": "data", "path": item["path"], "position": position,
                                               "data": stream.read(min(CHUNK, item["size"] - position))}
                            outgoing = chunks()
                        for frame in outgoing:
                            charged += frame_bytes(frame)
                            if charged > block["max_return_bytes"] or time.time() >= deadline - 5:
                                raise RuntimeError("Transfer byte or time limit reached.")
                            yield frame
                    expected = {"attempt_id": attempt, "sequence": sequence, "digest": digest}
                    ack = queue.get(partition="ack", timeout=max(.1, deadline - time.time() - 5))
                    if ack != expected:
                        raise RuntimeError("Missing or mismatched durable phase ACK.")
                    for item in files:
                        sent[item["path"]] = (item["offset"] + item["size"], item["final_sha256"], item["mtime_ns"])
                    (control / f"ack-{sequence}").touch(exist_ok=False)
                    sequence += 1
                    if request["kind"] == "terminal":
                        process.wait(timeout=max(.1, min(5, deadline - time.time())))
                        if process.returncode or request["status"] != "complete":
                            raise RuntimeError("Scientific block retained an incomplete terminal bundle.")
                        return
                raise TimeoutError("Unchanged absolute scientific deadline expired.")
            finally:
                (control / "abort").touch()
                stop_process(process)


class Receiver:
    """Accept one ordered stream; publish only byte-verified files, then fsync its receipt."""
    def __init__(self, root, audit, block):
        self.root, self.audit, self.block = Path(root), Path(audit), block
        self.used, self.sequence, self.attempt, self.manifest = 0, 0, None, None
        self.terminal = False

    def accept(self, frame):
        self.used += frame_bytes(frame)
        if self.used > self.block["max_return_bytes"] or self.terminal:
            raise ValueError("Transfer exceeded its bound or followed its terminal bundle.")
        if frame["kind"] == "begin":
            manifest = frame["manifest"]
            if self.manifest or manifest["sequence"] != self.sequence:
                raise ValueError("Duplicate or unordered phase.")
            self.attempt = self.attempt or manifest["attempt_id"]
            if manifest["attempt_id"] != self.attempt:
                raise ValueError("A replacement attempt cannot continue a scientific block.")
            self.pending = self.audit / f"phase-{self.sequence:03d}"; self.pending.mkdir()
            self.manifest, self.files = manifest, {}
            for item in manifest["files"]:
                path = Path(item["path"])
                if path.is_absolute() or ".." in path.parts or not path.is_relative_to(self.block["output"]):
                    raise ValueError("Returned path escapes the assigned run.")
                if str(path) in self.files or min(item["offset"], item["size"]) < 0:
                    raise ValueError("Duplicate or invalid file declaration.")
                self.files[str(path)] = {**item, "received": 0, "part": self.pending / f"{len(self.files)}.part"}
                self.files[str(path)]["part"].touch(exist_ok=False)
            if self.used + sum(item["size"] for item in self.files.values()) > self.block["max_return_bytes"]:
                raise ValueError("Declared transfer exceeds allowance.")
        elif frame["kind"] == "data":
            item = self.files[frame["path"]]; data = frame["data"]
            if frame["position"] != item["received"] or not 0 < len(data) <= min(CHUNK, item["size"] - item["received"]):
                raise ValueError("Corrupt, duplicate or oversized chunk.")
            with item["part"].open("ab") as stream:
                stream.write(data)
            item["received"] += len(data)
        elif frame["kind"] == "end":
            digest = hashlib.sha256(encoded(self.manifest)).hexdigest()
            if frame["digest"] != digest:
                raise ValueError("Phase manifest hash mismatch.")
            for name, item in self.files.items():
                if item["received"] != item["size"] or sha(item["part"]) != item["sha256"]:
                    raise ValueError("Incomplete or corrupt file content.")
                target = self.root / name; target.parent.mkdir(parents=True, exist_ok=True)
                if any(p.is_symlink() for p in [target, *target.parents]):
                    raise ValueError("Local artifact target contains a symlink.")
                if item["offset"]:
                    if target.stat().st_size != item["offset"] or sha(target) != item["base_sha256"]:
                        raise ValueError("Local append prefix mismatch.")
                elif target.exists():
                    raise ValueError("Refusing to replace an existing scientific artifact.")
                with target.open("ab" if item["offset"] else "xb") as stream, item["part"].open("rb") as delta:
                    while chunk := delta.read(CHUNK):
                        stream.write(chunk)
                    stream.flush(); os.fsync(stream.fileno())
                if sha(target) != item["final_sha256"]:
                    raise ValueError("Final artifact hash mismatch.")
            durable(self.pending / "receipt.json", {"manifest": self.manifest, "raw_bytes": self.used,
                                                   "wire_equivalent_bytes": 2 * self.used, "digest": digest})
            ack = {"attempt_id": self.attempt, "sequence": self.sequence, "digest": digest}
            self.terminal = self.manifest["kind"] == "terminal"
            if self.terminal and self.manifest["status"] == "complete":
                output = self.root / self.block["output"]
                end, arms = (json.loads((output / name).read_text()) for name in ("run-end.json", "arms.json"))
                if end["status"] != "complete" or set(arms) != {"rms", "ce", "trace", "random"} or any(v["status"] != "complete" for v in arms.values()):
                    raise ValueError("Terminal metadata does not describe four complete arms.")
            self.sequence += 1; self.manifest = None
            return ack
        else:
            raise ValueError("Unknown stream frame.")
        return None

    def finish(self):
        if not self.terminal or self.manifest:
            raise RuntimeError("Stream ended without a verified terminal bundle.")


def child(job, block, audit, deadline):
    import modal
    if modal.__version__ != "1.5.5" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Expected reviewed Modal 1.5.5 and Python 3.12.")
    class AppIdHandler(logging.Handler):
        def emit(self, record):
            if record.getMessage().startswith("Created new app with id ") and (match := APP_ID.search(record.getMessage())):
                durable(audit / "app-id.json", {"app_id": match.group()})
    logger = logging.getLogger("modal-client"); logger.setLevel(logging.DEBUG); logger.addHandler(AppIdHandler())
    image = modal.Image.debian_slim(python_version="3.12").run_commands(
        "timeout 300 python -m pip install 'tensorflow[and-cuda]==2.16.2' keras==3.13.2 numpy==1.26.4 "
        "scipy==1.12.0 scikit-learn==1.8.0").add_local_file(Path(job["payload_path"]).resolve(), "/payload.tar.gz")
    app = modal.App(f"fghpo-science-{block['dataset']}-{block['repetition_id']}")
    function = app.function(image=image, gpu="L4", cpu=(2, 2), memory=(8192, 16384), min_containers=0,
                            max_containers=1, buffer_containers=0, scaledown_window=2, startup_timeout=120,
                            timeout=block["provider_seconds"], serialized=True, include_source=False)(remote_block)
    permit = {"nonce": str(uuid.uuid4()), "deadline": deadline, "payload_sha256": job["payload_sha256"],
              "block": f"{block['dataset']}-{block['repetition_id']}"}
    receiver = Receiver(Path.cwd(), audit, block)
    with modal.Queue.ephemeral() as queue, modal.enable_output(), app.run():
        durable(audit / "app-id-after-build.json", {"app_id": app.app_id})
        queue.put(permit, partition="start")
        for frame in function.remote_gen(job, block, deadline, queue, permit):
            if ack := receiver.accept(frame):
                queue.put(ack, partition="ack")
                print(json.dumps({"phase_verified": ack, "raw_bytes": receiver.used}), flush=True)
        receiver.finish()
        durable(audit / "result.json", {"attempt_id": receiver.attempt, "raw_bytes": receiver.used, "terminal": True})


def supervise(job_path, audit, block):
    started = time.time(); deadline = started + block["hard_seconds"] - 30
    job = json.loads(job_path.read_text())
    if sha(job["payload_path"]) != job["payload_sha256"] or sum(b["hard_seconds"] for b in job["blocks"]) * 1.021392 / 3600 > 42:
        raise ValueError("Payload or cumulative scientific budget mismatch.")
    audit.mkdir(parents=True, exist_ok=False)
    Path(block["output"]).mkdir(parents=True, exist_ok=False)  # Permanent local exclusive attempt fence.
    durable(audit / "launch.json", {"started": started, "deadline": deadline, "job_sha256": sha(job_path), "block": block})
    ledger = {"started": started, "deadline": deadline, "cleanup_verified": False}
    print(json.dumps({"phase": "launch", "block": block["output"], "deadline": deadline}), flush=True)
    with (audit / "modal.log").open("wb") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(job_path.resolve()),
                                    str(audit.resolve()), "--block", f"{block['dataset']}-{block['repetition_id']}",
                                    "--child-deadline", str(deadline)], stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            process.wait(timeout=max(.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            ledger["error"] = "Local deadline expired."
        finally:
            stop_process(process)
            app_file = audit / "app-id.json"
            app_id = json.loads(app_file.read_text())["app_id"] if app_file.exists() else None
            ledger.update({"app_id": app_id, "returncode": process.returncode})
            try:
                if app_id is None:
                    raise RuntimeError("App identity unavailable; cleanup cannot be established.")
                result = subprocess.run([sys.executable, "-m", "modal", "app", "stop", app_id, "--yes"],
                                        capture_output=True, text=True, timeout=10, check=True)
                (audit / "stop.log").write_text(result.stdout + result.stderr)
                status = subprocess.run([sys.executable, "-m", "modal", "app", "list", "--json"],
                                        capture_output=True, text=True, timeout=5, check=True)
                containers = subprocess.run([sys.executable, "-m", "modal", "container", "list", "--app-id", app_id, "--json"],
                                            capture_output=True, text=True, timeout=5, check=True)
                (audit / "app-status.json").write_text(status.stdout); (audit / "container-status.json").write_text(containers.stdout)
                ledger["cleanup_verified"] = any(r["app_id"] == app_id and r["state"] == "stopped" for r in json.loads(status.stdout)) and json.loads(containers.stdout) == []
            except Exception as error:
                ledger["cleanup_error"] = repr(error)
            ledger["elapsed_seconds"] = time.time() - started
            durable(audit / "ledger.json", ledger)
    if process.returncode or not ledger["cleanup_verified"] or not (audit / "result.json").exists():
        raise RuntimeError("Scientific block or verified shutdown failed; stop the remaining cohort.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path); parser.add_argument("audit", type=Path)
    parser.add_argument("--block"); parser.add_argument("--child-deadline", type=float)
    args = parser.parse_args(); job = json.loads(args.job.read_text())
    blocks = [b for b in job["blocks"] if not args.block or f"{b['dataset']}-{b['repetition_id']}" == args.block]
    if not blocks or (args.child_deadline and len(blocks) != 1):
        parser.error("Select exactly one assigned block in child mode.")
    for block in blocks:
        if args.child_deadline:
            child(job, block, args.audit, args.child_deadline)
        else:
            supervise(args.job, args.audit / f"{block['dataset']}-{block['repetition_id']}", block)
