from __future__ import annotations

import json
import os
import py_compile
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import psutil


REMOTE_PROJECT_DIR = Path("/root/stagekv")
RUN_TAG = "run3"
SEQUENCE_LENGTH = 8192
DECODE_TOKENS = 32
WARMUP_REPEATS = 1
MEASURED_REPEATS = 1

CORE_FILES = (
    "stagekv_bidirectional_async_correctness.py",
    "stagekv_cross_layer_prefetch_correctness.py",
    "stagekv_cross_layer_calibrated_benchmark.py",
)
BENCHMARK_NAME = "stagekv_cross_layer_calibrated_benchmark.py"
RUNNER_NAME = "run_transfer_event_profile.py"
TMUX_SESSION = f"stagekv_transfer_profile_{RUN_TAG}"
CONTROLLER_DIR = REMOTE_PROJECT_DIR / f"day15_transfer_profile_controller_{RUN_TAG}"
CONTROLLER_LOG = CONTROLLER_DIR / "run.log"
STATUS_PATH = CONTROLLER_DIR / "status.json"


@dataclass(frozen=True)
class ProfileConfig:
    label: str
    model_path: Path
    resident_heads: int

    @property
    def results_dir(self) -> Path:
        return REMOTE_PROJECT_DIR / (
            f"day15_transfer_profile_{self.label}_8k_{RUN_TAG}"
        )


PROFILES = (
    ProfileConfig(
        label="qwen7b_r2",
        model_path=Path("/model/ModelScope/Qwen/Qwen2.5-7B-Instruct"),
        resident_heads=2,
    ),
    ProfileConfig(
        label="qwen3b_r1",
        model_path=Path("/model/ModelScope/Qwen/Qwen2.5-3B-Instruct"),
        resident_heads=1,
    ),
)


def find_existing_benchmarks() -> list[psutil.Process]:
    matches: list[psutil.Process] = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.pid == os.getpid():
            continue
        try:
            command = " ".join(process.info["cmdline"] or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if BENCHMARK_NAME in command:
            matches.append(process)
    return matches


def install_core_files() -> Path:
    source_dir = Path(__file__).resolve().parent
    if not REMOTE_PROJECT_DIR.is_dir():
        raise RuntimeError(f"Missing remote project directory: {REMOTE_PROJECT_DIR}")

    benchmark_source = source_dir / BENCHMARK_NAME
    benchmark_text = benchmark_source.read_text(encoding="utf-8")
    if "--transfer-event-timing" not in benchmark_text:
        raise RuntimeError("PyCharm has not synchronized the profiling benchmark")

    for name in CORE_FILES:
        source = source_dir / name
        target = REMOTE_PROJECT_DIR / name
        if not source.is_file():
            raise RuntimeError(f"Missing synchronized source file: {source}")
        if source.resolve() != target.resolve():
            backup = target.with_suffix(".pre_transfer_event_timing.py")
            if target.is_file() and not backup.exists():
                shutil.copy2(target, backup)
                print(f"backup={backup}", flush=True)
            shutil.copy2(source, target)
            print(f"installed={target}", flush=True)
        py_compile.compile(str(target), doraise=True)

    source_runner = source_dir / RUNNER_NAME
    target_runner = REMOTE_PROJECT_DIR / RUNNER_NAME
    if source_runner.resolve() != target_runner.resolve():
        shutil.copy2(source_runner, target_runner)
        print(f"installed={target_runner}", flush=True)
    py_compile.compile(str(target_runner), doraise=True)

    return REMOTE_PROJECT_DIR / BENCHMARK_NAME


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
        targets = [*parent.children(recursive=True), parent]
    except psutil.NoSuchProcess:
        return
    for target in reversed(targets):
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=10)
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass


def run_profile(benchmark: Path, profile: ProfileConfig) -> None:
    if not profile.model_path.is_dir():
        raise RuntimeError(f"Missing model directory: {profile.model_path}")
    if profile.results_dir.exists():
        raise RuntimeError(
            f"Results directory already exists: {profile.results_dir}. "
            "Change RUN_TAG so existing data is not overwritten."
        )
    profile.results_dir.mkdir(parents=True)
    log_path = profile.results_dir / "run.log"

    command = [
        sys.executable,
        str(benchmark),
        "--model-path",
        str(profile.model_path),
        "--results-dir",
        str(profile.results_dir),
        "--lengths",
        str(SEQUENCE_LENGTH),
        "--decode-tokens",
        str(DECODE_TOKENS),
        "--warmup-repeats",
        str(WARMUP_REPEATS),
        "--repeats",
        str(MEASURED_REPEATS),
        "--resident-heads",
        str(profile.resident_heads),
        "--stagekv-modes",
        "bidirectional",
        "cross_layer",
        "--smoke",
        "--transfer-event-timing",
    ]
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONFAULTHANDLER"] = "1"

    output_queue: queue.Queue[str | None] = queue.Queue()
    print(f"\nSTART profile={profile.label}", flush=True)
    print(f"results={profile.results_dir}", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=REMOTE_PROJECT_DIR,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None

        def read_output() -> None:
            for line in process.stdout:
                output_queue.put(line)
            output_queue.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        started = time.monotonic()
        try:
            while True:
                try:
                    line = output_queue.get(timeout=15)
                except queue.Empty:
                    elapsed_minutes = (time.monotonic() - started) / 60.0
                    heartbeat = (
                        f"[PyCharm heartbeat] profile={profile.label} "
                        f"elapsed={elapsed_minutes:.1f} min"
                    )
                    print(heartbeat, flush=True)
                    log.write(heartbeat + "\n")
                    continue
                if line is None:
                    break
                print(line, end="", flush=True)
                log.write(line)
        except BaseException:
            terminate_process_tree(process)
            raise
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Profile {profile.label} failed with return code {return_code}; "
            f"inspect {log_path}"
        )

    manifest_path = profile.results_dir / "day12_manifest.json"
    comparison_path = profile.results_dir / "day12_comparison.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest["transfer_event_timing_enabled"]:
        raise RuntimeError("Manifest did not record transfer event timing")
    if manifest["performance_claim_protocol"]["paper_ready"]:
        raise RuntimeError("Profiling run must never be marked paper_ready")
    print(f"PASS profile={profile.label}", flush=True)
    print(f"manifest={manifest_path}", flush=True)
    print(f"comparison={comparison_path}", flush=True)


def write_status(state: str, **extra: object) -> None:
    payload = {
        "state": state,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "tmux_session": TMUX_SESSION,
        "run_tag": RUN_TAG,
        "results": [str(profile.results_dir) for profile in PROFILES],
        **extra,
    }
    temporary = STATUS_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(STATUS_PATH)


def tmux_session_exists() -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", TMUX_SESSION],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def tmux_worker() -> int:
    CONTROLLER_DIR.mkdir(parents=True, exist_ok=True)
    with CONTROLLER_LOG.open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            print(
                "tmux_worker_started_at="
                f"{datetime.now().isoformat(timespec='seconds')}",
                flush=True,
            )
            write_status("running", current_profile=None)
            try:
                benchmark = REMOTE_PROJECT_DIR / BENCHMARK_NAME
                py_compile.compile(str(benchmark), doraise=True)
                for profile in PROFILES:
                    write_status("running", current_profile=profile.label)
                    run_profile(benchmark, profile)
                write_status("complete", current_profile=None, success=True)
                print("\nTRANSFER EVENT PROFILING SMOKE=PASS", flush=True)
                return 0
            except BaseException as exc:
                traceback.print_exc()
                write_status(
                    "failed",
                    current_profile=None,
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
                return 1


def print_existing_processes(processes: list[psutil.Process]) -> None:
    print("An existing StageKV benchmark is still running:", flush=True)
    for process in processes:
        try:
            print(
                f"PID={process.pid} CMD={' '.join(process.cmdline())}",
                flush=True,
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            print(f"PID={process.pid}", flush=True)
    previous_logs = sorted(
        REMOTE_PROJECT_DIR.glob(
            "day15_transfer_profile_qwen7b_r2_8k_run*/run.log"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if previous_logs:
        previous_log = previous_logs[0]
        content = previous_log.read_text(encoding="utf-8", errors="replace")
        print(f"\n===== LAST LOG: {previous_log} =====", flush=True)
        print(content[-12000:], flush=True)


def monitor_tmux() -> int:
    print(f"tmux_session={TMUX_SESSION}", flush=True)
    print(f"controller_log={CONTROLLER_LOG}", flush=True)
    print("This Run window only monitors the GPU job inside tmux.", flush=True)
    seen_characters = 0
    if CONTROLLER_LOG.is_file():
        content = CONTROLLER_LOG.read_text(encoding="utf-8", errors="replace")
        visible = content[-12000:]
        print("\n===== CURRENT LOG OUTPUT =====", flush=True)
        print(visible, end="" if visible.endswith("\n") else "\n", flush=True)
        seen_characters = len(content)

    started = time.monotonic()
    last_heartbeat = started
    while True:
        if CONTROLLER_LOG.is_file():
            content = CONTROLLER_LOG.read_text(encoding="utf-8", errors="replace")
            if len(content) > seen_characters:
                print(content[seen_characters:], end="", flush=True)
                seen_characters = len(content)

        status = None
        if STATUS_PATH.is_file():
            status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
            state = status.get("state")
            if state == "complete":
                print("\n===== PROFILING COMPLETE =====", flush=True)
                print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
                return 0
            if state == "failed":
                print("\n===== PROFILING FAILED =====", flush=True)
                print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
                return 1

        if not tmux_session_exists():
            print(
                "tmux session ended before a final status was written. "
                f"Inspect {CONTROLLER_LOG}",
                flush=True,
            )
            return 1

        now = time.monotonic()
        if now - last_heartbeat >= 15:
            current_profile = None if status is None else status.get("current_profile")
            print(
                f"[PyCharm monitor] session={TMUX_SESSION} "
                f"profile={current_profile} elapsed={(now - started) / 60:.1f} min",
                flush=True,
            )
            last_heartbeat = now
        time.sleep(3)


def controller_main() -> int:
    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is not installed on the remote host")

    if tmux_session_exists() or STATUS_PATH.is_file():
        return monitor_tmux()

    existing = find_existing_benchmarks()
    if existing:
        print_existing_processes(existing)
        print("No second benchmark was started.", flush=True)
        return 2

    install_core_files()
    if CONTROLLER_DIR.exists():
        raise RuntimeError(
            "Controller directory already exists without reusable status: "
            f"{CONTROLLER_DIR}. Change RUN_TAG; do not overwrite it."
        )
    for profile in PROFILES:
        if profile.results_dir.exists():
            raise RuntimeError(
                f"Results directory already exists: {profile.results_dir}. "
                "Change RUN_TAG; do not overwrite it."
            )

    CONTROLLER_DIR.mkdir(parents=True)
    CONTROLLER_LOG.write_text(
        f"controller_started_at={datetime.now().isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )
    target_runner = REMOTE_PROJECT_DIR / RUNNER_NAME
    worker_command = shlex.join(
        [sys.executable, str(target_runner), "--tmux-worker"]
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", TMUX_SESSION, worker_command],
        check=True,
    )
    time.sleep(1)
    if not tmux_session_exists() and not STATUS_PATH.is_file():
        raise RuntimeError(f"tmux worker did not start; inspect {CONTROLLER_LOG}")
    return monitor_tmux()


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--tmux-worker":
        return tmux_worker()
    if len(sys.argv) != 1:
        raise ValueError("Unsupported arguments")
    return controller_main()


if __name__ == "__main__":
    raise SystemExit(main())
