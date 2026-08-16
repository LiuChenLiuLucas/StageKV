from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil


ACTION = "start"  # 启动后改为 "status"，再次 Run 即可查看状态。

PROJECT_DIR = Path("/root/stagekv")
MODEL_PATH = Path("/model/ModelScope/Qwen/Qwen2.5-7B-Instruct")
BENCHMARK = PROJECT_DIR / "stagekv_cross_layer_calibrated_benchmark.py"
RESULTS_DIR = PROJECT_DIR / "results" / "day13_context_8k_run2"

SCRIPT_PATH = Path(__file__).resolve()
META_PATH = RESULTS_DIR / "background_process.json"
STATUS_PATH = RESULTS_DIR / "background_status.json"
LOG_PATH = RESULTS_DIR / "run.log"


def benchmark_command() -> list[str]:
    return [
        sys.executable,
        str(BENCHMARK),
        "--model-path", str(MODEL_PATH),
        "--results-dir", str(RESULTS_DIR),
        "--lengths", "8192",
        "--decode-tokens", "32",
        "--warmup-repeats", "2",
        "--repeats", "5",
        "--resident-heads", "2",
        "--stagekv-modes", "bidirectional", "cross_layer",
    ]


def detached_worker() -> int:
    started_at = datetime.now().isoformat(timespec="seconds")
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONFAULTHANDLER"] = "1"
    environment["TORCH_SHOW_CPP_STACKTRACES"] = "1"

    with LOG_PATH.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"background_started_at={started_at}\n")
        log.flush()

        completed = subprocess.run(
            benchmark_command(),
            cwd=PROJECT_DIR,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

        log.write(f"\nbenchmark_return_code={completed.returncode}\n")
        log.flush()

    status = {
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "return_code": completed.returncode,
        "success": completed.returncode == 0,
    }
    STATUS_PATH.write_text(
        json.dumps(status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return completed.returncode


def start() -> None:
    if not BENCHMARK.is_file():
        raise RuntimeError(f"Missing benchmark: {BENCHMARK}")
    if not MODEL_PATH.is_dir():
        raise RuntimeError(f"Missing model: {MODEL_PATH}")
    if RESULTS_DIR.exists():
        raise RuntimeError(
            f"Results directory already exists: {RESULTS_DIR}\n"
            "Use a new run name; do not overwrite existing data."
        )

    RESULTS_DIR.mkdir(parents=True)

    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        [sys.executable, str(SCRIPT_PATH), "--detached-worker"],
        cwd=PROJECT_DIR,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )

    metadata = {
        "pid": process.pid,
        "process_create_time": psutil.Process(process.pid).create_time(),
        "launched_at": datetime.now().isoformat(timespec="seconds"),
        "results_dir": str(RESULTS_DIR),
        "log": str(LOG_PATH),
    }
    META_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("8K benchmark started in background.")
    print("PID:", process.pid)
    print("Results:", RESULTS_DIR)
    print("Log:", LOG_PATH)
    print('Change ACTION = "status" and Run again to inspect progress.')


def status() -> None:
    if not META_PATH.is_file():
        raise RuntimeError(f"Missing process metadata: {META_PATH}")

    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    pid = int(metadata["pid"])
    expected_create_time = float(metadata["process_create_time"])

    running = False
    try:
        process = psutil.Process(pid)
        same_process = abs(process.create_time() - expected_create_time) < 1
        running = same_process and process.is_running()
        if running:
            print("Process status:", process.status())
    except psutil.NoSuchProcess:
        pass

    print("PID:", pid)
    print("Running:", running)

    if STATUS_PATH.is_file():
        print("\nCompletion status:")
        print(STATUS_PATH.read_text(encoding="utf-8"))

    if LOG_PATH.is_file():
        content = LOG_PATH.read_text(encoding="utf-8", errors="replace")
        print("\n===== LAST LOG OUTPUT =====")
        print(content[-12000:])

    manifest = RESULTS_DIR / "day12_manifest.json"
    if manifest.is_file():
        print("\nFormal manifest exists: benchmark completed successfully.")
    elif not running:
        print("\nProcess stopped without a manifest. Inspect the log above.")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--detached-worker":
        raise SystemExit(detached_worker())

    if ACTION == "start":
        start()
    elif ACTION == "status":
        status()
    else:
        raise ValueError('ACTION must be "start" or "status"')