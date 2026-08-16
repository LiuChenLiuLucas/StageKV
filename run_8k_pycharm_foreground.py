from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil


# 第一次设为 "stop"，运行一次；然后改成 "run" 正式实验。
MODE = "stop"

PROJECT_DIR = Path("/root/stagekv")
MODEL_PATH = Path("/model/ModelScope/Qwen/Qwen2.5-7B-Instruct")
BENCHMARK = PROJECT_DIR / "stagekv_cross_layer_calibrated_benchmark.py"

# 结果直接放在 /root/stagekv 下的新目录中。
RESULTS_DIR = PROJECT_DIR / "day13_context_8k_pycharm_run1"
LOG_PATH = RESULTS_DIR / "run.log"

active_process: subprocess.Popen[str] | None = None


def find_existing_benchmarks() -> list[psutil.Process]:
    targets = []
    benchmark_text = str(BENCHMARK)

    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.pid == os.getpid():
            continue

        try:
            command = " ".join(process.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

        if benchmark_text in command:
            targets.append(process)

    return targets


def stop_existing_benchmarks() -> None:
    targets = find_existing_benchmarks()

    if not targets:
        print("没有发现正在运行的 StageKV benchmark。")
        return

    print("将停止以下旧 benchmark 进程：")
    for process in targets:
        try:
            print(f"PID={process.pid} CMD={' '.join(process.cmdline())}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    for process in targets:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass

    _, alive = psutil.wait_procs(targets, timeout=15)

    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass

    psutil.wait_procs(alive, timeout=10)
    print("旧 benchmark 已停止。已有结果目录和日志没有删除。")


def terminate_process_tree() -> None:
    global active_process

    if active_process is None or active_process.poll() is not None:
        return

    try:
        parent = psutil.Process(active_process.pid)
        processes = [*parent.children(recursive=True), parent]
    except psutil.NoSuchProcess:
        return

    for process in reversed(processes):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass

    _, alive = psutil.wait_procs(processes, timeout=10)

    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


def handle_stop(signum: int, frame: object) -> None:
    del frame
    print(f"\n收到停止信号 {signum}，正在清理 benchmark 进程……")
    terminate_process_tree()
    raise SystemExit(128 + signum)


def run_foreground() -> None:
    global active_process

    existing = find_existing_benchmarks()
    if existing:
        pids = [process.pid for process in existing]
        raise RuntimeError(
            f"仍有 benchmark 正在运行，PID={pids}。"
            '请先把 MODE 改成 "stop" 并运行一次。'
        )

    if not BENCHMARK.is_file():
        raise RuntimeError(f"找不到 benchmark：{BENCHMARK}")
    if not MODEL_PATH.is_dir():
        raise RuntimeError(f"找不到模型：{MODEL_PATH}")
    if RESULTS_DIR.exists():
        raise RuntimeError(
            f"结果目录已经存在：{RESULTS_DIR}\n"
            "不要覆盖，请把 run1 改成 run2。"
        )

    RESULTS_DIR.mkdir(parents=True)

    command = [
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

    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONFAULTHANDLER"] = "1"
    environment["TORCH_SHOW_CPP_STACKTRACES"] = "1"

    output_queue: queue.Queue[str | None] = queue.Queue()

    with LOG_PATH.open("w", encoding="utf-8", buffering=1) as log:
        active_process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert active_process.stdout is not None

        def read_output() -> None:
            for line in active_process.stdout:
                output_queue.put(line)
            output_queue.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()

        started = time.monotonic()
        print("StageKV 8K foreground benchmark started.")
        print("PID:", active_process.pid)
        print("Results:", RESULTS_DIR)
        print("不要再次启动第二个实验。\n")

        while True:
            try:
                line = output_queue.get(timeout=15)
            except queue.Empty:
                elapsed = time.monotonic() - started
                heartbeat = (
                    f"[PyCharm heartbeat] benchmark仍在运行，"
                    f"elapsed={elapsed / 60:.1f} min"
                )
                print(heartbeat, flush=True)
                log.write(heartbeat + "\n")
                continue

            if line is None:
                break

            print(line, end="", flush=True)
            log.write(line)

        return_code = active_process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Benchmark失败，return_code={return_code}，日志：{LOG_PATH}"
        )

    print("\n===== BENCHMARK COMPLETED =====")
    print("Results:", RESULTS_DIR)

    comparison = RESULTS_DIR / "day12_comparison.csv"
    manifest = RESULTS_DIR / "day12_manifest.json"

    if comparison.is_file():
        print("\n===== COMPARISON =====")
        print(comparison.read_text(encoding="utf-8-sig"))

    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        print("\n===== PERFORMANCE DECISION =====")
        print(
            json.dumps(
                data["performance_decision"],
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    if MODE == "stop":
        stop_existing_benchmarks()
        print('\n下一步：把 MODE 改成 "run"，再点击 Run。')
    elif MODE == "run":
        run_foreground()
    else:
        raise ValueError('MODE 必须是 "stop" 或 "run"')