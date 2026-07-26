#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

"""
WhisperCLIBackend: 通过 subprocess 调用 whisper-cli 二进制的底层桥接层。

Tool Isolation: 所有与外部 whisper-cli 进程的交互封装在此模块中。
LRCer / Transcriber 等上层模块不直接接触 subprocess。
"""

from __future__ import annotations

import json
import logging
import queue
import re
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openlrc.workflow import CancellationToken, OwnedProcessRegistry

from openlrc.whisper_resources import resolve_vad_model_path, resolve_whisper_cli, resolve_whisper_model_path

logger = logging.getLogger(__name__)


class WhisperCLIBackend:
    """封装 whisper-cli subprocess 调用的底层通信接口。

    Design Decision (Subprocess 双管道模型):
      - stdout: 接收 JSON 输出（通过 -ojf -of -）
      - stderr: 接收 progress 回调和日志
      - 独立 daemon 线程持续排空 stderr，避免 OS pipe buffer 满导致死锁

    Args:
        cli_path: whisper-cli 可执行文件路径。
        model_path: Whisper GGML 模型文件路径。
        vad_model_path: Silero VAD 模型文件路径（为空则不启用 VAD）。
    """

    def __init__(self, cli_path: str, model_path: str, vad_model_path: str = ""):
        self.cli_path = resolve_whisper_cli(cli_path)
        self.model_path = resolve_whisper_model_path(model_path)
        self.vad_model_path = resolve_vad_model_path(vad_model_path) if vad_model_path else ""

    def transcribe(
        self,
        audio_path: str,
        lang: str | None = None,
        progress_cb: Callable[[int], None] | None = None,
        extra_args: list[str] | None = None,
        cancellation_token: CancellationToken | None = None,
        process_registry: OwnedProcessRegistry | None = None,
    ) -> dict:
        """调用 whisper-cli 进行转录推理。

        Args:
            audio_path: 音频文件路径。
            lang: 语言代码（None 则自动检测）。
            progress_cb: 进度回调函数，接收 0-100 的整数百分比。
            extra_args: 额外的 CLI 参数列表。

        Returns:
            whisper-cli 输出的 JSON dict。

        Raises:
            RuntimeError: whisper-cli 进程非零退出或无输出。
        """
        # Current whisper.cpp releases do not emit JSON when ``-of -`` is paired
        # with ``--no-prints``. Use an owned temporary output and retain stdout as
        # a compatibility fallback for older binaries.
        with tempfile.TemporaryDirectory(prefix="openlrc-whisper-") as output_dir:
            output_base = Path(output_dir) / "transcription"
            output_json = output_base.with_suffix(".json")
            cmd = [self.cli_path, "-m", self.model_path, "-f", audio_path, "-ojf", "-of", str(output_base), "-pp"]
            cmd.extend(["-l", lang or "auto"])
            if self.vad_model_path:
                cmd.extend(["--vad", "-vm", self.vad_model_path])
            if extra_args:
                cmd.extend(extra_args)
            logger.info(f"Running whisper-cli: {' '.join(cmd)}")
            return self._run_command(
                cmd,
                output_json,
                progress_cb=progress_cb,
                cancellation_token=cancellation_token,
                process_registry=process_registry,
            )

    @staticmethod
    def _run_command(
        cmd: list[str],
        output_json: Path,
        *,
        progress_cb: Callable[[int], None] | None,
        cancellation_token: CancellationToken | None,
        process_registry: OwnedProcessRegistry | None,
    ) -> dict:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if process_registry is not None:
            process_registry.register(proc)

        # Drain both pipes concurrently. whisper-cli can emit a JSON document larger
        # than the OS pipe buffer, so waiting to read stdout until exit can deadlock.
        stdout_chunks: list[str] = []
        stderr_lines: list[str] = []
        stderr_q: queue.Queue[str] = queue.Queue()

        def _drain_stdout() -> None:
            assert proc.stdout is not None
            while chunk := proc.stdout.read(65536):
                stdout_chunks.append(chunk)

        def _drain_stderr() -> None:
            """独立 daemon 线程持续排空 stderr，防止 pipe buffer 满导致死锁。"""
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_q.put(line)
                stderr_lines.append(line)

        stdout_thread = threading.Thread(target=_drain_stdout, name="WhisperStdout", daemon=True)
        stderr_thread = threading.Thread(target=_drain_stderr, name="WhisperStderr", daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        # 主线程轮询进度
        # whisper-cli 进度格式: "whisper_print_progress_callback: progress =  12%"
        progress_pattern = re.compile(r"progress\s*=\s*(\d+)%")
        try:
            while proc.poll() is None:
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled()
                try:
                    line = stderr_q.get(timeout=0.1)
                    match = progress_pattern.search(line)
                    if match and progress_cb:
                        progress_cb(int(match.group(1)))
                except queue.Empty:
                    continue
        finally:
            if proc.poll() is None:
                if process_registry is not None:
                    process_registry.terminate(proc)
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            elif process_registry is not None:
                process_registry.unregister(proc)

        stdout_thread.join(timeout=5.0)
        stderr_thread.join(timeout=5.0)
        while not stderr_q.empty():
            try:
                line = stderr_q.get_nowait()
                match = progress_pattern.search(line)
                if match and progress_cb:
                    progress_cb(int(match.group(1)))
            except queue.Empty:
                break

        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        stdout_data = "".join(stdout_chunks)

        if proc.returncode != 0:
            error_log = "".join(stderr_lines)
            raise RuntimeError(f"whisper-cli exited with code {proc.returncode}:\n{error_log}")

        if output_json.is_file():
            payload = output_json.read_text(encoding="utf-8")
        elif stdout_data.strip():
            payload = stdout_data
        else:
            raise RuntimeError("whisper-cli produced no output. stderr:\n" + "".join(stderr_lines))

        return json.loads(payload)
