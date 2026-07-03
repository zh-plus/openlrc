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
import threading
from collections.abc import Callable

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
        # === 构建命令 ===
        # -ojf: --output-json-full，含 token 级详细信息
        #        同时自动启用 token_timestamps (cli.cpp L1185)
        # -of -: 输出至 stdout (cli.cpp L1085: is_stdout{fname_out == "-"})
        # -pp:   --print-progress，启用进度打印至 stderr
        # --no-prints: 禁用普通推理日志（不影响 progress callback，
        #              因为 progress 通过 fprintf(stderr) 直接输出 (cli.cpp L353)）
        cmd = [
            self.cli_path,
            "-m",
            self.model_path,
            "-f",
            audio_path,
            "-ojf",  # --output-json-full
            "-of",
            "-",  # output to stdout
            "-pp",  # --print-progress
            "--no-prints",  # suppress verbose logs
        ]

        # 语言设置
        if lang:
            cmd.extend(["-l", lang])
        else:
            cmd.extend(["-l", "auto"])

        # VAD 设置
        if self.vad_model_path:
            cmd.extend(["--vad", "-vm", self.vad_model_path])

        # 额外参数
        if extra_args:
            cmd.extend(extra_args)

        logger.info(f"Running whisper-cli: {' '.join(cmd)}")

        # === 启动进程 ===
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # === stderr 进度解析线程 ===
        stderr_lines: list[str] = []
        stderr_q: queue.Queue[str] = queue.Queue()

        def _drain_stderr() -> None:
            """独立 daemon 线程持续排空 stderr，防止 pipe buffer 满导致死锁。"""
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_q.put(line)
                stderr_lines.append(line)

        t = threading.Thread(target=_drain_stderr, daemon=True)
        t.start()

        # 主线程轮询进度
        # whisper-cli 进度格式: "whisper_print_progress_callback: progress =  12%"
        progress_pattern = re.compile(r"progress\s*=\s*(\d+)%")
        while proc.poll() is None:
            try:
                line = stderr_q.get(timeout=0.1)
                m = progress_pattern.search(line)
                if m and progress_cb:
                    progress_cb(int(m.group(1)))
            except queue.Empty:
                continue

        # 排空残余 stderr（进程退出后可能还有未读取的行）
        t.join(timeout=5.0)
        while not stderr_q.empty():
            try:
                line = stderr_q.get_nowait()
                m = progress_pattern.search(line)
                if m and progress_cb:
                    progress_cb(int(m.group(1)))
            except queue.Empty:
                break

        # === 读取 stdout JSON ===
        assert proc.stdout is not None
        stdout_data = proc.stdout.read()

        if proc.returncode != 0:
            error_log = "".join(stderr_lines)
            raise RuntimeError(f"whisper-cli exited with code {proc.returncode}:\n{error_log}")

        if not stdout_data.strip():
            raise RuntimeError("whisper-cli produced no output. stderr:\n" + "".join(stderr_lines))

        return json.loads(stdout_data)
