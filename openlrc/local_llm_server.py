#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

"""Lifecycle management for a local llama.cpp OpenAI-compatible server."""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openlrc.workflow import ExecutionContext

import requests

from openlrc.llama_resources import (
    DEFAULT_LLAMA_CONTEXT_SIZE,
    DEFAULT_LLAMA_HOST,
    DEFAULT_LLAMA_IDLE_TIMEOUT,
    DEFAULT_LLAMA_MODEL_ALIAS,
    DEFAULT_LLAMA_MODEL_FILE,
    DEFAULT_LLAMA_PORT,
    DEFAULT_LLAMA_STARTUP_TIMEOUT,
    resolve_llama_model_path,
    resolve_llama_server,
)
from openlrc.logger import logger


class LocalLLMServer:
    """Start, reuse, and stop a local llama-server instance."""

    def __init__(
        self,
        *,
        server_path: str = "",
        model_path: str = DEFAULT_LLAMA_MODEL_FILE,
        host: str = DEFAULT_LLAMA_HOST,
        port: int = DEFAULT_LLAMA_PORT,
        alias: str = DEFAULT_LLAMA_MODEL_ALIAS,
        ctx_size: int = DEFAULT_LLAMA_CONTEXT_SIZE,
        gpu_layers: str | int = "all",
        idle_timeout: int = DEFAULT_LLAMA_IDLE_TIMEOUT,
        startup_timeout: int = DEFAULT_LLAMA_STARTUP_TIMEOUT,
        extra_args: list[str] | None = None,
        allow_external: bool = True,
        execution_context: ExecutionContext | None = None,
        role: str = "translation",
    ):
        self.server_path = server_path
        self.model_path = model_path
        self.host = host
        self.port = port
        self.alias = alias
        self.ctx_size = ctx_size
        self.gpu_layers = gpu_layers
        self.idle_timeout = idle_timeout
        self.startup_timeout = startup_timeout
        self.extra_args = list(extra_args or [])
        self.allow_external = allow_external
        self.execution_context = execution_context
        self.role = role

        self._process: subprocess.Popen | None = None
        self._owns_process = False
        self._active_sessions = 0
        self._idle_timer: threading.Timer | None = None
        self._lock = threading.RLock()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @contextmanager
    def session(self) -> Iterator[str]:
        """Ensure the server is running while work is active, then arm idle shutdown."""
        with self._lock:
            base_url = self.ensure_running(schedule_idle=False)
            self._active_sessions += 1
            self._cancel_idle_timer()

        try:
            yield base_url
        finally:
            with self._lock:
                self._active_sessions = max(0, self._active_sessions - 1)
                self._schedule_idle_shutdown()

    def ensure_running(self, *, schedule_idle: bool = True) -> str:
        """Return the OpenAI-compatible base URL, starting llama-server when needed."""
        with self._lock:
            if self._is_healthy():
                if self.execution_context is not None:
                    self.execution_context.model_event(
                        self.alias, "ready", owned=self._owns_process, role=self.role, endpoint=self.base_url
                    )
                if schedule_idle:
                    self._schedule_idle_shutdown()
                return self.base_url

            self._start()
            self._wait_until_ready()
            if schedule_idle:
                self._schedule_idle_shutdown()
            return self.base_url

    def close(self) -> None:
        """Stop the process if this manager started it. External servers are left alone."""
        with self._lock:
            self._cancel_idle_timer()
            process = self._process
            self._process = None
            self._active_sessions = 0

        if not process or not self._owns_process:
            return

        if process.poll() is not None:
            if self.execution_context is not None:
                self.execution_context.processes.unregister(process)
                self.execution_context.model_event(
                    self.alias, "stopped", owned=True, role=self.role, endpoint=self.base_url
                )
            return

        logger.info("Stopping local llama-server.")
        if self.execution_context is not None:
            self.execution_context.processes.terminate(process)
            self.execution_context.model_event(
                self.alias, "stopped", owned=True, role=self.role, endpoint=self.base_url
            )
        else:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("llama-server did not stop after terminate(); killing it.")
                process.kill()
                process.wait(timeout=5)

    def _start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        server = resolve_llama_server(self.server_path)
        model = resolve_llama_model_path(self.model_path)
        cmd = self._build_command(server=server, model=model)

        logger.info(f"Starting local llama-server at {self.base_url}.")
        if self.execution_context is not None:
            self.execution_context.check_cancelled()
            self.execution_context.model_event(
                self.alias, "starting", owned=True, role=self.role, endpoint=self.base_url
            )
        self._process = subprocess.Popen(
            cmd,
            cwd=str(Path(server).resolve().parents[2]) if "vendor/llama.cpp/build/bin" in server else None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._owns_process = True
        if self.execution_context is not None:
            self.execution_context.processes.register(self._process)

    def _build_command(self, *, server: str, model: str) -> list[str]:
        cmd = [
            server,
            "-m",
            model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--alias",
            self.alias,
            "-c",
            str(self.ctx_size),
            "-ngl",
            str(self.gpu_layers),
            "--jinja",
            "--reasoning",
            "off",
            "--reasoning-budget",
            "0",
            "--no-ui",
        ]
        cmd.extend(self.extra_args)
        return cmd

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                if self.execution_context is not None:
                    self.execution_context.processes.unregister(self._process)
                    self.execution_context.check_cancelled()
                raise RuntimeError("llama-server exited before becoming ready.")
            if self._is_healthy():
                logger.info(f"Local llama-server is ready at {self.base_url}.")
                if self.execution_context is not None:
                    self.execution_context.model_event(
                        self.alias, "ready", owned=True, role=self.role, endpoint=self.base_url
                    )
                return
            if self.execution_context is None:
                time.sleep(0.5)
            else:
                self.execution_context.cancellation_token.wait_or_raise(0.5)

        raise TimeoutError(f"llama-server did not become ready within {self.startup_timeout}s.")

    def _is_healthy(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/models", timeout=2)
            response.raise_for_status()
            if self._model_matches(response.json()):
                if not self.allow_external and self._process is None:
                    raise RuntimeError(
                        f"An external llama-server is already listening at {self.base_url}. "
                        "Staged local Hy-MT2 pipelines require an OpenLRC-owned server so the model can be unloaded."
                    )
                return True
            raise RuntimeError(
                f"A server is listening at {self.base_url}, but it does not expose model alias {self.alias!r}."
            )
        except RuntimeError:
            raise
        except requests.RequestException:
            return False

    def _model_matches(self, payload: dict) -> bool:
        candidates: list[str] = []
        for key in ("data", "models"):
            items = payload.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                for name_key in ("id", "name", "model"):
                    value = item.get(name_key)
                    if isinstance(value, str):
                        candidates.append(value)
                aliases = item.get("aliases")
                if isinstance(aliases, list):
                    candidates.extend(str(alias) for alias in aliases)

        return self.alias in candidates

    def _schedule_idle_shutdown(self) -> None:
        if not self._owns_process or self._active_sessions > 0 or self.idle_timeout <= 0:
            return

        self._cancel_idle_timer()
        self._idle_timer = threading.Timer(self.idle_timeout, self.close)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
