from __future__ import annotations

import sys
import threading
from pathlib import Path

from openlrc.application import (
    OperationGuard,
    SetupCancelledEvent,
    SetupCompletedEvent,
    SetupController,
    SetupLogEvent,
    SetupStartedEvent,
    SetupStatus,
    WhisperSetupRequest,
)
from openlrc.setup.whisper_cpp import WhisperSetupResult
from openlrc.workflow import CancellationToken


def test_setup_controller_returns_typed_result_and_events(tmp_path: Path, monkeypatch) -> None:
    cli = tmp_path / "whisper-cli"
    model = tmp_path / "ggml-base.bin"
    vad = tmp_path / "ggml-silero.bin"
    for path in (cli, model, vad):
        path.touch()

    def fake_setup(**kwargs):
        assert kwargs["model"] == "base"
        return WhisperSetupResult(cli_path=cli, whisper_model=model, vad_model=vad)

    monkeypatch.setattr("openlrc.application.setup.setup_whisper_cpp", fake_setup)
    events = []

    result = SetupController().run(WhisperSetupRequest(), on_event=events.append)

    assert result.status is SetupStatus.SUCCEEDED
    assert result.paths == (cli, model, vad)
    assert isinstance(events[-1], SetupCompletedEvent)
    assert [event.header.sequence for event in events] == list(range(1, len(events) + 1))


def test_setup_cancel_terminates_owned_process(tmp_path: Path, monkeypatch) -> None:
    entered = threading.Event()

    def blocking_setup(**kwargs):
        kwargs["runner"]([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path)
        return WhisperSetupResult()

    monkeypatch.setattr("openlrc.application.setup.setup_whisper_cpp", blocking_setup)
    controller = SetupController()
    events = []
    results = []

    def record(event) -> None:
        events.append(event)
        if isinstance(event, SetupLogEvent):
            entered.set()

    thread = threading.Thread(
        target=lambda: results.append(controller.run(WhisperSetupRequest(), on_event=record)), daemon=True
    )
    thread.start()
    assert entered.wait(timeout=2)
    assert controller.cancel()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert results[0].status is SetupStatus.CANCELLED
    assert any(isinstance(event, SetupLogEvent) for event in events)
    assert isinstance(events[-1], SetupCancelledEvent)


def test_setup_controller_honors_pre_cancelled_external_token_before_setup_service(monkeypatch) -> None:
    def unexpected_setup(**_kwargs):
        raise AssertionError("setup service must not run for a pre-cancelled operation")

    monkeypatch.setattr("openlrc.application.setup.setup_whisper_cpp", unexpected_setup)
    token = CancellationToken()
    token.cancel()
    events = []

    result = SetupController().run(WhisperSetupRequest(), on_event=events.append, cancellation_token=token)

    assert result.status is SetupStatus.CANCELLED
    assert not any(isinstance(event, SetupStartedEvent) for event in events)
    assert sum(isinstance(event, SetupCancelledEvent) for event in events) == 1


def test_operation_guard_rejects_parallel_resource_owner() -> None:
    guard = OperationGuard()

    with guard.acquire("workflow", "subtitle workflow"):
        try:
            with guard.acquire("setup", "llama setup"):
                raise AssertionError("second operation unexpectedly acquired the guard")
        except RuntimeError as exc:
            assert "subtitle workflow is already running" in str(exc)
