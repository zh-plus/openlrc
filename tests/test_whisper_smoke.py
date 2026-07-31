from __future__ import annotations

import json
import logging
import os
import platform
import shutil
from pathlib import Path

import pytest

from openlrc.config import TranscriptionConfig
from openlrc.transcribe import map_cli_json_to_segments
from openlrc.whisper_backend import WhisperCLIBackend
from openlrc.whisper_resources import (
    DEFAULT_MODEL_NAME,
    DEFAULT_VAD_MODEL_NAME,
    resolve_vad_model_path,
    resolve_whisper_cli,
    resolve_whisper_model_path,
)
from openlrc.workflow import ExecutionContext, TranscribeRequest, WorkflowExecutor, WorkflowKind, WorkflowStatus

DATA_DIR = Path(__file__).parent / "data"


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}


def _assert_raw_whisper_contract(payload: dict) -> None:
    assert payload.get("result", {}).get("language")
    assert payload.get("transcription")
    segments = map_cli_json_to_segments(payload)
    assert segments
    assert [segment.start for segment in segments] == sorted(segment.start for segment in segments)
    for segment in segments:
        assert 0 <= segment.start <= segment.end
        assert segment.words
        for word in segment.words:
            assert 0 <= word.start <= word.end


@pytest.mark.skipif(
    platform.system() != "Darwin" or not _enabled("OPENLRC_TEST_REAL_WHISPER"),
    reason="Set OPENLRC_TEST_REAL_WHISPER=1 on macOS with local Whisper resources.",
)
def test_real_whisper_audio_cpu_with_vad(tmp_path: Path) -> None:
    audio_path = tmp_path / "test_audio.wav"
    shutil.copy2(DATA_DIR / "test_audio.wav", audio_path)
    context = ExecutionContext(WorkflowKind.TRANSCRIBE)
    backend = WhisperCLIBackend(
        cli_path=resolve_whisper_cli(""),
        model_path=resolve_whisper_model_path(DEFAULT_MODEL_NAME),
        vad_model_path=resolve_vad_model_path(DEFAULT_VAD_MODEL_NAME),
    )

    try:
        payload = backend.transcribe(
            str(audio_path),
            extra_args=["-ng", "-nfa"],
            cancellation_token=context.cancellation_token,
            process_registry=context.processes,
        )
        _assert_raw_whisper_contract(payload)
        assert not context.processes._processes
    finally:
        context.close()


@pytest.mark.skipif(
    platform.system() != "Darwin" or not _enabled("OPENLRC_TEST_REAL_WHISPER"),
    reason="Set OPENLRC_TEST_REAL_WHISPER=1 on macOS with local Whisper resources.",
)
def test_real_whisper_video_cpu_exercises_ffmpeg_workflow(tmp_path: Path) -> None:
    video_path = tmp_path / "test_video.mp4"
    shutil.copy2(DATA_DIR / "test_video.mp4", video_path)
    request = TranscribeRequest(
        paths=(video_path,),
        transcription=TranscriptionConfig(
            whisper_model=DEFAULT_MODEL_NAME, vad_model="", asr_options={"use_gpu": False, "flash_attn": False}
        ),
        clear_temp=False,
    )

    result = WorkflowExecutor().execute(request)

    assert result.status is WorkflowStatus.SUCCEEDED, result.error
    assert len(result.outputs) == 1
    output = result.outputs[0].resolve()
    assert output.is_relative_to(tmp_path.resolve())
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["language"]
    assert payload["segments"]
    starts = [segment["start"] for segment in payload["segments"]]
    assert starts == sorted(starts)
    assert all(0 <= segment["start"] <= segment["end"] for segment in payload["segments"])
    assert all(artifact.path.resolve().is_relative_to(tmp_path.resolve()) for artifact in result.artifacts)


@pytest.mark.skipif(
    platform.system() != "Darwin" or not _enabled("OPENLRC_TEST_REAL_WHISPER_METAL"),
    reason="Set OPENLRC_TEST_REAL_WHISPER_METAL=1 in a non-restricted Apple Silicon environment.",
)
def test_real_whisper_metal_runtime(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    audio_path = tmp_path / "test_audio.wav"
    shutil.copy2(DATA_DIR / "test_audio.wav", audio_path)
    backend = WhisperCLIBackend(
        cli_path=resolve_whisper_cli(""), model_path=resolve_whisper_model_path(DEFAULT_MODEL_NAME), vad_model_path=""
    )
    caplog.set_level(logging.DEBUG, logger="openlrc.whisper_backend")

    payload = backend.transcribe(str(audio_path))

    _assert_raw_whisper_contract(payload)
    diagnostic = caplog.text.lower()
    assert "ggml_metal_buffer_init: error" not in diagnostic
    assert "use gpu    = 1" in diagnostic or "metal = 1" in diagnostic
