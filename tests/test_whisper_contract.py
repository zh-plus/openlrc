from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from openlrc.exceptions import TranscribeException
from openlrc.transcribe import map_cli_json_to_segments
from openlrc.whisper_backend import WhisperCLIBackend

FIXTURE_PATH = Path(__file__).parent / "data" / "whisper_cpp" / "v1.9.1" / "test_audio_vad_cpu.json"


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _backend(*, vad_model_path: str = "") -> WhisperCLIBackend:
    backend = object.__new__(WhisperCLIBackend)
    backend.cli_path = "/tmp/whisper-cli"
    backend.model_path = "/tmp/model.bin"
    backend.vad_model_path = vad_model_path
    return backend


def _owned_output_launcher(payload: str, commands: list[list[str]] | None = None):
    real_popen = subprocess.Popen

    def launch(command, **kwargs):
        if commands is not None:
            commands.append(command)
        output_base = Path(command[command.index("-of") + 1])
        output_base.with_suffix(".json").write_text(payload, encoding="utf-8")
        return real_popen([sys.executable, "-c", "pass"], **kwargs)

    return launch


def test_v191_fixture_preserves_real_multisegment_token_shape() -> None:
    payload = _fixture()

    segments = map_cli_json_to_segments(payload)

    assert payload["params"]["model"] == "<WHISPER_MODEL_PATH>"
    assert payload["result"]["language"] == "en"
    assert len(payload["transcription"]) == 3
    assert len(segments) == 3
    assert all(segment.words for segment in segments)
    assert [segment.start for segment in segments] == sorted(segment.start for segment in segments)
    assert segments[0].words[0].word == " How"


def test_empty_transcription_is_supported() -> None:
    payload = _fixture()
    payload["transcription"] = []

    assert map_cli_json_to_segments(payload) == []


def test_token_without_offsets_or_timestamps_uses_segment_range() -> None:
    payload = _fixture()
    token = payload["transcription"][0]["tokens"][1]
    token.pop("offsets")
    token.pop("timestamps")

    segments = map_cli_json_to_segments(payload)

    assert segments[0].words[0].start == segments[0].start
    assert segments[0].words[0].end == segments[0].end


def test_segment_without_offsets_uses_timestamp_range() -> None:
    payload = _fixture()
    payload["transcription"][0].pop("offsets")

    segment = map_cli_json_to_segments(payload)[0]

    assert segment.start == pytest.approx(0.19)
    assert segment.end == pytest.approx(2.79)


def test_segment_without_any_time_range_reports_its_index() -> None:
    payload = _fixture()
    payload["transcription"][1].pop("offsets")
    payload["transcription"][1].pop("timestamps")

    with pytest.raises(TranscribeException, match=r"segment 1.*offsets or timestamps"):
        map_cli_json_to_segments(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.__setitem__("transcription", {}), "'transcription' must be an array"),
        (lambda payload: payload["transcription"].__setitem__(0, []), "segment 0 must be an object"),
        (
            lambda payload: payload["transcription"][0].__setitem__("tokens", {}),
            "segment 0 field 'tokens' must be an array",
        ),
    ],
)
def test_unrecoverable_fixture_schema_errors_are_contextual(mutation, message: str) -> None:
    payload = _fixture()
    mutation(payload)

    with pytest.raises(TranscribeException, match=message):
        map_cli_json_to_segments(payload)


def test_backend_reports_invalid_json_without_echoing_payload() -> None:
    backend = _backend()
    payload = '{\n  "private": "do-not-echo",\n  broken\n}'

    with patch("openlrc.whisper_backend.subprocess.Popen", side_effect=_owned_output_launcher(payload)):
        with pytest.raises(TranscribeException) as caught:
            backend.transcribe("audio.wav")

    message = str(caught.value)
    assert "owned JSON output" in message
    assert "line 3" in message
    assert "column" in message
    assert "do-not-echo" not in message


def test_backend_rejects_non_object_json_root() -> None:
    backend = _backend()

    with patch("openlrc.whisper_backend.subprocess.Popen", side_effect=_owned_output_launcher("[]")):
        with pytest.raises(TranscribeException, match=r"must be an object, got list"):
            backend.transcribe("audio.wav")


def test_backend_reports_no_json_output_with_bounded_stderr() -> None:
    backend = _backend()
    real_popen = subprocess.Popen

    def launch(_command, **kwargs):
        script = "import sys; sys.stderr.write('prefix-' + 'x' * 6000 + '-tail')"
        return real_popen([sys.executable, "-c", script], **kwargs)

    with patch("openlrc.whisper_backend.subprocess.Popen", side_effect=launch):
        with pytest.raises(TranscribeException) as caught:
            backend.transcribe("audio.wav")

    message = str(caught.value)
    assert "produced no JSON output" in message
    assert "... [truncated]" in message
    assert message.endswith("-tail")
    assert len(message) < 4300


def test_backend_maps_rejected_vad_arguments_to_transcribe_exception() -> None:
    backend = _backend(vad_model_path="/tmp/vad.bin")
    real_popen = subprocess.Popen

    def launch(_command, **kwargs):
        script = "import sys; sys.stderr.write('unknown argument: --vad'); raise SystemExit(2)"
        return real_popen([sys.executable, "-c", script], **kwargs)

    with patch("openlrc.whisper_backend.subprocess.Popen", side_effect=launch):
        with pytest.raises(TranscribeException, match=r"(?s)exited with code 2.*unknown argument: --vad"):
            backend.transcribe("audio.wav")


def test_backend_adds_vad_flags_only_when_model_is_configured() -> None:
    commands: list[list[str]] = []
    payload = json.dumps({"result": {"language": "en"}, "transcription": []})

    with patch("openlrc.whisper_backend.subprocess.Popen", side_effect=_owned_output_launcher(payload, commands)):
        _backend(vad_model_path="/tmp/vad.bin").transcribe("audio.wav")
        _backend().transcribe("audio.wav")

    assert commands[0][commands[0].index("--vad") + 1 : commands[0].index("--vad") + 3] == ["-vm", "/tmp/vad.bin"]
    assert "--vad" not in commands[1]
    assert "-vm" not in commands[1]


def test_backend_constructor_surfaces_missing_vad_resource() -> None:
    with (
        patch("openlrc.whisper_backend.resolve_whisper_cli", return_value="/tmp/whisper-cli"),
        patch("openlrc.whisper_backend.resolve_whisper_model_path", return_value="/tmp/model.bin"),
        patch("openlrc.whisper_backend.resolve_vad_model_path", side_effect=FileNotFoundError("VAD model missing")),
    ):
        with pytest.raises(FileNotFoundError, match="VAD model missing"):
            WhisperCLIBackend("", "base", "silero-v6.2.0")


def test_fixture_language_can_change_without_changing_segment_mapping() -> None:
    payload = copy.deepcopy(_fixture())
    payload["result"]["language"] = "fr"

    segments = map_cli_json_to_segments(payload)

    assert payload["result"]["language"] == "fr"
    assert len(segments) == 3
