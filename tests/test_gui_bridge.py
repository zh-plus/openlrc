from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from openlrc.application.history import JobRecord, JobRecordStatus
from openlrc.gui_bridge.framing import read_json_lines
from openlrc.gui_bridge.protocol import MAX_FRAME_BYTES, ProtocolError, error_from_exception, parse_request
from openlrc.gui_bridge.serializers import history_job
from openlrc.gui_bridge.server import serve


def test_protocol_parser_rejects_unknown_version_and_unbounded_fields() -> None:
    with pytest.raises(ProtocolError, match="Unsupported protocol"):
        parse_request({"type": "request", "protocol": 2, "id": "a", "method": "app.hello", "params": {}})
    with pytest.raises(ProtocolError, match="Request id"):
        parse_request({"type": "request", "protocol": 1, "id": "", "method": "app.hello", "params": {}})


def test_json_line_reader_supports_multiple_frames_and_rejects_malformed_or_oversized() -> None:
    stream = io.BytesIO(b'{"one":1}\n{"two":2}\n')
    assert list(read_json_lines(stream)) == [{"one": 1}, {"two": 2}]
    with pytest.raises(ProtocolError, match="Malformed JSON"):
        list(read_json_lines(io.BytesIO(b"{broken}\n")))
    with pytest.raises(ProtocolError, match="1 MiB"):
        list(read_json_lines(io.BytesIO(b"x" * (MAX_FRAME_BYTES + 1) + b"\n")))
    with pytest.raises(ProtocolError, match="valid UTF-8"):
        list(read_json_lines(io.BytesIO(b"\xff\n")))
    with pytest.raises(ProtocolError, match="end with a newline"):
        list(read_json_lines(io.BytesIO(b"{}")))


def test_protocol_errors_redact_secrets() -> None:
    error = error_from_exception(ValueError("api_key=sk-super-secret-value"))
    message = error.to_dict()["message"]

    assert error.code == "INVALID_REQUEST"
    assert isinstance(message, str)
    assert "super-secret" not in message
    assert "<redacted>" in message


@pytest.mark.parametrize(
    ("status", "visible"),
    [
        (JobRecordStatus.RUNNING, False),
        (JobRecordStatus.CANCELLED, False),
        (JobRecordStatus.SUCCEEDED, True),
        (JobRecordStatus.SUCCEEDED_WITH_WARNINGS, True),
        (JobRecordStatus.FAILED, True),
        (JobRecordStatus.INTERRUPTED, True),
    ],
)
def test_gui_history_excludes_running_and_user_cancelled_records(status: JobRecordStatus, visible: bool) -> None:
    record = JobRecord(
        job_id="job-history",
        workflow="transcribe",
        name="History fixture",
        status=status,
        input_paths=["input.wav"],
        recipe={},
    )

    assert history_job(record) is visible


def test_stdio_server_sends_ready_round_trips_hello_and_shuts_down(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENLRC_APP_SUPPORT_DIR", str(tmp_path))
    requests = (
        {"type": "request", "protocol": 1, "id": "hello", "method": "app.hello", "params": {}},
        {"type": "request", "protocol": 1, "id": "bye", "method": "app.shutdown", "params": {}},
    )
    stdin = io.BytesIO("".join(json.dumps(item) + "\n" for item in requests).encode())
    stdout = io.StringIO()

    assert serve(stdin, stdout) == 0

    frames = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert frames[0]["event"] == "app.ready"
    assert frames[0]["payload"]["capabilities"] == [
        "workflow.transcribe",
        "workflow.translate",
        "workflow.run",
        "queue",
        "settings",
        "credentials",
        "resources",
        "jobs",
    ]
    assert next(frame for frame in frames if frame.get("id") == "hello")["ok"] is True
    assert next(frame for frame in frames if frame.get("id") == "bye")["result"] == {"accepted": True}


def test_stdio_server_reports_unknown_method_duplicate_id_and_continues(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENLRC_APP_SUPPORT_DIR", str(tmp_path))
    requests = (
        {"type": "request", "protocol": 1, "id": "same", "method": "missing.method", "params": {}},
        {"type": "request", "protocol": 1, "id": "same", "method": "app.hello", "params": {}},
        {"type": "request", "protocol": 1, "id": "settings", "method": "settings.get", "params": {}},
        {"type": "request", "protocol": 1, "id": "bye", "method": "app.shutdown", "params": {}},
    )
    stdin = io.BytesIO("".join(json.dumps(item) + "\n" for item in requests).encode())
    stdout = io.StringIO()

    assert serve(stdin, stdout) == 0

    frames = [json.loads(line) for line in stdout.getvalue().splitlines()]
    same = [frame for frame in frames if frame.get("id") == "same"]
    assert same[0]["error"]["code"] == "UNKNOWN_METHOD"
    assert same[1]["error"]["code"] == "DUPLICATE_REQUEST"
    assert next(frame for frame in frames if frame.get("id") == "settings")["ok"] is True
    assert next(frame for frame in frames if frame.get("id") == "bye")["ok"] is True


def test_shared_protocol_fixture_is_valid_for_python_contract() -> None:
    fixture_path = Path(__file__).parents[1] / "gui" / "tests" / "fixtures" / "protocol-v1.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert payload["ready"]["protocol"] == 1
    assert payload["ready"]["payload"]["capabilities"] == [
        "workflow.transcribe",
        "workflow.translate",
        "workflow.run",
        "queue",
        "settings",
        "credentials",
        "resources",
        "jobs",
    ]
    assert payload["queue"]["entries"][0]["state"] == "pending"
