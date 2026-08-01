"""Bounded JSON Lines reader/writer with one stdout owner."""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Iterator
from typing import BinaryIO, TextIO

from openlrc.gui_bridge.protocol import MAX_FRAME_BYTES, ProtocolError


def read_json_lines(stream: BinaryIO) -> Iterator[object]:
    while True:
        raw = stream.readline(MAX_FRAME_BYTES + 2)
        if not raw:
            return
        if len(raw) > MAX_FRAME_BYTES:
            raise ProtocolError("FRAME_TOO_LARGE", "Protocol frame exceeds the 1 MiB limit.")
        if not raw.endswith(b"\n"):
            raise ProtocolError("INVALID_FRAME", "Protocol frame must end with a newline.")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("INVALID_ENCODING", "Protocol frame must be valid UTF-8.") from exc
        try:
            yield json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProtocolError("INVALID_JSON", f"Malformed JSON frame: {exc.msg}.") from exc


class JsonLineWriter:
    """Serialize frames on one thread so worker events cannot interleave stdout."""

    def __init__(self, stream: TextIO, *, capacity: int = 512) -> None:
        self._stream = stream
        self._frames: queue.Queue[dict[str, object] | None] = queue.Queue(maxsize=capacity)
        self._thread = threading.Thread(target=self._run, name="openlrc-protocol-writer", daemon=True)
        self._closed = False

    def start(self) -> None:
        self._thread.start()

    def send(self, frame: dict[str, object], *, droppable: bool = False) -> None:
        if self._closed:
            return
        try:
            self._frames.put(frame, block=not droppable, timeout=0.25 if not droppable else None)
        except queue.Full:
            if not droppable:
                self._frames.put(frame)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._frames.put(None)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            frame = self._frames.get()
            if frame is None:
                return
            encoded = json.dumps(frame, ensure_ascii=False, separators=(",", ":"))
            self._stream.write(encoded + "\n")
            self._stream.flush()
