#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

import unittest
from pathlib import Path
from unittest.mock import patch

from openlrc.transcribe import Transcriber, TranscriptionInfo, _parse_timestamp_str, map_cli_json_to_segments
from openlrc.whisper_backend import WhisperCLIBackend
from openlrc.whisper_resources import DEFAULT_MODEL_NAME, DEFAULT_VAD_MODEL_NAME
from openlrc.whisper_types import Segment, Word

# === Shared mock return value (compatible with original test_transcribe.py) ===
return_tuple = (
    [
        Segment(
            0,
            0,
            0,
            3,
            "hello world",
            [],
            0.8,
            0,
            0,
            words=[Word(0, 1.5, "hello", probability=0.8), Word(1.6, 3, " world", probability=0.8)],
            temperature=1,
        ),
        Segment(
            0,
            0,
            3,
            6,
            "hello world",
            [],
            0.8,
            0,
            0,
            words=[Word(3, 4.5, "hello", probability=0.8), Word(4.6, 6, " world", probability=0.8)],
            temperature=1,
        ),
    ],
    TranscriptionInfo("en", 30, 30),
)


class TestTranscriber(unittest.TestCase):
    def setUp(self) -> None:
        self.audio_path = Path(__file__).parent / "data" / "test_audio.wav"

    @patch("openlrc.transcribe.WhisperCLIBackend")
    def test_init_uses_semantic_resource_defaults(self, MockBackend):
        """Transcriber should pass semantic defaults into the backend resolver path."""
        Transcriber()
        MockBackend.assert_called_once_with(
            cli_path="", model_path=DEFAULT_MODEL_NAME, vad_model_path=DEFAULT_VAD_MODEL_NAME
        )

    @patch("openlrc.transcribe.WhisperCLIBackend")
    def test_transcribe_success(self, MockBackend):
        """Test that transcribe() returns valid segments and info."""
        mock_backend_instance = MockBackend.return_value
        mock_backend_instance.transcribe.return_value = {
            "result": {"language": "xx"},
            "transcription": [
                {
                    "timestamps": {"from": "00:00:00,000", "to": "00:00:03,000"},
                    "offsets": {"from": 0, "to": 3000},
                    "text": " hello world",
                    "tokens": [
                        {"text": " hello", "offsets": {"from": 0, "to": 1500}, "id": 1, "p": 0.95, "t_dtw": -1.0},
                        {"text": " world", "offsets": {"from": 1500, "to": 3000}, "id": 2, "p": 0.90, "t_dtw": -1.0},
                    ],
                }
            ],
        }

        transcriber = Transcriber(model_name="tiny", cli_path="whisper-cli")
        result, info = transcriber.transcribe(self.audio_path)
        self.assertIsNotNone(result)
        self.assertEqual(round(info.duration), 30)

    @patch("openlrc.transcribe.WhisperCLIBackend")
    def test_audio_file_not_found(self, MockBackend):
        """Test FileNotFoundError for missing audio files."""
        transcriber = Transcriber(model_name="tiny", cli_path="whisper-cli")
        with self.assertRaises(FileNotFoundError):
            transcriber.transcribe("audio.wav")


class TestWhisperCLIBackend(unittest.TestCase):
    @patch("openlrc.whisper_backend.resolve_vad_model_path", return_value="/tmp/vad.bin")
    @patch("openlrc.whisper_backend.resolve_whisper_model_path", return_value="/tmp/model.bin")
    @patch("openlrc.whisper_backend.resolve_whisper_cli", return_value="/tmp/whisper-cli")
    def test_backend_resolves_cli_and_models(self, mock_cli, mock_model, mock_vad):
        backend = WhisperCLIBackend("", DEFAULT_MODEL_NAME, DEFAULT_VAD_MODEL_NAME)

        mock_cli.assert_called_once_with("")
        mock_model.assert_called_once_with(DEFAULT_MODEL_NAME)
        mock_vad.assert_called_once_with(DEFAULT_VAD_MODEL_NAME)
        self.assertEqual(backend.cli_path, "/tmp/whisper-cli")
        self.assertEqual(backend.model_path, "/tmp/model.bin")
        self.assertEqual(backend.vad_model_path, "/tmp/vad.bin")


class TestMapCliJsonToSegments(unittest.TestCase):
    """Unit tests for the whisper.cpp JSON -> Segment adapter."""

    def test_basic_conversion(self):
        """Test basic JSON -> Segment conversion with correct time units."""
        cli_json = {
            "result": {"language": "en"},
            "transcription": [
                {
                    "timestamps": {"from": "00:00:00,000", "to": "00:00:03,000"},
                    "offsets": {"from": 0, "to": 3000},
                    "text": " hello world",
                    "tokens": [
                        {"text": " hello", "offsets": {"from": 0, "to": 1500}, "id": 1, "p": 0.95, "t_dtw": -1.0},
                        {"text": " world", "offsets": {"from": 1500, "to": 3000}, "id": 2, "p": 0.90, "t_dtw": -1.0},
                    ],
                }
            ],
        }
        segments = map_cli_json_to_segments(cli_json)
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0].words), 2)
        self.assertAlmostEqual(segments[0].start, 0.0)
        self.assertAlmostEqual(segments[0].end, 3.0)
        self.assertAlmostEqual(segments[0].words[0].start, 0.0)
        self.assertAlmostEqual(segments[0].words[0].end, 1.5)
        self.assertEqual(segments[0].words[0].word, " hello")
        self.assertAlmostEqual(segments[0].words[0].probability, 0.95)
        self.assertAlmostEqual(segments[0].words[1].start, 1.5)
        self.assertAlmostEqual(segments[0].words[1].end, 3.0)
        self.assertEqual(segments[0].words[1].word, " world")

    def test_skip_special_tokens(self):
        """Test that special tokens like [SOT], [EOT] are filtered out."""
        cli_json = {
            "transcription": [
                {
                    "offsets": {"from": 0, "to": 3000},
                    "text": " hello",
                    "tokens": [
                        {"text": "[_BEG_]", "offsets": {"from": 0, "to": 0}, "p": 1.0},
                        {"text": " hello", "offsets": {"from": 0, "to": 1500}, "p": 0.95},
                        {"text": "[_EOT_]", "offsets": {"from": 1500, "to": 3000}, "p": 1.0},
                    ],
                }
            ]
        }
        segments = map_cli_json_to_segments(cli_json)
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0].words), 1)
        self.assertEqual(segments[0].words[0].word, " hello")

    def test_empty_segment_filtered(self):
        """Test that segments with no valid words are filtered out."""
        cli_json = {
            "transcription": [
                {
                    "offsets": {"from": 0, "to": 1000},
                    "text": "",
                    "tokens": [
                        {"text": "[_BEG_]", "offsets": {"from": 0, "to": 0}, "p": 1.0},
                        {"text": "[_EOT_]", "offsets": {"from": 500, "to": 1000}, "p": 1.0},
                    ],
                }
            ]
        }
        segments = map_cli_json_to_segments(cli_json)
        self.assertEqual(len(segments), 0)

    def test_multiple_segments(self):
        """Test conversion of multiple segments."""
        cli_json = {
            "transcription": [
                {
                    "offsets": {"from": 0, "to": 3000},
                    "text": " first",
                    "tokens": [{"text": " first", "offsets": {"from": 0, "to": 3000}, "p": 0.9}],
                },
                {
                    "offsets": {"from": 3000, "to": 6000},
                    "text": " second",
                    "tokens": [{"text": " second", "offsets": {"from": 3000, "to": 6000}, "p": 0.85}],
                },
            ]
        }
        segments = map_cli_json_to_segments(cli_json)
        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(segments[0].start, 0.0)
        self.assertAlmostEqual(segments[0].end, 3.0)
        self.assertAlmostEqual(segments[1].start, 3.0)
        self.assertAlmostEqual(segments[1].end, 6.0)

    def test_timestamp_string_fallback(self):
        """Test fallback to timestamps string when offsets are not available in tokens."""
        cli_json = {
            "transcription": [
                {
                    "offsets": {"from": 0, "to": 3000},
                    "text": " hello",
                    "tokens": [
                        {"text": " hello", "timestamps": {"from": "00:00:00,000", "to": "00:00:01,500"}, "p": 0.95}
                    ],
                }
            ]
        }
        segments = map_cli_json_to_segments(cli_json)
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0].words[0].start, 0.0)
        self.assertAlmostEqual(segments[0].words[0].end, 1.5)

    def test_empty_transcription(self):
        """Test handling of empty transcription array."""
        cli_json = {"transcription": []}
        segments = map_cli_json_to_segments(cli_json)
        self.assertEqual(len(segments), 0)


class TestParseTimestampStr(unittest.TestCase):
    """Unit tests for _parse_timestamp_str helper."""

    def test_hhmmss_dot(self):
        self.assertAlmostEqual(_parse_timestamp_str("00:00:01.500"), 1.5)

    def test_hhmmss_comma(self):
        self.assertAlmostEqual(_parse_timestamp_str("00:00:01,500"), 1.5)

    def test_mmss(self):
        self.assertAlmostEqual(_parse_timestamp_str("01:30.000"), 90.0)

    def test_hours(self):
        self.assertAlmostEqual(_parse_timestamp_str("01:00:00.000"), 3600.0)
