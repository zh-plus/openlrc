#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from openlrc.cli.main import CheckResult, app
from openlrc.setup.llama_cpp import LlamaSetupResult
from openlrc.setup.whisper_cpp import WhisperSetupResult


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help_and_version(self):
        help_result = self.runner.invoke(app, ["--help"])
        self.assertEqual(help_result.exit_code, 0)
        self.assertIn("OpenLRC Mac command line tools", help_result.output)

        version_result = self.runner.invoke(app, ["--version"])
        self.assertEqual(version_result.exit_code, 0)
        self.assertIn("OpenLRC Mac 0.2.0", version_result.output)
        self.assertIn("openlrc-mac", version_result.output)
        self.assertIn("OpenLRC 1.7.0a1", version_result.output)

    def test_doctor_strict_fails_on_missing_check(self):
        with patch(
            "openlrc.cli.main._doctor_checks", return_value=[CheckResult("ffmpeg", False, "missing", "install ffmpeg")]
        ):
            result = self.runner.invoke(app, ["doctor", "--strict"])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("ffmpeg", result.output)
        self.assertIn("Missing", result.output)

    def test_models_status_prints_model_checks(self):
        with patch(
            "openlrc.cli.main._model_checks", return_value=[CheckResult("Local Qwen GGUF", True, "/models/qwen.gguf")]
        ):
            result = self.runner.invoke(app, ["models", "status"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Local Qwen GGUF", result.output)
        self.assertIn("/models/qwen.gguf", result.output)

    def test_setup_whisper_passes_arguments(self):
        setup_result = WhisperSetupResult(cli_path=Path("/bin/whisper-cli"))
        with patch("openlrc.cli.main.setup_whisper_cpp", return_value=setup_result) as mock_setup:
            result = self.runner.invoke(
                app, ["setup", "whisper", "--model", "small", "--vad-model", "silero-v6.2.0", "--skip-build"]
            )

        self.assertEqual(result.exit_code, 0)
        mock_setup.assert_called_once()
        kwargs = mock_setup.call_args.kwargs
        self.assertEqual(kwargs["model"], "small")
        self.assertEqual(kwargs["vad_model"], "silero-v6.2.0")
        self.assertTrue(kwargs["skip_build"])
        self.assertFalse(kwargs["skip_models"])

    def test_setup_llama_passes_arguments(self):
        setup_result = LlamaSetupResult(server_path=Path("/bin/llama-server"))
        with patch("openlrc.cli.main.setup_llama_cpp", return_value=setup_result) as mock_setup:
            result = self.runner.invoke(
                app,
                [
                    "setup",
                    "llama",
                    "--model-repo",
                    "org/repo",
                    "--model-file",
                    "model.gguf",
                    "--revision",
                    "dev",
                    "--model-url",
                    "https://example.test/model.gguf",
                    "--force",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        mock_setup.assert_called_once()
        kwargs = mock_setup.call_args.kwargs
        self.assertEqual(kwargs["model_repo"], "org/repo")
        self.assertEqual(kwargs["model_file"], "model.gguf")
        self.assertEqual(kwargs["revision"], "dev")
        self.assertEqual(kwargs["model_url"], "https://example.test/model.gguf")
        self.assertTrue(kwargs["force"])

    def test_setup_all_calls_both_setup_helpers(self):
        with (
            patch("openlrc.cli.main.setup_whisper_cpp", return_value=WhisperSetupResult()) as mock_whisper,
            patch("openlrc.cli.main.setup_llama_cpp", return_value=LlamaSetupResult()) as mock_llama,
        ):
            result = self.runner.invoke(app, ["setup", "all", "--skip-models"])

        self.assertEqual(result.exit_code, 0)
        mock_whisper.assert_called_once_with(skip_build=False, skip_models=True)
        mock_llama.assert_called_once_with(skip_build=False, skip_models=True)

    def test_run_defaults_to_no_translation(self):
        lrcer = MagicMock()
        lrcer.run.return_value = [Path("output.lrc")]
        lrcer_cls = MagicMock(return_value=lrcer)
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(app, ["run", "input.wav"])

        self.assertEqual(result.exit_code, 0)
        lrcer_cls.assert_called_once()
        kwargs = lrcer.run.call_args.kwargs
        self.assertTrue(kwargs["skip_trans"])
        lrcer.close.assert_called_once()

    def test_run_local_translation_uses_local_lrcer_and_closes(self):
        lrcer = MagicMock()
        lrcer.run.return_value = [Path("output.lrc")]
        lrcer_cls = MagicMock()
        lrcer_cls.local.return_value = lrcer
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(app, ["run", "input.wav", "--translation", "local"])

        self.assertEqual(result.exit_code, 0)
        lrcer_cls.local.assert_called_once()
        kwargs = lrcer.run.call_args.kwargs
        self.assertFalse(kwargs["skip_trans"])
        lrcer.close.assert_called_once()

    def test_translate_local_uses_local_lrcer_and_closes(self):
        lrcer = MagicMock()
        lrcer.translate.return_value = [Path("output.lrc")]
        lrcer_cls = MagicMock()
        lrcer_cls.local.return_value = lrcer
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(app, ["translate", "input.json", "--translation", "local"])

        self.assertEqual(result.exit_code, 0)
        lrcer_cls.local.assert_called_once()
        lrcer.translate.assert_called_once()
        lrcer.close.assert_called_once()

    def test_transcribe_prints_outputs_and_closes(self):
        lrcer = MagicMock()
        lrcer.transcribe.return_value = [Path("input_transcribed.json")]
        lrcer_cls = MagicMock(return_value=lrcer)
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(app, ["transcribe", "input.wav", "--src-lang", "en"])

        self.assertEqual(result.exit_code, 0)
        lrcer_cls.assert_called_once()
        lrcer.transcribe.assert_called_once()
        self.assertIn("input_transcribed.json", result.output)
        lrcer.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
