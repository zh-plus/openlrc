#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openlrc.whisper_resources import (
    DEFAULT_MODEL_NAME,
    DEFAULT_VAD_MODEL_NAME,
    OPENLRC_WHISPER_CLI,
    OPENLRC_WHISPER_MODEL,
    OPENLRC_WHISPER_MODEL_DIR,
    OPENLRC_WHISPER_VAD_MODEL,
    SETUP_COMMAND,
    resolve_vad_model_path,
    resolve_whisper_cli,
    resolve_whisper_model_path,
)


def _touch(path: Path, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    if executable:
        path.chmod(0o755)
    return path


class TestWhisperResourceResolver(unittest.TestCase):
    def test_cli_explicit_and_env_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            explicit_cli = _touch(root / "explicit" / "whisper-cli", executable=True)
            env_cli = _touch(root / "env" / "whisper-cli", executable=True)
            vendor_cli = _touch(root / "vendor" / "whisper-cli", executable=True)
            path_cli = _touch(root / "path" / "whisper-cli", executable=True)

            with (
                patch.dict(os.environ, {OPENLRC_WHISPER_CLI: str(env_cli)}, clear=True),
                patch("openlrc.whisper_resources._app_bundle_cli_path", return_value=None),
                patch("openlrc.whisper_resources.vendor_cli_path", return_value=vendor_cli),
                patch("openlrc.whisper_resources.shutil.which", return_value=str(path_cli)),
            ):
                self.assertEqual(resolve_whisper_cli(str(explicit_cli)), str(explicit_cli.resolve()))
                self.assertEqual(resolve_whisper_cli(""), str(env_cli.resolve()))

    def test_cli_uses_vendor_before_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor_cli = _touch(root / "vendor" / "whisper-cli", executable=True)
            path_cli = _touch(root / "path" / "whisper-cli", executable=True)

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("openlrc.whisper_resources._app_bundle_cli_path", return_value=None),
                patch("openlrc.whisper_resources.vendor_cli_path", return_value=vendor_cli),
                patch("openlrc.whisper_resources.shutil.which", return_value=str(path_cli)),
            ):
                self.assertEqual(resolve_whisper_cli(""), str(vendor_cli.resolve()))

    def test_cli_uses_path_after_vendor_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path_cli = _touch(root / "path" / "whisper-cli", executable=True)

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("openlrc.whisper_resources._app_bundle_cli_path", return_value=None),
                patch("openlrc.whisper_resources.vendor_cli_path", return_value=root / "vendor" / "whisper-cli"),
                patch("openlrc.whisper_resources.shutil.which", return_value=str(path_cli)),
            ):
                self.assertEqual(resolve_whisper_cli(""), str(path_cli))

    def test_cli_missing_error_mentions_setup_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_vendor_cli = Path(tmp) / "missing" / "whisper-cli"
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("openlrc.whisper_resources._app_bundle_cli_path", return_value=None),
                patch("openlrc.whisper_resources.vendor_cli_path", return_value=missing_vendor_cli),
                patch("openlrc.whisper_resources.shutil.which", return_value=None),
            ):
                with self.assertRaises(FileNotFoundError) as ctx:
                    resolve_whisper_cli("")
                self.assertIn(SETUP_COMMAND, str(ctx.exception))
                self.assertIn(OPENLRC_WHISPER_CLI, str(ctx.exception))

    def test_model_precedence_and_default_env_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            explicit_model = _touch(root / "explicit" / "ggml-base.bin")
            env_model = _touch(root / "env" / "ggml-base.bin")
            user_model = _touch(root / "user" / DEFAULT_MODEL_NAME)
            vendor_model = _touch(root / "vendor" / DEFAULT_MODEL_NAME)

            with (
                patch.dict(
                    os.environ,
                    {OPENLRC_WHISPER_MODEL: str(env_model), OPENLRC_WHISPER_MODEL_DIR: str(root / "unused-model-dir")},
                    clear=True,
                ),
                patch("openlrc.whisper_resources.user_model_dir", return_value=user_model.parent),
                patch("openlrc.whisper_resources.vendor_model_dir", return_value=vendor_model.parent),
            ):
                self.assertEqual(resolve_whisper_model_path(str(explicit_model)), str(explicit_model.resolve()))
                self.assertEqual(resolve_whisper_model_path(DEFAULT_MODEL_NAME), str(env_model.resolve()))

    def test_model_uses_user_directory_before_vendor_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user_model = _touch(root / "user" / DEFAULT_MODEL_NAME)
            _touch(root / "vendor" / DEFAULT_MODEL_NAME)

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("openlrc.whisper_resources.user_model_dir", return_value=user_model.parent),
                patch("openlrc.whisper_resources.vendor_model_dir", return_value=root / "vendor"),
            ):
                self.assertEqual(resolve_whisper_model_path(DEFAULT_MODEL_NAME), str(user_model.resolve()))

    def test_model_uses_vendor_directory_when_user_model_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor_model = _touch(root / "vendor" / DEFAULT_MODEL_NAME)

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("openlrc.whisper_resources.user_model_dir", return_value=root / "user"),
                patch("openlrc.whisper_resources.vendor_model_dir", return_value=vendor_model.parent),
            ):
                self.assertEqual(resolve_whisper_model_path(DEFAULT_MODEL_NAME), str(vendor_model.resolve()))

    def test_vad_can_be_disabled_and_env_overrides_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_vad = _touch(root / "env" / DEFAULT_VAD_MODEL_NAME)

            with (
                patch.dict(os.environ, {OPENLRC_WHISPER_VAD_MODEL: str(env_vad)}, clear=True),
                patch("openlrc.whisper_resources.user_model_dir", return_value=root / "user"),
                patch("openlrc.whisper_resources.vendor_model_dir", return_value=root / "vendor"),
            ):
                self.assertEqual(resolve_vad_model_path(""), "")
                self.assertEqual(resolve_vad_model_path(DEFAULT_VAD_MODEL_NAME), str(env_vad.resolve()))

    def test_model_missing_error_mentions_setup_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("openlrc.whisper_resources.user_model_dir", return_value=root / "user"),
                patch("openlrc.whisper_resources.vendor_model_dir", return_value=root / "vendor"),
            ):
                with self.assertRaises(FileNotFoundError) as ctx:
                    resolve_whisper_model_path(DEFAULT_MODEL_NAME)
                self.assertIn(SETUP_COMMAND, str(ctx.exception))
                self.assertIn(OPENLRC_WHISPER_MODEL, str(ctx.exception))
