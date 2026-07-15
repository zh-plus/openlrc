#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openlrc.llama_resources import (
    DEFAULT_LLAMA_MODEL_FILE,
    HY_MT2_7B_MODEL_FILE,
    HY_MT2_7B_MODEL_REPO,
    HY_MT2_7B_PROFILE,
    HY_MT2_30B_A3B_PROFILE,
    OPENLRC_LLAMA_MODEL,
    OPENLRC_LLAMA_MODEL_DIR,
    OPENLRC_LLAMA_SERVER,
    SETUP_COMMAND,
    default_model_url,
    get_local_llm_profile,
    infer_local_llm_profile,
    normalize_llama_model_name,
    resolve_llama_cli,
    resolve_llama_model_path,
    resolve_llama_server,
)


def _touch(path: Path, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    if executable:
        path.chmod(0o755)
    return path


class TestLlamaResourceResolver(unittest.TestCase):
    def test_server_explicit_and_env_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            explicit_server = _touch(root / "explicit" / "llama-server", executable=True)
            env_server = _touch(root / "env" / "llama-server", executable=True)
            vendor_server = _touch(root / "vendor" / "llama-server", executable=True)
            path_server = _touch(root / "path" / "llama-server", executable=True)

            with (
                patch.dict(os.environ, {OPENLRC_LLAMA_SERVER: str(env_server)}, clear=True),
                patch("openlrc.llama_resources._app_bundle_binary_path", return_value=None),
                patch("openlrc.llama_resources.vendor_server_path", return_value=vendor_server),
                patch("openlrc.llama_resources.shutil.which", return_value=str(path_server)),
            ):
                self.assertEqual(resolve_llama_server(str(explicit_server)), str(explicit_server.resolve()))
                self.assertEqual(resolve_llama_server(""), str(env_server.resolve()))

    def test_server_uses_vendor_before_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor_server = _touch(root / "vendor" / "llama-server", executable=True)
            path_server = _touch(root / "path" / "llama-server", executable=True)

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("openlrc.llama_resources._app_bundle_binary_path", return_value=None),
                patch("openlrc.llama_resources.vendor_server_path", return_value=vendor_server),
                patch("openlrc.llama_resources.shutil.which", return_value=str(path_server)),
            ):
                self.assertEqual(resolve_llama_server(""), str(vendor_server.resolve()))

    def test_cli_uses_vendor_before_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor_cli = _touch(root / "vendor" / "llama-cli", executable=True)
            path_cli = _touch(root / "path" / "llama-cli", executable=True)

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("openlrc.llama_resources._app_bundle_binary_path", return_value=None),
                patch("openlrc.llama_resources.vendor_cli_path", return_value=vendor_cli),
                patch("openlrc.llama_resources.shutil.which", return_value=str(path_cli)),
            ):
                self.assertEqual(resolve_llama_cli(""), str(vendor_cli.resolve()))

    def test_missing_server_error_mentions_setup_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("openlrc.llama_resources._app_bundle_binary_path", return_value=None),
                patch("openlrc.llama_resources.vendor_server_path", return_value=Path(tmp) / "missing"),
                patch("openlrc.llama_resources.shutil.which", return_value=None),
            ):
                with self.assertRaises(FileNotFoundError) as ctx:
                    resolve_llama_server("")
                self.assertIn(SETUP_COMMAND, str(ctx.exception))
                self.assertIn(OPENLRC_LLAMA_SERVER, str(ctx.exception))

    def test_model_precedence_and_env_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            explicit_model = _touch(root / "explicit" / DEFAULT_LLAMA_MODEL_FILE)
            env_model = _touch(root / "env" / DEFAULT_LLAMA_MODEL_FILE)
            user_model = _touch(root / "user" / DEFAULT_LLAMA_MODEL_FILE)
            vendor_model = _touch(root / "vendor" / DEFAULT_LLAMA_MODEL_FILE)

            with (
                patch.dict(
                    os.environ,
                    {OPENLRC_LLAMA_MODEL: str(env_model), OPENLRC_LLAMA_MODEL_DIR: str(root / "unused")},
                    clear=True,
                ),
                patch("openlrc.llama_resources.user_llm_model_dir", return_value=user_model.parent),
                patch("openlrc.llama_resources.vendor_model_dir", return_value=vendor_model.parent),
            ):
                self.assertEqual(resolve_llama_model_path(str(explicit_model)), str(explicit_model.resolve()))
                self.assertEqual(resolve_llama_model_path(DEFAULT_LLAMA_MODEL_FILE), str(env_model.resolve()))

    def test_model_uses_user_directory_before_vendor_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user_model = _touch(root / "user" / DEFAULT_LLAMA_MODEL_FILE)
            _touch(root / "vendor" / DEFAULT_LLAMA_MODEL_FILE)

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("openlrc.llama_resources.user_llm_model_dir", return_value=user_model.parent),
                patch("openlrc.llama_resources.vendor_model_dir", return_value=root / "vendor"),
            ):
                self.assertEqual(resolve_llama_model_path(DEFAULT_LLAMA_MODEL_FILE), str(user_model.resolve()))

    def test_model_alias_and_url_helpers(self):
        self.assertEqual(normalize_llama_model_name("qwen3.5-9b"), DEFAULT_LLAMA_MODEL_FILE)
        self.assertEqual(normalize_llama_model_name("hy-mt2-7b"), HY_MT2_7B_MODEL_FILE)
        self.assertEqual(normalize_llama_model_name("hy-mt2-7b-q6"), HY_MT2_7B_MODEL_FILE)
        self.assertEqual(normalize_llama_model_name("hy-mt2-7b-q4"), "Hy-MT2-7B-Q4_K_M.gguf")
        self.assertEqual(normalize_llama_model_name("custom-model"), "custom-model.gguf")
        self.assertEqual(
            default_model_url("org/repo", "model file.gguf", "main"),
            "https://huggingface.co/org/repo/resolve/main/model%20file.gguf",
        )

    def test_hy_mt2_profile_registry(self):
        profile = get_local_llm_profile(HY_MT2_7B_PROFILE)
        self.assertEqual(profile.model_repo, HY_MT2_7B_MODEL_REPO)
        self.assertEqual(profile.model_file, HY_MT2_7B_MODEL_FILE)
        self.assertEqual(profile.temperature, 0.7)
        self.assertEqual(profile.top_p, 0.6)
        self.assertEqual(profile.top_k, 20)
        self.assertEqual(profile.repeat_penalty, 1.05)
        self.assertEqual(profile.max_tokens, 4096)
        self.assertEqual(infer_local_llm_profile("hy-mt2-7b"), HY_MT2_7B_PROFILE)
        self.assertIsNone(infer_local_llm_profile(HY_MT2_30B_A3B_PROFILE))

        profile_30b = get_local_llm_profile(HY_MT2_30B_A3B_PROFILE)
        self.assertIsNone(profile_30b.model_repo)
        self.assertIsNone(profile_30b.model_file)
