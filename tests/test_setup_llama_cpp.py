#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openlrc.setup import llama_cpp as setup_llama_cpp


class TestSetupLlamaCpp(unittest.TestCase):
    def test_download_model_skips_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            model_path = model_dir / "model.gguf"
            model_path.write_text("existing", encoding="utf-8")

            with patch("openlrc.setup.llama_cpp.run") as mock_run:
                result = setup_llama_cpp.download_model(
                    model_dir=model_dir,
                    model_repo="org/repo",
                    model_file="model.gguf",
                    model_url=None,
                    revision="main",
                    force=False,
                )

            self.assertEqual(result, model_path)
            mock_run.assert_not_called()

    def test_download_model_uses_huggingface_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)

            def fake_run(cmd, cwd=setup_llama_cpp.REPO_ROOT):
                Path(cmd[cmd.index("--output") + 1]).write_text("downloaded", encoding="utf-8")

            with patch("openlrc.setup.llama_cpp.run", side_effect=fake_run) as mock_run:
                result = setup_llama_cpp.download_model(
                    model_dir=model_dir,
                    model_repo="org/repo",
                    model_file="model file.gguf",
                    model_url=None,
                    revision="main",
                    force=False,
                )

            cmd = mock_run.call_args.args[0]
            self.assertEqual(result, model_dir / "model file.gguf")
            self.assertIn("curl", cmd)
            self.assertIn("https://huggingface.co/org/repo/resolve/main/model%20file.gguf", cmd)

    def test_build_checks_expected_binaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor_dir = Path(tmp)
            build_bin = vendor_dir / "build" / "bin"
            build_bin.mkdir(parents=True)
            (build_bin / "llama-server").write_text("", encoding="utf-8")
            (build_bin / "llama-cli").write_text("", encoding="utf-8")

            with (
                patch("openlrc.setup.llama_cpp.VENDOR_DIR", vendor_dir),
                patch("openlrc.setup.llama_cpp.run") as mock_run,
            ):
                server_path, cli_path = setup_llama_cpp.build_llama_cpp()

            self.assertEqual(server_path, build_bin / "llama-server")
            self.assertEqual(cli_path, build_bin / "llama-cli")
            self.assertEqual(mock_run.call_count, 2)
