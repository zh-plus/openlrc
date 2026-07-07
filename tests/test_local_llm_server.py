#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import unittest
from unittest.mock import MagicMock, patch

import requests

from openlrc.local_llm_server import LocalLLMServer


def _response_with_alias(alias: str = "qwen3.5-9b-local"):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"data": [{"id": alias, "aliases": [alias]}]}
    return response


class TestLocalLLMServer(unittest.TestCase):
    def test_reuses_matching_external_server(self):
        with (
            patch("openlrc.local_llm_server.requests.get", return_value=_response_with_alias()),
            patch("openlrc.local_llm_server.subprocess.Popen") as mock_popen,
        ):
            server = LocalLLMServer()
            self.assertEqual(server.ensure_running(), "http://127.0.0.1:8088/v1")
            mock_popen.assert_not_called()
            self.assertFalse(server._owns_process)

    def test_raises_when_existing_server_has_wrong_alias(self):
        with patch("openlrc.local_llm_server.requests.get", return_value=_response_with_alias("other-model")):
            server = LocalLLMServer()
            with self.assertRaisesRegex(RuntimeError, "does not expose model alias"):
                server.ensure_running()

    def test_starts_server_and_waits_until_ready(self):
        process = MagicMock()
        process.poll.return_value = None

        with (
            patch(
                "openlrc.local_llm_server.requests.get", side_effect=[requests.ConnectionError, _response_with_alias()]
            ),
            patch("openlrc.local_llm_server.resolve_llama_server", return_value="/tmp/llama-server"),
            patch("openlrc.local_llm_server.resolve_llama_model_path", return_value="/tmp/model.gguf"),
            patch("openlrc.local_llm_server.subprocess.Popen", return_value=process) as mock_popen,
            patch("openlrc.local_llm_server.time.sleep"),
        ):
            server = LocalLLMServer(idle_timeout=0)
            self.assertEqual(server.ensure_running(), "http://127.0.0.1:8088/v1")

            cmd = mock_popen.call_args.args[0]
            self.assertIn("/tmp/llama-server", cmd)
            self.assertIn("/tmp/model.gguf", cmd)
            self.assertIn("--reasoning", cmd)
            self.assertIn("--reasoning-budget", cmd)
            self.assertTrue(server._owns_process)

    def test_session_defers_idle_shutdown_until_work_finishes(self):
        process = MagicMock()
        process.poll.return_value = None

        with (
            patch(
                "openlrc.local_llm_server.requests.get", side_effect=[requests.ConnectionError, _response_with_alias()]
            ),
            patch("openlrc.local_llm_server.resolve_llama_server", return_value="/tmp/llama-server"),
            patch("openlrc.local_llm_server.resolve_llama_model_path", return_value="/tmp/model.gguf"),
            patch("openlrc.local_llm_server.subprocess.Popen", return_value=process),
            patch("openlrc.local_llm_server.threading.Timer") as mock_timer,
            patch("openlrc.local_llm_server.time.sleep"),
        ):
            timer = MagicMock()
            mock_timer.return_value = timer
            server = LocalLLMServer(idle_timeout=60)

            with server.session():
                self.assertEqual(server._active_sessions, 1)
                mock_timer.assert_not_called()

            self.assertEqual(server._active_sessions, 0)
            mock_timer.assert_called_once()
            timer.start.assert_called_once()

    def test_close_terminates_owned_process(self):
        process = MagicMock()
        process.poll.return_value = None

        server = LocalLLMServer()
        server._process = process
        server._owns_process = True

        server.close()

        process.terminate.assert_called_once()
        process.wait.assert_called()
        self.assertIsNone(server._process)
