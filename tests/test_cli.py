#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from openlrc.cli.main import CheckResult, app
from openlrc.config import ContextAssistance, SubtitleOptimizationMode
from openlrc.editing import EditAction, EditIssue, EditResult, EditSeverity
from openlrc.setup.llama_cpp import LlamaSetupResult
from openlrc.setup.whisper_cpp import WhisperSetupResult
from openlrc.workflow import (
    ReviewStatus,
    RunRequest,
    TranscribeRequest,
    TranslateRequest,
    TranslationMode,
    WorkflowKind,
    WorkflowResult,
    WorkflowStatus,
)

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _plain_output(output: str) -> str:
    return _ANSI_ESCAPE.sub("", output)


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.workflow_requests = []

        def execute(request, workflow):
            self.workflow_requests.append(request)
            if isinstance(request, TranscribeRequest):
                outputs = (Path("input_transcribed.json"),)
            elif isinstance(request, TranslateRequest):
                outputs = (Path(f"{Path(request.transcribed_paths[0]).stem}.lrc"),)
            else:
                outputs = (Path("output.lrc"),)
            return WorkflowResult(
                job_id="cli-test",
                workflow=WorkflowKind(workflow),
                status=WorkflowStatus.SUCCEEDED,
                translation_mode=(
                    request.translation.mode
                    if isinstance(request, TranslateRequest)
                    else request.translation.mode
                    if isinstance(request, RunRequest) and request.translation is not None
                    else None
                ),
                outputs=outputs,
            )

        self.execute_patcher = patch("openlrc.cli.main._execute_workflow", side_effect=execute)
        self.execute_patcher.start()

    def tearDown(self):
        self.execute_patcher.stop()

    def test_help_and_version(self):
        help_result = self.runner.invoke(app, ["--help"])
        self.assertEqual(help_result.exit_code, 0)
        self.assertIn("OpenLRC Mac command line tools", help_result.output)

        version_result = self.runner.invoke(app, ["--version"])
        self.assertEqual(version_result.exit_code, 0)
        self.assertIn("OpenLRC Mac 0.4.2", version_result.output)
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

    def test_setup_llama_profile_uses_hy_mt2_defaults(self):
        setup_result = LlamaSetupResult(server_path=Path("/bin/llama-server"))
        with patch("openlrc.cli.main.setup_llama_cpp", return_value=setup_result) as mock_setup:
            result = self.runner.invoke(app, ["setup", "llama", "--local-model-profile", "hy-mt2-7b"])

        self.assertEqual(result.exit_code, 0)
        kwargs = mock_setup.call_args.kwargs
        self.assertEqual(kwargs["model_repo"], "tencent/Hy-MT2-7B-GGUF")
        self.assertEqual(kwargs["model_file"], "HY-MT2-7B-Q6_K.gguf")

    def test_setup_llama_30b_profile_rejects_download(self):
        result = self.runner.invoke(app, ["setup", "llama", "--local-model-profile", "hy-mt2-30b-a3b"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("no downloadable default GGUF", result.output)

    def test_setup_llama_30b_profile_allows_skip_models(self):
        setup_result = LlamaSetupResult(server_path=Path("/bin/llama-server"))
        with patch("openlrc.cli.main.setup_llama_cpp", return_value=setup_result) as mock_setup:
            result = self.runner.invoke(
                app, ["setup", "llama", "--local-model-profile", "hy-mt2-30b-a3b", "--skip-models"]
            )

        self.assertEqual(result.exit_code, 0)
        kwargs = mock_setup.call_args.kwargs
        self.assertTrue(kwargs["skip_models"])

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
        lrcer.review_statuses = {}
        lrcer_cls = MagicMock(return_value=lrcer)
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(app, ["run", "input.wav"])

        self.assertEqual(result.exit_code, 0)
        request = self.workflow_requests[-1]
        self.assertIsInstance(request, RunRequest)
        self.assertIsNone(request.translation)
        self.assertTrue(request.clear_temp)
        self.assertIs(request.subtitle_optimization, SubtitleOptimizationMode.AGGRESSIVE)

    def test_translate_accepts_relaxed_subtitle_optimization(self):
        lrcer = MagicMock()
        lrcer.translate.return_value = [Path("sample.lrc")]
        lrcer.review_statuses = {}
        lrcer_cls = MagicMock(return_value=lrcer)
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(
                app, ["translate", "sample.json", "--translation", "online", "--subtitle-optimization", "relaxed"]
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIs(self.workflow_requests[-1].subtitle_optimization, SubtitleOptimizationMode.RELAXED)

    def test_translate_passes_glossary_and_edit_options(self):
        lrcer = MagicMock()
        lrcer.translate.return_value = [Path("sample.lrc")]
        lrcer.review_statuses = {}
        lrcer_cls = MagicMock()
        lrcer_cls.local_hy_mt2.return_value = lrcer
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(
                app,
                [
                    "translate",
                    "sample.json",
                    "--translation",
                    "local",
                    "--llama-model",
                    "hy-mt2-7b",
                    "--hy-mt2-mode",
                    "normal-plus",
                    "--context-provider",
                    "local",
                    "--context-model",
                    "qwen3.5-9b",
                    "--glossary",
                    "terms.json",
                    "--force-glossary",
                    "--edit-rounds",
                    "2",
                    "--enable-restore",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        config = self.workflow_requests[-1].translation.config
        self.assertEqual(config.glossary, "terms.json")
        self.assertTrue(config.glossary_options.force)
        self.assertEqual(config.edit_config.max_rounds, 2)
        self.assertTrue(config.edit_config.restore_enabled)

    def test_translate_passes_complete_manual_brief_without_context_model(self):
        lrcer = MagicMock()
        lrcer.translate.return_value = [Path("sample.lrc")]
        lrcer.review_statuses = {}
        lrcer_cls = MagicMock()
        lrcer_cls.local_hy_mt2.return_value = lrcer
        characters = json.dumps([{"source_name": "John", "target_name": "强尼"}], ensure_ascii=False)

        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(
                app,
                [
                    "translate",
                    "sample.json",
                    "--translation",
                    "local",
                    "--llama-model",
                    "hy-mt2-7b",
                    "--hy-mt2-mode",
                    "normal",
                    "--brief-summary",
                    "A courtroom drama.",
                    "--brief-characters",
                    characters,
                    "--brief-tone-style",
                    "Natural, restrained dialogue.",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        config = self.workflow_requests[-1].translation.config
        self.assertIsNone(config.context_llm)
        brief = config.translation_brief
        self.assertEqual(brief.summary, "A courtroom drama.")
        self.assertEqual(brief.characters[0].source_name, "John")
        self.assertEqual(brief.characters[0].target_name, "强尼")
        self.assertEqual(brief.tone_style, "Natural, restrained dialogue.")

    def test_translate_context_assistance_off_normalizes_empty_manual_brief_sections(self):
        result = self.runner.invoke(
            app,
            [
                "translate",
                "sample.json",
                "--translation",
                "local",
                "--llama-model",
                "hy-mt2-7b",
                "--hy-mt2-mode",
                "normal",
                "--context-assistance",
                "off",
                "--brief-summary",
                "A courtroom drama.",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        config = self.workflow_requests[-1].translation.config
        self.assertIs(config.context_assistance, ContextAssistance.OFF)
        self.assertIsNone(config.context_llm)
        self.assertEqual(config.translation_brief.characters, [])
        self.assertEqual(config.translation_brief.tone_style, "")

    def test_context_assistance_off_uses_shared_cli_restrictions(self):
        base = [
            "translate",
            "sample.json",
            "--translation",
            "local",
            "--llama-model",
            "hy-mt2-7b",
            "--context-assistance",
            "off",
            "--brief-summary",
            "Known story.",
        ]
        cases = [
            ([*base, "--hy-mt2-mode", "pro"], "Context assistance cannot be off"),
            ([*base, "--hy-mt2-mode", "normal-plus"], "Normal Plus semantic review"),
            (
                [*base, "--hy-mt2-mode", "normal", "--context-provider", "local", "--context-model", "qwen3.5-9b"],
                "combined with Context provider/model",
            ),
        ]

        for arguments, message in cases:
            with self.subTest(message=message):
                result = self.runner.invoke(app, arguments)
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn(message, result.output)

    def test_translate_loads_brief_characters_from_utf8_json_file(self):
        lrcer = MagicMock()
        lrcer.translate.return_value = [Path("sample.lrc")]
        lrcer.review_statuses = {}
        lrcer_cls = MagicMock()
        lrcer_cls.local_hy_mt2.return_value = lrcer

        with tempfile.TemporaryDirectory() as tmpdir:
            characters_path = Path(tmpdir) / "characters.json"
            characters_path.write_text(
                json.dumps([{"source_name": "Mei", "target_name": "梅"}], ensure_ascii=False), encoding="utf-8"
            )
            with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
                result = self.runner.invoke(
                    app,
                    [
                        "translate",
                        "sample.json",
                        "--translation",
                        "local",
                        "--llama-model",
                        "hy-mt2-7b",
                        "--hy-mt2-mode",
                        "normal",
                        "--context-provider",
                        "local",
                        "--context-model",
                        "qwen3.5-9b",
                        "--brief-characters",
                        f"@{characters_path}",
                    ],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        brief = self.workflow_requests[-1].translation.config.translation_brief
        self.assertIsNone(brief.summary)
        self.assertEqual(brief.characters[0].target_name, "梅")
        self.assertIsNone(brief.tone_style)

    def test_brief_characters_rejects_non_array_and_invalid_entries(self):
        base = [
            "translate",
            "sample.json",
            "--translation",
            "local",
            "--llama-model",
            "hy-mt2-7b",
            "--hy-mt2-mode",
            "normal",
        ]
        non_array = self.runner.invoke(app, [*base, "--brief-characters", '{"John":"强尼"}'])
        extra_field = self.runner.invoke(
            app, [*base, "--brief-characters", '[{"source_name":"John","target_name":"强尼","note":"lead"}]']
        )
        empty_name = self.runner.invoke(app, [*base, "--brief-characters", '[{"source_name":"","target_name":"强尼"}]'])

        self.assertNotEqual(non_array.exit_code, 0)
        self.assertIn("Brief characters JSON", non_array.output)
        self.assertIn("array.", non_array.output)
        self.assertNotEqual(extra_field.exit_code, 0)
        self.assertIn("Invalid Translation Brief", extra_field.output)
        self.assertNotEqual(empty_name.exit_code, 0)
        self.assertIn("Invalid Translation Brief", empty_name.output)

    def test_brief_characters_reports_invalid_json_and_missing_file(self):
        base = [
            "translate",
            "sample.json",
            "--translation",
            "local",
            "--llama-model",
            "hy-mt2-7b",
            "--hy-mt2-mode",
            "normal",
        ]
        invalid_json = self.runner.invoke(app, [*base, "--brief-characters", "[invalid"])
        missing_file = self.runner.invoke(app, [*base, "--brief-characters", "@does-not-exist.json"])

        self.assertNotEqual(invalid_json.exit_code, 0)
        self.assertIn("must be valid JSON", invalid_json.output)
        self.assertNotEqual(missing_file.exit_code, 0)
        self.assertIn("Cannot read brief characters file", missing_file.output)

    def test_manual_brief_rejects_fast_and_non_hymt2_backends(self):
        fast = self.runner.invoke(
            app,
            [
                "translate",
                "sample.json",
                "--translation",
                "local",
                "--llama-model",
                "hy-mt2-7b",
                "--brief-summary",
                "Known story.",
            ],
        )
        online = self.runner.invoke(
            app, ["translate", "sample.json", "--translation", "online", "--brief-summary", "Known story."]
        )

        self.assertNotEqual(fast.exit_code, 0)
        self.assertIn("not supported in Hy-MT2 fast mode", fast.output)
        self.assertNotEqual(online.exit_code, 0)
        self.assertIn("requires a Hy-MT2 local translation", online.output)
        self.assertIn("backend.", online.output)

    def test_glossary_validate_and_check_exit_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            glossary = root / "glossary.json"
            source = root / "source.json"
            target = root / "target.json"
            glossary.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_language": "en",
                        "target_language": "zh-cn",
                        "entries": [{"source": "case", "target": "案件", "required": True}],
                    }
                ),
                encoding="utf-8",
            )
            source.write_text(
                json.dumps({"language": "en", "segments": [{"start": 0, "end": 1, "text": "the case"}]}),
                encoding="utf-8",
            )
            target.write_text(
                json.dumps(
                    {"language": "zh-cn", "segments": [{"start": 0, "end": 1, "text": "这个箱子"}]}, ensure_ascii=False
                ),
                encoding="utf-8",
            )

            valid = self.runner.invoke(app, ["glossary", "validate", str(glossary)])
            checked = self.runner.invoke(
                app, ["glossary", "check", str(glossary), "--source", str(source), "--target", str(target)]
            )

            self.assertEqual(valid.exit_code, 0, valid.output)
            self.assertEqual(checked.exit_code, 1, checked.output)
            self.assertTrue((root / "target.glossary-report.json").exists())

    def test_edit_verify_returns_nonzero_for_hard_issues(self):
        lrcer = MagicMock()
        hard = EditIssue.create(
            segment_ids=[1],
            category="number",
            severity=EditSeverity.ERROR,
            source="number-preservation",
            message="missing number",
        )
        lrcer.edit.return_value = EditResult(
            action=EditAction.VERIFY,
            output_path=Path("target.json"),
            report_path=Path("target.edit-report.json"),
            unresolved_issues=[hard],
        )
        lrcer_cls = MagicMock(return_value=lrcer)
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(
                app, ["edit", "--source", "source.json", "--target", "target.json", "--action", "verify"]
            )

        self.assertEqual(result.exit_code, 1, result.output)
        lrcer.edit.assert_called_once()
        lrcer.close.assert_called_once()

    def test_edit_retranslate_requires_explicit_hymt2_model(self):
        result = self.runner.invoke(
            app,
            ["edit", "--source", "source.json", "--target", "target.json", "--action", "retranslate", "--ids", "1-2"],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("requires --llama-model", _plain_output(result.output))

    def test_edit_verify_and_restore_reject_manual_brief(self):
        for action in ("verify", "restore"):
            with self.subTest(action=action):
                result = self.runner.invoke(
                    app,
                    [
                        "edit",
                        "--source",
                        "source.json",
                        "--target",
                        "target.json",
                        "--action",
                        action,
                        "--brief-summary",
                        "Known story.",
                    ],
                )

                self.assertNotEqual(result.exit_code, 0)
                self.assertIn(f"not supported by edit {action}", result.output)

    def test_edit_review_passes_partial_manual_brief(self):
        lrcer = MagicMock()
        lrcer.edit.return_value = MagicMock(changed_ids=[], unresolved_issues=[], report_path=Path("report.json"))
        lrcer_cls = MagicMock(return_value=lrcer)
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(
                app,
                [
                    "edit",
                    "--source",
                    "source.json",
                    "--target",
                    "target.json",
                    "--action",
                    "review",
                    "--ids",
                    "1-2",
                    "--context-provider",
                    "local",
                    "--context-model",
                    "qwen3.5-9b",
                    "--brief-tone-style",
                    "Dry comedy.",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        config = lrcer_cls.call_args.kwargs["translation"]
        self.assertEqual(config.translation_brief.tone_style, "Dry comedy.")
        self.assertIsNone(config.translation_brief.summary)

    def test_edit_retranslate_complete_brief_skips_context_model(self):
        lrcer = MagicMock()
        lrcer.edit.return_value = MagicMock(changed_ids=[], unresolved_issues=[], report_path=Path("report.json"))
        lrcer_cls = MagicMock()
        lrcer_cls.local_hy_mt2.return_value = lrcer
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(
                app,
                [
                    "edit",
                    "--source",
                    "source.json",
                    "--target",
                    "target.json",
                    "--action",
                    "retranslate",
                    "--ids",
                    "1",
                    "--llama-model",
                    "hy-mt2-7b",
                    "--hy-mt2-mode",
                    "normal-plus",
                    "--brief-summary",
                    "Known story.",
                    "--brief-characters",
                    "[]",
                    "--brief-tone-style",
                    "",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        config = lrcer_cls.call_args.kwargs["translation"]
        self.assertIsNone(config.context_llm)
        self.assertEqual(config.edit_config.max_rounds, 0)
        self.assertFalse(config.edit_config.semantic_review)
        self.assertEqual(config.translation_brief.characters, [])

    def test_edit_retranslate_supports_context_assistance_off(self):
        lrcer = MagicMock()
        lrcer.edit.return_value = MagicMock(changed_ids=[], unresolved_issues=[], report_path=Path("report.json"))
        lrcer_cls = MagicMock(return_value=lrcer)
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(
                app,
                [
                    "edit",
                    "--source",
                    "source.json",
                    "--target",
                    "target.json",
                    "--action",
                    "retranslate",
                    "--ids",
                    "1",
                    "--llama-model",
                    "hy-mt2-7b",
                    "--hy-mt2-mode",
                    "normal",
                    "--context-assistance",
                    "off",
                    "--brief-summary",
                    "Known story.",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        config = lrcer_cls.call_args.kwargs["translation"]
        self.assertIs(config.context_assistance, ContextAssistance.OFF)
        self.assertIsNone(config.context_llm)
        self.assertEqual(config.translation_brief.characters, [])
        self.assertEqual(config.translation_brief.tone_style, "")

    def test_edit_non_retranslate_actions_reject_context_assistance_off(self):
        result = self.runner.invoke(
            app,
            [
                "edit",
                "--source",
                "source.json",
                "--target",
                "target.json",
                "--action",
                "review",
                "--ids",
                "1",
                "--context-assistance",
                "off",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--context-assistance off is only", _plain_output(result.output))

    def test_run_keep_temp_disables_cleanup(self):
        lrcer = MagicMock()
        lrcer.run.return_value = [Path("output.lrc")]
        lrcer.review_statuses = {}
        lrcer_cls = MagicMock(return_value=lrcer)
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(app, ["run", "input.wav", "--keep-temp"])

        self.assertEqual(result.exit_code, 0)
        self.assertFalse(self.workflow_requests[-1].clear_temp)

    def test_translate_prints_incomplete_review_status(self):
        lrcer = MagicMock()
        lrcer.translate.return_value = [Path("sample.lrc")]
        lrcer.review_statuses = {"sample": {"incomplete": True, "failed_chunks": [2], "total_chunks": 3}}
        lrcer_cls = MagicMock(return_value=lrcer)
        workflow_result = WorkflowResult(
            job_id="cli-test",
            workflow=WorkflowKind.TRANSLATE,
            status=WorkflowStatus.SUCCEEDED_WITH_WARNINGS,
            translation_mode=TranslationMode.STANDARD,
            outputs=(Path("sample.lrc"),),
            reviews=(ReviewStatus(item="sample", incomplete=True, details={"failed_chunks": [2], "total_chunks": 3}),),
        )
        with (
            patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls),
            patch("openlrc.cli.main._execute_workflow", return_value=workflow_result),
        ):
            result = self.runner.invoke(app, ["translate", "sample.json", "--translation", "online"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Editing incomplete", result.output)
        self.assertIn("chunks [2]", result.output)

    def test_run_local_translation_uses_local_lrcer_and_closes(self):
        lrcer = MagicMock()
        lrcer.run.return_value = [Path("output.lrc")]
        lrcer_cls = MagicMock()
        lrcer_cls.local.return_value = lrcer
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(app, ["run", "input.wav", "--translation", "local"])

        self.assertEqual(result.exit_code, 0)
        config = self.workflow_requests[-1].translation
        self.assertEqual(config.mode, TranslationMode.STANDARD)
        self.assertEqual(config.config.chatbot.provider.value, "local_llama")

    def test_run_local_translation_infers_hy_mt2_profile_from_model_alias(self):
        lrcer = MagicMock()
        lrcer.run.return_value = [Path("output.lrc")]
        lrcer_cls = MagicMock()
        lrcer_cls.local_hy_mt2.return_value = lrcer
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(
                app, ["run", "input.wav", "--translation", "local", "--llama-model", "hy-mt2-7b"]
            )

        self.assertEqual(result.exit_code, 0)
        config = self.workflow_requests[-1].translation.config
        self.assertEqual(config.local_llm.model_path, "HY-MT2-7B-Q6_K.gguf")
        self.assertEqual(config.hy_mt2_mode.value, "fast")

    def test_run_hy_mt2_context_requires_explicit_context_model(self):
        result = self.runner.invoke(
            app,
            ["run", "input.wav", "--translation", "local", "--llama-model", "hy-mt2-7b", "--hy-mt2-mode", "context"],
        )

        self.assertNotEqual(result.exit_code, 0)
        output = _plain_output(result.output)
        self.assertIn("requires both --context-provider", output)
        self.assertIn("--context-model", output)

    def test_run_hy_mt2_local_context_builds_staged_qwen_config(self):
        lrcer = MagicMock()
        lrcer.run.return_value = [Path("output.lrc")]
        lrcer_cls = MagicMock()
        lrcer_cls.local_hy_mt2.return_value = lrcer
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(
                app,
                [
                    "run",
                    "input.wav",
                    "--translation",
                    "local",
                    "--llama-model",
                    "hy-mt2-7b",
                    "--hy-mt2-mode",
                    "context-plus",
                    "--context-provider",
                    "local",
                    "--context-model",
                    "qwen3.5-9b",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        config = self.workflow_requests[-1].translation.config
        self.assertEqual(config.hy_mt2_mode.value, "normal-plus")
        self.assertIsNotNone(config.context_llm.local_llm)
        self.assertEqual(config.context_llm.local_llm.model_path, "Qwen3.5-9B-Q4_K_M.gguf")

    def test_run_passes_complete_manual_brief_without_context_model(self):
        lrcer = MagicMock()
        lrcer.run.return_value = [Path("output.lrc")]
        lrcer.review_statuses = {}
        lrcer_cls = MagicMock()
        lrcer_cls.local_hy_mt2.return_value = lrcer
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(
                app,
                [
                    "run",
                    "input.wav",
                    "--translation",
                    "local",
                    "--llama-model",
                    "hy-mt2-7b",
                    "--hy-mt2-mode",
                    "normal",
                    "--brief-summary",
                    "Known film.",
                    "--brief-characters",
                    "[]",
                    "--brief-tone-style",
                    "Conversational.",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        config = self.workflow_requests[-1].translation.config
        self.assertIsNone(config.context_llm)
        self.assertEqual(config.translation_brief.summary, "Known film.")

    def test_run_context_assistance_off_uses_manual_brief_without_context_model(self):
        result = self.runner.invoke(
            app,
            [
                "run",
                "input.wav",
                "--translation",
                "local",
                "--llama-model",
                "hy-mt2-7b",
                "--hy-mt2-mode",
                "normal",
                "--context-assistance",
                "off",
                "--brief-summary",
                "Known film.",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        config = self.workflow_requests[-1].translation.config
        self.assertIs(config.context_assistance, ContextAssistance.OFF)
        self.assertIsNone(config.context_llm)
        self.assertEqual(config.translation_brief.characters, [])
        self.assertEqual(config.translation_brief.tone_style, "")

    def test_run_hy_mt2_pro_builds_staged_context_config(self):
        lrcer = MagicMock()
        lrcer.run.return_value = [Path("output.lrc")]
        lrcer.review_statuses = {}
        lrcer_cls = MagicMock()
        lrcer_cls.local_hy_mt2.return_value = lrcer
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(
                app,
                [
                    "run",
                    "input.wav",
                    "--translation",
                    "local",
                    "--llama-model",
                    "hy-mt2-7b",
                    "--hy-mt2-mode",
                    "pro",
                    "--context-provider",
                    "local",
                    "--context-model",
                    "qwen3.5-9b",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        config = self.workflow_requests[-1].translation.config
        self.assertEqual(config.hy_mt2_mode.value, "pro")
        self.assertIsNotNone(config.context_llm.local_llm)

    def test_run_hy_mt2_online_context_passes_provider_and_model(self):
        lrcer = MagicMock()
        lrcer.run.return_value = [Path("output.lrc")]
        lrcer_cls = MagicMock()
        lrcer_cls.local_hy_mt2.return_value = lrcer
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(
                app,
                [
                    "run",
                    "input.wav",
                    "--translation",
                    "local",
                    "--llama-model",
                    "hy-mt2-7b",
                    "--hy-mt2-mode",
                    "context",
                    "--context-provider",
                    "openai",
                    "--context-model",
                    "gpt-4.1-nano",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        context_llm = self.workflow_requests[-1].translation.config.context_llm
        self.assertIsNone(context_llm.local_llm)
        self.assertEqual(context_llm.chatbot.name, "gpt-4.1-nano")

    def test_run_help_hides_internal_translation_engine(self):
        result = self.runner.invoke(app, ["run", "--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertNotIn("--translate-mode", _plain_output(result.output))

    def test_run_local_translation_30b_requires_model_path(self):
        result = self.runner.invoke(
            app, ["run", "input.wav", "--translation", "local", "--local-model-profile", "hy-mt2-30b-a3b"]
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("requires --llama-model", _plain_output(result.output))

    def test_run_local_translation_rejects_conflicting_profile_alias(self):
        result = self.runner.invoke(
            app,
            [
                "run",
                "input.wav",
                "--translation",
                "local",
                "--local-model-profile",
                "hy-mt2-7b",
                "--llama-model",
                "hy-mt2-30b-a3b",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("conflicts with", result.output)
        self.assertIn("--local-model-profile", _plain_output(result.output))

    def test_run_local_translation_rejects_qwen_profile_with_hy_mt2_alias(self):
        result = self.runner.invoke(
            app,
            [
                "run",
                "input.wav",
                "--translation",
                "local",
                "--local-model-profile",
                "qwen3.5-9b",
                "--llama-model",
                "hy-mt2-7b",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("conflicts with", result.output)
        self.assertIn("--local-model-profile", _plain_output(result.output))

    def test_translate_local_uses_local_lrcer_and_closes(self):
        lrcer = MagicMock()
        lrcer.translate.return_value = [Path("output.lrc")]
        lrcer_cls = MagicMock()
        lrcer_cls.local.return_value = lrcer
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(app, ["translate", "input.json", "--translation", "local"])

        self.assertEqual(result.exit_code, 0)
        config = self.workflow_requests[-1].translation
        self.assertEqual(config.mode, TranslationMode.STANDARD)
        self.assertEqual(config.config.chatbot.provider.value, "local_llama")

    def test_translate_local_hy_mt2_30b_uses_explicit_model(self):
        lrcer = MagicMock()
        lrcer.translate.return_value = [Path("output.lrc")]
        lrcer_cls = MagicMock()
        lrcer_cls.local_hy_mt2.return_value = lrcer
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(
                app,
                [
                    "translate",
                    "input.json",
                    "--translation",
                    "local",
                    "--local-model-profile",
                    "hy-mt2-30b-a3b",
                    "--llama-model",
                    "/models/hy-mt2-30b-a3b-q6.gguf",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        config = self.workflow_requests[-1].translation.config
        self.assertEqual(config.local_llm.model_path, "/models/hy-mt2-30b-a3b-q6.gguf")
        self.assertEqual(config.hy_mt2_mode.value, "fast")

    def test_transcribe_prints_outputs_and_closes(self):
        lrcer = MagicMock()
        lrcer.transcribe.return_value = [Path("input_transcribed.json")]
        lrcer_cls = MagicMock(return_value=lrcer)
        with patch("openlrc.cli.main._lrcer_cls", return_value=lrcer_cls):
            result = self.runner.invoke(app, ["transcribe", "input.wav", "--src-lang", "en"])

        self.assertEqual(result.exit_code, 0)
        self.assertIsInstance(self.workflow_requests[-1], TranscribeRequest)
        self.assertEqual(self.workflow_requests[-1].src_lang, "en")
        self.assertIn("input_transcribed.json", result.output)

    def test_transcribe_exposes_explicit_whisper_cpu_compatibility_flags(self):
        result = self.runner.invoke(app, ["transcribe", "input.wav", "--no-whisper-gpu", "--no-whisper-flash-attn"])

        self.assertEqual(result.exit_code, 0, result.output)
        options = self.workflow_requests[-1].transcription.asr_options
        self.assertEqual(options, {"use_gpu": False, "flash_attn": False})

    def test_run_forwards_whisper_acceleration_defaults_and_overrides(self):
        default_result = self.runner.invoke(app, ["run", "input.wav"])
        default_options = self.workflow_requests[-1].transcription.asr_options
        cpu_result = self.runner.invoke(app, ["run", "input.wav", "--no-whisper-gpu", "--no-whisper-flash-attn"])
        cpu_options = self.workflow_requests[-1].transcription.asr_options

        self.assertEqual(default_result.exit_code, 0, default_result.output)
        self.assertEqual(default_options, {"use_gpu": True, "flash_attn": True})
        self.assertEqual(cpu_result.exit_code, 0, cpu_result.output)
        self.assertEqual(cpu_options, {"use_gpu": False, "flash_attn": False})

    def test_transcribe_help_documents_whisper_acceleration_flags(self):
        result = self.runner.invoke(app, ["transcribe", "--help"], terminal_width=160)

        self.assertEqual(result.exit_code, 0, result.output)
        output = _plain_output(result.output)
        self.assertIn("--whisper-gpu", output)
        self.assertIn("--no-whisper-gpu", output)
        self.assertIn("--whisper-flash-attn", output)
        self.assertIn("--no-whisper-flash", output)
        self.assertNotIn("--noise-suppress", output)

    def test_removed_noise_suppress_option_is_rejected(self):
        for command in ("transcribe", "run"):
            result = self.runner.invoke(app, [command, "input.wav", "--noise-suppress"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("No such option", result.output)


if __name__ == "__main__":
    unittest.main()
