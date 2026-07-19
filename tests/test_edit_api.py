#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from openlrc.config import GlossaryOptions, TranslationConfig
from openlrc.context import TranslationBrief
from openlrc.edit_pipeline import EditPipeline
from openlrc.edit_validators import DeterministicValidatorSuite
from openlrc.editing import (
    EditAction,
    EditIssue,
    EditPatch,
    EditRound,
    EditSessionStatus,
    EditSeverity,
    apply_patch_transaction,
    save_edit_session,
)
from openlrc.openlrc import LRCer
from openlrc.subtitle import Subtitle


def write_subtitle(path: Path, texts: list[str], *, language: str, offset: float = 0.0) -> None:
    path.write_text(
        json.dumps(
            {
                "language": language,
                "segments": [
                    {"start": offset + index, "end": offset + index + 0.8, "text": text}
                    for index, text in enumerate(texts)
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class TestStandaloneEditAPI(unittest.TestCase):
    def test_verify_writes_report_and_does_not_load_a_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            target = root / "target.json"
            write_subtitle(source, ["Pay $25 on 2026-07-18."], language="en")
            write_subtitle(target, ["请付款。"], language="zh-cn")
            original = target.read_bytes()

            lrcer = LRCer()
            result = lrcer.edit(source, target, action="verify")

            self.assertEqual(target.read_bytes(), original)
            self.assertTrue(result.report_path.exists())
            self.assertTrue(result.session_path and result.session_path.exists())
            self.assertTrue(any(issue.severity is EditSeverity.ERROR for issue in result.unresolved_issues))
            self.assertIsNone(lrcer._chatbot)
            self.assertTrue((root / "target.edit-checkpoint.json").exists())

    def test_complete_verify_removes_process_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            target = root / "target.json"
            write_subtitle(source, ["Hello."], language="en")
            write_subtitle(target, ["你好。"], language="zh-cn")

            result = LRCer().edit(source, target, action="verify")

            self.assertEqual(result.unresolved_issues, [])
            self.assertFalse((root / "target.edit-checkpoint.json").exists())

    def test_report_matches_false_hides_match_details_but_keeps_compliance_issue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            target = root / "target.json"
            write_subtitle(source, ["Open the case."], language="en")
            write_subtitle(target, ["打开箱子。"], language="zh-cn")
            translation = TranslationConfig(
                glossary={"schema_version": 1, "entries": [{"source": "case", "target": "案件", "required": True}]},
                glossary_options=GlossaryOptions(report_matches=False),
            )

            result = LRCer(translation=translation).edit(source, target, action="verify")
            session_payload = json.loads(result.session_path.read_text(encoding="utf-8"))

            self.assertEqual(session_payload["glossary_state"]["matches"], [])
            self.assertEqual(session_payload["metrics"]["glossary"]["required_noncompliant"], 1)
            self.assertTrue(any(issue.category == "glossary" for issue in result.unresolved_issues))

    def test_restore_validates_and_restores_historical_round(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source.json"
            target_path = root / "target.json"
            session_path = root / "saved.edit-session.json"
            write_subtitle(source_path, ["Hello"], language="en")
            write_subtitle(target_path, ["您好"], language="zh-cn")
            source = Subtitle.from_json(source_path)
            suite = DeterministicValidatorSuite(source)
            pipeline = EditPipeline(source, suite, max_rounds=1, restore_enabled=True)
            session = pipeline.new_session(["你好"])
            patch = EditPatch.create(
                issue_ids=[], segment_id=1, before="你好", after="您好", reason="tone", action=EditAction.REVIEW
            )
            transaction = apply_patch_transaction(
                session.current_translations, [patch], scope_ids=[1], baseline_issues=[], validate=lambda _: []
            )
            session.current_translations = transaction.translations
            session.rounds.append(
                EditRound(round_index=1, scope_ids=[1], proposed_patches=[patch], applied_patches=[patch])
            )
            session.status = EditSessionStatus.COMPLETE
            pipeline._save(session)
            save_edit_session(session_path, session)

            result = LRCer().edit(
                source_path, target_path, action="restore", restore_round=0, session_path=session_path
            )

            self.assertEqual(result.changed_ids, [1])
            self.assertEqual(Subtitle.from_json(target_path).texts, ["你好"])

    def test_restore_rejects_stale_current_translation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source.json"
            target_path = root / "target.json"
            session_path = root / "saved.edit-session.json"
            write_subtitle(source_path, ["Hello"], language="en")
            write_subtitle(target_path, ["外部修改"], language="zh-cn")
            source = Subtitle.from_json(source_path)
            session = EditPipeline(source, DeterministicValidatorSuite(source)).new_session(["你好"])
            save_edit_session(session_path, session)

            with self.assertRaisesRegex(ValueError, "Current translation has changed"):
                LRCer().edit(source_path, target_path, action="restore", restore_round=0, session_path=session_path)

    def test_review_requires_explicit_context_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            target = root / "target.json"
            write_subtitle(source, ["Hello"], language="en")
            write_subtitle(target, ["你好"], language="zh-cn")

            with self.assertRaisesRegex(ValueError, "explicitly configured context model"):
                LRCer().edit(source, target, action="review", segment_ids=[1])

    def test_retranslate_requires_explicit_hymt2(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            target = root / "target.json"
            write_subtitle(source, ["Hello"], language="en")
            write_subtitle(target, ["你好"], language="zh-cn")

            with self.assertRaisesRegex(ValueError, "local Hy-MT2"):
                LRCer().edit(source, target, action="retranslate", segment_ids=[1])

    def test_review_resumes_only_unfinished_chunks_and_keeps_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            target = root / "target.json"
            texts = [f"line {index}" for index in range(1, 32)]
            write_subtitle(source, texts, language="en")
            write_subtitle(target, [f"译文 {index}" for index in range(1, 32)], language="zh-cn")
            bot = MagicMock(api_fees=[])
            bot.close = MagicMock()
            lrcer = LRCer()
            lrcer.context_llm = object()

            with (
                patch.object(lrcer, "_create_context_chatbot", return_value=(bot, None)),
                patch(
                    "openlrc.hymt2_pipeline.TranslationBriefAgent.build",
                    return_value=TranslationBrief(summary="summary"),
                ),
                patch(
                    "openlrc.hymt2_pipeline.HyMT2RiskReviewAgent.review_patches",
                    side_effect=[([], [], {"chunk": 1}), RuntimeError("interrupted")],
                ) as first_review,
            ):
                first = lrcer.edit(source, target, action="review")

            checkpoint_path = root / "target.edit-checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(len(first.rounds), 1)
            self.assertEqual(first_review.call_count, 2)
            self.assertEqual(checkpoint["model_progress"]["completed_chunks"], [1])
            self.assertTrue(checkpoint_path.exists())

            with (
                patch.object(lrcer, "_create_context_chatbot", return_value=(bot, None)),
                patch(
                    "openlrc.hymt2_pipeline.HyMT2RiskReviewAgent.review_patches", return_value=([], [], {"chunk": 2})
                ) as resumed_review,
            ):
                resumed = lrcer.edit(source, target, action="review")

            resumed_review.assert_called_once()
            self.assertEqual(resumed.unresolved_issues, [])
            self.assertFalse(checkpoint_path.exists())

    def test_committed_checkpoint_recovers_if_subtitle_replace_was_interrupted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            target = root / "target.json"
            write_subtitle(source, ["Hello"], language="en")
            write_subtitle(target, ["你好"], language="zh-cn")
            bot = MagicMock(api_fees=[])
            bot.close = MagicMock()
            lrcer = LRCer()
            lrcer.context_llm = object()
            found = EditIssue.create(
                segment_ids=[1],
                category="semantic",
                severity=EditSeverity.ERROR,
                source="semantic-review",
                message="use formal greeting",
                evidence={},
            )
            revision = EditPatch.create(
                issue_ids=[found.issue_id],
                segment_id=1,
                before="你好",
                after="您好",
                reason="formal tone",
                action=EditAction.REVIEW,
            )

            with (
                patch.object(lrcer, "_create_context_chatbot", return_value=(bot, None)),
                patch(
                    "openlrc.hymt2_pipeline.TranslationBriefAgent.build",
                    return_value=TranslationBrief(summary="summary"),
                ),
                patch(
                    "openlrc.hymt2_pipeline.HyMT2RiskReviewAgent.review_patches", return_value=([found], [revision], {})
                ),
                patch("openlrc.editing.atomic_save_subtitle", side_effect=OSError("replace interrupted")),
                self.assertRaisesRegex(OSError, "replace interrupted"),
            ):
                lrcer.edit(source, target, action="review")

            checkpoint_path = root / "target.edit-checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["phase"], "complete")
            self.assertEqual(Subtitle.from_json(target).texts, ["你好"])

            with (
                patch.object(lrcer, "_create_context_chatbot") as no_model,
                patch("openlrc.hymt2_pipeline.HyMT2RiskReviewAgent.review_patches") as no_review,
            ):
                result = lrcer.edit(source, target, action="review")

            no_model.assert_not_called()
            no_review.assert_not_called()
            self.assertEqual(result.changed_ids, [1])
            self.assertEqual(Subtitle.from_json(target).texts, ["您好"])
            self.assertFalse(checkpoint_path.exists())


if __name__ == "__main__":
    unittest.main()
