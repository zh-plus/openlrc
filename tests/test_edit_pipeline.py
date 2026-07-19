#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from openlrc.edit_pipeline import EditPipeline
from openlrc.edit_validators import DeterministicValidatorSuite
from openlrc.editing import (
    EditAction,
    EditIssue,
    EditIssueStatus,
    EditPatch,
    EditRound,
    EditRoundStatus,
    EditSessionStatus,
    EditSeverity,
    EditStopReason,
)
from openlrc.glossary import GlossaryCatalog, GlossaryEntry, GlossaryService
from openlrc.subtitle import Subtitle


def subtitle(texts):
    return Subtitle(
        language="en",
        segments=[{"start": float(index), "end": float(index + 1), "text": text} for index, text in enumerate(texts)],
        filename=Path("source.json"),
    )


def semantic_issue(segment_id, message="meaning error"):
    return EditIssue.create(
        segment_ids=[segment_id],
        category="semantic",
        severity=EditSeverity.ERROR,
        source="semantic-review",
        message=message,
        evidence={},
    )


class TestEditPipeline(unittest.TestCase):
    def test_deterministic_repair_targets_only_hard_issue_ids(self):
        source = subtitle(["There are 2 cases.", "Alex arrived."])
        service = GlossaryService()
        state = service.merge(GlossaryCatalog(entries=[GlossaryEntry(source="cases", target="案件", required=True)]))
        validators = DeterministicValidatorSuite(source, glossary_service=service, glossary_state=state)
        checkpoint = MagicMock()
        pipeline = EditPipeline(source, validators, checkpoint_hook=checkpoint)
        session = pipeline.new_session(["有一些事情。", "亚历克斯到了。"])
        repair = MagicMock(return_value={1: "有2个案件。"})

        pipeline.run_deterministic_repair(session, repair=repair)

        repair.assert_called_once_with([1], ["有一些事情。", "亚历克斯到了。"])
        self.assertEqual(session.current_translations, ["有2个案件。", "亚历克斯到了。"])
        self.assertEqual(session.rounds[0].round_index, 0)
        checkpoint.assert_called()

    def test_first_semantic_round_scans_all_and_second_expands_neighbors(self):
        source = subtitle(["one", "two", "three", "four"])
        validators = DeterministicValidatorSuite(source)
        pipeline = EditPipeline(source, validators, max_rounds=2)
        session = pipeline.new_session(["一", "二", "三", "四"])
        scopes = []

        def review(scope, translations, round_index):
            scopes.append(scope)
            if round_index == 1:
                found = semantic_issue(2)
                return (
                    [found],
                    [
                        EditPatch.create(
                            issue_ids=[found.issue_id],
                            segment_id=2,
                            before=translations[1],
                            after="第二",
                            reason="meaning",
                            action=EditAction.REVIEW,
                        )
                    ],
                    {"round": round_index},
                )
            return [], [], {"round": round_index}

        pipeline.run_semantic_rounds(session, review=review)

        self.assertEqual(scopes, [[1, 2, 3, 4], [1, 2, 3]])
        self.assertEqual(session.current_translations[1], "第二")
        self.assertEqual(session.rounds[-1].stop_reason, EditStopReason.NO_HIGH_RISK)
        self.assertEqual(session.status, EditSessionStatus.COMPLETE)

    def test_no_patch_and_max_rounds_have_bounded_stop_reasons(self):
        source = subtitle(["one"])
        validators = DeterministicValidatorSuite(source)

        no_patch = EditPipeline(source, validators, max_rounds=2).new_session(["一"])
        EditPipeline(source, validators, max_rounds=2).run_semantic_rounds(
            no_patch, review=lambda scope, translations, round_index: ([semantic_issue(1)], [], {})
        )
        self.assertEqual(no_patch.rounds[-1].stop_reason, EditStopReason.NO_EFFECTIVE_PATCH)

        one_round_pipeline = EditPipeline(source, validators, max_rounds=1)
        maxed = one_round_pipeline.new_session(["一"])

        def review(scope, translations, round_index):
            found = semantic_issue(1)
            return (
                [found],
                [
                    EditPatch.create(
                        issue_ids=[found.issue_id],
                        segment_id=1,
                        before=translations[0],
                        after="第一",
                        reason="meaning",
                        action=EditAction.REVIEW,
                    )
                ],
                {},
            )

        one_round_pipeline.run_semantic_rounds(maxed, review=review)
        self.assertEqual(maxed.rounds[-1].stop_reason, EditStopReason.MAX_ROUNDS)

    def test_zero_rounds_runs_no_semantic_model(self):
        source = subtitle(["one"])
        pipeline = EditPipeline(source, DeterministicValidatorSuite(source), max_rounds=0)
        session = pipeline.new_session(["一"])
        reviewer = MagicMock()

        pipeline.run_semantic_rounds(session, review=reviewer)

        reviewer.assert_not_called()
        self.assertEqual(session.status, EditSessionStatus.COMPLETE)

    def test_resume_starts_after_successful_round_and_uses_changed_neighbors(self):
        source = subtitle(["one", "two", "three"])
        pipeline = EditPipeline(source, DeterministicValidatorSuite(source), max_rounds=2)
        session = pipeline.new_session(["一", "二", "三"])
        found = semantic_issue(2)
        patch_item = EditPatch.create(
            issue_ids=[found.issue_id],
            segment_id=2,
            before="二",
            after="第二",
            reason="meaning",
            action=EditAction.REVIEW,
        )
        session.current_translations[1] = "第二"
        session.rounds.append(
            EditRound(round_index=1, scope_ids=[1, 2, 3], issues_before=[found], applied_patches=[patch_item])
        )
        scopes = []

        pipeline.run_semantic_rounds(
            session, review=lambda scope, translations, round_index: scopes.append((round_index, scope)) or ([], [], {})
        )

        self.assertEqual(scopes, [(2, [1, 2, 3])])
        self.assertEqual(session.rounds[-1].round_index, 2)

    def test_failed_round_is_retried_without_consuming_round_budget(self):
        source = subtitle(["one"])
        pipeline = EditPipeline(source, DeterministicValidatorSuite(source), max_rounds=1)
        session = pipeline.new_session(["一"])
        calls = []

        def fail(scope, translations, round_index):
            calls.append(round_index)
            return (
                [
                    EditIssue.create(
                        segment_ids=[1],
                        category="semantic",
                        severity=EditSeverity.ERROR,
                        source="semantic-review",
                        message="model failed",
                        evidence={},
                        status=EditIssueStatus.FAILED,
                    )
                ],
                [],
                {},
            )

        pipeline.run_semantic_rounds(session, review=fail)
        pipeline.run_semantic_rounds(session, review=fail)

        self.assertEqual(calls, [1, 1])
        self.assertEqual([item.status for item in session.rounds], [EditRoundStatus.FAILED] * 2)

    def test_started_round_is_checkpointed_before_model_call(self):
        source = subtitle(["one"])
        saved_statuses = []
        pipeline = EditPipeline(
            source,
            DeterministicValidatorSuite(source),
            max_rounds=1,
            checkpoint_hook=lambda session: saved_statuses.append(
                session.rounds[-1].status if session.rounds else None
            ),
        )
        session = pipeline.new_session(["一"])

        pipeline.run_semantic_rounds(session, review=lambda scope, translations, round_index: ([], [], {}))

        self.assertIn(EditRoundStatus.STARTED, saved_statuses)

    def test_model_failure_rolls_back_other_patches_in_same_round(self):
        source = subtitle(["one", "two"])
        pipeline = EditPipeline(source, DeterministicValidatorSuite(source), max_rounds=1)
        session = pipeline.new_session(["一", "二"])
        found = semantic_issue(1)
        failed = EditIssue.create(
            segment_ids=[2],
            category="semantic",
            severity=EditSeverity.ERROR,
            source="semantic-review",
            message="second chunk failed",
            evidence={},
            status=EditIssueStatus.FAILED,
        )
        proposed = EditPatch.create(
            issue_ids=[found.issue_id],
            segment_id=1,
            before="一",
            after="第一",
            reason="meaning",
            action=EditAction.REVIEW,
        )

        pipeline.run_semantic_rounds(
            session, review=lambda scope, translations, round_index: ([found, failed], [proposed], {})
        )

        self.assertEqual(session.current_translations, ["一", "二"])
        self.assertEqual(session.rounds[-1].status, EditRoundStatus.FAILED)
        self.assertEqual(session.rounds[-1].applied_patches, [])


if __name__ == "__main__":
    unittest.main()
