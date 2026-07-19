#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openlrc.editing import (
    EditAction,
    EditIssue,
    EditPatch,
    EditRound,
    EditRoundStatus,
    EditSession,
    EditSeverity,
    apply_patch_transaction,
    atomic_save_subtitle,
    load_edit_session,
    parse_segment_ids,
    save_edit_session,
    translations_at_round,
)


def issue(segment_id=1, *, message="missing number"):
    return EditIssue.create(
        segment_ids=[segment_id],
        category="number",
        severity=EditSeverity.ERROR,
        source="number-preservation",
        message=message,
        evidence={"missing": ["2"]},
    )


def edit_patch(segment_id=1, before="old", after="new", issue_ids=()):
    return EditPatch.create(
        issue_ids=issue_ids, segment_id=segment_id, before=before, after=after, reason="fix", action=EditAction.REVIEW
    )


class TestPatchTransaction(unittest.TestCase):
    def test_applies_valid_patch_without_touching_other_lines(self):
        original_issue = issue()
        patch_item = edit_patch(after="new 2", issue_ids=[original_issue.issue_id])

        result = apply_patch_transaction(
            ["old", "untouched"],
            [patch_item],
            scope_ids=[1],
            baseline_issues=[original_issue],
            validate=lambda texts: [],
        )

        self.assertTrue(result.committed)
        self.assertEqual(result.translations, ["new 2", "untouched"])

    def test_rejects_out_of_scope_stale_duplicate_and_no_change_patches(self):
        cases = [
            ([edit_patch(segment_id=2)], [1], "outside"),
            ([edit_patch(before="stale")], [1], "stale"),
            ([edit_patch(), edit_patch(after="other")], [1], "Multiple"),
            ([edit_patch(after="old")], [1], "did not change"),
        ]
        for patches, scope, message in cases:
            with self.subTest(message=message):
                result = apply_patch_transaction(
                    ["old", "second"], patches, scope_ids=scope, baseline_issues=[], validate=lambda texts: []
                )
                self.assertFalse(result.committed)
                self.assertIn(message, result.reason)

    def test_rolls_back_entire_round_on_new_hard_issue(self):
        patch_item = edit_patch()
        new_issue = issue(message="new problem")

        result = apply_patch_transaction(
            ["old"], [patch_item], scope_ids=[1], baseline_issues=[], validate=lambda texts: [new_issue]
        )

        self.assertFalse(result.committed)
        self.assertEqual(result.translations, ["old"])
        self.assertEqual(result.rejected_patches[0].rejection_reason, result.reason)

    def test_allows_unchanged_unrelated_baseline_error(self):
        unrelated = issue(segment_id=2)
        patch_item = edit_patch(segment_id=1)

        result = apply_patch_transaction(
            ["old", "broken"],
            [patch_item],
            scope_ids=[1],
            baseline_issues=[unrelated],
            validate=lambda texts: [unrelated],
        )

        self.assertTrue(result.committed)

    def test_allows_same_baseline_error_inside_broad_scope_when_unaddressed(self):
        existing = issue(segment_id=2)
        patch_item = edit_patch(segment_id=1)

        result = apply_patch_transaction(
            ["old", "broken"],
            [patch_item],
            scope_ids=[1, 2],
            baseline_issues=[existing],
            validate=lambda texts: [existing],
        )

        self.assertTrue(result.committed)


class TestEditSession(unittest.TestCase):
    def test_round_restore_rebuilds_history(self):
        first = edit_patch(before="draft", after="round one")
        second = edit_patch(before="round one", after="round two")
        session = EditSession(
            source_fingerprint="source",
            translation_fingerprint="translation",
            raw_translations=["draft"],
            current_translations=["round two"],
            rounds=[
                EditRound(round_index=1, scope_ids=[1], applied_patches=[first]),
                EditRound(round_index=2, scope_ids=[1], applied_patches=[second]),
            ],
        )

        self.assertEqual(translations_at_round(session, 0), ["draft"])
        self.assertEqual(translations_at_round(session, 1), ["round one"])
        self.assertEqual(translations_at_round(session, 2), ["round two"])

    def test_round_zero_is_raw_draft_and_later_round_includes_deterministic_repair(self):
        deterministic = edit_patch(before="draft", after="deterministic")
        semantic = edit_patch(before="deterministic", after="semantic")
        session = EditSession(
            source_fingerprint="source",
            translation_fingerprint="translation",
            raw_translations=["draft"],
            current_translations=["semantic"],
            rounds=[
                EditRound(
                    round_index=0, status=EditRoundStatus.COMMITTED, scope_ids=[1], applied_patches=[deterministic]
                ),
                EditRound(round_index=1, status=EditRoundStatus.COMMITTED, scope_ids=[1], applied_patches=[semantic]),
            ],
        )

        self.assertEqual(translations_at_round(session, 0), ["draft"])
        self.assertEqual(translations_at_round(session, 1), ["semantic"])

    def test_failed_semantic_round_is_not_a_restore_target(self):
        session = EditSession(
            source_fingerprint="source",
            translation_fingerprint="translation",
            raw_translations=["draft"],
            current_translations=["draft"],
            rounds=[
                EditRound(round_index=1, status=EditRoundStatus.FAILED, scope_ids=[1], stop_reason="model-failure")
            ],
        )

        with self.assertRaisesRegex(ValueError, "one of: 0"):
            translations_at_round(session, 1)

    def test_corrupt_history_is_rejected(self):
        session = EditSession(
            source_fingerprint="source",
            translation_fingerprint="translation",
            raw_translations=["draft"],
            current_translations=["wrong"],
            rounds=[EditRound(round_index=1, scope_ids=[1], applied_patches=[edit_patch(before="stale")])],
        )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            translations_at_round(session, 1)

    def test_session_round_trip_and_invalid_session(self):
        session = EditSession(
            source_fingerprint="source",
            translation_fingerprint="translation",
            raw_translations=["draft"],
            current_translations=["draft"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.edit-session.json"
            save_edit_session(path, session)
            loaded = load_edit_session(path)
            self.assertEqual(loaded, session)

            path.write_text("{bad", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid edit session"):
                load_edit_session(path)

    def test_atomic_subtitle_write_preserves_previous_file_on_replace_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "translated.json"
            path.write_text('{"old": true}', encoding="utf-8")
            with patch("openlrc.checkpoint.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    atomic_save_subtitle(path, language="zh", timestamps=[(0.0, 1.0)], texts=["新"])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"old": True})

    def test_parse_segment_ids(self):
        self.assertEqual(parse_segment_ids("1,3-5,4", maximum=5), [1, 3, 4, 5])
        for invalid in ("", "0", "3-2", "1,,2", "x"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                parse_segment_ids(invalid, maximum=5)


if __name__ == "__main__":
    unittest.main()
