#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openlrc.checkpoint import HYM_T2_CHECKPOINT_KIND, migrate_hymt2_checkpoint, save_json_checkpoint


class TestAtomicCheckpoint(unittest.TestCase):
    def test_atomically_replaces_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "compare.json"
            path.write_text('{"old": true}', encoding="utf-8")

            save_json_checkpoint(path, {"new": True})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"new": True})
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])


class TestCheckpointMigration(unittest.TestCase):
    def test_v2_alias_checkpoint_migrates_to_canonical_v4(self):
        payload = {
            "schema_version": 2,
            "source_fingerprint": "legacy",
            "hymt2_mode": "context-plus",
            "translation_brief": {"summary": "summary"},
            "raw_hymt2_translations": ["草稿"],
        }

        migrated = migrate_hymt2_checkpoint(
            payload,
            canonical_mode="normal-plus",
            canonical_fingerprint="canonical",
            accepted_legacy_fingerprints={"legacy"},
            glossary_fingerprint="glossary",
        )

        self.assertEqual(migrated["checkpoint_kind"], HYM_T2_CHECKPOINT_KIND)
        self.assertEqual(migrated["schema_version"], 4)
        self.assertEqual(migrated["mode"], "normal-plus")
        self.assertEqual(migrated["source_fingerprint"], "canonical")

    def test_stale_fingerprint_is_not_migrated(self):
        self.assertEqual(
            migrate_hymt2_checkpoint(
                {"schema_version": 2, "source_fingerprint": "stale"},
                canonical_mode="normal",
                canonical_fingerprint="current",
                accepted_legacy_fingerprints={"legacy"},
            ),
            {},
        )

    def test_legacy_review_is_adapted_without_fake_evidence(self):
        payload = {
            "schema_version": 3,
            "source_fingerprint": "same",
            "raw_hymt2_translations": ["草稿"],
            "final_translations": ["修订"],
            "review_protocol_version": 2,
            "review_results": {"1": [{"id": 1, "risk": "high", "issues": ["meaning"], "revised_translation": "修订"}]},
        }

        migrated = migrate_hymt2_checkpoint(payload, canonical_mode="pro", canonical_fingerprint="same")

        edit_round = migrated["edit_session"]["rounds"][0]
        self.assertEqual(edit_round["issues_before"][0]["evidence"], {})
        self.assertEqual(edit_round["applied_patches"][0]["before"], "草稿")

    def test_failed_replace_preserves_previous_checkpoint_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "compare.json"
            path.write_text('{"old": true}', encoding="utf-8")

            with patch("openlrc.checkpoint.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    save_json_checkpoint(path, {"new": True})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"old": True})
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])
