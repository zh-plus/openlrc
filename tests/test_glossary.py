#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import json
import tempfile
import unittest
from pathlib import Path

from openlrc.context import CharacterBrief, GlossaryBrief, TranslationBrief
from openlrc.glossary import GlossaryCatalog, GlossaryEntry, GlossaryOrigin, GlossaryService


class TestGlossaryService(unittest.TestCase):
    def test_loads_legacy_mapping_and_versioned_catalog(self):
        service = GlossaryService()

        legacy, _ = service.load({"case": "案件"})
        versioned, _ = service.load(
            {
                "schema_version": 1,
                "name": "project",
                "entries": [{"source": "OpenLRC", "target": "OpenLRC", "required": True}],
            }
        )

        self.assertEqual(legacy.entries[0].source, "case")
        self.assertFalse(legacy.entries[0].required)
        self.assertTrue(versioned.entries[0].required)

    def test_file_errors_are_user_readable(self):
        service = GlossaryService()
        with self.assertRaisesRegex(FileNotFoundError, "Glossary file not found"):
            service.load("missing-glossary.json")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "glossary.json"
            path.write_text("{bad", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid glossary JSON"):
                service.load(path)

    def test_language_mismatch_is_always_rejected(self):
        catalog = {"schema_version": 1, "source_language": "ja", "entries": []}

        with self.assertRaisesRegex(ValueError, "does not match task language"):
            GlossaryService(strict=False).load(catalog, source_language="en")

    def test_same_priority_conflict_is_strict_or_first_wins(self):
        payload = {
            "schema_version": 1,
            "entries": [{"source": "case", "target": "案件"}, {"source": "case", "target": "案子"}],
        }
        with self.assertRaisesRegex(ValueError, "Conflicting task glossary"):
            GlossaryService(strict=True).load(payload)

        catalog, conflicts = GlossaryService(strict=False).load(payload)
        self.assertEqual([entry.target for entry in catalog.entries], ["案件"])
        self.assertEqual(conflicts[0].selected_target, "案件")

    def test_task_entries_override_brief_and_force_only_user_entries(self):
        service = GlossaryService(force=True)
        catalog = GlossaryCatalog(entries=[GlossaryEntry(source="case", target="案件")])
        brief = TranslationBrief(
            summary="summary",
            glossary=[GlossaryBrief(source="case", target="案子"), GlossaryBrief(source="clue", target="线索")],
        )

        state = service.merge(catalog, brief=brief)

        by_source = {entry.source: entry for entry in state.merged_entries}
        self.assertEqual(by_source["case"].origin, GlossaryOrigin.TASK)
        self.assertTrue(by_source["case"].required)
        self.assertEqual(by_source["clue"].origin, GlossaryOrigin.BRIEF)
        self.assertFalse(by_source["clue"].required)
        self.assertEqual(state.metrics.brief_entries_overridden, 1)

    def test_task_entry_overrides_matching_brief_character(self):
        service = GlossaryService()
        catalog = GlossaryCatalog(entries=[GlossaryEntry(source="John", target="强尼", aliases=["John Smith"])])
        brief = TranslationBrief(
            summary="summary", characters=[CharacterBrief(source_name="john smith", target_name="约翰")]
        )

        state = service.merge(catalog, brief=brief)

        self.assertEqual(state.metrics.brief_characters_overridden, 1)
        self.assertEqual(state.metrics.conflict_count, 1)
        self.assertEqual(state.conflicts[0].selected_target, "强尼")
        self.assertEqual(state.conflicts[0].reason, "higher-priority-task-entry-over-character")

    def test_character_cross_section_check_is_skipped_without_task_glossary(self):
        brief = TranslationBrief(
            summary="summary",
            characters=[CharacterBrief(source_name="John", target_name="约翰")],
            glossary=[GlossaryBrief(source="John", target="强尼")],
        )

        state = GlossaryService().merge(GlossaryCatalog(), brief=brief)

        self.assertEqual(state.metrics.brief_characters_overridden, 0)
        self.assertEqual(state.conflicts, [])

    def test_matches_unicode_aliases_cjk_and_accepted_targets(self):
        service = GlossaryService()
        catalog = GlossaryCatalog(
            entries=[
                GlossaryEntry(source="OpenLRC", target="OpenLRC", aliases=["open lrc"], required=True),
                GlossaryEntry(source="案件", target="case", accepted_targets=["the case"]),
            ]
        )
        state = service.merge(catalog)

        checked = service.check(state, ["Use Ｏｐｅｎ　ＬＲＣ", "这个案件"], ["使用 OpenLRC", "the case"])

        self.assertTrue(checked.metrics.final_compliance_passed)
        self.assertEqual(checked.metrics.source_matched_entries, 2)
        self.assertTrue(all(match.target_found for match in checked.matches if match.segment_ids))

    def test_cross_line_source_and_target_window(self):
        service = GlossaryService()
        state = service.merge(
            GlossaryCatalog(entries=[GlossaryEntry(source="machine learning", target="机器学习", required=True)])
        )

        checked = service.check(state, ["machine", "learning is useful"], ["这就是", "机器学习"])

        match = checked.matches[0]
        self.assertEqual(match.segment_ids, [1, 2])
        self.assertTrue(match.target_found)

    def test_latin_target_matches_when_adjacent_to_cjk(self):
        service = GlossaryService()
        state = service.merge(
            GlossaryCatalog(entries=[GlossaryEntry(source="API", target="API", required=True, case_sensitive=True)])
        )

        checked = service.check(state, ["Use the API."], ["当前API可用。"])

        self.assertTrue(checked.matches[0].target_found)

    def test_required_and_preferred_noncompliance_have_different_severity(self):
        service = GlossaryService()
        state = service.merge(
            GlossaryCatalog(
                entries=[
                    GlossaryEntry(source="case", target="案件", required=True),
                    GlossaryEntry(source="clue", target="线索"),
                ]
            )
        )

        checked = service.check(state, ["case", "clue"], ["事情", "提示"])

        self.assertEqual([match.severity for match in checked.matches], ["error", "warning"])
        self.assertFalse(checked.metrics.final_compliance_passed)

    def test_required_compliance_is_counted_per_source_occurrence(self):
        service = GlossaryService()
        state = service.merge(GlossaryCatalog(entries=[GlossaryEntry(source="case", target="案件", required=True)]))

        checked = service.check(state, ["The first case.", "The second case."], ["第一个案件。", "第二个漏译。"])

        occurrence_matches = [match for match in checked.matches if match.occurrence_index is not None]
        self.assertEqual([match.target_found for match in occurrence_matches], [True, False])
        self.assertEqual(checked.metrics.source_matched_occurrences, 2)
        self.assertEqual(checked.metrics.required_occurrences_compliant, 1)
        self.assertEqual(checked.metrics.required_occurrences_noncompliant, 1)
        self.assertFalse(checked.metrics.final_compliance_passed)

    def test_one_target_occurrence_cannot_satisfy_two_source_occurrences_on_one_line(self):
        service = GlossaryService()
        state = service.merge(GlossaryCatalog(entries=[GlossaryEntry(source="case", target="案件", required=True)]))

        checked = service.check(state, ["case and case"], ["案件"])

        self.assertEqual([match.target_found for match in checked.matches], [True, False])
        self.assertEqual(checked.metrics.required_noncompliant, 1)

    def test_case_insensitive_entries_conflict_after_casefold(self):
        payload = {
            "schema_version": 1,
            "entries": [{"source": "Case", "target": "案件"}, {"source": "case", "target": "案子"}],
        }

        with self.assertRaisesRegex(ValueError, "Conflicting task glossary"):
            GlossaryService(strict=True).load(payload)
        catalog, conflicts = GlossaryService(strict=False).load(payload)

        self.assertEqual([(entry.source, entry.target) for entry in catalog.entries], [("Case", "案件")])
        self.assertEqual(len(conflicts), 1)

    def test_cjk_phrase_matches_across_line_without_inserted_space(self):
        service = GlossaryService()
        state = service.merge(
            GlossaryCatalog(entries=[GlossaryEntry(source="机器学习", target="machine learning", required=True)])
        )

        checked = service.check(state, ["机器", "学习很有用"], ["machine", "learning is useful"])

        self.assertEqual(checked.matches[0].segment_ids, [1, 2])
        self.assertTrue(checked.matches[0].target_found)

    def test_report_matches_can_hide_details_without_losing_metrics(self):
        service = GlossaryService(report_matches=False)
        state = service.merge(GlossaryCatalog(entries=[GlossaryEntry(source="case", target="案件", required=True)]))
        checked = service.check(state, ["case"], ["漏译"])

        reported = service.report_state(checked)

        self.assertEqual(reported.matches, [])
        self.assertEqual(reported.metrics.required_noncompliant, 1)
        self.assertFalse(reported.metrics.final_compliance_passed)

    def test_fingerprint_changes_with_semantic_entry_fields(self):
        service = GlossaryService()
        first = service.merge(GlossaryCatalog(entries=[GlossaryEntry(source="case", target="案件")]))
        second = service.merge(
            GlossaryCatalog(entries=[GlossaryEntry(source="case", target="案件", accepted_targets=["案子"])])
        )

        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_catalog_round_trip_is_json_serializable(self):
        service = GlossaryService(force=True)
        catalog, conflicts = service.load({"case": "案件"})
        state = service.merge(catalog, load_conflicts=conflicts)

        payload = json.loads(state.model_dump_json())

        self.assertEqual(payload["merged_entries"][0]["origin"], "task")


if __name__ == "__main__":
    unittest.main()
