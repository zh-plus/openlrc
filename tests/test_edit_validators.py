#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import unittest
from pathlib import Path

from openlrc.context import CharacterBrief, TranslationBrief
from openlrc.edit_validators import DeterministicValidatorSuite, ImmutableFieldValidator
from openlrc.editing import EditSeverity
from openlrc.glossary import GlossaryCatalog, GlossaryEntry, GlossaryService
from openlrc.subtitle import Subtitle


def subtitle(texts, *, language="en", starts=None):
    starts = starts or list(range(len(texts)))
    return Subtitle(
        language=language,
        segments=[{"start": float(start), "end": float(start + 1), "text": text} for start, text in zip(starts, texts)],
        filename=Path("sample.json"),
    )


class TestDeterministicValidators(unittest.TestCase):
    def test_structure_reports_count_and_empty_lines(self):
        source = subtitle(["one", "two"])
        suite = DeterministicValidatorSuite(source)

        count_issues = suite.validate(["一"])
        empty_issues = suite.validate(["一", " "])

        self.assertEqual(count_issues[0].category, "structure")
        self.assertEqual(empty_issues[0].segment_ids, [2])
        self.assertEqual(empty_issues[0].severity, EditSeverity.ERROR)

    def test_number_placeholder_percent_and_currency_loss_is_error(self):
        source = subtitle(["Pay $20 (50%) to {name} on 2026-07-18."])
        suite = DeterministicValidatorSuite(source)

        issues = suite.validate(["付款给他。"])

        kinds = {issue.evidence.get("kind") for issue in issues if issue.category == "number"}
        self.assertEqual(kinds, {"placeholder", "number", "percent", "currency"})
        self.assertTrue(all(issue.severity is EditSeverity.ERROR for issue in issues))

    def test_number_format_normalizes_grouping_commas(self):
        source = subtitle(["The total is 1,000."])
        suite = DeterministicValidatorSuite(source)

        issues = suite.validate(["总数是1000。"])

        self.assertEqual(issues, [])

    def test_percent_phrase_is_not_placeholder_and_localized_currency_is_accepted(self):
        source = subtitle(["Pay $25.50.", "The refund is 10% of 1,200 AUD."])
        suite = DeterministicValidatorSuite(source)

        issues = suite.validate(["支付25.50美元。", "退款金额为1,200澳元的10%。"])

        self.assertFalse(any(issue.severity is EditSeverity.ERROR for issue in issues))
        self.assertFalse(any(issue.evidence.get("kind") == "placeholder" for issue in issues))

    def test_percent_notations_are_semantically_equivalent(self):
        source = subtitle(["Half is 50%.", "Half is 50 percent.", "Half is 50 per cent."])
        suite = DeterministicValidatorSuite(source)

        issues = suite.validate(["一半是百分之50。", "一半是50％。", "一半是50%。"])

        self.assertFalse(any(issue.evidence.get("kind") == "percent" for issue in issues))

    def test_printf_placeholder_is_not_counted_as_percent(self):
        source = subtitle(["Value: %s"])
        suite = DeterministicValidatorSuite(source)

        issues = suite.validate(["值：%s"])

        self.assertEqual(issues, [])

    def test_glossary_and_entity_issues_share_protocol(self):
        source = subtitle(["Alex opened the case."])
        service = GlossaryService()
        state = service.merge(GlossaryCatalog(entries=[GlossaryEntry(source="case", target="案件", required=True)]))
        brief = TranslationBrief(
            summary="summary", characters=[CharacterBrief(source_name="Alex", target_name="亚历克斯")]
        )
        suite = DeterministicValidatorSuite(source, glossary_service=service, glossary_state=state, brief=brief)

        issues = suite.validate(["他打开了箱子。"])

        self.assertEqual({issue.category for issue in issues}, {"glossary", "entity"})
        self.assertEqual(next(issue for issue in issues if issue.category == "glossary").severity, EditSeverity.ERROR)
        self.assertEqual(next(issue for issue in issues if issue.category == "entity").severity, EditSeverity.ERROR)

    def test_issue_ids_are_stable(self):
        source = subtitle(["There are 2 items."])
        suite = DeterministicValidatorSuite(source)

        first = suite.validate(["有一些项目。"])
        second = suite.validate(["有一些项目。"])

        self.assertEqual([issue.issue_id for issue in first], [issue.issue_id for issue in second])

    def test_immutable_validator_reports_timeline_changes(self):
        source = subtitle(["one", "two"])
        target = subtitle(["一", "二"], language="zh", starts=[0, 9])

        issues = ImmutableFieldValidator().validate(source, target, scope_ids={2})

        self.assertEqual(issues[0].category, "immutable")
        self.assertEqual(issues[0].segment_ids, [2])


if __name__ == "__main__":
    unittest.main()
