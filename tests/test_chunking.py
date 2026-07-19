#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import unittest

from openlrc.chunking import CHUNK_PLANNER_VERSION, chunk_plan_signature, plan_translation_chunks
from openlrc.utils import get_text_token_number


class TestTranslationChunkPlanner(unittest.TestCase):
    def test_scene_boundary_is_preserved_in_plan(self):
        texts = ["one", "two", "three", "four"]
        timestamps = [(0.0, 1.0), (1.0, 2.0), (62.0, 63.0), (63.0, 64.0)]

        plans = plan_translation_chunks(texts, timestamps=timestamps, token_budget=10000)

        self.assertEqual([plan.segment_ids for plan in plans], [[1, 2], [3, 4]])
        self.assertFalse(plans[0].hard_scene_break_before)
        self.assertTrue(plans[1].hard_scene_break_before)

    def test_small_tail_does_not_merge_across_scene_boundary(self):
        texts = [f"line {index}" for index in range(7)]
        timestamps = [(float(index), float(index + 1)) for index in range(5)] + [(100.0, 101.0), (101.0, 102.0)]

        plans = plan_translation_chunks(texts, timestamps=timestamps, chunk_size=5, token_budget=10000)

        self.assertEqual([plan.segment_ids for plan in plans], [[1, 2, 3, 4, 5], [6, 7]])
        self.assertTrue(plans[1].hard_scene_break_before)

    def test_signature_changes_with_boundaries_or_source(self):
        texts = ["one", "two", "three"]
        first = plan_translation_chunks(texts, chunk_size=2, token_budget=10000)
        second = plan_translation_chunks(texts, chunk_size=3, token_budget=10000)
        changed = plan_translation_chunks(["one", "TWO", "three"], chunk_size=2, token_budget=10000)

        self.assertNotEqual(chunk_plan_signature(first), chunk_plan_signature(second))
        self.assertNotEqual(chunk_plan_signature(first), chunk_plan_signature(changed))

    def test_pairs_preserve_exact_ids_and_texts(self):
        plans = plan_translation_chunks(["alpha", "beta"], chunk_size=30, token_budget=10000)

        self.assertEqual(plans[0].pairs(), [(1, "alpha"), (2, "beta")])

    def test_equal_timestamp_gaps_do_not_create_single_line_chunks(self):
        texts = ["x"] * 100
        timestamps = [(float(index), float(index) + 0.5) for index in range(100)]

        plans = plan_translation_chunks(texts, timestamps=timestamps, chunk_size=30, token_budget=1000)

        self.assertEqual([len(plan.segment_ids) for plan in plans], [30, 30, 30, 10])

    def test_small_tail_never_breaks_line_cap(self):
        plans = plan_translation_chunks(["x"] * 31, chunk_size=30, token_budget=1000)

        self.assertEqual([len(plan.segment_ids) for plan in plans], [30, 1])

    def test_small_tail_never_breaks_token_cap(self):
        plans = plan_translation_chunks(["x"] * 13, chunk_size=30, token_budget=10)

        self.assertEqual([sum(get_text_token_number(text) for text in plan.texts) for plan in plans], [10, 3])

    def test_timestamp_split_rechecks_token_budget(self):
        texts = ["x"] * 10 + ["one two three four five"]
        timestamps = [(float(index), float(index) + 0.5) for index in range(len(texts))]

        plans = plan_translation_chunks(texts, timestamps=timestamps, chunk_size=30, token_budget=10)

        self.assertTrue(
            all(len(plan.texts) == 1 or sum(get_text_token_number(text) for text in plan.texts) <= 10 for plan in plans)
        )

    def test_planner_preserves_partition_and_caps_across_boundary_matrix(self):
        """Every plan remains an exact ordered partition under varied inputs."""
        for line_count in (1, 2, 29, 30, 31, 59, 60, 61, 100):
            for chunk_size in (1, 7, 30):
                for token_budget in (1, 10, 1000):
                    texts = ["x" if index % 5 else "one two three" for index in range(line_count)]
                    timestamp_options = (
                        None,
                        [(float(index), float(index) + 0.5) for index in range(line_count)],
                        [
                            (
                                float(index + (40 if index >= line_count // 2 else 0)),
                                float(index + (40 if index >= line_count // 2 else 0)) + 0.5,
                            )
                            for index in range(line_count)
                        ],
                    )
                    for timestamps in timestamp_options:
                        with self.subTest(
                            lines=line_count,
                            chunk_size=chunk_size,
                            token_budget=token_budget,
                            timestamps=timestamps is not None,
                        ):
                            plans = plan_translation_chunks(
                                texts, timestamps=timestamps, chunk_size=chunk_size, token_budget=token_budget
                            )
                            flattened_ids = [line_id for plan in plans for line_id in plan.segment_ids]
                            flattened_texts = [text for plan in plans for text in plan.texts]
                            self.assertEqual(flattened_ids, list(range(1, line_count + 1)))
                            self.assertEqual(flattened_texts, texts)
                            self.assertTrue(all(plan.segment_ids for plan in plans))
                            self.assertTrue(all(len(plan.segment_ids) <= chunk_size for plan in plans))
                            self.assertTrue(
                                all(
                                    len(plan.texts) == 1
                                    or sum(get_text_token_number(text) for text in plan.texts) <= token_budget
                                    for plan in plans
                                )
                            )

    def test_planner_version_tracks_boundary_algorithm(self):
        self.assertEqual(CHUNK_PLANNER_VERSION, 2)


if __name__ == "__main__":
    unittest.main()
