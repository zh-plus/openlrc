#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

"""Shared subtitle chunk planning for translation and contextual stages."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from openlrc.logger import logger
from openlrc.utils import get_text_token_number

CHUNK_PLANNER_VERSION = 2


class TranslationChunkPlan(BaseModel):
    """Stable chunk boundary shared by context planning, translation, and review."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: int = Field(ge=1)
    segment_ids: list[int]
    texts: list[str]
    hard_scene_break_before: bool = False

    def pairs(self) -> list[tuple[int, str]]:
        return list(zip(self.segment_ids, self.texts))


def _chunk_token_count(chunk: list[tuple[int, str]], token_counts: list[int]) -> int:
    return sum(token_counts[line_id - 1] for line_id, _ in chunk)


def _fits_limits(chunk: list[tuple[int, str]], *, token_counts: list[int], chunk_size: int, token_budget: int) -> bool:
    if not chunk:
        return True
    within_token_budget = _chunk_token_count(chunk, token_counts) <= token_budget or len(chunk) == 1
    return len(chunk) <= chunk_size and within_token_budget


def _find_best_split(
    chunk: list[tuple[int, str]],
    timestamps: list[tuple[float, float | None]] | None,
    *,
    token_counts: list[int],
    chunk_size: int,
    token_budget: int,
) -> int | None:
    """Choose a valid large-gap boundary, preferring the latest equal gap."""
    if not timestamps or len(chunk) < 2:
        return None

    best_key: tuple[float, int] | None = None
    best_idx: int | None = None
    for index in range(1, len(chunk)):
        left = chunk[:index]
        right = chunk[index:]
        if not _fits_limits(left, token_counts=token_counts, chunk_size=chunk_size, token_budget=token_budget):
            continue
        if not _fits_limits(right, token_counts=token_counts, chunk_size=chunk_size, token_budget=token_budget):
            continue

        previous = chunk[index - 1][0] - 1
        current = chunk[index][0] - 1
        previous_end = timestamps[previous][1]
        current_start = timestamps[current][0]
        if previous_end is not None:
            gap = current_start - previous_end
            key = (gap, index)
            if best_key is None or key > best_key:
                best_key = key
                best_idx = index
    return best_idx


def _is_hard_scene_break(
    previous_id: int, current_id: int, timestamps: list[tuple[float, float | None]] | None, scene_threshold: float
) -> bool:
    if timestamps is None:
        return False
    previous_end = timestamps[previous_id - 1][1]
    current_start = timestamps[current_id - 1][0]
    return previous_end is not None and (current_start - previous_end) > scene_threshold


def plan_translation_chunks(
    texts: list[str],
    *,
    timestamps: list[tuple[float, float | None]] | None = None,
    chunk_size: int = 30,
    token_budget: int = 1000,
    scene_threshold: float = 30.0,
) -> list[TranslationChunkPlan]:
    """Plan translation chunks without depending on a translator instance."""
    if not texts:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive.")

    valid_timestamps = timestamps
    if valid_timestamps is not None and len(valid_timestamps) != len(texts):
        logger.warning(
            f"Timestamps length ({len(valid_timestamps)}) != texts length ({len(texts)}), ignoring timestamps."
        )
        valid_timestamps = None

    token_counts = [get_text_token_number(text) for text in texts]
    chunks: list[list[tuple[int, str]]] = []
    current_chunk: list[tuple[int, str]] = []

    for index, text in enumerate(texts):
        line_id = index + 1

        if current_chunk and index > 0 and _is_hard_scene_break(index, line_id, valid_timestamps, scene_threshold):
            chunks.append(current_chunk)
            current_chunk = []

        prospective = [*current_chunk, (line_id, text)]
        if _fits_limits(prospective, token_counts=token_counts, chunk_size=chunk_size, token_budget=token_budget):
            current_chunk = prospective
            continue

        split_index = _find_best_split(
            prospective, valid_timestamps, token_counts=token_counts, chunk_size=chunk_size, token_budget=token_budget
        )
        if split_index is None:
            split_index = len(current_chunk)
        chunks.append(prospective[:split_index])
        current_chunk = prospective[split_index:]

    if current_chunk:
        chunks.append(current_chunk)

    if len(chunks) >= 2:
        tail = chunks[-1]
        small_tail = len(tail) < chunk_size / 2 and _chunk_token_count(tail, token_counts) < token_budget / 2
        hard_break = small_tail and _is_hard_scene_break(
            chunks[-2][-1][0], tail[0][0], valid_timestamps, scene_threshold
        )
        merged = [*chunks[-2], *tail]
        if (
            small_tail
            and not hard_break
            and _fits_limits(merged, token_counts=token_counts, chunk_size=chunk_size, token_budget=token_budget)
        ):
            chunks[-2] = merged
            chunks.pop()

    plans: list[TranslationChunkPlan] = []
    for chunk_index, chunk in enumerate(chunks, 1):
        hard_break_before = chunk_index > 1 and _is_hard_scene_break(
            chunks[chunk_index - 2][-1][0], chunk[0][0], valid_timestamps, scene_threshold
        )
        plans.append(
            TranslationChunkPlan(
                chunk_id=chunk_index,
                segment_ids=[line_id for line_id, _ in chunk],
                texts=[text for _, text in chunk],
                hard_scene_break_before=hard_break_before,
            )
        )
    return plans


def chunk_plan_signature(
    plans: list[TranslationChunkPlan], *, timestamps: list[tuple[float, float | None]] | None = None
) -> str:
    """Fingerprint ordered chunk boundaries and the source data that produced them."""
    valid_timestamps = (
        timestamps
        if timestamps is not None and sum(len(plan.segment_ids) for plan in plans) == len(timestamps)
        else None
    )
    payload = {
        "planner_version": CHUNK_PLANNER_VERSION,
        "chunks": [
            {
                **plan.model_dump(),
                "timestamps": (
                    [valid_timestamps[line_id - 1] for line_id in plan.segment_ids]
                    if valid_timestamps is not None
                    else None
                ),
            }
            for plan in plans
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
