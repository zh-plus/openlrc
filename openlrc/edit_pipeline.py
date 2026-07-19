#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

"""Model-agnostic orchestration for deterministic repair and bounded edit rounds."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from openlrc.edit_validators import DeterministicValidatorSuite
from openlrc.editing import (
    EditAction,
    EditIssue,
    EditIssueStatus,
    EditPatch,
    EditRound,
    EditRoundStatus,
    EditSession,
    EditSessionStatus,
    EditSeverity,
    EditStopReason,
    apply_patch_transaction,
    source_fingerprint,
    subtitle_fingerprint,
)
from openlrc.subtitle import Subtitle

TargetedRepair = Callable[[list[int], list[str]], dict[int, str]]
SemanticReview = Callable[[list[int], list[str], int], tuple[list[EditIssue], list[EditPatch], dict[str, Any]]]
CheckpointHook = Callable[[EditSession], None]


class EditPipeline:
    """Apply every model proposal through the same deterministic transaction."""

    def __init__(
        self,
        source: Subtitle,
        validator_suite: DeterministicValidatorSuite,
        *,
        glossary_fingerprint: str = "",
        max_rounds: int = 1,
        restore_enabled: bool = False,
        report_matches: bool = True,
        checkpoint_hook: CheckpointHook | None = None,
    ):
        if not 0 <= max_rounds <= 3:
            raise ValueError("EditPipeline max_rounds must be between 0 and 3.")
        self.source = source
        self.validators = validator_suite
        self.glossary_fingerprint = glossary_fingerprint
        self.max_rounds = max_rounds
        self.restore_enabled = restore_enabled
        self.report_matches = report_matches
        self.checkpoint_hook = checkpoint_hook
        self.timestamps = [(segment.start, segment.end) for segment in source.segments]

    def new_session(self, translations: list[str]) -> EditSession:
        return EditSession(
            source_fingerprint=source_fingerprint(self.source.texts, self.timestamps, language=self.source.lang),
            translation_fingerprint=subtitle_fingerprint(translations, self.timestamps),
            glossary_fingerprint=self.glossary_fingerprint,
            status=EditSessionStatus.CHECKING,
            max_rounds=self.max_rounds,
            restore_enabled=self.restore_enabled,
            raw_translations=list(translations),
            current_translations=list(translations),
            source_timestamps=self.timestamps,
        )

    def _save(self, session: EditSession) -> None:
        session.translation_fingerprint = subtitle_fingerprint(session.current_translations, self.timestamps)
        glossary_state = self.validators.effective_glossary_state
        if glossary_state is not None:
            reported_state = (
                glossary_state if self.report_matches else glossary_state.model_copy(update={"matches": []})
            )
            session.glossary_state = reported_state.model_dump(mode="json")
            session.metrics["glossary"] = glossary_state.metrics.model_dump(mode="json")
        session.metrics.update(
            changed_ids=sorted(
                {patch.segment_id for edit_round in session.rounds for patch in edit_round.applied_patches}
            ),
            applied_patches=sum(len(edit_round.applied_patches) for edit_round in session.rounds),
            rejected_patches=sum(len(edit_round.rejected_patches) for edit_round in session.rounds),
            unresolved_issues=len(session.unresolved_issues),
        )
        if self.checkpoint_hook is not None:
            self.checkpoint_hook(session)

    @staticmethod
    def _repairable_ids(issues: list[EditIssue]) -> list[int]:
        repairable = {
            line_id
            for issue in issues
            if issue.severity is EditSeverity.ERROR and issue.category in {"glossary", "number", "entity"}
            for line_id in issue.segment_ids
        }
        return sorted(repairable)

    def run_deterministic_repair(self, session: EditSession, *, repair: TargetedRepair | None) -> EditSession:
        """Run offline checks and at most one batched Hy-MT2 repair pass."""
        before = self.validators.validate(session.current_translations)
        scope = self._repairable_ids(before)
        if not scope or repair is None:
            session.unresolved_issues = before
            session.status = EditSessionStatus.EDITING if self.max_rounds else EditSessionStatus.COMPLETE
            self._save(session)
            return session

        replacements = repair(scope, session.current_translations)
        patches: list[EditPatch] = []
        for segment_id in scope:
            related = [issue for issue in before if segment_id in issue.segment_ids]
            patches.append(
                EditPatch.create(
                    issue_ids=[issue.issue_id for issue in related],
                    segment_id=segment_id,
                    before=session.current_translations[segment_id - 1],
                    after=replacements[segment_id],
                    reason="; ".join(issue.message for issue in related) or "Deterministic repair",
                    action=EditAction.RETRANSLATE,
                )
            )
        result = apply_patch_transaction(
            session.current_translations,
            patches,
            scope_ids=scope,
            baseline_issues=before,
            validate=self.validators.validate,
        )
        validation_after = result.validation_after or self.validators.validate(result.translations)
        edit_round = EditRound(
            round_index=0,
            status=EditRoundStatus.COMMITTED if result.committed else EditRoundStatus.FAILED,
            scope_ids=scope,
            issues_before=before,
            proposed_patches=patches,
            applied_patches=result.applied_patches,
            rejected_patches=result.rejected_patches,
            validation_after=validation_after,
            model_metadata={"stage": "hy-mt2-deterministic-repair"},
            stop_reason=None if result.committed else EditStopReason.VALIDATION_FAILURE,
        )
        session.rounds.append(edit_round)
        session.current_translations = result.translations
        session.unresolved_issues = validation_after
        session.status = (
            EditSessionStatus.EDITING
            if self.max_rounds
            else (EditSessionStatus.COMPLETE if result.committed else EditSessionStatus.INCOMPLETE)
        )
        self._save(session)
        return session

    @staticmethod
    def _next_scope(*, changed_ids: list[int], unresolved: list[EditIssue], maximum: int) -> list[int]:
        ids = set(changed_ids)
        ids.update(line_id for issue in unresolved for line_id in issue.segment_ids)
        expanded = set(ids)
        for line_id in ids:
            if line_id > 1:
                expanded.add(line_id - 1)
            if line_id < maximum:
                expanded.add(line_id + 1)
        return sorted(expanded)

    def run_semantic_rounds(self, session: EditSession, *, review: SemanticReview | None) -> EditSession:
        if review is None or self.max_rounds == 0:
            session.status = (
                EditSessionStatus.INCOMPLETE
                if any(issue.severity is EditSeverity.ERROR for issue in session.unresolved_issues)
                else EditSessionStatus.COMPLETE
            )
            self._save(session)
            return session

        semantic_rounds = [item for item in session.rounds if item.round_index > 0]
        successful_indexes = {
            item.round_index
            for item in semantic_rounds
            if item.status in {EditRoundStatus.COMMITTED, EditRoundStatus.COMPLETED}
        }
        retry_round = next(
            (
                item
                for item in reversed(semantic_rounds)
                if item.status in {EditRoundStatus.STARTED, EditRoundStatus.FAILED}
                and item.round_index not in successful_indexes
            ),
            None,
        )
        start_round = retry_round.round_index if retry_round is not None else max(successful_indexes, default=0) + 1
        if start_round > self.max_rounds:
            self._save(session)
            return session
        successful_rounds = [
            item for item in semantic_rounds if item.status in {EditRoundStatus.COMMITTED, EditRoundStatus.COMPLETED}
        ]
        if retry_round is not None:
            scope = list(retry_round.scope_ids)
        elif successful_rounds:
            last_round = max(successful_rounds, key=lambda item: item.round_index)
            scope = self._next_scope(
                changed_ids=[patch.segment_id for patch in last_round.applied_patches],
                unresolved=session.unresolved_issues,
                maximum=len(self.source),
            )
        else:
            scope = list(range(1, len(self.source) + 1))
        for round_index in range(start_round, self.max_rounds + 1):
            session.status = EditSessionStatus.EDITING
            attempt_position = len(session.rounds)
            session.rounds.append(EditRound(round_index=round_index, status=EditRoundStatus.STARTED, scope_ids=scope))
            self._save(session)
            before_hash = subtitle_fingerprint(session.current_translations, self.timestamps)
            semantic_issues, patches, metadata = review(scope, session.current_translations, round_index)
            high_risk = [issue for issue in semantic_issues if issue.severity is EditSeverity.ERROR]
            model_failed = any(issue.status is EditIssueStatus.FAILED for issue in semantic_issues)
            deterministic_before = self.validators.validate(session.current_translations)
            issues_before = [*deterministic_before, *semantic_issues]
            if not high_risk:
                edit_round = EditRound(
                    round_index=round_index,
                    status=EditRoundStatus.COMPLETED,
                    scope_ids=scope,
                    issues_before=issues_before,
                    model_metadata=metadata,
                    stop_reason=EditStopReason.NO_HIGH_RISK,
                )
                session.rounds[attempt_position] = edit_round
                session.unresolved_issues = deterministic_before
                session.status = (
                    EditSessionStatus.INCOMPLETE
                    if any(issue.severity is EditSeverity.ERROR for issue in deterministic_before)
                    else EditSessionStatus.COMPLETE
                )
                self._save(session)
                return session
            if not patches:
                session.rounds[attempt_position] = EditRound(
                    round_index=round_index,
                    status=EditRoundStatus.FAILED,
                    scope_ids=scope,
                    issues_before=issues_before,
                    proposed_patches=patches,
                    model_metadata=metadata,
                    stop_reason=(EditStopReason.MODEL_FAILURE if model_failed else EditStopReason.NO_EFFECTIVE_PATCH),
                )
                session.unresolved_issues = issues_before
                session.status = EditSessionStatus.INCOMPLETE
                self._save(session)
                return session

            if model_failed:
                session.rounds[attempt_position] = EditRound(
                    round_index=round_index,
                    status=EditRoundStatus.FAILED,
                    scope_ids=scope,
                    issues_before=issues_before,
                    proposed_patches=patches,
                    validation_after=deterministic_before,
                    model_metadata=metadata,
                    stop_reason=EditStopReason.MODEL_FAILURE,
                )
                session.unresolved_issues = issues_before
                session.status = EditSessionStatus.INCOMPLETE
                self._save(session)
                return session

            result = apply_patch_transaction(
                session.current_translations,
                patches,
                scope_ids=scope,
                baseline_issues=deterministic_before,
                validate=self.validators.validate,
            )
            after_hash = subtitle_fingerprint(result.translations, self.timestamps)
            if result.committed:
                resolved_ids = {issue_id for patch in result.applied_patches for issue_id in patch.issue_ids}
                semantic_after = [
                    issue.model_copy(update={"status": EditIssueStatus.DEFERRED})
                    for issue in high_risk
                    if issue.issue_id not in resolved_ids
                ]
                unresolved = [*result.validation_after, *semantic_after]
                stop_reason = EditStopReason.UNCHANGED_HASH if after_hash == before_hash else None
            else:
                unresolved = issues_before
                stop_reason = EditStopReason.VALIDATION_FAILURE

            edit_round = EditRound(
                round_index=round_index,
                status=(
                    EditRoundStatus.COMMITTED
                    if result.committed and after_hash != before_hash
                    else EditRoundStatus.FAILED
                ),
                scope_ids=scope,
                issues_before=issues_before,
                proposed_patches=patches,
                applied_patches=result.applied_patches,
                rejected_patches=result.rejected_patches,
                validation_after=result.validation_after,
                model_metadata=metadata,
                stop_reason=stop_reason,
            )
            session.rounds[attempt_position] = edit_round
            session.current_translations = result.translations
            session.unresolved_issues = unresolved
            self._save(session)

            if not result.committed or after_hash == before_hash:
                session.status = EditSessionStatus.INCOMPLETE
                self._save(session)
                return session
            if round_index == self.max_rounds:
                edit_round.stop_reason = EditStopReason.MAX_ROUNDS
                session.status = (
                    EditSessionStatus.INCOMPLETE
                    if any(issue.severity is EditSeverity.ERROR for issue in unresolved)
                    else EditSessionStatus.COMPLETE
                )
                self._save(session)
                return session
            scope = self._next_scope(
                changed_ids=[patch.segment_id for patch in result.applied_patches],
                unresolved=unresolved,
                maximum=len(self.source),
            )

        return session
