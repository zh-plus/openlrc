"""Persistent Jobs, recovery, artifacts, and contextual glossary detail."""

from __future__ import annotations

import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding

from openlrc.application import JobRecord, JobRecordStatus, WorkflowDraft
from openlrc.tui.modals import ConfirmModal, DetailModal
from openlrc.tui.navigation import ActionItem, ActionList, action_group
from openlrc.tui.screens.base import OpenLRCScreen
from openlrc.tui.screens.workflow import ConfirmWorkflowScreen
from openlrc.tui.widgets import PageFooter, PageHeader


class JobsScreen(OpenLRCScreen):
    def compose(self) -> ComposeResult:
        items = []
        for record in self.app.job_controller.records:
            completed = sum(item.state in {"completed", "warning"} for item in record.items.values())
            items.append(
                ActionItem(
                    f"job:{record.job_id}",
                    record.name,
                    f"{_status_label(record)} · {completed}/{max(1, len(record.items))}",
                )
            )
        if not items:
            items.append(
                ActionItem(
                    "empty", "No jobs yet", "Completed and recoverable workflows will appear here", disabled=True
                )
            )
        yield PageHeader("Jobs & Recovery", f"{len(self.app.job_controller.records)} records")
        yield ActionList(*action_group("History", *items), id="jobs-list", classes="page-list")
        yield PageFooter("[UP/DOWN] Select   [ENTER] Details   [ESC] Back")

    def on_mount(self) -> None:
        self.call_after_refresh(self.query_one(ActionList).focus)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        if event.action_id.startswith("job:"):
            record = self.app.find_job(event.action_id.partition(":")[2])
            if record is not None:
                self.app.push_screen(JobDetailScreen(record.job_id))


class JobDetailScreen(OpenLRCScreen):
    BINDINGS = [
        *OpenLRCScreen.BINDINGS,
        Binding("c", "cancel_job", "Cancel", show=False, priority=True),
        Binding("r", "resume_job", "Resume", show=False, priority=True),
        Binding("d", "delete_job", "Delete", show=False, priority=True),
        Binding("o", "open_output", "Open", show=False, priority=True),
        Binding("f", "reveal_output", "Reveal", show=False, priority=True),
        Binding("y", "copy_path", "Copy", show=False, priority=True),
    ]

    def __init__(self, job_id: str) -> None:
        super().__init__()
        self.job_id = job_id

    @property
    def record(self) -> JobRecord | None:
        return self.app.find_job(self.job_id)

    def compose(self) -> ComposeResult:
        record = self.record
        if record is None:
            items = [ActionItem("missing", "History record unavailable", "It may have been deleted", disabled=True)]
            title = "Job Detail"
            context = "Missing"
        else:
            title = record.name
            context = _status_label(record)
            items = [
                ActionItem("overview", "Overview", f"{record.workflow} · {record.progress:.1f}%"),
                ActionItem("files", "Files", f"{len(record.items)} items"),
                ActionItem("logs", "Logs", f"{len(record.event_log)} lines"),
            ]
            if record.outputs or record.artifacts:
                items.append(
                    ActionItem(
                        "outputs", "Outputs", f"{len(record.outputs)} primary · {len(record.artifacts)} artifacts"
                    )
                )
            glossary = self.app.glossaries.for_job(record)
            if glossary is not None:
                items.append(
                    ActionItem(
                        "glossary", "Glossary", f"{glossary.entry_count} terms · {glossary.violation_count} violations"
                    )
                )
            if record.reviews or self.app.job_controller.can_resume(record):
                items.append(ActionItem("recovery", "Recovery", _recovery_detail(record)))
            if record.status is JobRecordStatus.RUNNING:
                items.append(ActionItem("cancel", "Cancel", "Stop the active workflow"))
            if self.app.job_controller.can_resume(record):
                items.append(ActionItem("resume", "Resume", "Create a new Draft and run Preflight again"))
            if record.outputs or record.artifacts:
                items.extend(
                    [
                        ActionItem("open", "Open first output", "Open with the default macOS application"),
                        ActionItem("reveal", "Reveal first output", "Show in Finder"),
                        ActionItem("copy", "Copy first output path", "Copy to terminal clipboard"),
                    ]
                )
            items.append(ActionItem("delete", "Delete history record", "Does not delete outputs or recovery files"))
            action_ids = {"cancel", "resume", "open", "reveal", "copy", "delete"}
            detail_items = [item for item in items if item.action_id not in action_ids]
            action_items = [item for item in items if item.action_id in action_ids]
            items = [
                *action_group("Summary & artifacts", *detail_items),
                *action_group("Recovery & actions", *action_items),
            ]
        yield PageHeader(title, context)
        yield ActionList(*items, id="job-detail", classes="page-list")
        yield PageFooter("[R] Resume   [C] Cancel   [O/F/Y] Output   [D] Delete   [ESC] Back")

    def on_mount(self) -> None:
        self.call_after_refresh(self.query_one(ActionList).focus)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        record = self.record
        if record is None:
            return
        action = event.action_id
        if action == "overview":
            self.app.push_screen(DetailModal("Overview", _overview(record)))
        elif action == "files":
            self.app.push_screen(DetailModal("Files", _files(record)))
        elif action == "logs":
            self.app.push_screen(DetailModal("Logs", "\n".join(record.event_log[-300:]) or "No logs."))
        elif action == "outputs":
            self.app.push_screen(DetailModal("Outputs & Artifacts", _outputs(record)))
        elif action == "glossary":
            self._show_glossary(record)
        elif action == "recovery":
            self.app.push_screen(DetailModal("Recovery", _recovery(record)))
        elif action == "cancel":
            self.action_cancel_job()
        elif action == "resume":
            self.action_resume_job()
        elif action == "open":
            self.action_open_output()
        elif action == "reveal":
            self.action_reveal_output()
        elif action == "copy":
            self.action_copy_path()
        elif action == "delete":
            self.action_delete_job()

    def action_cancel_job(self) -> None:
        record = self.record
        if record is None or record.status is not JobRecordStatus.RUNNING:
            return
        self.app.push_screen(
            ConfirmModal(
                "Cancel active job?", "Owned processes will be terminated safely.", confirm_label="Cancel Job"
            ),
            lambda confirmed: self.app.cancel_active_operation() if confirmed else None,
        )

    def action_resume_job(self) -> None:
        record = self.record
        if record is None or not self.app.job_controller.can_resume(record):
            return
        draft = WorkflowDraft.from_recipe(record.recipe)
        draft.paths = list(record.input_paths)
        self.app.begin_workflow(draft)
        self.app.push_screen(ConfirmWorkflowScreen(resumed_from=record.job_id))

    def action_delete_job(self) -> None:
        record = self.record
        if record is None or record.status is JobRecordStatus.RUNNING:
            return
        self.app.push_screen(
            ConfirmModal(
                "Delete history record?",
                "This removes only local history. Outputs, checkpoints, and edit sessions remain on disk.",
                confirm_label="Delete Record",
            ),
            self._delete_confirmed,
        )

    def _delete_confirmed(self, confirmed: bool | None) -> None:
        if not confirmed:
            return
        try:
            self.app.job_controller.delete_record(self.job_id)
        except Exception as exc:
            self.app.notify(str(exc), severity="error")
            return
        self.app.notify("History record deleted.")
        self.app.pop_screen()

    def action_open_output(self) -> None:
        path = self._first_output()
        if path is not None:
            subprocess.Popen(["open", str(path)])

    def action_reveal_output(self) -> None:
        path = self._first_output()
        if path is not None:
            subprocess.Popen(["open", "-R", str(path)])

    def action_copy_path(self) -> None:
        path = self._first_output()
        if path is not None:
            self.app.copy_to_clipboard(str(path))
            self.app.notify("Output path copied.")

    def _first_output(self) -> Path | None:
        record = self.record
        if record is None:
            return None
        raw = next(iter(record.outputs), None)
        if raw is None and record.artifacts:
            candidate = record.artifacts[0].get("path")
            raw = candidate if isinstance(candidate, str) else None
        if raw is None:
            self.app.notify("No output path is available.", severity="warning")
            return None
        return Path(raw)

    def _show_glossary(self, record: JobRecord) -> None:
        inspection = self.app.glossaries.for_job(record)
        if inspection is None:
            self.app.notify("No glossary data is available for this Job.", severity="warning")
            return
        lines = [
            f"Fingerprint: {inspection.state.fingerprint}",
            f"Entries: {inspection.entry_count}",
            f"Violations: {inspection.violation_count}",
            "",
        ]
        lines.extend(
            f"{entry.source} -> {entry.target} · {entry.origin.value}{' · required' if entry.required else ''}"
            for entry in inspection.state.merged_entries
        )
        if inspection.state.matches:
            lines.append("")
            lines.extend(
                f"{match.severity.upper()} · {match.entry_source} · {match.message}"
                for match in inspection.state.matches
            )
        self.app.push_screen(DetailModal("Effective Glossary", "\n".join(lines)))


def _status_label(record: JobRecord) -> str:
    if record.status is JobRecordStatus.SUCCEEDED_WITH_WARNINGS or any(
        bool(review.get("incomplete")) for review in record.reviews
    ):
        return "Needs review"
    return record.status.value.replace("_", " ").title()


def _overview(record: JobRecord) -> str:
    parts = [
        f"Status: {_status_label(record)}",
        f"Workflow: {record.workflow}",
        f"Started: {record.started_at}",
        f"Completed: {record.completed_at or 'Not completed'}",
        f"Progress: {record.progress:.1f}%",
        f"Current stage: {record.current_stage or 'None'}",
        f"Elapsed: {record.elapsed_seconds:.1f}s",
        f"API fee: {record.api_fee:.4f}",
    ]
    if record.error:
        parts.append(f"Error: {record.error.get('message', 'Unknown error')}")
    return "\n".join(parts)


def _files(record: JobRecord) -> str:
    return (
        "\n".join(
            f"{item.display_name} · {item.state} · {item.progress:.1f}%\n  {item.path}"
            for item in record.items.values()
        )
        or "No file details."
    )


def _outputs(record: JobRecord) -> str:
    lines = [f"PRIMARY · {path}" for path in record.outputs]
    lines.extend(f"{artifact.get('kind', 'artifact')} · {artifact.get('path', '')}" for artifact in record.artifacts)
    return "\n".join(lines) or "No outputs."


def _recovery_detail(record: JobRecord) -> str:
    incomplete = sum(bool(review.get("incomplete")) for review in record.reviews)
    return f"{incomplete} incomplete review(s)" if incomplete else "Draft can be rebuilt from the secret-free recipe"


def _recovery(record: JobRecord) -> str:
    lines = [f"Can resume: {_status_label(record)}", f"Resumed from: {record.resumed_from or 'No'}"]
    for review in record.reviews:
        lines.append(
            f"{review.get('item', 'item')} · incomplete={review.get('incomplete', False)} · "
            f"checkpoint={review.get('checkpoint') or 'none'}"
        )
    return "\n".join(lines)
