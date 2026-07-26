"""Single-screen branded card Home for TUI v2."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from openlrc import __version__
from openlrc.application import JobRecordStatus
from openlrc.tui.i18n import tr
from openlrc.tui.navigation import ActionList
from openlrc.tui.screens.base import OpenLRCScreen
from openlrc.tui.widgets.home_card import HomeCard
from openlrc.tui.widgets.logo import LogoWidget
from openlrc.tui.widgets.status import PageFooter


class HomeScreen(OpenLRCScreen):
    BINDINGS = []

    def __init__(self) -> None:
        super().__init__(id="home")

    def compose(self) -> ComposeResult:
        with Vertical(id="home-shell"):
            with Horizontal(id="home-topbar"):
                yield Static(f"OpenLRC Mac  {__version__}", id="home-version", markup=False)
                yield Button("LOCAL READY", id="doctor-status", compact=True, flat=True)
            with Horizontal(id="home-main"):
                with Vertical(id="home-brand"):
                    yield LogoWidget(fixed_frame=self.app.fixed_logo_frame, id="home-logo")
                    yield Static(tr("Local subtitle pipeline"), classes="brand-line", markup=False)
                    yield Static(tr("Transcribe and translate on Mac"), classes="brand-line muted", markup=False)
                yield ActionList(
                    HomeCard("new-workflow", "New Work"),
                    HomeCard("edit", "Edit Subtitle", "Coming later", disabled=True),
                    HomeCard("jobs", "Jobs & Recovery"),
                    HomeCard("models", "Models & Setup"),
                    HomeCard("settings", "Settings"),
                    id="home-cards",
                )
            yield PageFooter("[UP/DOWN] Select   [ENTER] Open   [D] Doctor   [?] Help   [Q] Quit")

    def on_mount(self) -> None:
        self.refresh_status()
        self.query_one("#home-cards", ActionList).focus()

    def on_screen_resume(self) -> None:
        self.app.discard_clean_draft()
        self.refresh(recompose=True)
        self.call_after_refresh(self._resume_after_recompose)

    def _resume_after_recompose(self) -> None:
        self.refresh_status()
        self.focus_default_action_list()

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        self.app.open_route(event.action_id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "doctor-status":
            self.app.open_route("doctor")

    def refresh_status(self) -> None:
        cards = {item.action_id: item for item in self.query(HomeCard)}
        cards["jobs"].set_detail(self._jobs_detail())
        cards["models"].set_detail(self._models_detail())
        cards["settings"].set_detail(self._settings_detail())
        doctor = self.app.doctor_statuses()
        essential = {"ffmpeg", "whisper-cli", "Whisper model"}
        ready = all(status.available for status in doctor if status.name in essential)
        label = "LOCAL WORKING" if self.app.operation_state != "idle" else "LOCAL READY" if ready else "LOCAL SETUP"
        self.query_one("#doctor-status", Button).label = tr(label)

    def _jobs_detail(self) -> str:
        records = self.app.job_controller.records
        review_count = sum(
            record.status is JobRecordStatus.SUCCEEDED_WITH_WARNINGS
            or any(bool(review.get("incomplete")) for review in record.reviews)
            for record in records
        )
        failed_count = sum(record.status in {JobRecordStatus.FAILED, JobRecordStatus.INTERRUPTED} for record in records)
        running_count = sum(record.status is JobRecordStatus.RUNNING for record in records)
        if review_count:
            return f"{review_count} needs review"
        if failed_count:
            return f"{failed_count} failed"
        if running_count:
            return f"{running_count} running"
        return ""

    def _models_detail(self) -> str:
        statuses = self.app.model_statuses()
        setup_names = {"whisper-cli", "llama-server"}
        if any(not status.available and status.name in setup_names for status in statuses):
            return "Setup required"
        missing = sum(not status.available for status in statuses if "Hy-MT2" not in status.name)
        return f"{missing} model missing" if missing else ""

    def _settings_detail(self) -> str:
        if self.app.credentials.last_error:
            return "Credential error"
        if self.app.settings_dirty:
            return "Unsaved changes"
        return ""
