"""OpenLRC Mac TUI v2 application shell and long-operation lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from textual import work
from textual.app import App
from textual.binding import Binding
from textual.notifications import SeverityLevel
from textual.theme import Theme
from textual.widgets import Input, TextArea

from openlrc.application import (
    AppSettings,
    CredentialStore,
    GlossaryApplicationService,
    JobController,
    JobRecord,
    OperationGuard,
    ResourceStatus,
    ResourceStatusService,
    SettingsStore,
    SetupController,
    SetupEvent,
    SetupFailedEvent,
    SetupLogEvent,
    SetupRequest,
    SetupResult,
    SetupStageEvent,
    WorkflowDraft,
    build_provider_model,
    preflight,
    test_provider_connection,
)
from openlrc.application.settings import GeneralSettings
from openlrc.tui.i18n import tr
from openlrc.tui.modals import ChoiceModal, ConfirmModal, DetailModal
from openlrc.tui.modals.file_picker import choose_files_native
from openlrc.tui.screens.doctor import DoctorScreen
from openlrc.tui.screens.home import HomeScreen
from openlrc.tui.screens.jobs import JobsScreen
from openlrc.tui.screens.models import ModelsScreen, SetupRunningScreen
from openlrc.tui.screens.settings import SettingsScreen
from openlrc.tui.screens.workflow import RunningWorkflowScreen, WorkflowOptionsScreen, WorkflowTypeScreen
from openlrc.tui.widgets.logo import LogoWidget
from openlrc.workflow import WorkflowEvent, WorkflowResult

OPENLRC_DARK = Theme(
    name="openlrc-dark",
    primary="#48c6dc",
    secondary="#318fa7",
    warning="#f0aa4b",
    error="#dc6b72",
    success="#63c174",
    accent="#f0aa4b",
    foreground="#d9e1e8",
    background="#0b0f14",
    surface="#111820",
    panel="#0e151c",
    dark=True,
)


class OpenLRCTUI(App[None]):
    """A page-stack TUI that consumes shared application and Workflow contracts."""

    CSS_PATH = "openlrc.tcss"
    TITLE = "OpenLRC Mac"
    BINDINGS = [
        Binding("q", "quit_requested", "Quit", show=False, priority=True),
        Binding("d", "doctor", "Doctor", show=False, priority=True),
        Binding("n", "new_workflow", "New Work", show=False, priority=True),
        Binding("question_mark", "help", "Help", show=False, priority=True),
        Binding("ctrl+c", "interrupt", "Cancel / Quit", show=False, priority=True),
        Binding("g", "goto_prefix", "Go to", show=False, priority=True),
        Binding("h", "goto_home", "Home", show=False, priority=True),
        Binding("j", "goto_jobs", "Jobs", show=False, priority=True),
    ]

    def __init__(
        self,
        *,
        settings_store: SettingsStore | None = None,
        credentials: CredentialStore | None = None,
        job_controller: JobController | None = None,
        setup_controller: SetupController | None = None,
        resources: ResourceStatusService | None = None,
        glossaries: GlossaryApplicationService | None = None,
        preflight_service: Callable = preflight,
        fixed_logo_frame: int | None = None,
    ) -> None:
        super().__init__()
        self.register_theme(OPENLRC_DARK)
        guard = OperationGuard()
        self.settings_store = settings_store or SettingsStore()
        self.settings = self.settings_store.load()
        self.working_settings = AppSettings.from_dict(self.settings.to_dict())
        self._visual_general = self.settings.general
        self.credentials = credentials or CredentialStore()
        self.job_controller = job_controller or JobController(operation_guard=guard)
        self.setup_controller = setup_controller or SetupController(operation_guard=guard)
        self.resources = resources or ResourceStatusService()
        self.glossaries = glossaries or GlossaryApplicationService()
        self.preflight_service = preflight_service
        self.fixed_logo_frame = fixed_logo_frame

        self._draft: WorkflowDraft | None = None
        self._draft_baseline: dict[str, object] | None = None
        self.input_issues: list[str] = []
        self.last_picker_directory = Path.cwd()
        self.operation_state = "idle"
        self.active_operation_kind: str | None = None
        self.last_job_record: JobRecord | None = None
        self.running_workflow_screen: RunningWorkflowScreen | None = None
        self.setup_running_screen: SetupRunningScreen | None = None
        self.setup_stage = ""
        self.setup_logs: list[str] = []
        self.exit_after_operation = False
        self._native_picker_callback: Callable[[list[str]], None] | None = None
        self._doctor_cache: list[ResourceStatus] | None = None
        self._model_cache: list[ResourceStatus] | None = None
        self._goto_pending = False

    @property
    def settings_dirty(self) -> bool:
        return self.working_settings.to_dict() != self.settings.to_dict()

    @property
    def visual_general(self) -> GeneralSettings:
        return self._visual_general

    @property
    def ui_language(self) -> str:
        return self._visual_general.language

    @property
    def draft(self) -> WorkflowDraft:
        """Return the Draft required by every workflow screen."""
        if self._draft is None:
            raise RuntimeError("No Workflow Draft is active.")
        return self._draft

    @property
    def draft_dirty(self) -> bool:
        if self._draft is None or self._draft_baseline is None:
            return False
        return self._draft.to_recipe() != self._draft_baseline

    def on_mount(self) -> None:
        self.apply_visual_settings(self.settings.general)
        self.push_screen(HomeScreen())

    def apply_visual_settings(self, general: GeneralSettings | None = None) -> None:
        """Preview or restore presentation-only settings without touching Workflow state."""

        self._visual_general = general or self.settings.general
        self.theme = self._visual_general.theme
        for logo in self.query(LogoWidget):
            logo.refresh_visual_settings()

    def notify(
        self,
        message: str,
        *,
        title: str = "",
        severity: SeverityLevel = "information",
        timeout: float | None = None,
        markup: bool = True,
    ) -> None:
        super().notify(
            tr(message, language=self.ui_language),
            title=tr(title, language=self.ui_language),
            severity=severity,
            timeout=timeout,
            markup=markup,
        )

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        del parameters
        if action in {"quit_requested", "doctor", "new_workflow", "help", "goto_prefix", "goto_home", "goto_jobs"}:
            if isinstance(self.focused, (Input, TextArea)):
                return False
        if action in {"goto_home", "goto_jobs"}:
            return self._goto_pending
        return True

    def open_route(self, action_id: str) -> None:
        if action_id == "edit":
            return
        if action_id == "new-workflow":
            self._open_new_workflow()
        elif action_id == "jobs":
            self.push_screen(JobsScreen())
        elif action_id == "models":
            self.push_screen(ModelsScreen())
        elif action_id == "settings":
            self.working_settings = AppSettings.from_dict(self.settings.to_dict())
            self.apply_visual_settings(self.working_settings.general)
            self.push_screen(SettingsScreen())
        elif action_id == "doctor":
            self.push_screen(DoctorScreen())

    def _open_new_workflow(self) -> None:
        if self.draft_dirty:
            assert self._draft is not None
            choices = [
                ("continue", "Continue Draft", self._draft.task.replace("-", " ").title()),
                ("discard", "Discard and Start New", "Remove the current unsaved Draft"),
            ]
            self.push_screen(ChoiceModal("Existing Workflow Draft", choices), self._draft_choice)
        else:
            self.discard_draft()
            self.push_screen(WorkflowTypeScreen())

    def _draft_choice(self, choice: str | None) -> None:
        if choice == "continue":
            self.push_screen(WorkflowOptionsScreen())
        elif choice == "discard":
            self.discard_draft()
            self.push_screen(WorkflowTypeScreen())

    def begin_workflow(self, draft: WorkflowDraft) -> None:
        self._draft = draft
        self._draft_baseline = None
        self.input_issues = []

    def establish_draft_baseline(self) -> None:
        if self._draft is not None and self._draft_baseline is None:
            self._draft_baseline = self._draft.to_recipe()

    def discard_clean_draft(self) -> None:
        if self._draft is not None and not self.draft_dirty:
            self.discard_draft()

    def discard_draft(self) -> None:
        self._draft = None
        self._draft_baseline = None
        self.input_issues = []

    def go_back(self) -> None:
        if self.screen.id == "home":
            return
        if self.screen.id == "settings-root" and self.settings_dirty:
            choices = [
                ("save", "Save and Leave", "Write the settings working copy"),
                ("discard", "Discard and Leave", "Restore saved settings"),
                ("stay", "Stay", "Continue editing"),
            ]
            self.push_screen(ChoiceModal("Unsaved Settings", choices), self._settings_leave)
            return
        self.pop_screen()

    def _settings_leave(self, choice: str | None) -> None:
        if choice == "save":
            self.save_settings()
            self.pop_screen()
        elif choice == "discard":
            self.discard_settings()
            self.pop_screen()

    def save_settings(self) -> None:
        try:
            self.settings_store.save(self.working_settings)
        except Exception as exc:
            self.notify(f"Settings were not saved: {exc}", severity="error", timeout=8)
            return
        self.settings = AppSettings.from_dict(self.working_settings.to_dict())
        self.apply_visual_settings(self.settings.general)
        self.clear_resource_cache()
        self.notify("Settings saved.")

    def discard_settings(self) -> None:
        self.working_settings = AppSettings.from_dict(self.settings.to_dict())
        self.apply_visual_settings(self.working_settings.general)
        self.notify("Unsaved settings discarded.")

    def action_quit_requested(self) -> None:
        if self.operation_state != "idle":
            message = "The active operation must be cancelled and cleaned up before OpenLRC exits."
        else:
            message = "Exit OpenLRC Mac?"
        self.push_screen(ConfirmModal("Quit?", message, confirm_label="Quit"), self._quit_confirmed)

    def _quit_confirmed(self, confirmed: bool | None) -> None:
        if not confirmed:
            return
        if self.operation_state == "idle":
            self.exit()
        else:
            self.exit_after_operation = True
            self.cancel_active_operation()

    def action_interrupt(self) -> None:
        if self.operation_state == "idle":
            self.action_quit_requested()
            return
        self.push_screen(
            ConfirmModal(
                "Cancel active operation?",
                "Cleanup completes before the operation reaches Cancelled.",
                confirm_label="Cancel",
            ),
            lambda confirmed: self.cancel_active_operation() if confirmed else None,
        )

    def action_doctor(self) -> None:
        self.open_route("doctor")

    def action_new_workflow(self) -> None:
        self.open_route("new-workflow")

    def action_help(self) -> None:
        content = (
            "UP/DOWN or j/k  Move through lists\n"
            "ENTER            Open the selected row\n"
            "SPACE            Activate or toggle a row\n"
            "ESC              Return one page\n"
            "g h              Home\n"
            "g j              Jobs & Recovery\n"
            "d                Doctor\n"
            "n                New Work\n"
            "Ctrl-C           Cancel active operation or quit\n"
            "q                Quit\n\n"
            "Single-character shortcuts are disabled while editing text."
        )
        self.push_screen(DetailModal("Keyboard Help", content))

    def action_goto_prefix(self) -> None:
        self._goto_pending = True
        self.set_timer(1.0, self._clear_goto_prefix)

    def action_goto_home(self) -> None:
        self._goto_pending = False
        stack = list(self.screen_stack)
        home_index = next((index for index, screen in enumerate(stack) if screen.id == "home"), len(stack) - 1)
        for _ in range(len(stack) - home_index - 1):
            self.pop_screen()

    def action_goto_jobs(self) -> None:
        self._goto_pending = False
        self.push_screen(JobsScreen())

    def _clear_goto_prefix(self) -> None:
        self._goto_pending = False

    def doctor_statuses(self) -> list[ResourceStatus]:
        if self._doctor_cache is None:
            self._doctor_cache = self.resources.doctor()
        return self._doctor_cache

    def model_statuses(self) -> list[ResourceStatus]:
        if self._model_cache is None:
            self._model_cache = self.resources.models(self.settings.local_models)
        return self._model_cache

    def clear_resource_cache(self) -> None:
        self._doctor_cache = None
        self._model_cache = None

    def find_job(self, job_id: str) -> JobRecord | None:
        return next((record for record in self.job_controller.records if record.job_id == job_id), None)

    def preflight(self, draft: WorkflowDraft):
        return self.preflight_service(draft, self.settings, self.credentials)

    def open_native_picker(self, callback: Callable[[list[str]], None]) -> None:
        self._native_picker_callback = callback
        self._choose_files_native()

    @work(thread=True, group="native-picker", exclusive=True)
    def _choose_files_native(self) -> None:
        json_only = bool(self._draft and self._draft.workflow == "translate")
        try:
            paths = choose_files_native(
                self.last_picker_directory,
                json_only=json_only,
                language=self.ui_language,
            )
        except Exception as exc:
            self.call_from_thread(self._native_picker_finished, [], str(exc))
        else:
            self.call_from_thread(self._native_picker_finished, paths, None)

    def _native_picker_finished(self, paths: list[str], error: str | None) -> None:
        callback, self._native_picker_callback = self._native_picker_callback, None
        if error:
            self.notify(error, severity="error", timeout=8)
        elif callback is not None:
            callback(paths)

    def start_workflow(self, *, resumed_from: str | None = None) -> None:
        if self._draft is None or self.operation_state != "idle":
            self.notify("Another operation is already active.", severity="warning")
            return
        self.operation_state = "starting"
        self.active_operation_kind = "workflow"
        self.last_job_record = None
        self.running_workflow_screen = RunningWorkflowScreen()
        self.push_screen(self.running_workflow_screen)
        draft = WorkflowDraft.from_recipe(self._draft.to_recipe())
        self._run_workflow(draft, resumed_from)

    @work(thread=True, group="workflow", exclusive=True)
    def _run_workflow(self, draft: WorkflowDraft, resumed_from: str | None) -> None:
        try:
            result = self.job_controller.run(
                draft,
                self.settings,
                self.credentials,
                on_event=lambda event: self.call_from_thread(self._workflow_event, event),
                resumed_from=resumed_from,
            )
        except Exception as exc:
            self.call_from_thread(self._operation_start_failed, str(exc))
        else:
            self.call_from_thread(self._workflow_finished, result)

    def _workflow_event(self, event: WorkflowEvent) -> None:
        del event
        if self.operation_state == "starting":
            self.operation_state = "running"
        if self.running_workflow_screen is not None and self.running_workflow_screen.is_mounted:
            self.running_workflow_screen.refresh_from_record()

    def _workflow_finished(self, result: WorkflowResult) -> None:
        self.last_job_record = self.find_job(result.job_id)
        self.operation_state = "idle"
        self.active_operation_kind = None
        if self._draft is not None:
            self._draft_baseline = self._draft.to_recipe()
        if self.running_workflow_screen is not None and self.running_workflow_screen.is_mounted:
            self.running_workflow_screen.finish(result)
        self._operation_complete()

    def start_setup(self, request: SetupRequest) -> None:
        if self.operation_state != "idle":
            self.notify("Another operation is already active.", severity="warning")
            return
        self.operation_state = "starting"
        self.active_operation_kind = "setup"
        self.setup_stage = "Starting setup"
        self.setup_logs = []
        self.setup_running_screen = SetupRunningScreen(request)
        self.push_screen(self.setup_running_screen)
        self._run_setup(request)

    @work(thread=True, group="setup", exclusive=True)
    def _run_setup(self, request: SetupRequest) -> None:
        try:
            result = self.setup_controller.run(
                request, on_event=lambda event: self.call_from_thread(self._setup_event, event)
            )
        except Exception as exc:
            self.call_from_thread(self._operation_start_failed, str(exc))
        else:
            self.call_from_thread(self._setup_finished, result)

    def _setup_event(self, event: SetupEvent) -> None:
        if self.operation_state == "starting":
            self.operation_state = "running"
        if isinstance(event, SetupStageEvent):
            self.setup_stage = event.message
        elif isinstance(event, SetupLogEvent):
            self.setup_logs.append(event.message)
            del self.setup_logs[:-1000]
        elif isinstance(event, SetupFailedEvent):
            self.setup_stage = event.message
        if self.setup_running_screen is not None and self.setup_running_screen.is_mounted:
            self.setup_running_screen.update_event()

    def _setup_finished(self, result: SetupResult) -> None:
        self.operation_state = "idle"
        self.active_operation_kind = None
        self.clear_resource_cache()
        if self.setup_running_screen is not None and self.setup_running_screen.is_mounted:
            self.setup_running_screen.finish(result)
        self._operation_complete()

    def cancel_active_operation(self) -> None:
        if self.operation_state == "idle" or self.operation_state == "cancelling":
            return
        self.operation_state = "cancelling"
        if self.active_operation_kind == "workflow":
            token = self.job_controller.prepare_cancel()
            if token is None:
                return
            self._cancel_token(token)
        elif self.active_operation_kind == "setup":
            self._cancel_setup()

    @work(thread=True, group="cancellation", exclusive=True)
    def _cancel_token(self, token) -> None:
        token.cancel()

    @work(thread=True, group="cancellation", exclusive=True)
    def _cancel_setup(self) -> None:
        self.setup_controller.cancel()

    def _operation_start_failed(self, message: str) -> None:
        operation_kind = self.active_operation_kind
        self.operation_state = "idle"
        self.active_operation_kind = None
        if operation_kind == "workflow" and self.running_workflow_screen is not None:
            if self.running_workflow_screen.is_mounted:
                self.running_workflow_screen.fail(message)
        elif operation_kind == "setup" and self.setup_running_screen is not None:
            if self.setup_running_screen.is_mounted:
                self.setup_running_screen.fail(message)
        self.notify(message, severity="error", timeout=10)
        self._operation_complete()

    def _operation_complete(self) -> None:
        home = next((screen for screen in self.screen_stack if isinstance(screen, HomeScreen)), None)
        if home is not None and home.is_mounted:
            home.refresh_status()
        if self.exit_after_operation:
            self.exit()

    @work(thread=True, group="provider-test", exclusive=True)
    def test_provider(self, provider: str) -> None:
        try:
            profile = self.working_settings.providers[provider]
            model = build_provider_model(provider, profile, self.credentials)
            result = test_provider_connection(model)
        except Exception as exc:
            self.call_from_thread(self.notify, str(exc), severity="error", timeout=8)
        else:
            self.call_from_thread(self.notify, result, timeout=8)


def run() -> None:
    OpenLRCTUI().run()
