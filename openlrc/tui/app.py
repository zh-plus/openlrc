"""OpenLRC Mac TUI v2 application shell and long-operation lifecycle."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from textual import work
from textual.app import App
from textual.await_complete import AwaitComplete
from textual.binding import Binding
from textual.message import Message
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
from openlrc.logger import handler as openlrc_terminal_handler
from openlrc.logger import logger as openlrc_logger
from openlrc.tui.i18n import tr
from openlrc.tui.modals import ChoiceModal, ConfirmModal, DetailModal
from openlrc.tui.modals.file_picker import choose_files_native
from openlrc.tui.screens.doctor import DoctorScreen
from openlrc.tui.screens.home import HomeScreen
from openlrc.tui.screens.jobs import JobsScreen
from openlrc.tui.screens.models import ModelsScreen, SetupRunningScreen
from openlrc.tui.screens.settings import SettingsScreen
from openlrc.tui.screens.workflow import (
    ConfirmWorkflowScreen,
    RunningWorkflowScreen,
    WorkflowOptionsScreen,
    WorkflowTypeScreen,
)
from openlrc.tui.widgets.logo import LogoWidget
from openlrc.workflow import CancellationToken, WorkflowEvent, WorkflowResult


class RuntimeLogLine(Message):
    """A thread-safe OpenLRC log record routed into the active TUI operation."""

    def __init__(self, operation_kind: str | None, line: str) -> None:
        super().__init__()
        self.operation_kind = operation_kind
        self.line = line


class _TextualLogHandler(logging.Handler):
    """Forward OpenLRC records without writing through the terminal stream."""

    def __init__(self, sink: Callable[[str], None]) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sink(self.format(record))
        except Exception:
            # Logging must never interrupt the Workflow or corrupt the TUI.
            return


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
        Binding("d", "doctor", "Doctor", show=False),
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
        self.workflow_output: list[str] = []
        self._active_cancellation_token: CancellationToken | None = None
        self.exit_after_operation = False
        self._native_picker_callback: Callable[[list[str]], None] | None = None
        self._doctor_cache: list[ResourceStatus] | None = None
        self._model_cache: list[ResourceStatus] | None = None
        self._goto_pending = False
        self._finished_workflow_close_pending = False
        self._terminal_log_handler_removed = False
        self._tui_log_handler = _TextualLogHandler(self._post_runtime_log)
        self._tui_log_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

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
        self._install_log_capture()
        self.apply_visual_settings(self.settings.general)
        self.push_screen(HomeScreen())
        history_warning = self.job_controller.repository.last_warning
        if history_warning:
            self.job_controller.repository.last_warning = None
            self.notify(history_warning, severity="warning", timeout=10)

    def on_unmount(self) -> None:
        self._uninstall_log_capture()

    def _install_log_capture(self) -> None:
        if self._tui_log_handler in openlrc_logger.handlers:
            return
        if openlrc_terminal_handler in openlrc_logger.handlers:
            openlrc_logger.removeHandler(openlrc_terminal_handler)
            self._terminal_log_handler_removed = True
        openlrc_logger.addHandler(self._tui_log_handler)

    def _uninstall_log_capture(self) -> None:
        if self._tui_log_handler in openlrc_logger.handlers:
            openlrc_logger.removeHandler(self._tui_log_handler)
        if self._terminal_log_handler_removed and openlrc_terminal_handler not in openlrc_logger.handlers:
            openlrc_logger.addHandler(openlrc_terminal_handler)
        self._terminal_log_handler_removed = False

    def _post_runtime_log(self, line: str) -> None:
        self.post_message(RuntimeLogLine(self.active_operation_kind, line))

    def on_runtime_log_line(self, message: RuntimeLogLine) -> None:
        if message.operation_kind == "workflow":
            self.workflow_output.append(message.line)
            del self.workflow_output[:-500]
            screen = self.running_workflow_screen
            if screen is not None and screen.is_mounted:
                screen.append_runtime_output(message.line)
        elif message.operation_kind == "setup":
            self.setup_logs.append(message.line)
            del self.setup_logs[:-1000]
            screen = self.setup_running_screen
            if screen is not None and screen.is_mounted:
                screen.update_event()

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

    def resume_workflow(self, record: JobRecord) -> None:
        draft = WorkflowDraft.from_recipe(record.recipe)
        context_defaults = WorkflowDraft.defaults(self.settings)
        draft.context_provider = draft.context_provider or context_defaults.context_provider
        draft.context_model = draft.context_model or context_defaults.context_model
        draft.paths = list(record.input_paths)
        if not self.draft_dirty:
            self._open_resumed_workflow(draft, record.job_id)
            return
        choices = [
            ("resume", "Discard Draft and Resume", "Replace the current unsaved Draft"),
            ("keep", "Keep Current Draft", "Cancel Resume without changing the Draft"),
        ]
        self.push_screen(
            ChoiceModal("Unsaved Workflow Draft", choices),
            lambda choice: self._resume_choice(choice, draft, record.job_id),
        )

    def _resume_choice(self, choice: str | None, draft: WorkflowDraft, job_id: str) -> None:
        if choice == "resume":
            self._open_resumed_workflow(draft, job_id)

    def _open_resumed_workflow(self, draft: WorkflowDraft, job_id: str) -> None:
        self.begin_workflow(draft)
        self.push_screen(ConfirmWorkflowScreen(resumed_from=job_id))

    def establish_draft_baseline(self) -> None:
        if self._draft is not None and self._draft_baseline is None:
            self._draft_baseline = self._draft.to_recipe()

    def discard_clean_draft(self) -> None:
        if not self._finished_workflow_close_pending and self._draft is not None and not self.draft_dirty:
            self.discard_draft()

    def discard_draft(self) -> None:
        self._draft = None
        self._draft_baseline = None
        self.input_issues = []

    def close_finished_workflow(self, destination: str | None = None) -> None:
        """Leave a terminal Workflow result without retaining its configuration stack."""
        self._finished_workflow_close_pending = True
        completions = self._pop_to_home()
        self.run_worker(
            self._finalize_finished_workflow_close(completions, destination),
            group="finished-workflow-close",
            exclusive=True,
        )

    async def _finalize_finished_workflow_close(
        self, completions: list[AwaitComplete], destination: str | None
    ) -> None:
        # pop_screen() removes stack entries synchronously but unmounts them
        # asynchronously. Keep the consumed Draft alive until every old screen
        # is detached, then make the discard atomic from the user's perspective.
        for completion in completions:
            await completion
        self._finished_workflow_close_pending = False
        self.discard_draft()
        self.running_workflow_screen = None
        if destination is not None:
            self.open_route(destination)

    def go_back(self) -> None:
        if self.screen.id == "home":
            return
        if self.screen.id == "settings-root":
            self._request_settings_leave(self.pop_screen)
            return
        self.pop_screen()

    def _settings_open(self) -> bool:
        return any(screen.id == "settings-root" for screen in self.screen_stack)

    def _request_settings_leave(self, continuation: Callable[[], object]) -> None:
        if not self.settings_dirty:
            continuation()
            return
        choices = [
            ("save", "Save and Leave", "Write the settings working copy"),
            ("discard", "Discard and Leave", "Restore saved settings"),
            ("stay", "Stay", "Continue editing"),
        ]
        self.push_screen(
            ChoiceModal("Unsaved Settings", choices), lambda choice: self._settings_leave(choice, continuation)
        )

    def _settings_leave(self, choice: str | None, continuation: Callable[[], object]) -> None:
        if choice == "save":
            if self.save_settings():
                continuation()
        elif choice == "discard":
            self.discard_settings()
            continuation()

    def save_settings(self) -> bool:
        try:
            self.settings_store.save(self.working_settings)
        except Exception as exc:
            self.notify(f"Settings were not saved: {exc}", severity="error", timeout=8)
            return False
        self.settings = AppSettings.from_dict(self.working_settings.to_dict())
        self.apply_visual_settings(self.settings.general)
        self.clear_resource_cache()
        self.notify("Settings saved.")
        return True

    def discard_settings(self) -> None:
        self.working_settings = AppSettings.from_dict(self.settings.to_dict())
        self.apply_visual_settings(self.working_settings.general)
        self.notify("Unsaved settings discarded.")

    def action_quit_requested(self) -> None:
        if self._settings_open() and self.settings_dirty:
            self._request_settings_leave(self._show_quit_confirmation)
            return
        self._show_quit_confirmation()

    def _show_quit_confirmation(self) -> None:
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
        if self._finished_workflow_is_current():
            self.close_finished_workflow("new-workflow")
            return
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
        if self._finished_workflow_is_current():
            self.close_finished_workflow()
            return
        if self._settings_open() and self.settings_dirty:
            self._request_settings_leave(self._pop_to_home)
            return
        self._pop_to_home()

    def _pop_to_home(self) -> list[AwaitComplete]:
        stack = list(self.screen_stack)
        home_index = next((index for index, screen in enumerate(stack) if screen.id == "home"), len(stack) - 1)
        return [self.pop_screen() for _ in range(len(stack) - home_index - 1)]

    def action_goto_jobs(self) -> None:
        self._goto_pending = False
        if self._finished_workflow_is_current():
            self.close_finished_workflow("jobs")
            return
        self.push_screen(JobsScreen())

    def _finished_workflow_is_current(self) -> bool:
        return isinstance(self.screen, RunningWorkflowScreen) and self.screen.result is not None

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
            paths = choose_files_native(self.last_picker_directory, json_only=json_only, language=self.ui_language)
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
        self._active_cancellation_token = CancellationToken()
        self.last_job_record = None
        self.workflow_output = []
        self.running_workflow_screen = RunningWorkflowScreen(self._draft)
        self.push_screen(self.running_workflow_screen)
        draft = WorkflowDraft.from_recipe(self._draft.to_recipe())
        self._run_workflow(draft, resumed_from, self._active_cancellation_token)

    @work(thread=True, group="workflow", exclusive=True)
    def _run_workflow(
        self, draft: WorkflowDraft, resumed_from: str | None, cancellation_token: CancellationToken
    ) -> None:
        try:
            result = self.job_controller.run(
                draft,
                self.settings,
                self.credentials,
                on_event=lambda event: self.call_from_thread(self._workflow_event, event),
                resumed_from=resumed_from,
                cancellation_token=cancellation_token,
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
        self._active_cancellation_token = None
        if self._draft is not None:
            # Keep the consumed Draft only while its result screen is mounted;
            # leaving the result removes the entire configuration stack.
            self._draft_baseline = self._draft.to_recipe()
        if self.running_workflow_screen is not None and self.running_workflow_screen.is_mounted:
            self.running_workflow_screen.finish(result)
        persistence_warning = self.job_controller.persistence_warning
        if persistence_warning:
            self.job_controller.persistence_warning = None
            self.notify(persistence_warning, severity="warning", timeout=10)
        self._operation_complete()

    def start_setup(self, request: SetupRequest) -> None:
        if self.operation_state != "idle":
            self.notify("Another operation is already active.", severity="warning")
            return
        self.operation_state = "starting"
        self.active_operation_kind = "setup"
        self._active_cancellation_token = CancellationToken()
        self.setup_stage = "Starting setup"
        self.setup_logs = []
        self.setup_running_screen = SetupRunningScreen(request)
        self.push_screen(self.setup_running_screen)
        self._run_setup(request, self._active_cancellation_token)

    @work(thread=True, group="setup", exclusive=True)
    def _run_setup(self, request: SetupRequest, cancellation_token: CancellationToken) -> None:
        try:
            result = self.setup_controller.run(
                request,
                on_event=lambda event: self.call_from_thread(self._setup_event, event),
                cancellation_token=cancellation_token,
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
        self._active_cancellation_token = None
        self.clear_resource_cache()
        if self.setup_running_screen is not None and self.setup_running_screen.is_mounted:
            self.setup_running_screen.finish(result)
        self._operation_complete()

    def cancel_active_operation(self) -> None:
        if self.operation_state == "idle" or self.operation_state == "cancelling":
            return
        token = self._active_cancellation_token
        if token is None:
            self.notify("The active operation has no cancellation token.", severity="error")
            return
        self.operation_state = "cancelling"
        if self.active_operation_kind == "workflow":
            self.job_controller.prepare_cancel()
        self._cancel_token(token)

    @work(thread=True, group="cancellation", exclusive=True)
    def _cancel_token(self, token: CancellationToken) -> None:
        token.cancel()

    def _operation_start_failed(self, message: str) -> None:
        operation_kind = self.active_operation_kind
        self.operation_state = "idle"
        self.active_operation_kind = None
        self._active_cancellation_token = None
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
