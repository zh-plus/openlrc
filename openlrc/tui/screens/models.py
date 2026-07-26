"""Model status plus typed, cancellable Whisper and Llama setup flows."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Static

from openlrc.application import LlamaSetupRequest, SetupAllRequest, SetupRequest, SetupResult, WhisperSetupRequest
from openlrc.llama_resources import HY_MT2_7B_PROFILE, QWEN35_9B_PROFILE, get_local_llm_profile
from openlrc.tui.i18n import tr
from openlrc.tui.modals import ChoiceModal, ConfirmModal, DetailModal, TextInputModal
from openlrc.tui.navigation import ActionItem, ActionList, action_group
from openlrc.tui.screens.base import OpenLRCScreen
from openlrc.tui.widgets import PageFooter, PageHeader


class ModelsScreen(OpenLRCScreen):
    def compose(self) -> ComposeResult:
        yield PageHeader("Models & Setup", "Local resources only")
        yield ActionList(
            *action_group(
                "Resources",
                ActionItem("status", "Model Status", "Installed binaries, models, and locations"),
            ),
            *action_group(
                "Setup",
                ActionItem("whisper", "Setup Whisper", "Build whisper.cpp and download ASR models"),
                ActionItem("llama", "Setup Llama", "Build llama.cpp and download one local model"),
                ActionItem("all", "Setup All", "Whisper then Llama · one guarded operation"),
            ),
            id="models-menu",
            classes="page-list",
        )
        yield PageFooter("[UP/DOWN] Select   [ENTER] Open   [ESC] Back")

    def on_mount(self) -> None:
        self.call_after_refresh(self.focus_default_action_list)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        if event.action_id == "status":
            self.app.push_screen(ModelStatusScreen())
        else:
            self.app.push_screen(SetupConfigScreen(event.action_id))


class ModelStatusScreen(OpenLRCScreen):
    def compose(self) -> ComposeResult:
        statuses = self.app.model_statuses()
        transcription_items = []
        translation_items = []
        for index, status in enumerate(statuses):
            item = ActionItem(
                f"resource:{index}",
                f"{tr('OK') if status.available else tr('MISSING')}  {status.name}",
                status.detail,
            )
            if status.role == "transcription" or "whisper" in status.name.lower():
                transcription_items.append(item)
            else:
                translation_items.append(item)
        items = [
            *action_group("Transcription", *transcription_items),
            *action_group("Translation", *translation_items),
            *action_group(
                "Actions",
                ActionItem("refresh", "Run checks again", "Refresh the shared resource snapshot"),
            ),
        ]
        yield PageHeader("Model Status", f"{sum(status.available for status in statuses)}/{len(statuses)} available")
        yield ActionList(*items, id="model-status", classes="page-list")
        yield PageFooter("[ENTER] Details/Refresh   [ESC] Back")

    def on_mount(self) -> None:
        self.call_after_refresh(self.focus_default_action_list)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        if event.action_id == "refresh":
            self.app.clear_resource_cache()
            self.refresh(recompose=True)
            return
        index = int(event.action_id.partition(":")[2])
        status = self.app.model_statuses()[index]
        self.app.push_screen(DetailModal(status.name, f"{status.detail}\n\n{status.hint}"))


class SetupConfigScreen(OpenLRCScreen):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.profile = QWEN35_9B_PROFILE
        if kind == "whisper":
            self.request: SetupRequest = WhisperSetupRequest()
        elif kind == "llama":
            self.request = LlamaSetupRequest()
        else:
            self.request = SetupAllRequest()

    def compose(self) -> ComposeResult:
        items: list[ActionItem]
        if isinstance(self.request, WhisperSetupRequest):
            items = self._whisper_items(self.request)
        elif isinstance(self.request, LlamaSetupRequest):
            items = self._llama_items(self.request)
        else:
            items = [
                ActionItem("skip-build", "Skip builds", _on_off(self.request.whisper.skip_build)),
                ActionItem("skip-models", "Skip model downloads", _on_off(self.request.whisper.skip_models)),
            ]
        grouped_items = [
            *action_group("Download & build options", *items),
            *action_group(
                "Actions",
                ActionItem(
                    "start",
                    "Review and Start",
                    "Setup may build tools or download large model files",
                    classes="action-primary",
                ),
            ),
        ]
        yield PageHeader(f"Setup {self.kind.title()}", "Explicit local changes")
        yield ActionList(*grouped_items, id="setup-config", classes="page-list")
        yield PageFooter("[ENTER] Edit/Start   [SPACE] Toggle   [ESC] Back")

    def _whisper_items(self, request: WhisperSetupRequest) -> list[ActionItem]:
        return [
            ActionItem("model", "Whisper model", request.model),
            ActionItem("vad-model", "VAD model", request.vad_model),
            ActionItem("model-dir", "Model directory", str(request.model_dir or "Default Application Support")),
            ActionItem("skip-build", "Skip build", _on_off(request.skip_build)),
            ActionItem("skip-models", "Skip model downloads", _on_off(request.skip_models)),
        ]

    def _llama_items(self, request: LlamaSetupRequest) -> list[ActionItem]:
        return [
            ActionItem("profile", "Local model profile", self.profile),
            ActionItem("repo", "Model repository", request.model_repo),
            ActionItem("file", "Model file", request.model_file),
            ActionItem("revision", "Revision", request.revision),
            ActionItem("url", "Override URL", request.model_url or "Automatic Hugging Face URL"),
            ActionItem("model-dir", "Model directory", str(request.model_dir or "Default Application Support")),
            ActionItem("skip-build", "Skip build", _on_off(request.skip_build)),
            ActionItem("skip-models", "Skip model download", _on_off(request.skip_models)),
            ActionItem("force", "Replace existing model", _on_off(request.force)),
        ]

    def on_mount(self) -> None:
        self.call_after_refresh(self.focus_default_action_list)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        action = event.action_id
        if action == "start":
            self.app.push_screen(
                ConfirmModal("Start setup?", self._summary(), confirm_label="Start Setup"), self._start_confirmed
            )
        elif action == "profile" and isinstance(self.request, LlamaSetupRequest):
            choices = [
                (QWEN35_9B_PROFILE, "Qwen 3.5 9B", "Classic translation and Context model"),
                (HY_MT2_7B_PROFILE, "Hy-MT2 7B", "Local subtitle translation"),
            ]
            self.app.push_screen(
                ChoiceModal("Local Model Profile", choices, current=self.profile), self._profile_selected
            )
        elif action in {"skip-build", "skip-models", "force"}:
            self._toggle(action)
        elif action in {"model", "vad-model", "model-dir", "repo", "file", "revision", "url"}:
            value = self._field_value(action)
            self.app.push_screen(TextInputModal(action.replace("-", " ").title(), value), self._text_setter(action))

    def _toggle(self, action: str) -> None:
        attribute = action.replace("-", "_")
        if isinstance(self.request, SetupAllRequest):
            value = not getattr(self.request.whisper, attribute)
            self.request = replace(
                self.request,
                whisper=replace(self.request.whisper, **{attribute: value}),
                llama=replace(self.request.llama, **{attribute: value}),
            )
        else:
            self.request = replace(self.request, **{attribute: not getattr(self.request, attribute)})
        self._recompose()

    def _field_value(self, action: str) -> str:
        assert not isinstance(self.request, SetupAllRequest)
        attribute = {"repo": "model_repo", "file": "model_file", "url": "model_url"}.get(
            action, action.replace("-", "_")
        )
        value = getattr(self.request, attribute)
        return "" if value is None else str(value)

    def _text_setter(self, action: str):
        def resolved(value: str | None) -> None:
            if value is None or isinstance(self.request, SetupAllRequest):
                return
            attribute = {"repo": "model_repo", "file": "model_file", "url": "model_url"}.get(
                action, action.replace("-", "_")
            )
            parsed = Path(value).expanduser() if attribute == "model_dir" and value.strip() else value.strip() or None
            self.request = replace(self.request, **{attribute: parsed})
            self._recompose()

        return resolved

    def _profile_selected(self, profile_name: str | None) -> None:
        if profile_name is None or not isinstance(self.request, LlamaSetupRequest):
            return
        profile = get_local_llm_profile(profile_name)
        if profile.model_repo is None or profile.model_file is None:
            self.app.notify("This profile requires an explicit repository and model file.", severity="warning")
            return
        self.profile = profile_name
        self.request = replace(self.request, model_repo=profile.model_repo, model_file=profile.model_file)
        self._recompose()

    def _summary(self) -> str:
        if isinstance(self.request, WhisperSetupRequest):
            return (
                f"Whisper model: {self.request.model}\nVAD model: {self.request.vad_model}\n"
                f"Build: {'skip' if self.request.skip_build else 'yes'}\n"
                f"Download: {'skip' if self.request.skip_models else 'yes'}\n"
                f"Target: {self.request.model_dir or 'default Application Support directory'}"
            )
        if isinstance(self.request, LlamaSetupRequest):
            return (
                f"Profile: {self.profile}\nModel: {self.request.model_repo}/{self.request.model_file}\n"
                f"Build: {'skip' if self.request.skip_build else 'yes'}\n"
                f"Download: {'skip' if self.request.skip_models else 'yes'}\n"
                f"Replace existing: {_on_off(self.request.force)}\n"
                f"Target: {self.request.model_dir or 'default Application Support directory'}"
            )
        return (
            f"Setup Whisper then Llama\nBuilds: {'skip' if self.request.whisper.skip_build else 'yes'}\n"
            f"Downloads: {'skip' if self.request.whisper.skip_models else 'yes'}"
        )

    def _start_confirmed(self, confirmed: bool | None) -> None:
        if confirmed:
            self.app.start_setup(self.request)

    def _recompose(self) -> None:
        self.refresh(recompose=True)
        self.call_after_refresh(self.focus_default_action_list)


class SetupRunningScreen(OpenLRCScreen):
    BINDINGS = [*OpenLRCScreen.BINDINGS, Binding("c", "cancel", "Cancel", show=False)]

    def __init__(self, request: SetupRequest) -> None:
        super().__init__()
        self.request = request
        self.result: SetupResult | None = None
        self.start_error: str | None = None

    def compose(self) -> ComposeResult:
        status = (
            "failed"
            if self.start_error
            else self.result.status.value
            if self.result is not None
            else self.app.operation_state
        )
        stage = self.start_error or self.app.setup_stage or "Starting"
        items = [
            ActionItem("logs", "Logs", f"{len(self.app.setup_logs)} lines"),
            ActionItem("paths", "Installed paths", f"{len(self.result.paths)} available" if self.result else "Pending"),
        ]
        if self.result is None and self.start_error is None:
            items.append(ActionItem("cancel", "Cancel", "Terminate the owned setup process"))
        else:
            items.append(ActionItem("models", "Return to Model Status", "Refresh resource checks"))
        yield PageHeader(f"Setup · {status.title()}", self.request.kind.value.title())
        with Vertical(id="setup-running-body"):
            yield Static(stage, id="setup-stage", markup=False)
            yield ActionList(*items, id="setup-running-actions", classes="page-list")
        yield PageFooter("[ENTER] Details   [C] Cancel   [ESC] Leave setup running")

    def on_mount(self) -> None:
        self.call_after_refresh(self.query_one(ActionList).focus)

    def update_event(self) -> None:
        if self.query("#setup-stage"):
            self.query_one("#setup-stage", Static).update(self.app.setup_stage or "Working")
        for item in self.query(ActionItem):
            if item.action_id == "logs":
                item.set_detail(f"{len(self.app.setup_logs)} lines")

    def finish(self, result: SetupResult) -> None:
        self.result = result
        self.refresh(recompose=True)

    def fail(self, message: str) -> None:
        self.start_error = message
        self.refresh(recompose=True)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        if event.action_id == "logs":
            self.app.push_screen(DetailModal("Setup Logs", "\n".join(self.app.setup_logs[-500:]) or "No logs."))
        elif event.action_id == "paths":
            paths = "\n".join(str(path) for path in self.result.paths) if self.result else "Setup is still running."
            self.app.push_screen(DetailModal("Installed Paths", paths))
        elif event.action_id == "cancel":
            self.action_cancel()
        elif event.action_id == "models":
            self.app.clear_resource_cache()
            self.app.push_screen(ModelStatusScreen())

    def action_cancel(self) -> None:
        if self.result is not None:
            return
        self.app.push_screen(
            ConfirmModal(
                "Cancel setup?", "The owned build/download process will be terminated.", confirm_label="Cancel Setup"
            ),
            lambda confirmed: self.app.cancel_active_operation() if confirmed else None,
        )


def _on_off(value: bool) -> str:
    return "On" if value else "Off"
