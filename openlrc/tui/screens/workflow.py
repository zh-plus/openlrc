"""Draft, preflight, execution, and result screens for four Workflow tasks."""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import RichLog, Static

from openlrc.application import PreflightReport, WorkflowDraft, normalize_input_paths, parse_brief_characters
from openlrc.config import ContextAssistance
from openlrc.llama_resources import HY_MT2_7B_PROFILE, HY_MT2_30B_A3B_PROFILE
from openlrc.tui.i18n import tr
from openlrc.tui.modals import ChoiceModal, ConfirmModal, DetailModal, MultilineTextModal, PathsModal, TextInputModal
from openlrc.tui.modals.file_picker import TerminalFilePickerModal
from openlrc.tui.navigation import ActionItem, ActionList, action_group
from openlrc.tui.screens.base import OpenLRCScreen
from openlrc.tui.widgets import PageFooter, PageHeader, WorkflowSteps
from openlrc.workflow import TranslationMode, WorkflowKind, WorkflowResult


def workflow_label(draft: WorkflowDraft) -> str:
    return {
        "transcribe-json": "Transcribe to JSON",
        "source-subtitle": "Transcribe to subtitle",
        "translate-existing": "Translate existing JSON",
        "full-run": "Full run",
    }.get(draft.task, draft.workflow.replace("-", " ").title())


def engine_label(draft: WorkflowDraft) -> str:
    if draft.translation_backend == "online":
        return "Online Provider · Classic"
    if draft.translation_backend == "local-qwen":
        return "Local Qwen · Classic"
    if draft.translation_backend == "local":
        return "Hy-MT2 · " + draft.mode.replace("-", " ").title()
    return "None"


class RuntimeOutputLog(RichLog):
    """Read-only auto-scrolling output that stays out of the keyboard focus chain."""

    can_focus = False


class WorkflowTypeScreen(OpenLRCScreen):
    def compose(self) -> ComposeResult:
        yield PageHeader("New Work", "Choose a backend capability")
        yield WorkflowSteps(1)
        yield ActionList(
            *action_group(
                "Work type",
                ActionItem(
                    "transcribe-json",
                    _accent_label("Transcribe to JSON", "Transcribe"),
                    "Transcribe media and keep raw OpenLRC JSON for later work",
                ),
                ActionItem(
                    "source-subtitle",
                    _accent_label("Transcribe to subtitle", "Transcribe"),
                    "Transcribe media into source-language LRC/SRT without translation",
                ),
                ActionItem(
                    "translate-existing",
                    _accent_label("Translate existing JSON", "Translate"),
                    "Translate an existing OpenLRC transcription JSON",
                ),
                ActionItem(
                    "full-run",
                    _accent_label("Full run", "Full run"),
                    "Transcribe media and create a translated subtitle in one run",
                ),
            ),
            id="workflow-types",
            classes="page-list",
        )
        yield PageFooter("[UP/DOWN] Select   [ENTER] Continue   [ESC] Back   [?] Help")

    def on_mount(self) -> None:
        self.call_after_refresh(self.focus_default_action_list)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        action = event.action_id
        if action == "transcribe-json":
            draft = WorkflowDraft.defaults(self.app.settings, WorkflowKind.TRANSCRIBE)
            draft.task = action
            self.app.begin_workflow(draft)
            self.app.push_screen(WorkflowOptionsScreen())
        elif action == "source-subtitle":
            draft = WorkflowDraft.defaults(self.app.settings, WorkflowKind.RUN)
            draft.task = action
            draft.translation_backend = "none"
            self.app.begin_workflow(draft)
            self.app.push_screen(WorkflowOptionsScreen())
        else:
            kind = WorkflowKind.TRANSLATE if action == "translate-existing" else WorkflowKind.RUN
            draft = WorkflowDraft.defaults(self.app.settings, kind)
            draft.task = action
            self.app.begin_workflow(draft)
            self.app.push_screen(TranslationEngineScreen())


class TranslationEngineScreen(OpenLRCScreen):
    def compose(self) -> ComposeResult:
        yield PageHeader("Translation Engine", workflow_label(self.app.draft))
        yield WorkflowSteps(2)
        yield ActionList(
            *action_group(
                "Translation engine",
                ActionItem("online", "Online Provider", "Classic · may incur provider fees"),
                ActionItem("local-qwen", "Local Qwen", "Classic · managed llama.cpp"),
                ActionItem("local", "Hy-MT2", "Fast / Normal / Normal Plus / Pro"),
            ),
            id="translation-engines",
            classes="page-list",
        )
        yield PageFooter("[UP/DOWN] Select   [ENTER] Continue   [ESC] Back")

    def on_mount(self) -> None:
        self.call_after_refresh(self.focus_default_action_list)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        draft = self.app.draft
        draft.translation_backend = event.action_id
        if event.action_id == "online":
            draft.mode = TranslationMode.STANDARD.value
            enabled = next((name for name, profile in self.app.settings.providers.items() if profile.enabled), "openai")
            draft.provider = enabled
            draft.primary_model = self.app.settings.providers[enabled].model
            self.app.push_screen(WorkflowOptionsScreen())
        elif event.action_id == "local-qwen":
            draft.mode = TranslationMode.STANDARD.value
            draft.qwen_model = self.app.settings.local_models.qwen_model
            self.app.push_screen(WorkflowOptionsScreen())
        else:
            draft.mode = TranslationMode.FAST.value
            self.app.push_screen(HyMT2ModeScreen())


class HyMT2ModeScreen(OpenLRCScreen):
    def compose(self) -> ComposeResult:
        yield PageHeader("Hy-MT2 Mode", workflow_label(self.app.draft))
        yield WorkflowSteps(2)
        yield ActionList(
            *action_group(
                "Hy-MT2 mode",
                ActionItem("fast", "Fast", "Translation only · no Context model"),
                ActionItem("normal", "Normal", "Context model when the Brief is incomplete"),
                ActionItem("normal-plus", "Normal Plus", "Optional semantic review"),
                ActionItem("pro", "Pro", "Timeline and high-risk review"),
            ),
            id="hymt2-modes",
            classes="page-list",
        )
        yield PageFooter("[UP/DOWN] Select   [ENTER] Continue   [ESC] Back")

    def on_mount(self) -> None:
        self.call_after_refresh(self.focus_default_action_list)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        draft = self.app.draft
        if (
            event.action_id == TranslationMode.NORMAL_PLUS.value
            and draft.context_assistance == ContextAssistance.OFF.value
            and draft.edit_rounds > 0
        ):
            self.app.notify(
                "Enable Context assistance Auto before using Normal Plus semantic review.", severity="error"
            )
            return
        draft.mode = event.action_id
        if event.action_id == TranslationMode.PRO.value and draft.context_assistance == ContextAssistance.OFF.value:
            draft.context_assistance = ContextAssistance.AUTO.value
            self.app.notify("Context assistance changed to Auto because Pro requires it.")
        if event.action_id == TranslationMode.FAST.value:
            draft.context_provider = ""
            draft.context_model = ""
        self.app.push_screen(WorkflowOptionsScreen())


class WorkflowOptionsScreen(OpenLRCScreen):
    def compose(self) -> ComposeResult:
        draft = self.app.draft
        basic_items = [
            ActionItem(
                "inputs",
                "Input Files",
                f"{len(draft.paths)} selected"
                + (f" · {len(self.app.input_issues)} issue(s)" if self.app.input_issues else ""),
            )
        ]
        advanced_items: list[ActionItem] = []
        if draft.workflow in {WorkflowKind.TRANSCRIBE.value, WorkflowKind.RUN.value}:
            basic_items.append(ActionItem("source-language", "Source language", draft.source_language or "Auto detect"))
            advanced_items.extend(
                [
                    ActionItem("whisper-model", "Whisper model", draft.whisper_model),
                    ActionItem("vad-model", "VAD model", draft.vad_model or "Disabled"),
                    ActionItem("skip-preprocess", "Skip preprocess", _on_off(draft.skip_preprocess)),
                    ActionItem("whisper-gpu", "Whisper GPU", _on_off(draft.whisper_use_gpu)),
                    ActionItem("whisper-flash-attn", "Whisper flash attention", _on_off(draft.whisper_flash_attn)),
                ]
            )
        if draft.translation_backend not in {"", "none"}:
            basic_items.extend(
                [
                    ActionItem("engine", "Translation engine", engine_label(draft)),
                    ActionItem("target-language", "Target language", draft.target_language),
                ]
            )
            if draft.translation_backend == "online":
                advanced_items.extend(
                    [
                        ActionItem("provider", "Provider", draft.provider),
                        ActionItem("primary-model", "Model", draft.primary_model or "Not set"),
                        ActionItem("retry-provider", "Retry provider", draft.retry_provider or "None"),
                        *(
                            [ActionItem("retry-model", "Retry model", draft.retry_model or "Not set")]
                            if draft.retry_provider
                            else []
                        ),
                        ActionItem("reviewer-provider", "Reviewer provider", draft.reviewer_provider or "None"),
                        *(
                            [ActionItem("reviewer-model", "Reviewer model", draft.reviewer_model or "Not set")]
                            if draft.reviewer_provider
                            else []
                        ),
                        ActionItem("fee-limit", "Fee limit", f"{draft.fee_limit:.2f}"),
                        ActionItem("consumer-thread", "Consumers", str(draft.consumer_thread)),
                    ]
                )
            elif draft.translation_backend == "local-qwen":
                advanced_items.append(ActionItem("qwen-model", "Qwen model", draft.qwen_model or "Default"))
            else:
                advanced_items.extend(
                    [
                        ActionItem("local-profile", "Hy-MT2 profile", draft.local_profile),
                        ActionItem("hymt2-model", "Hy-MT2 model", draft.hymt2_model or "Profile default"),
                    ]
                )
                if draft.mode != TranslationMode.FAST.value:
                    if draft.mode == TranslationMode.PRO.value:
                        assistance_detail = "Required by Pro"
                        assistance_disabled = True
                    elif draft.mode == TranslationMode.NORMAL_PLUS.value and draft.edit_rounds > 0:
                        assistance_detail = "Required by semantic review"
                        assistance_disabled = True
                    else:
                        assistance_detail = (
                            "Auto" if draft.context_assistance == ContextAssistance.AUTO.value else "Off"
                        )
                        assistance_disabled = False
                    advanced_items.append(
                        ActionItem(
                            "context-assistance", "Context assistance", assistance_detail, disabled=assistance_disabled
                        )
                    )
                requires_context_model = False
                if draft.context_assistance == ContextAssistance.AUTO.value:
                    try:
                        requires_context_model = draft.requires_context_model()
                    except ValueError:
                        # Invalid legacy/user text remains editable instead of
                        # crashing the asynchronous Textual recompose.
                        requires_context_model = True
                if requires_context_model:
                    advanced_items.extend(
                        [
                            ActionItem("context-provider", "Context provider", draft.context_provider or "Not set"),
                            ActionItem("context-model", "Context model", draft.context_model or "Not set"),
                        ]
                    )
                    if draft.context_provider != "local":
                        advanced_items.extend(
                            [
                                ActionItem(
                                    "context-base-url", "Context base URL", draft.context_base_url or "Provider default"
                                ),
                                ActionItem("context-fee-limit", "Context fee limit", f"{draft.context_fee_limit:.2f}"),
                            ]
                        )
                if draft.mode != TranslationMode.FAST.value:
                    off = draft.context_assistance == ContextAssistance.OFF.value
                    advanced_items.extend(
                        [
                            ActionItem(
                                "brief-summary",
                                "Brief summary",
                                _inline_text(draft.brief_summary) or ("Required" if off else "Not set"),
                            ),
                            ActionItem(
                                "brief-characters",
                                "Brief characters",
                                _brief_characters_detail(draft.brief_characters, off=off),
                            ),
                            ActionItem(
                                "brief-tone",
                                "Brief tone & style",
                                _inline_text(draft.brief_tone_style)
                                or ("No additional guidance" if off else "Not set"),
                            ),
                        ]
                    )
                if draft.mode in {TranslationMode.NORMAL_PLUS.value, TranslationMode.PRO.value}:
                    advanced_items.extend(
                        [
                            ActionItem("edit-rounds", "Semantic review rounds", str(draft.edit_rounds)),
                            ActionItem("enable-restore", "Enable restore", _on_off(draft.enable_restore)),
                        ]
                    )
            advanced_items.extend(
                [
                    ActionItem(
                        "glossary", "Glossary", Path(draft.glossary_path).name if draft.glossary_path else "Not set"
                    ),
                    ActionItem("glossary-strict", "Strict glossary conflicts", _on_off(draft.glossary_strict)),
                    ActionItem("force-glossary", "Force glossary terms", _on_off(draft.force_glossary)),
                    ActionItem("bilingual", "Bilingual subtitle", _on_off(draft.bilingual_subtitle)),
                    ActionItem("clear-checkpoint", "Clear checkpoints", _on_off(draft.clear_checkpoint)),
                ]
            )
        advanced_items.append(ActionItem("optimization", "Subtitle optimization", draft.subtitle_optimization.title()))
        if draft.workflow != WorkflowKind.TRANSLATE.value:
            advanced_items.append(ActionItem("clear-temp", "Clear temporary files", _on_off(draft.clear_temp)))
        continue_item = ActionItem(
            "continue", "Continue to Preflight", "Validate the real request and resource plan", classes="action-primary"
        )
        items = [
            *action_group("Basic settings", *basic_items),
            *action_group("Advanced settings", *advanced_items),
            *action_group("Actions", continue_item),
        ]
        yield PageHeader(
            workflow_label(draft), engine_label(draft) if draft.translation_backend not in {"", "none"} else ""
        )
        yield WorkflowSteps(2)
        yield ActionList(*items, id="workflow-options", classes="page-list")
        yield PageFooter("[UP/DOWN] Select   [ENTER] Edit/Open   [SPACE] Toggle   [ESC] Back")

    def on_mount(self) -> None:
        self.app.establish_draft_baseline()
        self.call_after_refresh(self.focus_default_action_list)

    def on_screen_resume(self) -> None:
        """Refresh values changed on a nested editor before restoring focus."""

        self.recompose_preserving_action()

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        action = event.action_id
        draft = self.app.draft
        if action == "inputs":
            self.app.push_screen(InputFilesScreen())
        elif action == "engine":
            self.app.push_screen(TranslationEngineScreen())
        elif action == "provider":
            choices = self._provider_choices()
            self.app.push_screen(
                ChoiceModal("Online Provider", choices, current=draft.provider), self._provider_selected
            )
        elif action in {"retry-provider", "reviewer-provider"}:
            attribute = action.replace("-", "_")
            choices = [("", "None", "Disable this fallback role"), *self._provider_choices()]
            self.app.push_screen(
                ChoiceModal(event.item.label_text, choices, current=str(getattr(draft, attribute))),
                self._provider_role_selected(attribute),
            )
        elif action == "context-provider":
            choices = [("local", "Local Qwen", "Managed local context model")]
            choices.extend(
                (name, name.replace("_", " ").title(), profile.model)
                for name, profile in self.app.settings.providers.items()
                if profile.enabled
            )
            self.app.push_screen(
                ChoiceModal("Context Provider", choices, current=draft.context_provider),
                self._context_provider_selected,
            )
        elif action == "context-assistance":
            choices = [
                (ContextAssistance.AUTO.value, "Auto", "Use a Context model when required"),
                (ContextAssistance.OFF.value, "Off", "Use only the complete manual Brief"),
            ]
            self.app.push_screen(
                ChoiceModal("Context Assistance", choices, current=draft.context_assistance),
                self._context_assistance_selected,
            )
        elif action == "local-profile":
            choices = [
                (HY_MT2_7B_PROFILE, "Hy-MT2 7B", "Recommended local model"),
                (HY_MT2_30B_A3B_PROFILE, "Hy-MT2 30B A3B", "Requires an explicit model path"),
            ]
            self.app.push_screen(
                ChoiceModal("Hy-MT2 Profile", choices, current=draft.local_profile), self._local_profile_selected
            )
        elif action == "optimization":
            choices = [
                ("aggressive", "Aggressive", "Historical subtitle optimization"),
                ("relaxed", "Relaxed", "Never merge, delete, truncate, reorder, or retime"),
            ]
            self.app.push_screen(
                ChoiceModal("Subtitle Optimization", choices, current=draft.subtitle_optimization),
                self._optimization_selected,
            )
        elif action in _MULTILINE_TEXT_FIELDS:
            attribute, title, help_text, placeholder = _MULTILINE_TEXT_FIELDS[action]
            self.app.push_screen(
                MultilineTextModal(title, str(getattr(draft, attribute)), help_text=help_text, placeholder=placeholder),
                self._setter(attribute),
            )
        elif action == "brief-characters":
            self.app.push_screen(
                MultilineTextModal(
                    "Brief Characters",
                    draft.brief_characters,
                    help_text="Enter one mapping per line. Empty lines are ignored.",
                    placeholder="John = 强尼\nMary = 玛丽",
                    validator=parse_brief_characters,
                ),
                self._setter("brief_characters"),
            )
        elif action in _TEXT_FIELDS:
            attribute, title = _TEXT_FIELDS[action]
            self.app.push_screen(TextInputModal(title, str(getattr(draft, attribute))), self._setter(attribute))
        elif action == "glossary":
            self.app.push_screen(
                TextInputModal("Glossary JSON path", draft.glossary_path, placeholder="/path/to/glossary.json"),
                self._glossary_selected,
            )
        elif action in _TOGGLE_FIELDS:
            attribute = _TOGGLE_FIELDS[action]
            setattr(draft, attribute, not bool(getattr(draft, attribute)))
            self._recompose()
        elif action == "continue":
            self.app.push_screen(ConfirmWorkflowScreen())

    def _setter(self, attribute: str):
        def resolved(value: str | None) -> None:
            if value is None:
                return
            parsed: str | float | int = value
            if attribute in {"fee_limit", "context_fee_limit"}:
                try:
                    parsed = float(value)
                except ValueError:
                    self.app.notify("This fee limit must be a number.", severity="error")
                    return
                if parsed <= 0:
                    self.app.notify("Fee limits must be greater than zero.", severity="error")
                    return
            elif attribute in {"consumer_thread", "edit_rounds"}:
                try:
                    parsed = int(value)
                except ValueError:
                    self.app.notify("This field must be an integer.", severity="error")
                    return
                if attribute == "consumer_thread" and parsed < 1:
                    self.app.notify("Consumers must be at least 1.", severity="error")
                    return
                if attribute == "edit_rounds" and not 0 <= parsed <= 3:
                    self.app.notify("Semantic rounds must be between 0 and 3.", severity="error")
                    return
                if (
                    attribute == "edit_rounds"
                    and parsed > 0
                    and self.app.draft.mode == TranslationMode.NORMAL_PLUS.value
                    and self.app.draft.context_assistance == ContextAssistance.OFF.value
                ):
                    self.app.notify(
                        "Enable Context assistance Auto before adding semantic review rounds.", severity="error"
                    )
                    return
            setattr(self.app.draft, attribute, parsed)
            self._recompose()

        return resolved

    def _provider_selected(self, provider: str | None) -> None:
        if provider is None:
            return
        self.app.draft.provider = provider
        self.app.draft.primary_model = self.app.settings.providers[provider].model
        self._recompose()

    def _provider_choices(self) -> list[tuple[str, str, str]]:
        return [
            (name, name.replace("_", " ").title(), profile.model)
            for name, profile in self.app.settings.providers.items()
            if profile.enabled
        ]

    def _provider_role_selected(self, attribute: str):
        def resolved(provider: str | None) -> None:
            if provider is None:
                return
            setattr(self.app.draft, attribute, provider)
            model_attribute = attribute.replace("provider", "model")
            setattr(self.app.draft, model_attribute, self.app.settings.providers[provider].model if provider else "")
            self._recompose()

        return resolved

    def _context_provider_selected(self, provider: str | None) -> None:
        if provider is None:
            return
        self.app.draft.context_provider = provider
        if provider == "local":
            self.app.draft.context_model = self.app.settings.local_models.qwen_model
        else:
            self.app.draft.context_model = self.app.settings.providers[provider].model
        self._recompose()

    def _context_assistance_selected(self, assistance: str | None) -> None:
        if assistance is None:
            return
        draft = self.app.draft
        if assistance == ContextAssistance.OFF.value:
            if draft.mode == TranslationMode.PRO.value:
                self.app.notify("Pro requires Context assistance Auto.", severity="error")
                return
            if draft.mode == TranslationMode.NORMAL_PLUS.value and draft.edit_rounds > 0:
                self.app.notify(
                    "Set semantic review rounds to 0 before turning Context assistance Off.", severity="error"
                )
                return
        draft.context_assistance = assistance
        self._recompose()

    def _local_profile_selected(self, profile: str | None) -> None:
        if profile is not None:
            self.app.draft.local_profile = profile
            self._recompose()

    def _optimization_selected(self, value: str | None) -> None:
        if value is not None:
            self.app.draft.subtitle_optimization = value
            self._recompose()

    def _glossary_selected(self, value: str | None) -> None:
        if value is None:
            return
        if value.strip():
            try:
                inspection = self.app.glossaries.inspect(
                    value,
                    source_language=self.app.draft.source_language,
                    target_language=self.app.draft.target_language,
                    strict=self.app.draft.glossary_strict,
                    force=self.app.draft.force_glossary,
                )
            except Exception as exc:
                self.app.notify(str(exc), severity="error", timeout=8)
                return
            self.app.notify(f"Glossary valid · {inspection.entry_count} entries")
        self.app.draft.glossary_path = value.strip()
        self._recompose()

    def _recompose(self) -> None:
        self.recompose_preserving_action()


class InputFilesScreen(OpenLRCScreen):
    BINDINGS = [
        *OpenLRCScreen.BINDINGS,
        Binding("delete,backspace", "remove", "Remove", show=False),
        Binding("alt+up", "move_up", "Move up", show=False),
        Binding("alt+down", "move_down", "Move down", show=False),
    ]

    def compose(self) -> ComposeResult:
        add_items = [
            ActionItem("native", "Add with macOS Picker", "System file chooser"),
            ActionItem("terminal", "Browse in Terminal", "Works over a terminal session"),
            ActionItem("paste", "Paste Paths", "One path per line"),
        ]
        issue_text = "\n".join(self.app.input_issues)
        path_items: list[ActionItem] = []
        for index, raw_path in enumerate(self.app.draft.paths):
            path = Path(raw_path)
            state = "Ready"
            if not path.is_file():
                state = "Missing · remove or replace"
            elif self.app.draft.workflow == WorkflowKind.TRANSLATE.value and path.suffix.lower() != ".json":
                state = "Invalid · JSON required"
            path_items.append(ActionItem(f"path:{index}", path.name or str(path), state))
        if not path_items:
            path_items.append(
                ActionItem("empty", "No input files selected", "Choose one of the add methods above", disabled=True)
            )
        items = [
            *action_group("Add input", *add_items),
            *action_group("Selected files", *path_items),
            *action_group(
                "Actions",
                ActionItem("continue", "Continue", f"{len(self.app.draft.paths)} selected", classes="action-primary"),
            ),
        ]
        yield PageHeader("Input Files", f"{len(self.app.draft.paths)} selected")
        yield WorkflowSteps(2)
        with Vertical(id="input-files-body"):
            yield ActionList(*items, id="input-files", classes="page-list")
            yield Static(issue_text, id="input-issues", classes="issue-summary", markup=False)
        yield PageFooter("[ENTER] Add/Open   [DEL] Remove   [ALT+UP/DOWN] Reorder   [ESC] Back")

    def on_mount(self) -> None:
        self.call_after_refresh(self.focus_default_action_list)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        if event.action_id == "native":
            self.app.open_native_picker(self._add_paths)
        elif event.action_id == "terminal":
            self.app.push_screen(TerminalFilePickerModal(self.app.last_picker_directory), self._add_paths)
        elif event.action_id == "paste":
            self.app.push_screen(PathsModal(), self._paths_pasted)
        elif event.action_id == "continue":
            self.app.pop_screen()
        elif event.action_id.startswith("path:"):
            index = int(event.action_id.partition(":")[2])
            self.app.push_screen(DetailModal("Input Path", self.app.draft.paths[index]))

    def _paths_pasted(self, value: str | None) -> None:
        if value is not None:
            self._add_paths(value.splitlines())

    def _add_paths(self, paths: list[str] | None) -> None:
        if not paths:
            return
        normalized, issues = normalize_input_paths([*self.app.draft.paths, *paths], self.app.draft.workflow)
        self.app.draft.paths = normalized
        self.app.input_issues = issues
        first = Path(paths[0]).expanduser()
        if first.parent.is_dir():
            self.app.last_picker_directory = first.parent.resolve()
        self._recompose()

    def action_remove(self) -> None:
        action = self.query_one(ActionList).selected_action()
        if action is None or not action.startswith("path:"):
            return
        index = int(action.partition(":")[2])
        self.app.draft.paths.pop(index)
        _, self.app.input_issues = normalize_input_paths(self.app.draft.paths, self.app.draft.workflow)
        self._recompose()

    def action_move_up(self) -> None:
        self._move(-1)

    def action_move_down(self) -> None:
        self._move(1)

    def _move(self, offset: int) -> None:
        action = self.query_one(ActionList).selected_action()
        if action is None or not action.startswith("path:"):
            return
        index = int(action.partition(":")[2])
        target = index + offset
        if 0 <= target < len(self.app.draft.paths):
            self.app.draft.paths[index], self.app.draft.paths[target] = (
                self.app.draft.paths[target],
                self.app.draft.paths[index],
            )
            self._recompose()

    def _recompose(self) -> None:
        self.recompose_preserving_action()


class ConfirmWorkflowScreen(OpenLRCScreen):
    BINDINGS = [
        *OpenLRCScreen.BINDINGS,
        Binding("pageup", "preview_up", "Preview up", show=False, priority=True),
        Binding("pagedown", "preview_down", "Preview down", show=False, priority=True),
    ]

    def __init__(self, *, resumed_from: str | None = None) -> None:
        super().__init__()
        self.resumed_from = resumed_from
        self.report = None

    def compose(self) -> ComposeResult:
        self.report = self.app.preflight(self.app.draft)
        actions = action_group(
            "Actions",
            ActionItem(
                "start",
                "Start",
                "Resolve blocked items first" if self.report.blocked else "Run this workflow",
                disabled=self.report.blocked,
                classes="action-primary",
            ),
            ActionItem("back", "Back", "Return to Configure"),
        )
        yield PageHeader("Preflight", f"{workflow_label(self.app.draft)} · {self.report.status}")
        yield WorkflowSteps(3)
        with Vertical(id="preflight-body"):
            yield Static(_preflight_text(self.report), id="preflight-readonly", markup=False)
            yield ActionList(*actions, id="preflight-list")
        yield PageFooter("[UP/DOWN] Select   [PGUP/PGDN] Preview   [ENTER] Activate   [ESC] Back")

    def on_mount(self) -> None:
        self.call_after_refresh(self.query_one(ActionList).focus)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        assert self.report is not None
        if event.action_id == "back":
            self.app.pop_screen()
        elif event.action_id == "start":
            if self.report.warnings:
                message = "\n".join(f"• {issue.message}" for issue in self.report.warnings)
                self.app.push_screen(
                    ConfirmModal("Start with warnings?", message, confirm_label="Start"), self._warning_confirmed
                )
            else:
                self.app.start_workflow(resumed_from=self.resumed_from)

    def action_preview_up(self) -> None:
        self.query_one("#preflight-readonly", Static).scroll_page_up(animate=False, force=True)

    def action_preview_down(self) -> None:
        self.query_one("#preflight-readonly", Static).scroll_page_down(animate=False, force=True)

    def _warning_confirmed(self, confirmed: bool | None) -> None:
        if confirmed:
            self.app.start_workflow(resumed_from=self.resumed_from)


class RunningWorkflowScreen(OpenLRCScreen):
    BINDINGS = [
        Binding("escape", "back", "Back", show=False, priority=True),
        Binding("question_mark", "help", "Help", show=False),
        Binding("l", "logs", "Logs", show=False),
        Binding("o", "outputs", "Outputs", show=False),
        Binding("c", "cancel", "Cancel", show=False),
    ]

    def __init__(self, draft: WorkflowDraft) -> None:
        super().__init__()
        self.result: WorkflowResult | None = None
        self.start_error: str | None = None
        self.workflow_title = workflow_label(draft)
        self.engine_title = engine_label(draft)

    def compose(self) -> ComposeResult:
        record = self.app.job_controller.active_record
        if record is None and self.app.last_job_record is not None:
            record = self.app.last_job_record
        progress = record.progress if record is not None else 0.0
        stage = record.current_stage if record is not None else "starting"
        status = (
            "failed"
            if self.start_error
            else self.result.status.value
            if self.result is not None
            else self.app.operation_state
        )
        bar = _progress_bar(progress)
        items = [
            ActionItem("files", "Files", str(len(record.items)) if record is not None else "Preparing"),
            ActionItem("logs", "Logs", f"{len(record.event_log)} lines" if record is not None else "0 lines"),
            ActionItem(
                "outputs", "Outputs", f"{len(record.outputs)} available" if record is not None else "0 available"
            ),
        ]
        if self.result is None and self.start_error is None:
            items.append(ActionItem("cancel", "Cancel", "Stop future work and clean up owned processes"))
        else:
            items.extend(
                [ActionItem("jobs", "Open Jobs & Recovery", "View persisted details"), ActionItem("new", "New Work")]
            )
        yield PageHeader(f"{status.replace('_', ' ').title()} · {self.workflow_title}", self.engine_title)
        with Vertical(id="running-body"):
            yield Static(
                self.start_error or f"{bar} {progress:5.1f}%  {stage or 'working'}", id="running-summary", markup=False
            )
            yield ActionList(*items, id="running-actions", classes="page-list")
            with Vertical(id="workflow-runtime-frame"):
                yield Static(tr("RUNTIME OUTPUT"), id="workflow-runtime-title", markup=False)
                yield RuntimeOutputLog(
                    max_lines=500, min_width=1, wrap=True, highlight=False, markup=False, id="workflow-runtime-output"
                )
        footer = (
            "[L] Logs   [O] Outputs   [ESC] Home"
            if self.result is not None
            else "[L] Logs   [O] Outputs   [C] Cancel   [ESC] Leave running"
        )
        yield PageFooter(footer)

    def on_mount(self) -> None:
        self.call_after_refresh(self.query_one(ActionList).focus)
        self.call_after_refresh(self._restore_runtime_output)
        self.set_interval(0.25, self.refresh_from_record)

    def action_back(self) -> None:
        if self.result is not None:
            self.app.close_finished_workflow()
        else:
            super().action_back()

    def append_runtime_output(self, line: str) -> None:
        if not self.query("#workflow-runtime-output"):
            return
        output = self.query_one("#workflow-runtime-output", RichLog)
        if len(self.app.workflow_output) == 1:
            output.clear()
        output.write(line)

    def _restore_runtime_output(self) -> None:
        if not self.query("#workflow-runtime-output"):
            return
        output = self.query_one("#workflow-runtime-output", RichLog)
        output.clear()
        if self.app.workflow_output:
            for line in self.app.workflow_output:
                output.write(line)
        else:
            output.write(tr("Waiting for runtime output..."))

    def refresh_from_record(self) -> None:
        if self.result is not None or self.start_error is not None:
            return
        record = self.app.job_controller.active_record or self.app.last_job_record
        progress = record.progress if record is not None else 0.0
        stage = record.current_stage if record is not None else "starting"
        if self.query("#running-summary"):
            self.query_one("#running-summary", Static).update(
                f"{_progress_bar(progress)} {progress:5.1f}%  {stage or 'working'}"
            )
        for item in self.query(ActionItem):
            if item.action_id == "files":
                item.set_detail(str(len(record.items)) if record is not None else "Preparing")
            elif item.action_id == "logs":
                item.set_detail(f"{len(record.event_log)} lines" if record is not None else "0 lines")
            elif item.action_id == "outputs":
                item.set_detail(f"{len(record.outputs)} available" if record is not None else "0 available")

    def finish(self, result: WorkflowResult) -> None:
        self.result = result
        self._recompose_terminal_view()

    def fail(self, message: str) -> None:
        self.start_error = message
        self._recompose_terminal_view()

    def _recompose_terminal_view(self) -> None:
        self.recompose_preserving_action(after_recompose=self._restore_runtime_output)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        if event.action_id == "logs":
            self.action_logs()
        elif event.action_id == "outputs":
            self.action_outputs()
        elif event.action_id == "files":
            record = self.app.job_controller.active_record or self.app.last_job_record
            content = (
                "\n".join(
                    f"{item.display_name} · {item.state} · {item.progress:.1f}%" for item in record.items.values()
                )
                if record
                else "No file state is available yet."
            )
            self.app.push_screen(DetailModal("Files", content))
        elif event.action_id == "cancel":
            self.action_cancel()
        elif event.action_id == "jobs":
            if self.result is not None:
                self.app.close_finished_workflow("jobs")
            else:
                self.app.open_route("jobs")
        elif event.action_id == "new":
            if self.result is not None:
                self.app.close_finished_workflow("new-workflow")
            else:
                self.app.open_route("new-workflow")

    def action_logs(self) -> None:
        record = self.app.job_controller.active_record or self.app.last_job_record
        content = "\n".join(record.event_log[-200:]) if record and record.event_log else "No log lines yet."
        self.app.push_screen(DetailModal("Workflow Logs", content))

    def action_outputs(self) -> None:
        record = self.app.job_controller.active_record or self.app.last_job_record
        content = "\n".join(record.outputs) if record and record.outputs else "No outputs are available yet."
        self.app.push_screen(DetailModal("Outputs", content))

    def action_cancel(self) -> None:
        if self.result is not None:
            return
        self.app.push_screen(
            ConfirmModal(
                "Cancel workflow?",
                "OpenLRC will stop future work, terminate owned processes, and retain compatible recovery data.",
                confirm_label="Cancel Workflow",
            ),
            self._cancel_confirmed,
        )

    def _cancel_confirmed(self, confirmed: bool | None) -> None:
        if confirmed:
            self.app.cancel_active_operation()


def _on_off(value: bool) -> str:
    return "On" if value else "Off"


def _accent_label(label: str, accent: str) -> Text:
    localized_label = tr(label)
    localized_accent = tr(accent)
    text = Text(localized_label)
    start = localized_label.index(localized_accent)
    text.stylize("#f0aa4b bold", start, start + len(localized_accent))
    return text


def _preflight_text(report: PreflightReport) -> Text:
    text = Text()
    text.append(f"{tr('REQUEST PREVIEW')}\n", style="#f0aa4b bold")
    if report.summary:
        for key, value in report.summary.items():
            text.append(f"{tr(key.replace('_', ' ').title()):<18}", style="#71808d")
            text.append(f"{value}\n", style="#d9e1e8")
    else:
        text.append(f"{tr('No resolved request summary is available.')}\n", style="#71808d")
    text.append(f"\n{tr('CHECKS')}\n", style="#f0aa4b bold")
    if not report.issues:
        text.append(f"{tr('READY'):<9}", style="#48c6dc bold")
        text.append(f"{tr('No blocking issues found')}\n", style="#d9e1e8")
    for issue in report.issues:
        color = "#dc6b72" if issue.severity == "blocked" else "#f0aa4b"
        text.append(f"{tr(issue.severity.title()).upper():<9}", style=f"{color} bold")
        text.append(f"{issue.message}\n", style="#d9e1e8")
    return text


def _progress_bar(percent: float, width: int = 20) -> str:
    complete = min(width, max(0, round(percent / 100 * width)))
    return "[" + "#" * complete + "-" * (width - complete) + "]"


def _inline_text(value: str) -> str:
    return " ".join(part.strip() for part in value.splitlines() if part.strip())


def _brief_characters_detail(value: str, *, off: bool) -> str:
    if not value.strip():
        return "None · explicit" if off else "Not set"
    try:
        count = len(parse_brief_characters(value))
    except ValueError:
        return "Invalid · edit to fix"
    return f"{count} mappings"


_TEXT_FIELDS = {
    "source-language": ("source_language", "Source language (blank = auto)"),
    "whisper-model": ("whisper_model", "Whisper model"),
    "vad-model": ("vad_model", "VAD model (blank = disabled)"),
    "target-language": ("target_language", "Target language"),
    "primary-model": ("primary_model", "Online model"),
    "retry-model": ("retry_model", "Retry model"),
    "reviewer-model": ("reviewer_model", "Reviewer model"),
    "fee-limit": ("fee_limit", "Fee limit"),
    "consumer-thread": ("consumer_thread", "Consumers"),
    "qwen-model": ("qwen_model", "Local Qwen model"),
    "hymt2-model": ("hymt2_model", "Hy-MT2 model or path"),
    "context-model": ("context_model", "Context model"),
    "context-base-url": ("context_base_url", "Context base URL"),
    "context-fee-limit": ("context_fee_limit", "Context fee limit"),
    "edit-rounds": ("edit_rounds", "Semantic review rounds"),
}

_MULTILINE_TEXT_FIELDS = {
    "brief-summary": (
        "brief_summary",
        "Translation Brief summary",
        "Describe the story, setting, and context. Multiple lines are supported.",
        "Story summary and important context",
    ),
    "brief-tone": (
        "brief_tone_style",
        "Translation Brief tone and style",
        "Add optional tone, register, or style guidance. Multiple lines are supported.",
        "Tone, register, and style guidance",
    ),
}

_TOGGLE_FIELDS = {
    "skip-preprocess": "skip_preprocess",
    "whisper-gpu": "whisper_use_gpu",
    "whisper-flash-attn": "whisper_flash_attn",
    "bilingual": "bilingual_subtitle",
    "glossary-strict": "glossary_strict",
    "force-glossary": "force_glossary",
    "clear-checkpoint": "clear_checkpoint",
    "enable-restore": "enable_restore",
    "clear-temp": "clear_temp",
}
