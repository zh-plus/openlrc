"""Working-copy settings, Keychain credentials, and contextual glossary inspection."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult

from openlrc.application import CredentialSource
from openlrc.tui.modals import ChoiceModal, ConfirmModal, DetailModal, TextInputModal
from openlrc.tui.navigation import ActionItem, ActionList, action_group
from openlrc.tui.screens.base import OpenLRCScreen
from openlrc.tui.widgets import PageFooter, PageHeader


class SettingsScreen(OpenLRCScreen):
    def __init__(self) -> None:
        super().__init__(id="settings-root")

    def compose(self) -> ComposeResult:
        settings = self.app.working_settings
        enabled = sum(profile.enabled for profile in settings.providers.values())
        local = settings.local_models
        configured = sum(
            bool(value)
            for value in (
                local.whisper_model,
                local.vad_model,
                local.qwen_model,
                local.hymt2_profile,
                local.hymt2_model,
                local.llama_server,
            )
        )
        glossary_detail = "Not set"
        if settings.workflow.default_glossary:
            try:
                inspection = self.app.glossaries.inspect(settings.workflow.default_glossary)
                glossary_detail = f"{inspection.entry_count} entries"
            except Exception:
                glossary_detail = "Invalid glossary"
        yield PageHeader("Settings", "Unsaved changes" if self.app.settings_dirty else "Local working copy")
        yield ActionList(
            *action_group(
                "Resources & providers",
                ActionItem("providers", "Providers", f"{enabled} enabled"),
                ActionItem("local-models", "Local Models", f"{configured} configured"),
            ),
            *action_group(
                "Workflow defaults",
                ActionItem("transcription", "Transcription Defaults"),
                ActionItem("translation", "Translation Defaults"),
                ActionItem("glossary", "Default Glossary", glossary_detail),
            ),
            *action_group(
                "Application",
                ActionItem(
                    "appearance",
                    "Appearance",
                    f"{_theme_label(settings.general.theme)} · {_language_label(settings.general.language)}",
                ),
            ),
            *action_group(
                "Actions",
                ActionItem(
                    "save",
                    "Save changes",
                    "Write settings.json",
                    disabled=not self.app.settings_dirty,
                    classes="action-primary",
                ),
                ActionItem("discard", "Discard changes", "Restore saved values", disabled=not self.app.settings_dirty),
            ),
            id="settings-menu",
            classes="page-list",
        )
        yield PageFooter("[ENTER] Open/Save   [ESC] Back   Unsaved changes prompt on exit")

    def on_mount(self) -> None:
        self.call_after_refresh(self.focus_default_action_list)

    def on_screen_resume(self) -> None:
        self.recompose_preserving_action()

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        pages = {
            "providers": ProvidersScreen,
            "local-models": LocalModelsSettingsScreen,
            "transcription": TranscriptionSettingsScreen,
            "translation": TranslationSettingsScreen,
            "glossary": DefaultGlossaryScreen,
            "appearance": AppearanceSettingsScreen,
        }
        if event.action_id in pages:
            self.app.push_screen(pages[event.action_id]())
        elif event.action_id == "save":
            self.app.save_settings()
            self.recompose_preserving_action("save")
        elif event.action_id == "discard":
            self.app.discard_settings()
            self.recompose_preserving_action("discard")


class ProvidersScreen(OpenLRCScreen):
    def compose(self) -> ComposeResult:
        items = []
        for name, profile in self.app.working_settings.providers.items():
            credential = self.app.credentials.resolve(name)
            state = "Enabled" if profile.enabled else "Disabled"
            key_state = credential.source.value.replace("_", " ").title()
            items.append(
                ActionItem(
                    name, name.replace("_", " ").title(), f"{state} · {profile.model or 'No model'} · {key_state}"
                )
            )
        yield PageHeader("Providers", "Settings values remain unsaved until Save")
        yield ActionList(*action_group("Providers", *items), id="providers-list", classes="page-list")
        yield PageFooter("[ENTER] Configure   [ESC] Back")

    def on_mount(self) -> None:
        self.call_after_refresh(self.focus_default_action_list)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        self.app.push_screen(ProviderDetailScreen(event.action_id))


class ProviderDetailScreen(OpenLRCScreen):
    def __init__(self, provider: str) -> None:
        super().__init__()
        self.provider = provider

    def compose(self) -> ComposeResult:
        profile = self.app.working_settings.providers[self.provider]
        credential = self.app.credentials.resolve(self.provider)
        credential_detail = {
            CredentialSource.KEYCHAIN: "Stored in System Keychain",
            CredentialSource.ENVIRONMENT: f"From {credential.environment_name}",
            CredentialSource.MISSING: "Missing",
        }[credential.source]
        yield PageHeader(self.provider.replace("_", " ").title(), "Provider working values")
        yield ActionList(
            *action_group(
                "Basic",
                ActionItem("enabled", "Enabled", _on_off(profile.enabled)),
                ActionItem("model", "Model", profile.model or "Not set"),
            ),
            *action_group(
                "Connection",
                ActionItem("base-url", "Base URL", profile.base_url or "Provider default"),
                ActionItem("proxy", "Proxy", profile.proxy or "Not set"),
            ),
            *action_group(
                "Credentials",
                ActionItem("credential", "Credential", credential_detail),
                ActionItem("remove-credential", "Remove Keychain credential", "Environment variables are not modified"),
            ),
            *action_group(
                "Actions",
                ActionItem(
                    "test",
                    "Test connection",
                    "Uses these unsaved values and may incur a minimal fee",
                    classes="action-primary",
                ),
            ),
            id="provider-detail",
            classes="page-list",
        )
        yield PageFooter("[ENTER] Edit/Test   [SPACE] Toggle   [ESC] Back")

    def on_mount(self) -> None:
        self.call_after_refresh(self.focus_default_action_list)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        profile = self.app.working_settings.providers[self.provider]
        if event.action_id == "enabled":
            profile.enabled = not profile.enabled
            self._recompose()
        elif event.action_id in {"model", "base-url", "proxy"}:
            attribute = event.action_id.replace("-", "_")
            self.app.push_screen(
                TextInputModal(event.item.label_text, getattr(profile, attribute)), self._profile_setter(attribute)
            )
        elif event.action_id == "credential":
            self.app.push_screen(
                TextInputModal("API key", placeholder="Stored in System Keychain", password=True), self._credential_set
            )
        elif event.action_id == "remove-credential":
            self.app.push_screen(
                ConfirmModal(
                    "Remove Keychain credential?",
                    "This does not remove environment variables or save provider settings.",
                    confirm_label="Remove Credential",
                ),
                self._credential_remove,
            )
        elif event.action_id == "test":
            self.app.test_provider(self.provider)

    def _profile_setter(self, attribute: str):
        def resolved(value: str | None) -> None:
            if value is not None:
                setattr(self.app.working_settings.providers[self.provider], attribute, value.strip())
                self._recompose()

        return resolved

    def _credential_set(self, value: str | None) -> None:
        if not value:
            return
        try:
            self.app.credentials.set(self.provider, value)
        except Exception as exc:
            self.app.notify(str(exc), severity="error", timeout=8)
        else:
            self.app.notify("Credential saved to System Keychain.")
            self._recompose()

    def _credential_remove(self, confirmed: bool | None) -> None:
        if not confirmed:
            return
        try:
            self.app.credentials.delete(self.provider)
        except Exception as exc:
            self.app.notify(str(exc), severity="error", timeout=8)
        else:
            self.app.notify("Keychain credential removed.")
            self._recompose()

    def _recompose(self) -> None:
        self.recompose_preserving_action()


class LocalModelsSettingsScreen(OpenLRCScreen):
    FIELDS = {
        "whisper-model": ("whisper_model", "Whisper model"),
        "vad-model": ("vad_model", "Whisper VAD model"),
        "whisper-cli": ("whisper_cli", "whisper-cli path"),
        "qwen-model": ("qwen_model", "Local Qwen model"),
        "hymt2-profile": ("hymt2_profile", "Hy-MT2 profile"),
        "hymt2-model": ("hymt2_model", "Hy-MT2 model override"),
        "llama-server": ("llama_server", "llama-server path"),
        "host": ("host", "Local server host"),
        "port": ("port", "Local server port"),
        "context-size": ("context_size", "Context size"),
        "gpu-layers": ("gpu_layers", "GPU layers"),
        "idle-timeout": ("idle_timeout", "Idle timeout"),
        "startup-timeout": ("startup_timeout", "Startup timeout"),
    }

    def compose(self) -> ComposeResult:
        local = self.app.working_settings.local_models

        def rows(actions: tuple[str, ...]) -> list[ActionItem]:
            return [
                ActionItem(action, self.FIELDS[action][1], str(getattr(local, self.FIELDS[action][0])) or "Not set")
                for action in actions
            ]

        items = [
            *action_group("Transcription", *rows(("whisper-model", "vad-model", "whisper-cli"))),
            *action_group("Translation", *rows(("qwen-model", "hymt2-profile", "hymt2-model", "llama-server"))),
            *action_group(
                "Runtime", *rows(("host", "port", "context-size", "gpu-layers", "idle-timeout", "startup-timeout"))
            ),
        ]
        yield PageHeader("Local Models", "Shared by Workflow and future GUI")
        yield ActionList(*items, id="local-settings", classes="page-list")
        yield PageFooter("[ENTER] Edit   [ESC] Back")

    def on_mount(self) -> None:
        self.call_after_refresh(self.focus_default_action_list)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        attribute, title = self.FIELDS[event.action_id]
        value = str(getattr(self.app.working_settings.local_models, attribute))
        self.app.push_screen(TextInputModal(title, value), self._setter(attribute))

    def _setter(self, attribute: str):
        def resolved(value: str | None) -> None:
            if value is None:
                return
            current = getattr(self.app.working_settings.local_models, attribute)
            try:
                parsed = int(value) if isinstance(current, int) else value.strip()
            except ValueError:
                self.app.notify("This field requires an integer.", severity="error")
                return
            setattr(self.app.working_settings.local_models, attribute, parsed)
            self.recompose_preserving_action()

        return resolved


class TranscriptionSettingsScreen(OpenLRCScreen):
    def compose(self) -> ComposeResult:
        settings = self.app.working_settings.transcription
        yield PageHeader("Transcription Defaults")
        yield ActionList(
            *action_group("Basic", ActionItem("source", "Source language", settings.source_language or "Auto detect")),
            *action_group("Processing", ActionItem("skip", "Skip preprocess", _on_off(settings.skip_preprocess))),
            *action_group(
                "Hardware",
                ActionItem("gpu", "Whisper GPU", _on_off(settings.use_gpu)),
                ActionItem("flash", "Whisper flash attention", _on_off(settings.flash_attn)),
            ),
            id="transcription-settings",
            classes="page-list",
        )
        yield PageFooter("[ENTER] Edit/Toggle   [ESC] Back")

    def on_mount(self) -> None:
        self.call_after_refresh(self.focus_default_action_list)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        settings = self.app.working_settings.transcription
        if event.action_id == "source":
            self.app.push_screen(TextInputModal("Source language", settings.source_language), self._source_set)
        elif event.action_id == "skip":
            settings.skip_preprocess = not settings.skip_preprocess
            self.recompose_preserving_action()
        elif event.action_id == "gpu":
            settings.use_gpu = not settings.use_gpu
            self.recompose_preserving_action()
        elif event.action_id == "flash":
            settings.flash_attn = not settings.flash_attn
            self.recompose_preserving_action()

    def _source_set(self, value: str | None) -> None:
        if value is not None:
            self.app.working_settings.transcription.source_language = value.strip()
            self.recompose_preserving_action()


class TranslationSettingsScreen(OpenLRCScreen):
    def compose(self) -> ComposeResult:
        settings = self.app.working_settings.workflow
        yield PageHeader("Translation Defaults")
        yield ActionList(
            *action_group(
                "Basic",
                ActionItem("target", "Target language", settings.target_language),
                ActionItem("bilingual", "Bilingual subtitle", _on_off(settings.bilingual_subtitle)),
                ActionItem("optimization", "Subtitle optimization", settings.subtitle_optimization.title()),
            ),
            *action_group(
                "Glossary",
                ActionItem("strict", "Strict glossary conflicts", _on_off(settings.glossary_strict)),
                ActionItem("force", "Force glossary terms", _on_off(settings.force_glossary)),
            ),
            *action_group(
                "Review & recovery",
                ActionItem("rounds", "Semantic edit rounds", str(settings.edit_rounds)),
                ActionItem("restore", "Enable restore", _on_off(settings.enable_restore)),
            ),
            *action_group(
                "Output & cleanup",
                ActionItem("clear-temp", "Clear temporary files", _on_off(settings.clear_temp)),
                ActionItem("clear-checkpoint", "Clear checkpoints", _on_off(settings.clear_checkpoint)),
            ),
            id="translation-settings",
            classes="page-list",
        )
        yield PageFooter("[ENTER] Edit/Toggle   [ESC] Back")

    def on_mount(self) -> None:
        self.call_after_refresh(self.focus_default_action_list)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        settings = self.app.working_settings.workflow
        if event.action_id == "target":
            self.app.push_screen(TextInputModal("Target language", settings.target_language), self._target_set)
        elif event.action_id == "optimization":
            choices = [
                ("aggressive", "Aggressive", "Historical optimization"),
                ("relaxed", "Relaxed", "No merge, delete, truncate, reorder, or retime"),
            ]
            self.app.push_screen(
                ChoiceModal("Subtitle Optimization", choices, current=settings.subtitle_optimization),
                self._optimization_set,
            )
        elif event.action_id == "rounds":
            self.app.push_screen(TextInputModal("Semantic edit rounds", str(settings.edit_rounds)), self._rounds_set)
        else:
            attributes = {
                "bilingual": "bilingual_subtitle",
                "clear-temp": "clear_temp",
                "clear-checkpoint": "clear_checkpoint",
                "strict": "glossary_strict",
                "force": "force_glossary",
                "restore": "enable_restore",
            }
            attribute = attributes[event.action_id]
            setattr(settings, attribute, not getattr(settings, attribute))
            self.recompose_preserving_action()

    def _target_set(self, value: str | None) -> None:
        if value is not None:
            self.app.working_settings.workflow.target_language = value.strip()
            self.recompose_preserving_action()

    def _optimization_set(self, value: str | None) -> None:
        if value is not None:
            self.app.working_settings.workflow.subtitle_optimization = value
            self.recompose_preserving_action()

    def _rounds_set(self, value: str | None) -> None:
        if value is None:
            return
        try:
            rounds = int(value)
            if not 0 <= rounds <= 3:
                raise ValueError
        except ValueError:
            self.app.notify("Edit rounds must be between 0 and 3.", severity="error")
            return
        self.app.working_settings.workflow.edit_rounds = rounds
        self.recompose_preserving_action()


class DefaultGlossaryScreen(OpenLRCScreen):
    def compose(self) -> ComposeResult:
        path = self.app.working_settings.workflow.default_glossary
        detail = Path(path).name if path else "Not set"
        inspect_detail = "Select and validate a glossary first"
        inspect_valid = False
        if path:
            try:
                inspection = self.app.glossaries.inspect(path)
                inspect_detail = f"{inspection.entry_count} entries · {len(inspection.conflicts)} conflicts"
                inspect_valid = True
            except Exception as exc:
                inspect_detail = f"Invalid · {exc}"
        yield PageHeader("Default Glossary", detail)
        yield ActionList(
            *action_group(
                "Glossary",
                ActionItem("select", "Select glossary JSON", path or "Not set"),
                ActionItem("inspect", "Inspect normalized entries", inspect_detail, disabled=not inspect_valid),
            ),
            *action_group(
                "Actions",
                ActionItem(
                    "clear",
                    "Clear default glossary",
                    "Workflow-specific glossaries are unchanged",
                    disabled=not bool(path),
                ),
            ),
            id="default-glossary",
            classes="page-list",
        )
        yield PageFooter("[ENTER] Select/Inspect   [ESC] Back")

    def on_mount(self) -> None:
        self.call_after_refresh(self.focus_default_action_list)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        if event.action_id == "select":
            self.app.push_screen(
                TextInputModal(
                    "Default glossary JSON path",
                    self.app.working_settings.workflow.default_glossary,
                    placeholder="/path/to/glossary.json",
                ),
                self._selected,
            )
        elif event.action_id == "inspect":
            self._inspect()
        elif event.action_id == "clear":
            self.app.working_settings.workflow.default_glossary = ""
            self.recompose_preserving_action()

    def _selected(self, value: str | None) -> None:
        if value is None:
            return
        try:
            inspection = self.app.glossaries.inspect(value)
        except Exception as exc:
            self.app.notify(str(exc), severity="error", timeout=8)
            return
        self.app.working_settings.workflow.default_glossary = str(inspection.path)
        self.app.notify(f"Glossary valid · {inspection.entry_count} entries")
        self.recompose_preserving_action()

    def _inspect(self) -> None:
        try:
            inspection = self.app.glossaries.inspect(self.app.working_settings.workflow.default_glossary)
        except Exception as exc:
            self.app.notify(str(exc), severity="error", timeout=8)
            return
        lines = [
            f"Name: {inspection.catalog.name or 'Unnamed'}",
            f"Source language: {inspection.catalog.source_language or 'Any'}",
            f"Target language: {inspection.catalog.target_language or 'Any'}",
            f"Fingerprint: {inspection.state.fingerprint}",
            "",
        ]
        lines.extend(
            f"{entry.source} -> {entry.target} · required={entry.required} · aliases={', '.join(entry.aliases) or 'none'}"
            for entry in inspection.state.merged_entries
        )
        self.app.push_screen(DetailModal("Normalized Glossary", "\n".join(lines)))


class AppearanceSettingsScreen(OpenLRCScreen):
    def compose(self) -> ComposeResult:
        general = self.app.working_settings.general
        yield PageHeader("Appearance", "Visual settings never alter Workflow behavior")
        yield ActionList(
            *action_group(
                "Appearance",
                ActionItem("theme", "Theme", _theme_label(general.theme)),
                ActionItem("language", "Language", _language_label(general.language)),
                ActionItem("animation", "Logo animation", _on_off(general.logo_animation)),
                ActionItem("motion", "Reduced motion", _on_off(general.reduce_motion)),
            ),
            id="appearance-settings",
            classes="page-list",
        )
        yield PageFooter("[ENTER] Select/Toggle   [ESC] Back")

    def on_mount(self) -> None:
        self.call_after_refresh(self.query_one(ActionList).focus)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        general = self.app.working_settings.general
        if event.action_id == "theme":
            choices = [
                ("openlrc-dark", "OpenLRC Dark", "Cyan and amber"),
                ("textual-dark", "Textual Dark", "Framework dark palette"),
            ]
            self.app.push_screen(ChoiceModal("Theme", choices, current=general.theme), self._theme_set)
        elif event.action_id == "language":
            choices = [("en", "English", "English"), ("zh-cn", "Simplified Chinese", "简体中文")]
            self.app.push_screen(ChoiceModal("Language", choices, current=general.language), self._language_set)
        elif event.action_id == "animation":
            general.logo_animation = not general.logo_animation
            self.app.apply_visual_settings(general)
            self._update_detail("animation", _on_off(general.logo_animation))
        elif event.action_id == "motion":
            general.reduce_motion = not general.reduce_motion
            self.app.apply_visual_settings(general)
            self._update_detail("motion", _on_off(general.reduce_motion))

    def _theme_set(self, value: str | None) -> None:
        if value is not None:
            self.app.working_settings.general.theme = value
            self.app.apply_visual_settings(self.app.working_settings.general)
            self._update_detail("theme", _theme_label(value))
        self.call_after_refresh(self.focus_action, "theme")

    def _language_set(self, value: str | None) -> None:
        if value is not None:
            self.app.working_settings.general.language = value
            self.app.apply_visual_settings(self.app.working_settings.general)
            self.recompose_preserving_action("language")
        else:
            self.call_after_refresh(self.focus_action, "language")

    def _update_detail(self, action_id: str, detail: str) -> None:
        item = next((item for item in self.query(ActionItem) if item.action_id == action_id), None)
        if item is not None:
            item.set_detail(detail)
        self.call_after_refresh(self.focus_action, action_id)


def _on_off(value: bool) -> str:
    return "On" if value else "Off"


def _theme_label(value: str) -> str:
    return {"openlrc-dark": "OpenLRC Dark", "textual-dark": "Textual Dark"}.get(value, value)


def _language_label(value: str) -> str:
    return "简体中文" if value == "zh-cn" else "English"
