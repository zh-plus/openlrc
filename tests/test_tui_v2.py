from __future__ import annotations

import asyncio
import threading
from collections import Counter
from pathlib import Path

import pytest
from rich.cells import cell_len
from rich.text import Text
from textual.widgets import Button, Input, RichLog, Static, TextArea

from openlrc.application import (
    CredentialSource,
    JobController,
    JobRecord,
    JobRecordStatus,
    JobRepository,
    PreflightReport,
    ResolvedCredential,
    ResourceStatus,
    SettingsStore,
    SetupController,
    SetupStatus,
    WhisperSetupRequest,
    WorkflowDraft,
    preflight,
)
from openlrc.config import ContextAssistance
from openlrc.logger import handler as openlrc_terminal_handler
from openlrc.logger import logger as openlrc_logger
from openlrc.tui import OpenLRCTUI
from openlrc.tui.i18n import tr
from openlrc.tui.modals import ChoiceModal, ConfirmModal, MultilineTextModal, TextInputModal
from openlrc.tui.navigation import ActionGroupHeading, ActionGroupSpacer, ActionItem, ActionList
from openlrc.tui.screens.jobs import JobDetailScreen, JobsScreen
from openlrc.tui.screens.models import SetupRunningScreen
from openlrc.tui.screens.settings import ProviderDetailScreen
from openlrc.tui.screens.workflow import (
    ConfirmWorkflowScreen,
    HyMT2ModeScreen,
    RunningWorkflowScreen,
    WorkflowOptionsScreen,
    WorkflowTypeScreen,
)
from openlrc.tui.widgets.home_card import HomeCard
from openlrc.tui.widgets.logo import WIDE_LOGO, LogoWidget, render_logo
from openlrc.tui.widgets.status import WorkflowSteps
from openlrc.workflow import CancellationToken, WorkflowResult, WorkflowStatus


class StaticResources:
    def doctor(self):
        return [
            ResourceStatus("ffmpeg", True, "/opt/homebrew/bin/ffmpeg"),
            ResourceStatus("whisper-cli", True, "/app/whisper-cli"),
            ResourceStatus("Whisper model", True, "/models/ggml-base.bin"),
        ]

    def models(self, _settings):
        return [
            ResourceStatus("whisper-cli", True, "/app/whisper-cli", role="transcription"),
            ResourceStatus("Whisper default model", True, "/models/base.bin", role="transcription"),
            ResourceStatus("Whisper VAD model", True, "/models/vad.bin", role="transcription"),
            ResourceStatus("Local Qwen GGUF", True, "/models/qwen.gguf", role="translation"),
            ResourceStatus("Local Hy-MT2 7B Q6_K GGUF", False, "Missing", role="translation"),
            ResourceStatus("llama-server", True, "/app/llama-server", role="translation"),
        ]


class StaticCredentials:
    last_error = None

    def resolve(self, _provider: str) -> ResolvedCredential:
        return ResolvedCredential("test-key", CredentialSource.KEYCHAIN)

    def set(self, _provider: str, _value: str) -> None:
        return None

    def delete(self, _provider: str) -> None:
        return None


def _app(tmp_path: Path, *, ready_preflight: bool = False, fixed_logo_frame: int = 5) -> OpenLRCTUI:
    repository = JobRepository(tmp_path / "jobs.json")
    controller = JobController(repository, executor=SuccessfulExecutor() if ready_preflight else None)

    def report(*_args):
        return PreflightReport(summary={"pipeline": "Transcribe", "inputs": "1", "strategy": "transcribe-only"})

    return OpenLRCTUI(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        credentials=StaticCredentials(),
        job_controller=controller,
        resources=StaticResources(),
        preflight_service=report if ready_preflight else preflight,
        fixed_logo_frame=fixed_logo_frame,
    )


class SuccessfulExecutor:
    def execute(self, _request, context):
        context.workflow_started()
        context.workflow_completed(WorkflowStatus.SUCCEEDED, 0.01)
        context.close()
        return WorkflowResult(
            job_id=context.job_id, workflow=context.workflow, status=WorkflowStatus.SUCCEEDED, elapsed_seconds=0.01
        )


def test_home_is_single_screen_and_disabled_edit_is_skipped(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            cards = list(app.screen.query(HomeCard))
            assert [card.action_id for card in cards] == ["new-workflow", "edit", "jobs", "models", "settings"]
            assert cards[0].label_text == "NEW WORK"
            assert cards[1].disabled
            assert app.screen.query_one("#home-cards", ActionList).index == 0

            await pilot.press("up")
            assert app.screen.query_one("#home-cards", ActionList).selected_action() == "settings"
            await pilot.press("down")
            assert app.screen.query_one("#home-cards", ActionList).selected_action() == "new-workflow"
            await pilot.press("down")
            assert app.screen.query_one("#home-cards", ActionList).index == 2
            await pilot.press("up")
            await pilot.click(".home-card-disabled")
            assert app.screen.id == "home"

            await pilot.press("down", "enter")
            await pilot.pause()
            assert isinstance(app.screen, JobsScreen)

    asyncio.run(scenario())


@pytest.mark.parametrize("size", [(80, 24), (100, 30), (160, 50)])
def test_home_cards_are_uniform_and_vertically_centered(tmp_path: Path, size: tuple[int, int]) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            cards = list(app.screen.query(HomeCard))
            action_list = app.screen.query_one("#home-cards", ActionList)
            brand = app.screen.query_one("#home-brand")
            logo = app.screen.query_one(LogoWidget)
            brand_lines = list(app.screen.query(".brand-line"))

            top_gap = cards[0].region.y - action_list.content_region.y
            bottom_gap = action_list.content_region.bottom - cards[-1].region.bottom
            assert abs(top_gap - bottom_gap) <= 1
            brand_center = brand.content_region.x + brand.content_region.width / 2
            assert logo.content_region.x + logo.content_region.width / 2 == brand_center
            assert all(line.region.x + line.region.width / 2 == brand_center for line in brand_lines)
            assert [card.region.height for card in cards] == [4] * 5
            for card in cards:
                label = card.query_one(".action-label", Static)
                assert label.region.y - card.region.y == 1
                assert label.region.bottom < card.region.bottom

    asyncio.run(scenario())


def test_active_home_card_mouse_click_uses_same_route(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.click(".home-card")
            await pilot.pause()
            assert isinstance(app.screen, WorkflowTypeScreen)

    asyncio.run(scenario())


def test_home_cards_show_only_high_priority_actionable_status(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path / "jobs.json")
    repository.save(
        [
            JobRecord(
                job_id="review-job",
                workflow="run",
                name="episode.mp4",
                status=JobRecordStatus.SUCCEEDED_WITH_WARNINGS,
                input_paths=["episode.mp4"],
                recipe={},
                reviews=[{"incomplete": True}],
            )
        ]
    )

    class MissingResources(StaticResources):
        def models(self, _settings):
            return [ResourceStatus("whisper-cli", False, "Missing")]

    credentials = StaticCredentials()
    credentials.last_error = "Keychain unavailable"
    app = OpenLRCTUI(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        credentials=credentials,
        job_controller=JobController(repository),
        resources=MissingResources(),
        fixed_logo_frame=0,
    )

    async def scenario() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            cards = {card.action_id: card for card in app.screen.query(HomeCard)}
            assert cards["new-workflow"].detail_text == ""
            assert cards["jobs"].detail_text == "1 needs review"
            assert cards["models"].detail_text == "Setup required"
            assert cards["settings"].detail_text == "Credential error"

    asyncio.run(scenario())


def test_logo_sweep_preserves_glyphs_wraps_and_degrades_without_color() -> None:
    width = max(map(len, WIDE_LOGO))
    first = render_logo("wide", frame=0)
    wrapped = render_logo("wide", frame=width)
    static = render_logo("wide", frame=27, color=False)

    assert first.plain == wrapped.plain == static.plain
    assert first.spans == wrapped.spans
    assert static.spans != first.spans
    for layout in ("wide", "compact"):
        current_frame = render_logo(layout, frame=14)
        next_frame = render_logo(layout, frame=15)
        assert current_frame.plain == next_frame.plain
        assert len({cell_len(line) for line in current_frame.plain.split("\n")}) == 1

    populated_columns = Counter(
        column for line in WIDE_LOGO for column, character in enumerate(line) if not character.isspace()
    )
    column = max(populated_columns, key=populated_columns.get)
    column_frame = render_logo("wide", frame=column)
    amber_spans = [span for span in column_frame.spans if getattr(span.style.color, "name", None) == "#f0aa4b"]
    assert len(amber_spans) == populated_columns[column]
    assert len(amber_spans) > 1
    fixed_width = cell_len(column_frame.plain.split("\n")[0])
    assert {span.start % (fixed_width + 1) for span in amber_spans} == {column}

    blue_static = render_logo("wide", frame=column, sweep=False)
    assert blue_static.plain == column_frame.plain
    assert all(getattr(span.style.color, "name", None) != "#f0aa4b" for span in blue_static.spans)
    assert any(getattr(span.style.color, "name", None) == "#48c6dc" for span in blue_static.spans)


def test_appearance_theme_language_focus_persistence_and_discard(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = SettingsStore(tmp_path / "settings.json")
        app = _app(tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("end", "enter", "end", "enter")
            await pilot.pause()
            appearance = app.screen.query_one(ActionList)
            assert appearance.selected_action() == "theme"

            openlrc_svg = app.export_screenshot()
            await pilot.press("enter", "down", "enter")
            await pilot.pause()
            assert app.theme == "textual-dark"
            assert app.export_screenshot() != openlrc_svg
            assert app.focused is appearance
            assert appearance.selected_action() == "theme"

            await pilot.press("down")
            assert appearance.selected_action() == "language"
            await pilot.press("enter", "down", "enter")
            await pilot.pause()
            appearance = app.screen.query_one(ActionList)
            assert app.ui_language == "zh-cn"
            assert isinstance(app.focused, ActionList)
            assert appearance.selected_action() == "language"
            assert [item.label_text for item in app.screen.query(ActionItem)] == [
                "主题",
                "语言",
                "Logo 动效",
                "减少动效",
                "仅使用 ASCII 状态符号",
            ]
            await pilot.press("down")
            appearance = app.screen.query_one(ActionList)
            assert appearance.selected_action() == "animation"
            await pilot.press("up")

            await pilot.press("escape", "end", "up", "enter")
            await pilot.pause()
            assert store.load().general.theme == "textual-dark"
            assert store.load().general.language == "zh-cn"
            await pilot.press("escape")
            await pilot.pause()
            cards = list(app.screen.query(HomeCard))
            assert cards[0].label_text == "新建任务"
            assert app.screen.query_one("#doctor-status", Button).label == "本地就绪"

        restarted = _app(tmp_path)
        async with restarted.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert restarted.theme == "textual-dark"
            assert restarted.ui_language == "zh-cn"
            await pilot.press("end", "enter", "end", "enter")
            await pilot.pause()
            appearance = restarted.screen.query_one(ActionList)
            await pilot.press("enter", "enter")
            await pilot.pause()
            assert restarted.theme == "openlrc-dark"
            await pilot.press("down", "enter", "enter")
            await pilot.pause()
            assert restarted.ui_language == "en"
            await pilot.press("escape", "end", "enter")
            await pilot.pause()
            assert restarted.theme == "textual-dark"
            assert restarted.ui_language == "zh-cn"

    asyncio.run(scenario())


def test_brief_character_validation_error_is_localized() -> None:
    assert (
        tr("Brief character line 2 must use 'Source Name = Target Name'.", language="zh-cn")
        == "人物映射第 2 行必须使用“源名称 = 目标名称”。"
    )


def test_disabling_logo_animation_renders_a_fresh_static_blue_logo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)

    async def scenario() -> None:
        app = _app(tmp_path, fixed_logo_frame=None)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.2)
            logo = app.screen.query_one(LogoWidget)
            logo.refresh_visual_settings()
            assert logo._showing_sweep

            await pilot.press("end", "enter", "end", "enter", "down", "down", "enter")
            await pilot.pause()
            assert app.working_settings.general.logo_animation is False
            assert logo._showing_sweep is False
            content = logo.content
            assert isinstance(content, Text)
            assert all(getattr(span.style.color, "name", None) != "#f0aa4b" for span in content.spans)
            assert any(getattr(span.style.color, "name", None) == "#48c6dc" for span in content.spans)

            await pilot.press("enter")
            assert app.working_settings.general.logo_animation is True
            await pilot.press("escape", "escape")
            await pilot.pause(0.2)
            logo = app.screen.query_one(LogoWidget)
            logo.refresh_visual_settings()
            assert logo._showing_sweep

    asyncio.run(scenario())


def test_compact_action_rows_reflow_detail_and_keep_one_line_group_spacing(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("end", "enter")
            await pilot.pause()

            rows = {item.action_id: item for item in app.screen.query(ActionItem)}
            headings = list(app.screen.query(ActionGroupHeading))
            group_ends = [item for item in app.screen.query(ActionItem) if item.has_class("action-group-last")]

            assert rows["transcription"].region.height == 1
            assert rows["providers"].region.height == 2
            assert rows["local-models"].region.height == 3
            assert all(
                heading.region.y - previous_group.region.bottom == 1
                for previous_group, heading in zip(group_ends, headings[1:], strict=False)
            )

            rows["providers"].set_detail("")
            await pilot.pause()
            assert not rows["providers"].has_class("action-has-detail")
            assert rows["providers"].has_class("action-no-detail")
            assert rows["providers"].region.height == 1
            assert rows["providers"].query_one(".action-detail", Static).styles.display == "none"

            rows["providers"].set_detail("2 enabled")
            await pilot.pause()
            assert rows["providers"].has_class("action-has-detail")
            assert not rows["providers"].has_class("action-no-detail")
            assert rows["providers"].region.height == 2
            assert rows["providers"].query_one(".action-detail", Static).styles.display == "block"

    asyncio.run(scenario())


def test_work_type_group_spacers_fit_compact_terminal_and_stay_inert(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            action_list = app.screen.query_one("#workflow-types", ActionList)
            children = list(action_list.children)
            assert [type(child) for child in children] == [
                ActionGroupHeading,
                ActionGroupSpacer,
                ActionItem,
                ActionGroupSpacer,
                ActionItem,
                ActionGroupSpacer,
                ActionItem,
                ActionGroupSpacer,
                ActionItem,
            ]
            spacers = list(app.screen.query(ActionGroupSpacer))
            assert all(spacer.disabled and not spacer.can_focus for spacer in spacers)
            assert all(spacer.region.height == 1 for spacer in spacers)
            assert action_list.max_scroll_y == 0
            assert children[-1].region.bottom == action_list.content_region.bottom
            assert "ActionGroupSpacer" not in app.export_screenshot()

            await pilot.press("end", "down")
            assert action_list.selected_action() == "transcribe-json"

    asyncio.run(scenario())


def test_settings_save_is_neutral_when_clean_and_accented_only_when_dirty(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)

        def assert_clean_save() -> None:
            rows = {item.action_id: item for item in app.screen.query(ActionItem)}
            save = rows["save"]
            discard = rows["discard"]
            save_label = save.query_one(".action-label", Static)
            save_detail = save.query_one(".action-detail", Static)
            discard_label = discard.query_one(".action-label", Static)
            accent_reference = app.screen.query_one(".action-group-title", Static)

            assert save.disabled
            assert discard.disabled
            assert save_label.styles.color == discard_label.styles.color
            assert save_detail.styles.color == discard_label.styles.color
            assert save_label.styles.color != accent_reference.styles.color
            assert save.styles.border_left == discard.styles.border_left
            assert save.styles.border_right == discard.styles.border_right
            assert save.styles.background == discard.styles.background

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("end", "enter")
            await pilot.pause()
            assert_clean_save()

            await pilot.press("end", "enter", "down", "down", "enter", "escape")
            await pilot.pause()
            rows = {item.action_id: item for item in app.screen.query(ActionItem)}
            save = rows["save"]
            assert not save.disabled
            assert (
                save.query_one(".action-label", Static).styles.color
                == app.screen.query_one(".action-group-title", Static).styles.color
            )

            await pilot.press("down", "enter")
            await pilot.pause()
            assert_clean_save()

            await pilot.press("end", "enter", "down", "down", "enter", "escape", "down", "down", "enter")
            await pilot.pause()
            assert_clean_save()

    asyncio.run(scenario())


def test_settings_leave_guard_blocks_failed_save_and_wraps_home_and_quit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("end", "enter")
            await pilot.pause()
            app.working_settings.general.reduce_motion = not app.settings.general.reduce_motion
            working_copy = app.working_settings.to_dict()
            original_save = app.settings_store.save

            def fail_save(_settings) -> None:
                raise OSError("disk full")

            monkeypatch.setattr(app.settings_store, "save", fail_save)
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, ChoiceModal)
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen.id == "settings-root"
            assert app.settings_dirty
            assert app.working_settings.to_dict() == working_copy

            monkeypatch.setattr(app.settings_store, "save", original_save)
            await pilot.press("g", "h")
            await pilot.pause()
            assert isinstance(app.screen, ChoiceModal)
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen.id == "home"
            assert not app.settings_dirty

            await pilot.press("end", "enter")
            await pilot.pause()
            app.working_settings.general.reduce_motion = not app.settings.general.reduce_motion
            await pilot.press("q")
            await pilot.pause()
            assert isinstance(app.screen, ChoiceModal)
            await pilot.press("end", "enter")
            await pilot.pause()
            assert app.screen.id == "settings-root"
            assert app.settings_dirty

            await pilot.press("q")
            await pilot.pause()
            await pilot.press("down", "enter")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmModal)
            assert not app.settings_dirty

    asyncio.run(scenario())


@pytest.mark.parametrize("theme", ["openlrc-dark", "textual-dark"])
def test_disabled_settings_actions_keep_the_group_frame_continuous(tmp_path: Path, theme: str) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            app.theme = theme
            await pilot.pause()
            await pilot.press("end", "enter")
            await pilot.pause()

            rows = {item.action_id: item for item in app.screen.query(ActionItem)}
            actions_heading = list(app.screen.query(ActionGroupHeading))[-1]
            actions_spacer = list(app.screen.query(ActionGroupSpacer))[-2]
            frame_color = actions_heading.styles.border_left[1]
            assert actions_spacer.styles.border_left[1] == frame_color
            assert rows["save"].styles.border_left[1] == frame_color
            assert rows["save"].styles.border_right[1] == frame_color
            assert rows["discard"].styles.border_left[1] == frame_color
            assert rows["discard"].styles.border_right[1] == frame_color

    asyncio.run(scenario())


def test_new_work_copy_groups_accent_and_wraparound(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, WorkflowTypeScreen)
            action_list = app.screen.query_one(ActionList)
            items = list(app.screen.query(ActionItem))
            assert [item.label_text for item in items] == [
                "Transcribe to JSON",
                "Transcribe to subtitle",
                "Translate existing JSON",
                "Full run",
            ]
            assert [
                str(heading.query_one(".action-group-title").content)
                for heading in app.screen.query(ActionGroupHeading)
            ] == ["WORK TYPE"]
            expected_accents = ("Transcribe", "Transcribe", "Translate", "Full run")
            for item, expected in zip(items, expected_accents, strict=True):
                content = item.query_one(".action-label").content
                assert isinstance(content, Text)
                assert content.spans[0].start == 0
                assert content.spans[0].end == len(expected)
                assert "#f0aa4b" in str(content.spans[0].style)

            await pilot.press("end")
            assert action_list.selected_action() == "full-run"
            await pilot.press("down")
            assert action_list.selected_action() == "transcribe-json"
            await pilot.press("up")
            assert action_list.selected_action() == "full-run"

    asyncio.run(scenario())


def test_new_work_steps_and_readonly_preflight_actions(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path, ready_preflight=True)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen.query_one(WorkflowSteps).current_step == 1

            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, WorkflowOptionsScreen)
            assert app.screen.query_one(WorkflowSteps).current_step == 2
            headings = [
                str(heading.query_one(".action-group-title").content)
                for heading in app.screen.query(ActionGroupHeading)
            ]
            assert headings == ["BASIC SETTINGS", "ADVANCED SETTINGS", "ACTIONS"]

            await pilot.press("end", "enter")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmWorkflowScreen)
            assert app.screen.query_one(WorkflowSteps).current_step == 3
            assert [item.action_id for item in app.screen.query(ActionItem)] == ["start", "back"]
            actions = app.screen.query_one(ActionList)
            assert actions.selected_action() == "start"
            assert not app.screen.query_one("#preflight-readonly").can_focus
            preview = app.screen.query_one("#preflight-readonly", Static)
            assert preview.max_scroll_y > 0

            await pilot.press("pagedown")
            await pilot.pause()
            assert preview.scroll_y > 0
            assert actions.has_focus
            assert actions.selected_action() == "start"
            await pilot.press("pageup")
            await pilot.pause()
            assert preview.scroll_y == 0

            await pilot.press("down")
            assert actions.selected_action() == "back"
            await pilot.press("down")
            assert actions.selected_action() == "start"
            await pilot.press("up")
            assert actions.selected_action() == "back"
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, WorkflowOptionsScreen)
            assert app.screen.query_one(WorkflowSteps).current_step == 2

    asyncio.run(scenario())


def test_workflow_options_refresh_input_count_after_nested_editor(tmp_path: Path) -> None:
    media = tmp_path / "episode.mp4"
    media.touch()

    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("enter", "enter", "enter")
            await pilot.pause()

            await pilot.press("down", "down", "enter")
            await pilot.pause()
            app.screen.query_one(TextArea).text = str(media)
            await pilot.press("tab", "space")
            await pilot.pause()
            await pilot.press("end", "enter")
            await pilot.pause()

            assert isinstance(app.screen, WorkflowOptionsScreen)
            input_item = next(item for item in app.screen.query(ActionItem) if item.action_id == "inputs")
            assert input_item.detail_text.startswith("1 selected")

    asyncio.run(scenario())


def test_context_assistance_tui_hides_context_fields_and_enforces_required_modes(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        draft = WorkflowDraft(
            task="translate-existing",
            workflow="translate",
            translation_backend="local",
            mode="normal",
            context_assistance="off",
            context_provider="local",
            context_model="preserved-context.gguf",
            brief_summary="Known story.",
        )
        async with app.run_test(size=(100, 36)) as pilot:
            app.begin_workflow(draft)
            app.push_screen(WorkflowOptionsScreen())
            await pilot.pause()
            rows = {item.action_id: item for item in app.screen.query(ActionItem)}
            assert rows["context-assistance"].detail_text == "Off"
            assert "context-provider" not in rows
            assert "context-model" not in rows
            assert rows["brief-characters"].detail_text == "None · explicit"
            assert rows["brief-tone"].detail_text == "No additional guidance"

            options = app.screen
            assert isinstance(options, WorkflowOptionsScreen)
            options._context_assistance_selected(ContextAssistance.AUTO.value)
            await pilot.pause()
            assert app.draft.context_provider == "local"
            assert app.draft.context_model == "preserved-context.gguf"
            rows = {item.action_id: item for item in app.screen.query(ActionItem)}
            assert "context-provider" in rows
            assert "context-model" in rows

            options._context_assistance_selected(ContextAssistance.OFF.value)
            app.draft.mode = "normal-plus"
            app.draft.edit_rounds = 0
            options._recompose()
            await pilot.pause()
            options._setter("edit_rounds")("1")
            assert app.draft.edit_rounds == 0

            app.draft.mode = "normal"
            app.draft.edit_rounds = 1
            app.push_screen(HyMT2ModeScreen())
            await pilot.pause()
            await pilot.press("down", "down", "enter")
            await pilot.pause()
            assert isinstance(app.screen, HyMT2ModeScreen)
            assert app.draft.mode == "normal"
            assert app.draft.context_assistance == ContextAssistance.OFF.value

            await pilot.press("end", "enter")
            await pilot.pause()
            assert isinstance(app.screen, WorkflowOptionsScreen)
            assert app.draft.mode == "pro"
            assert app.draft.context_assistance == ContextAssistance.AUTO.value
            rows = {item.action_id: item for item in app.screen.query(ActionItem)}
            assert rows["context-assistance"].disabled
            assert rows["context-assistance"].detail_text == "Required by Pro"

    asyncio.run(scenario())


def test_blocked_preflight_skips_disabled_start_and_focuses_back(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("enter", "enter")
            await pilot.pause()
            await pilot.press("end", "enter")
            await pilot.pause()

            assert isinstance(app.screen, ConfirmWorkflowScreen)
            actions = app.screen.query_one(ActionList)
            assert actions.selected_action() == "back"
            assert app.screen.query_one("#preflight-readonly").content.plain
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, WorkflowOptionsScreen)

    asyncio.run(scenario())


def test_clean_draft_is_discarded_but_edited_draft_prompts(tmp_path: Path) -> None:
    media = tmp_path / "draft.mp4"
    media.touch()

    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("enter", "enter")
            await pilot.pause()
            assert isinstance(app.screen, WorkflowOptionsScreen)
            assert not app.draft_dirty

            await pilot.press("escape", "escape")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, WorkflowTypeScreen)
            with pytest.raises(RuntimeError, match="No Workflow Draft"):
                _ = app.draft

            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("down", "down", "enter")
            await pilot.pause()
            text_area = app.screen.query_one(TextArea)
            text_area.text = str(media)
            await pilot.press("tab")
            assert app.screen.query_one(ActionList).has_focus
            await pilot.press("space")
            await pilot.pause()
            assert app.draft_dirty

            await pilot.press("escape", "escape", "escape")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ChoiceModal)

    asyncio.run(scenario())


def test_resume_preserves_dirty_draft_until_explicit_discard_and_replaces_clean_draft(tmp_path: Path) -> None:
    history_input = tmp_path / "history.mp4"
    current_input = tmp_path / "current.mp4"
    history_input.touch()
    current_input.touch()
    history = WorkflowDraft(
        task="transcribe-json", workflow="transcribe", paths=[str(history_input)], whisper_model="history-model.bin"
    )
    record = JobRecord(
        job_id="resume-source",
        workflow="transcribe",
        name=history_input.name,
        status=JobRecordStatus.CANCELLED,
        input_paths=[str(history_input)],
        recipe=history.to_recipe(),
    )
    JobRepository(tmp_path / "jobs.json").save([record])

    async def dirty_scenario() -> None:
        app = _app(tmp_path, ready_preflight=True)
        current = WorkflowDraft(
            task="transcribe-json", workflow="transcribe", paths=[str(current_input)], whisper_model="current-model.bin"
        )
        app.begin_workflow(current)
        app.establish_draft_baseline()
        app.draft.source_language = "ja"
        before_recipe = app.draft.to_recipe()
        before_baseline = dict(app._draft_baseline or {})

        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(JobDetailScreen(record.job_id))
            await pilot.pause()
            detail = app.screen
            assert isinstance(detail, JobDetailScreen)
            detail.action_resume_job()
            await pilot.pause()
            assert isinstance(app.screen, ChoiceModal)
            assert app.draft.to_recipe() == before_recipe
            assert app._draft_baseline == before_baseline

            await pilot.press("down", "enter")
            await pilot.pause()
            assert isinstance(app.screen, JobDetailScreen)
            assert app.draft.to_recipe() == before_recipe
            assert app._draft_baseline == before_baseline

            detail = app.screen
            assert isinstance(detail, JobDetailScreen)
            detail.action_resume_job()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmWorkflowScreen)
            assert app.screen.resumed_from == record.job_id
            assert app.draft.paths == [str(history_input)]
            assert app.draft.whisper_model == "history-model.bin"
            assert app._draft_baseline is None

    async def clean_scenario() -> None:
        app = _app(tmp_path, ready_preflight=True)
        app.begin_workflow(
            WorkflowDraft(
                task="transcribe-json",
                workflow="transcribe",
                paths=[str(current_input)],
                whisper_model="current-model.bin",
            )
        )
        app.establish_draft_baseline()

        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(JobDetailScreen(record.job_id))
            await pilot.pause()
            detail = app.screen
            assert isinstance(detail, JobDetailScreen)
            detail.action_resume_job()
            await pilot.pause()
            assert isinstance(app.screen, ConfirmWorkflowScreen)
            assert app.screen.resumed_from == record.job_id
            assert app.draft.paths == [str(history_input)]

    asyncio.run(dirty_scenario())
    asyncio.run(clean_scenario())


def test_settings_text_modal_is_visible_and_keyboard_operable_at_80x24(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("end", "enter")
            await pilot.pause()
            await pilot.press("enter", "enter", "down", "enter")
            await pilot.pause()

            assert isinstance(app.screen, TextInputModal)
            input_widget = app.screen.query_one(Input)
            actions = app.screen.query_one(ActionList)
            assert input_widget.has_focus
            assert actions.region.y + actions.region.height <= app.size.height
            assert [item.action_id for item in app.screen.query(ActionItem)] == ["apply", "cancel"]

            input_widget.value = "keyboard-model"
            await pilot.press("tab")
            assert actions.has_focus
            assert actions.selected_action() == "apply"
            await pilot.press("up")
            assert actions.selected_action() == "cancel"
            await pilot.press("down", "space")
            await pilot.pause()

            assert isinstance(app.screen, ProviderDetailScreen)
            assert app.working_settings.providers["openai"].model == "keyboard-model"

            provider_actions = app.screen.query_one(ActionList)
            provider_actions.action_first_enabled()
            await pilot.press("down", "enter")
            await pilot.pause()
            assert isinstance(app.screen, TextInputModal)
            app.screen.query_one(Input).value = "mouse-model"
            await pilot.click(".modal-action-list .action-primary")
            await pilot.pause()
            assert isinstance(app.screen, ProviderDetailScreen)
            assert app.working_settings.providers["openai"].model == "mouse-model"

            provider_actions = app.screen.query_one(ActionList)
            provider_actions.action_first_enabled()
            await pilot.press("down", "down", "down", "down", "enter")
            await pilot.pause()
            assert isinstance(app.screen, TextInputModal)
            assert app.screen.query_one(Input).password
            await pilot.press("escape")

    asyncio.run(scenario())


def test_brief_multiline_editor_keeps_actions_visible_and_rejects_invalid_characters(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        draft = WorkflowDraft(
            task="translate-existing",
            workflow="translate",
            translation_backend="local",
            mode="normal",
            context_provider="local",
            context_model="qwen.gguf",
            brief_summary="Existing summary",
            brief_characters="111111",
        )
        async with app.run_test(size=(80, 24)) as pilot:
            app.begin_workflow(draft)
            app.push_screen(WorkflowOptionsScreen())
            await pilot.pause()
            rows = {item.action_id: item for item in app.screen.query(ActionItem)}
            assert rows["brief-characters"].detail_text == "Invalid · edit to fix"

            options = app.screen
            assert isinstance(options, WorkflowOptionsScreen)
            options.focus_action("brief-summary")
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, MultilineTextModal)
            summary_editor = app.screen.query_one(TextArea)
            actions = app.screen.query_one(ActionList)
            dialog = app.screen.query_one(".editor-dialog")
            assert summary_editor.has_focus
            assert summary_editor.region.height >= 5
            assert actions.region.y + actions.region.height <= dialog.region.y + dialog.region.height
            assert dialog.region.y + dialog.region.height <= app.size.height
            assert "Ctrl+Enter Apply" in str(app.screen.query_one(".modal-editor-hint", Static).content)

            summary_editor.text = "Discard this edit."
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, WorkflowOptionsScreen)
            assert app.draft.brief_summary == "Existing summary"

            app.screen.focus_action("brief-summary")
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, MultilineTextModal)
            summary_editor = app.screen.query_one(TextArea)
            actions = app.screen.query_one(ActionList)
            summary_editor.text = "First line.\nSecond line."
            await pilot.press("tab")
            assert actions.has_focus
            await pilot.press("shift+tab")
            assert summary_editor.has_focus
            await pilot.press("ctrl+enter")
            await pilot.pause()

            assert isinstance(app.screen, WorkflowOptionsScreen)
            assert app.draft.brief_summary == "First line.\nSecond line."
            rows = {item.action_id: item for item in app.screen.query(ActionItem)}
            assert rows["brief-summary"].detail_text == "First line. Second line."

            app.screen.focus_action("brief-characters")
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, MultilineTextModal)
            characters_editor = app.screen.query_one(TextArea)
            assert characters_editor.text == "111111"

            await pilot.press("ctrl+enter")
            await pilot.pause()
            assert isinstance(app.screen, MultilineTextModal)
            error = app.screen.query_one(".modal-validation-error", Static)
            assert error.has_class("visible")
            assert "Brief character line 1" in str(error.content)
            assert characters_editor.has_class("input-invalid")
            assert characters_editor.has_focus
            assert app.draft.brief_characters == "111111"
            dialog = app.screen.query_one(".editor-dialog")
            actions = app.screen.query_one(ActionList)
            assert actions.region.y + actions.region.height <= dialog.region.y + dialog.region.height

            characters_editor.text = "John = 强尼"
            await pilot.pause()
            assert not error.has_class("visible")
            assert not characters_editor.has_class("input-invalid")

            characters_editor.text = "John = 强尼\nMary = 玛丽"
            await pilot.press("ctrl+enter")
            await pilot.pause()
            assert isinstance(app.screen, WorkflowOptionsScreen)
            assert app.draft.brief_characters == "John = 强尼\nMary = 玛丽"

    asyncio.run(scenario())


def test_keyboard_builds_raw_transcription_draft(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, WorkflowTypeScreen)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, WorkflowOptionsScreen)
            request = app.draft
            assert request is not None
            assert request.task == "transcribe-json"
            media = tmp_path / "draft.mp4"
            media.touch()
            request.paths = [str(media)]
            assert request.build_request(app.settings, app.credentials).subtitle_output is False

    asyncio.run(scenario())


def test_text_input_suppresses_single_character_shortcuts(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("end", "enter")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("down", "enter")
            await pilot.pause()
            input_widget = app.screen.query_one(Input)
            input_widget.value = ""
            await pilot.press("q", "d", "n", "j", "k")
            assert input_widget.value == "qdnjk"
            assert app.screen.query_one(Input) is input_widget

    asyncio.run(scenario())


def test_workflow_starts_once_and_reaches_persisted_result(tmp_path: Path) -> None:
    media = tmp_path / "episode.mp4"
    media.touch()

    async def scenario() -> None:
        app = _app(tmp_path, ready_preflight=True)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("down", "down", "enter")
            await pilot.pause()
            paths = app.screen.query_one(TextArea)
            paths.text = str(media)
            await pilot.press("ctrl+enter")
            await pilot.pause()
            await pilot.press("end", "enter")
            await pilot.pause()
            await pilot.press("end", "enter")
            await pilot.pause()
            await pilot.press("enter")
            for _ in range(20):
                if isinstance(app.screen, RunningWorkflowScreen) and app.screen.result is not None:
                    break
                await pilot.pause(0.05)

            assert isinstance(app.screen, RunningWorkflowScreen)
            assert app.screen.start_error is None, app.screen.start_error
            assert app.screen.result is not None
            assert app.screen.result.status is WorkflowStatus.SUCCEEDED
            assert len(app.job_controller.records) == 1
            assert app.operation_state == "idle"

    asyncio.run(scenario())


def test_workflow_runtime_logs_stay_in_frame_and_completed_escape_returns_home(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    media = tmp_path / "episode.mp4"
    media.touch()

    class LoggingExecutor:
        def execute(self, _request, context):
            context.workflow_started()
            openlrc_logger.info("captured workflow runtime output")
            context.workflow_completed(WorkflowStatus.SUCCEEDED, 0.01)
            context.close()
            return WorkflowResult(
                job_id=context.job_id, workflow=context.workflow, status=WorkflowStatus.SUCCEEDED, elapsed_seconds=0.01
            )

    app = OpenLRCTUI(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        credentials=StaticCredentials(),
        job_controller=JobController(JobRepository(tmp_path / "jobs.json"), executor=LoggingExecutor()),
        resources=StaticResources(),
        fixed_logo_frame=0,
    )

    async def scenario() -> None:
        draft = WorkflowDraft(workflow="transcribe", paths=[str(media)])
        async with app.run_test(size=(80, 24)) as pilot:
            assert openlrc_terminal_handler not in openlrc_logger.handlers
            app.begin_workflow(draft)
            app.push_screen(WorkflowOptionsScreen())
            await pilot.pause()
            app.start_workflow()

            for _ in range(200):
                screen = app.running_workflow_screen
                if screen is not None and screen.result is not None:
                    break
                await pilot.pause(0.01)
            await pilot.pause()

            assert isinstance(app.screen, RunningWorkflowScreen)
            assert app.screen.result is not None
            assert app.workflow_output == ["INFO captured workflow runtime output"]
            output = app.screen.query_one("#workflow-runtime-output", RichLog)
            assert "captured workflow runtime output" in "\n".join(line.text for line in output.lines)
            actions = app.screen.query_one("#running-actions", ActionList)
            frame = app.screen.query_one("#workflow-runtime-frame")
            footer = app.screen.query_one("PageFooter")
            assert actions.region.bottom <= frame.region.y
            assert frame.region.bottom <= footer.region.y

            escape_binding = app.active_bindings["escape"]
            assert escape_binding.node is app.screen
            assert escape_binding.binding.action == "back"
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen.id == "home"
            assert app._draft is None
            assert not any(isinstance(screen, WorkflowOptionsScreen) for screen in app.screen_stack)

    asyncio.run(scenario())
    assert openlrc_terminal_handler in openlrc_logger.handlers
    assert "captured workflow runtime output" not in capsys.readouterr().err


def test_tui_starting_workflow_cancel_reaches_controller_before_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "episode.mp4"
    media.touch()
    entered = threading.Event()
    release = threading.Event()
    captured_tokens: list[CancellationToken] = []

    class DelayedJobController(JobController):
        def run(self, draft, settings, credentials, *, on_event=None, resumed_from=None, cancellation_token=None):
            assert cancellation_token is not None
            captured_tokens.append(cancellation_token)
            entered.set()
            release.wait(timeout=5)
            return super().run(
                draft,
                settings,
                credentials,
                on_event=on_event,
                resumed_from=resumed_from,
                cancellation_token=cancellation_token,
            )

    def unexpected_lrcer(*_args, **_kwargs):
        raise AssertionError("LRCer must not be created for a pre-cancelled Workflow")

    monkeypatch.setattr("openlrc.workflow.executor.LRCer", unexpected_lrcer)
    controller = DelayedJobController(JobRepository(tmp_path / "jobs.json"))
    app = OpenLRCTUI(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        credentials=StaticCredentials(),
        job_controller=controller,
        resources=StaticResources(),
        fixed_logo_frame=0,
    )

    async def scenario() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.begin_workflow(WorkflowDraft(workflow="transcribe", paths=[str(media)]))
            app.start_workflow()
            for _ in range(100):
                if entered.is_set():
                    break
                await pilot.pause(0.01)
            assert entered.is_set()
            assert app.operation_state == "starting"

            app.cancel_active_operation()
            for _ in range(100):
                if captured_tokens[0].is_cancelled:
                    break
                await pilot.pause(0.01)
            assert captured_tokens[0].is_cancelled
            release.set()

            for _ in range(100):
                if app.running_workflow_screen is not None and app.running_workflow_screen.result is not None:
                    break
                await pilot.pause(0.01)
            assert isinstance(app.screen, RunningWorkflowScreen)
            assert app.screen.result is not None
            assert app.screen.result.status is WorkflowStatus.CANCELLED
            assert controller.records[0].status is JobRecordStatus.CANCELLED
            assert app.operation_state == "idle"
            assert app._active_cancellation_token is None

    asyncio.run(scenario())


def test_tui_starting_setup_cancel_skips_setup_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entered = threading.Event()
    release = threading.Event()
    captured_tokens: list[CancellationToken] = []

    class DelayedSetupController(SetupController):
        def run(self, request, *, on_event=None, cancellation_token=None):
            assert cancellation_token is not None
            captured_tokens.append(cancellation_token)
            entered.set()
            release.wait(timeout=5)
            return super().run(request, on_event=on_event, cancellation_token=cancellation_token)

    def unexpected_setup(**_kwargs):
        raise AssertionError("setup service must not run for a pre-cancelled Setup")

    monkeypatch.setattr("openlrc.application.setup.setup_whisper_cpp", unexpected_setup)
    app = OpenLRCTUI(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        credentials=StaticCredentials(),
        setup_controller=DelayedSetupController(),
        resources=StaticResources(),
        fixed_logo_frame=0,
    )

    async def scenario() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            app.start_setup(WhisperSetupRequest())
            for _ in range(100):
                if entered.is_set():
                    break
                await pilot.pause(0.01)
            assert entered.is_set()
            assert app.operation_state == "starting"

            app.cancel_active_operation()
            for _ in range(100):
                if captured_tokens[0].is_cancelled:
                    break
                await pilot.pause(0.01)
            assert captured_tokens[0].is_cancelled
            release.set()

            for _ in range(100):
                if app.setup_running_screen is not None and app.setup_running_screen.result is not None:
                    break
                await pilot.pause(0.01)
            assert isinstance(app.screen, SetupRunningScreen)
            assert app.screen.result is not None
            assert app.screen.result.status is SetupStatus.CANCELLED
            assert app.operation_state == "idle"
            assert app._active_cancellation_token is None

    asyncio.run(scenario())
