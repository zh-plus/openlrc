from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from openlrc.application import AppSettings, SettingsStore, WorkflowDraft
from openlrc.tui.navigation import ActionItem, ActionList
from openlrc.tui.screens.workflow import WorkflowOptionsScreen
from tests.test_tui_v2 import _app

_TERMINAL_ID = re.compile(r"terminal-\d+")
_STYLE_RULE = re.compile(r"\.(terminal-SNAPSHOT-r\d+)\s*\{\s*([^}]+?)\s*\}")
_STYLE_CLASS = re.compile(r"terminal-SNAPSHOT-r\d+")


@pytest.fixture(autouse=True)
def _render_colored_logo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)


def _normalize_svg(svg: str) -> str:
    normalized = _TERMINAL_ID.sub("terminal-SNAPSHOT", svg)
    rules = list(_STYLE_RULE.finditer(normalized))
    declarations = sorted({match.group(2).strip() for match in rules})
    style_names = {
        declaration: f"terminal-SNAPSHOT-style-{index}" for index, declaration in enumerate(declarations, start=1)
    }
    class_names = {match.group(1): style_names[match.group(2).strip()] for match in rules}
    normalized = _STYLE_CLASS.sub(lambda match: class_names.get(match.group(0), match.group(0)), normalized)

    # Rich numbers style classes by first paint order, which can change when a
    # focused widget repaints without changing the resulting SVG appearance.
    # Compare canonical declaration-backed classes while preserving every
    # declaration and every use site.
    lines = normalized.splitlines()
    canonical_rule = re.compile(r"\.terminal-SNAPSHOT-style-\d+\s*\{[^}]+\}")
    rule_indexes = [index for index, line in enumerate(lines) if canonical_rule.fullmatch(line.strip())]
    if rule_indexes:
        first_rule = rule_indexes[0]
        rule_index_set = set(rule_indexes)
        canonical_lines = [f".{style_names[declaration]} {{ {declaration} }}" for declaration in declarations]
        lines = [line for index, line in enumerate(lines) if index not in rule_index_set]
        lines[first_rule:first_rule] = canonical_lines
    suffix = "\n" if normalized.endswith("\n") else ""
    return "\n".join(line.rstrip() for line in lines) + suffix


def _assert_svg_snapshot(rendered: str, baseline: Path) -> None:
    expected = baseline.read_text(encoding="utf-8")
    assert _normalize_svg(rendered) == _normalize_svg(expected)


@pytest.mark.parametrize("size", [(80, 24), (100, 30), (160, 50)])
def test_home_svg_snapshot(tmp_path: Path, size: tuple[int, int]) -> None:
    async def render() -> str:
        app = _app(tmp_path)
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            return app.export_screenshot()

    width, height = size
    baseline = Path(__file__).parent / "snapshots" / "tui_v2" / f"home-{width}x{height}.svg"
    _assert_svg_snapshot(asyncio.run(render()), baseline)


@pytest.mark.parametrize("frame", [14, 15, 16])
def test_wide_logo_frame_svg_snapshot(tmp_path: Path, frame: int) -> None:
    async def render() -> str:
        app = _app(tmp_path, fixed_logo_frame=frame)
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause()
            return app.export_screenshot()

    baseline = Path(__file__).parent / "snapshots" / "tui_v2" / f"home-wide-frame-{frame}-160x50.svg"
    _assert_svg_snapshot(asyncio.run(render()), baseline)


@pytest.mark.parametrize(
    ("name", "keys", "ready_preflight"),
    [
        ("new-work", ("enter",), False),
        ("workflow-options", ("enter", "enter"), False),
        ("preflight-ready", ("enter", "enter", "end", "enter"), True),
        ("jobs", ("down", "enter"), False),
        ("models", ("down", "down", "enter"), False),
        ("settings", ("end", "enter"), False),
        ("doctor", ("d",), False),
        ("quit-confirm", ("q",), False),
    ],
)
def test_representative_page_svg_snapshot(
    tmp_path: Path, name: str, keys: tuple[str, ...], ready_preflight: bool
) -> None:
    async def render() -> str:
        app = _app(tmp_path, ready_preflight=ready_preflight)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press(*keys)
            await pilot.pause()
            return app.export_screenshot()

    baseline = Path(__file__).parent / "snapshots" / "tui_v2" / f"{name}-100x30.svg"
    _assert_svg_snapshot(asyncio.run(render()), baseline)


def test_settings_input_modal_svg_snapshot(tmp_path: Path) -> None:
    async def render() -> str:
        app = _app(tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("end", "enter", "enter", "enter", "down", "enter")
            await pilot.pause()
            return app.export_screenshot()

    baseline = Path(__file__).parent / "snapshots" / "tui_v2" / "settings-input-modal-80x24.svg"
    _assert_svg_snapshot(asyncio.run(render()), baseline)


def test_brief_character_error_modal_svg_snapshot(tmp_path: Path) -> None:
    async def render() -> str:
        app = _app(tmp_path)
        draft = WorkflowDraft(
            task="translate-existing",
            workflow="translate",
            translation_backend="local",
            mode="normal",
            context_provider="local",
            context_model="qwen.gguf",
            brief_summary="A two-scene workplace comedy.",
            brief_characters="111111",
        )
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app.begin_workflow(draft)
            app.push_screen(WorkflowOptionsScreen())
            await pilot.pause()
            actions = app.screen.query_one(ActionList)
            actions.index = next(
                index
                for index, item in enumerate(actions.children)
                if isinstance(item, ActionItem) and item.action_id == "brief-characters"
            )
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("ctrl+enter")
            await pilot.press("tab")
            await pilot.pause()
            return app.export_screenshot()

    baseline = Path(__file__).parent / "snapshots" / "tui_v2" / "brief-character-error-modal-80x24.svg"
    _assert_svg_snapshot(asyncio.run(render()), baseline)


@pytest.mark.parametrize(
    ("theme", "keys"),
    [
        ("openlrc-dark", ("end", "enter", "end", "enter")),
        ("textual-dark", ("end", "enter", "end", "enter", "enter", "down", "enter")),
    ],
)
def test_appearance_theme_svg_snapshot(tmp_path: Path, theme: str, keys: tuple[str, ...]) -> None:
    async def render() -> str:
        app = _app(tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press(*keys)
            await pilot.pause()
            assert app.theme == theme
            return app.export_screenshot()

    baseline = Path(__file__).parent / "snapshots" / "tui_v2" / f"appearance-{theme}-80x24.svg"
    _assert_svg_snapshot(asyncio.run(render()), baseline)


def test_simplified_chinese_home_svg_snapshot(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.general.language = "zh-cn"
    SettingsStore(tmp_path / "settings.json").save(settings)

    async def render() -> str:
        app = _app(tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            return app.export_screenshot()

    baseline = Path(__file__).parent / "snapshots" / "tui_v2" / "home-zh-cn-80x24.svg"
    _assert_svg_snapshot(asyncio.run(render()), baseline)


def test_static_logo_svg_snapshot(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.general.logo_animation = False
    SettingsStore(tmp_path / "settings.json").save(settings)

    async def render() -> str:
        app = _app(tmp_path, fixed_logo_frame=None)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            return app.export_screenshot()

    baseline = Path(__file__).parent / "snapshots" / "tui_v2" / "home-static-logo-100x30.svg"
    _assert_svg_snapshot(asyncio.run(render()), baseline)
