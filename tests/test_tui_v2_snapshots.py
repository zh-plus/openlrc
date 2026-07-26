from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from openlrc.application import AppSettings, SettingsStore
from tests.test_tui_v2 import _app

_TERMINAL_ID = re.compile(r"terminal-\d+")


@pytest.fixture(autouse=True)
def _render_colored_logo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)


def _normalize_svg(svg: str) -> str:
    normalized = _TERMINAL_ID.sub("terminal-SNAPSHOT", svg)
    suffix = "\n" if normalized.endswith("\n") else ""
    return "\n".join(line.rstrip() for line in normalized.splitlines()) + suffix


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
