"""Fixed-cell ASCII logo with a style-only accent sweep."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from openlrc.tui.app import OpenLRCTUI

WIDE_LOGO = (
    r"  ___  ____  _____ _   _ _     ____   ____",
    r" / _ \|  _ \| ____| \ | | |   |  _ \ / ___|",
    r"| | | | |_) |  _| |  \| | |   | |_) | |",
    r"| |_| |  __/| |___| |\  | |___|  _ <| |___",
    r" \___/|_|   |_____|_| \_|_____|_| \_\____|",
)
COMPACT_LOGO = (
    " __  ___  ___ _  _ |  ___  __ ",
    "/  \\ |__] |__ |\\ | |  |__] /  `",
    "\\__/ |    |__ | \\| |__| \\_ \\__,",
)
MINIMAL_LOGO = ("OpenLRC Mac",)


def _fixed_cell_lines(lines: tuple[str, ...]) -> tuple[str, ...]:
    """Pad ASCII art rows so Rich centers one fixed-width character matrix."""

    width = max((cell_len(line) for line in lines), default=0)
    return tuple(f"{line}{' ' * (width - cell_len(line))}" for line in lines)


def render_logo(layout: str, *, frame: int = 0, color: bool = True, sweep: bool = True) -> Text:
    source_lines = {"wide": WIDE_LOGO, "compact": COMPACT_LOGO, "minimal": MINIMAL_LOGO}[layout]
    lines = source_lines if layout == "minimal" else _fixed_cell_lines(source_lines)
    text = Text("\n".join(lines), no_wrap=True, overflow="crop")
    if not color:
        text.stylize(Style(color="#7f8b96"))
        return text
    if layout == "minimal":
        text.stylize(Style(color="#48c6dc", bold=True))
        return text
    if not sweep:
        text.stylize(Style(color="#48c6dc", bold=True))
        return text

    width = max((cell_len(line) for line in lines), default=0)
    if width == 0:
        return text
    text.stylize(Style(color="#318fa7"))
    positions_by_column: dict[int, list[int]] = {}
    plain_index = 0
    for line in lines:
        for column, char in enumerate(line):
            if not char.isspace():
                positions_by_column.setdefault(column, []).append(plain_index + column)
        plain_index += len(line) + 1

    position = frame % width
    for offset, color_value, bold in (
        (-3, "#3c6471", False),
        (-2, "#3f8e9f", False),
        (-1, "#49c6dc", True),
        (0, "#f0aa4b", True),
        (1, "#f6cf78", True),
        (2, "#49c6dc", False),
        (3, "#3f8e9f", False),
    ):
        column = (position + offset) % width
        for index in positions_by_column.get(column, ()):
            text.stylize(Style(color=color_value, bold=bold), index, index + 1)
    return text


class LogoWidget(Static):
    """Refresh only the logo while Home is visible and the application is idle."""

    def __init__(self, *, fixed_frame: int | None = None, id: str | None = None) -> None:
        super().__init__(id=id)
        self.frame = fixed_frame or 0
        self.fixed_frame = fixed_frame
        self.layout_name = "compact"
        self._showing_sweep = False

    def on_mount(self) -> None:
        self._update_layout()
        self.set_interval(0.125, self._advance)

    def on_resize(self) -> None:
        self._update_layout()

    def _update_layout(self) -> None:
        width = self.app.size.width
        self.layout_name = "wide" if width >= 100 else "compact" if width >= 80 else "minimal"
        self._render_frame()

    def _advance(self) -> None:
        if self.fixed_frame is not None:
            return
        if not self._animation_enabled():
            if self._showing_sweep:
                self._render_frame()
            return
        self.frame += 1
        self._render_frame()

    def _animation_enabled(self) -> bool:
        general = self.app.visual_general
        return bool(
            self.app.screen.id == "home"
            and self.layout_name != "minimal"
            and self.app.operation_state == "idle"
            and general.logo_animation
            and not general.reduce_motion
            and "NO_COLOR" not in os.environ
        )

    def _render_frame(self) -> None:
        color = "NO_COLOR" not in os.environ
        sweep = self.fixed_frame is not None or self._animation_enabled()
        self._showing_sweep = bool(color and sweep and self.layout_name != "minimal")
        self.update(render_logo(self.layout_name, frame=self.frame, color=color, sweep=sweep))

    def refresh_visual_settings(self) -> None:
        self._render_frame()

    if TYPE_CHECKING:

        @property
        def app(self) -> OpenLRCTUI:
            return cast("OpenLRCTUI", super().app)
