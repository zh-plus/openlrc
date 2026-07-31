"""macOS and terminal-native file selection for TUI v2."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, Static

from openlrc.tui.i18n import tr


def native_picker_availability() -> tuple[bool, str]:
    if sys.platform != "darwin":
        return False, "Native file selection is available on macOS only."
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return False, "Native file selection is unavailable in an SSH session."
    if shutil.which("osascript") is None:
        return False, "osascript is not available."
    return True, ""


def choose_files_native(initial_directory: Path, *, json_only: bool = False, language: str | None = None) -> list[str]:
    """Open the macOS picker without a shell and return absolute paths."""
    available, reason = native_picker_availability()
    if not available:
        raise RuntimeError(reason)
    start = initial_directory.expanduser().resolve(strict=False)
    if not start.is_dir():
        start = Path.home()
    prompt = tr("Choose transcription JSON files" if json_only else "Choose audio or video files", language=language)
    script = """
on run argv
    set startFolder to POSIX file (item 1 of argv) as alias
    set chooserPrompt to item 2 of argv
    try
        set chosenFiles to choose file with prompt chooserPrompt default location startFolder with multiple selections allowed
    on error number -128
        return ""
    end try
    set resultText to ""
    repeat with chosenFile in chosenFiles
        set resultText to resultText & POSIX path of chosenFile & linefeed
    end repeat
    return resultText
end run
"""
    result = subprocess.run(
        ["osascript", "-e", script, "--", str(start), prompt], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "macOS file selection failed.")
    return [str(Path(line).resolve(strict=False)) for line in result.stdout.splitlines() if line.strip()]


class VisibleDirectoryTree(DirectoryTree):
    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return [path for path in paths if not path.name.startswith(".")]


class _PickerButton(Button):
    BINDINGS = [
        Binding("enter", "press", "Press button", show=False),
        Binding("space", "press", "Press button", show=False),
    ]


class TerminalFilePickerModal(ModalScreen[list[str]]):
    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, initial_directory: Path) -> None:
        super().__init__()
        self.initial_directory = initial_directory if initial_directory.is_dir() else Path.home()
        self.selected: list[str] = []

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-dialog file-picker-dialog"):
            yield Static(tr("Browse in Terminal"), classes="modal-title", markup=False)
            with Horizontal(classes="file-picker-location-row"):
                yield Input(value=str(self.initial_directory.resolve()), id="file-picker-location")
                yield _PickerButton(tr("Go"), id="file-picker-go")
                yield _PickerButton(tr("Up"), id="file-picker-up")
            yield VisibleDirectoryTree(self.initial_directory, id="file-tree")
            yield Static(tr("No files selected"), id="file-picker-selected", classes="muted")
            with Horizontal(classes="modal-actions"):
                yield _PickerButton(tr("Cancel"), id="file-picker-cancel")
                yield _PickerButton(tr("Use Selected"), id="file-picker-done", variant="primary")

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        self.query_one("#file-picker-location", Input).value = str(event.path.resolve())

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        path = str(event.path.resolve())
        if path not in self.selected:
            self.selected.append(path)
        self.query_one("#file-picker-selected", Static).update(
            tr("Selected: ") + ", ".join(Path(item).name for item in self.selected)
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "file-picker-done":
            self.dismiss(self.selected)
        elif event.button.id == "file-picker-cancel":
            self.dismiss([])
        elif event.button.id == "file-picker-go":
            self._go_to(Path(self.query_one("#file-picker-location", Input).value).expanduser())
        elif event.button.id == "file-picker-up":
            self._go_to(Path(self.query_one("#file-picker-location", Input).value).expanduser().parent)

    def _go_to(self, path: Path) -> None:
        resolved = path.resolve(strict=False)
        if not resolved.is_dir():
            self.query_one("#file-picker-selected", Static).update(f"{tr('Not a directory: ')}{resolved}")
            return
        self.query_one("#file-picker-location", Input).value = str(resolved)
        self.query_one("#file-tree", VisibleDirectoryTree).path = resolved

    def action_cancel(self) -> None:
        self.dismiss([])
