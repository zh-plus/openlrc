"""List-based choice modal."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from openlrc.tui.i18n import tr
from openlrc.tui.navigation import ActionItem, ActionList, action_group


class ChoiceModal(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, title: str, choices: list[tuple[str, str, str]], *, current: str = "") -> None:
        super().__init__()
        self.dialog_title = title
        self.choices = choices
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-dialog choice-dialog"):
            yield Static(tr(self.dialog_title), classes="modal-title", markup=False)
            items = [
                ActionItem(value, label, f"{detail}{' · Current' if value == self.current else ''}")
                for value, label, detail in self.choices
            ]
            yield ActionList(*action_group("Options", *items), id="choice-list")

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        self.dismiss(event.action_id)

    def action_cancel(self) -> None:
        self.dismiss(None)
