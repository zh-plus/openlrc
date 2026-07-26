"""Confirmation and read-only detail modals."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from openlrc.tui.i18n import tr
from openlrc.tui.navigation import ActionItem, ActionList


class ConfirmModal(ModalScreen[bool | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, title: str, message: str, *, confirm_label: str = "Confirm") -> None:
        super().__init__()
        self.dialog_title = title
        self.message = message
        self.confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-dialog confirm-dialog"):
            yield Static(tr(self.dialog_title), classes="modal-title", markup=False)
            yield Static(tr(self.message), classes="modal-message", markup=False)
            yield ActionList(
                ActionItem("cancel", "Cancel"),
                ActionItem("confirm", self.confirm_label, classes="action-primary"),
                classes="modal-action-list",
            )

    def on_mount(self) -> None:
        self.query_one(ActionList).focus()

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        self.dismiss(event.action_id == "confirm")

    def action_cancel(self) -> None:
        self.dismiss(False)


class DetailModal(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "Close", show=False)]

    def __init__(self, title: str, content: str) -> None:
        super().__init__()
        self.dialog_title = title
        self.content = content

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-dialog detail-dialog"):
            yield Static(tr(self.dialog_title), classes="modal-title", markup=False)
            yield Static(tr(self.content), classes="modal-message detail-content", markup=False)
            yield ActionList(ActionItem("close", "Close", classes="action-primary"), classes="modal-action-list")

    def on_action_list_activated(self) -> None:
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)
