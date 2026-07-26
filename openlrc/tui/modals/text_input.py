"""Single-line and multiline text entry modals."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static, TextArea

from openlrc.tui.i18n import tr
from openlrc.tui.navigation import ActionItem, ActionList


class TextInputModal(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, title: str, value: str = "", *, placeholder: str = "", password: bool = False) -> None:
        super().__init__()
        self.dialog_title = title
        self.value = value
        self.placeholder = placeholder
        self.password = password

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-dialog text-dialog"):
            yield Static(tr(self.dialog_title), classes="modal-title", markup=False)
            with Vertical(classes="modal-field-frame"):
                yield Static(tr("VALUE"), classes="modal-field-title", markup=False)
                yield Input(
                    value=self.value,
                    placeholder=tr(self.placeholder),
                    password=self.password,
                    id="text-value",
                )
            yield _modal_actions("Apply")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        self.dismiss(self.query_one(Input).value if event.action_id == "apply" else None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class PathsModal(ModalScreen[str | None]):
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+enter", "apply", "Apply", show=False, priority=True),
        Binding("tab", "focus_actions", "Actions", show=False, priority=True),
        Binding("shift+tab", "focus_editor", "Editor", show=False, priority=True),
    ]

    def __init__(self, value: str = "") -> None:
        super().__init__()
        self.value = value

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-dialog paths-dialog"):
            yield Static(tr("Paste Paths"), classes="modal-title", markup=False)
            yield Static(tr("Enter one path per line. Missing paths stay visible for correction."), classes="muted")
            yield TextArea(self.value, id="paths-value")
            yield _modal_actions("Add Paths")

    def on_mount(self) -> None:
        self.query_one(TextArea).focus()

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        self.dismiss(self.query_one(TextArea).text if event.action_id == "apply" else None)

    def action_apply(self) -> None:
        self.dismiss(self.query_one(TextArea).text)

    def action_focus_actions(self) -> None:
        self.query_one(ActionList).focus()

    def action_focus_editor(self) -> None:
        self.query_one(TextArea).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)


class MultilineTextModal(ModalScreen[str | None]):
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+enter", "apply", "Apply", show=False, priority=True),
        Binding("tab", "focus_actions", "Actions", show=False, priority=True),
        Binding("shift+tab", "focus_editor", "Editor", show=False, priority=True),
    ]

    def __init__(self, title: str, value: str = "", *, help_text: str = "") -> None:
        super().__init__()
        self.dialog_title = title
        self.value = value
        self.help_text = help_text

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-dialog paths-dialog"):
            yield Static(tr(self.dialog_title), classes="modal-title", markup=False)
            yield Static(tr(self.help_text), classes="muted", markup=False)
            yield TextArea(self.value, id="multiline-value")
            yield _modal_actions("Apply")

    def on_mount(self) -> None:
        self.query_one(TextArea).focus()

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        self.dismiss(self.query_one(TextArea).text if event.action_id == "apply" else None)

    def action_apply(self) -> None:
        self.dismiss(self.query_one(TextArea).text)

    def action_focus_actions(self) -> None:
        self.query_one(ActionList).focus()

    def action_focus_editor(self) -> None:
        self.query_one(TextArea).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)


def _modal_actions(apply_label: str) -> ActionList:
    return ActionList(
        ActionItem("apply", apply_label, classes="action-primary"),
        ActionItem("cancel", "Cancel"),
        classes="modal-action-list",
    )
