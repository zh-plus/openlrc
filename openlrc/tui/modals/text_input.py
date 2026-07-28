"""Single-line and multiline text entry modals."""

from collections.abc import Callable

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
                yield Input(value=self.value, placeholder=tr(self.placeholder), password=self.password, id="text-value")
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
        with Vertical(classes="modal-dialog editor-dialog"):
            yield Static(tr("Paste Paths"), classes="modal-title", markup=False)
            yield Static(
                tr("Enter one path per line. Missing paths stay visible for correction."),
                classes="modal-editor-help",
                markup=False,
            )
            yield TextArea(self.value, id="paths-value", tab_behavior="focus")
            yield Static(
                tr("Ctrl+Enter Apply · Esc Cancel · Tab Actions · Shift+Tab Editor"),
                classes="modal-editor-hint",
                markup=False,
            )
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

    def __init__(
        self,
        title: str,
        value: str = "",
        *,
        help_text: str = "",
        placeholder: str = "",
        validator: Callable[[str], object] | None = None,
    ) -> None:
        super().__init__()
        self.dialog_title = title
        self.value = value
        self.help_text = help_text
        self.placeholder = placeholder
        self.validator = validator

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-dialog editor-dialog"):
            yield Static(tr(self.dialog_title), classes="modal-title", markup=False)
            yield Static(tr(self.help_text), classes="modal-editor-help", markup=False)
            yield TextArea(self.value, id="multiline-value", placeholder=tr(self.placeholder), tab_behavior="focus")
            yield Static("", classes="modal-validation-error", markup=False)
            yield Static(
                tr("Ctrl+Enter Apply · Esc Cancel · Tab Actions · Shift+Tab Editor"),
                classes="modal-editor-hint",
                markup=False,
            )
            yield _modal_actions("Apply")

    def on_mount(self) -> None:
        self.query_one(TextArea).focus()

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        if event.action_id == "apply":
            self._submit()
        else:
            self.dismiss(None)

    def action_apply(self) -> None:
        self._submit()

    def action_focus_actions(self) -> None:
        self.query_one(ActionList).focus()

    def action_focus_editor(self) -> None:
        self.query_one(TextArea).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "multiline-value":
            return
        event.text_area.remove_class("input-invalid")
        error = self.query_one(".modal-validation-error", Static)
        error.remove_class("visible")
        error.update("")

    def _submit(self) -> None:
        value = self.query_one(TextArea).text
        if self.validator is not None:
            try:
                self.validator(value)
            except ValueError as exc:
                editor = self.query_one(TextArea)
                editor.add_class("input-invalid")
                error = self.query_one(".modal-validation-error", Static)
                error.update(tr(str(exc)))
                error.add_class("visible")
                editor.focus()
                return
        self.dismiss(value)


def _modal_actions(apply_label: str) -> ActionList:
    return ActionList(
        ActionItem("apply", apply_label, classes="action-primary"),
        ActionItem("cancel", "Cancel"),
        classes="modal-action-list",
    )
