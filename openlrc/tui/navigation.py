"""Stable action IDs and public list navigation helpers."""

from __future__ import annotations

from collections.abc import Sequence

from rich.text import Text
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import ListItem, ListView, Static

from openlrc.tui.i18n import tr


class ActionItem(ListItem):
    """One label/value row that carries a stable application action ID."""

    def __init__(
        self, action_id: str, label: str | Text, detail: str = "", *, disabled: bool = False, classes: str = ""
    ) -> None:
        label = label if isinstance(label, Text) else tr(label)
        detail = tr(detail)
        self.action_id = action_id
        self.label_text = label.plain if isinstance(label, Text) else label
        self.detail_text = detail
        detail_class = "action-has-detail" if detail else "action-no-detail"
        item_classes = f"{classes} {detail_class}".strip()
        content = Vertical(
            Static(label, classes="action-label", markup=False),
            Static(detail, classes="action-detail", markup=False),
            classes="action-content",
        )
        super().__init__(content, disabled=disabled, classes=item_classes)

    def set_detail(self, detail: str) -> None:
        detail = tr(detail)
        self.detail_text = detail
        self.set_class(bool(detail), "action-has-detail")
        self.set_class(not detail, "action-no-detail")
        self.query_one(".action-detail", Static).update(detail)


class ActionGroupHeading(ListItem):
    """A visible group boundary that never enters the focus sequence."""

    def __init__(self, title: str) -> None:
        super().__init__(
            Static(tr(title).upper(), classes="action-group-title", markup=False),
            disabled=True,
            classes="action-group-heading",
        )


class ActionGroupSpacer(ListItem):
    """One inert row that separates content without breaking a group frame."""

    def __init__(self) -> None:
        super().__init__(Static("", markup=False), disabled=True, classes="action-group-spacer")


def action_group(title: str, *items: ActionItem) -> list[ListItem]:
    """Return one non-focusable heading plus rows in a shared visual frame."""

    if not items:
        return []
    for item in items:
        item.add_class("action-group-row")
    items[-1].add_class("action-group-last")
    group: list[ListItem] = [ActionGroupHeading(title), ActionGroupSpacer()]
    for index, item in enumerate(items):
        if index:
            group.append(ActionGroupSpacer())
        group.append(item)
    return group


class ActionList(ListView):
    """A one-dimensional keyboard and mouse action surface."""

    def __init__(
        self,
        *children: ListItem,
        initial_index: int | None = 0,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        for child in reversed(children):
            if isinstance(child, ActionItem):
                if child.has_class("action-group-last"):
                    child.add_class("action-list-terminal")
                break
        super().__init__(*children, initial_index=initial_index, name=name, id=id, classes=classes, disabled=disabled)

    class Activated(Message):
        def __init__(self, action_id: str, item: ActionItem) -> None:
            super().__init__()
            self.action_id = action_id
            self.item = item

    BINDINGS = [
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("space", "select_cursor", "Open", show=False),
        Binding("home", "first_enabled", "First", show=False),
        Binding("end", "last_enabled", "Last", show=False),
    ]

    def on_mount(self) -> None:
        if self.index not in self._enabled_indices():
            self.action_first_enabled()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, ActionItem) and not event.item.disabled:
            self.post_message(self.Activated(event.item.action_id, event.item))

    def selected_action(self) -> str | None:
        item = self.highlighted_child
        return item.action_id if isinstance(item, ActionItem) and not item.disabled else None

    def action_cursor_down(self) -> None:
        self._move_enabled(1)

    def action_cursor_up(self) -> None:
        self._move_enabled(-1)

    def action_first_enabled(self) -> None:
        indices = self._enabled_indices()
        self.index = indices[0] if indices else None

    def action_last_enabled(self) -> None:
        indices = self._enabled_indices()
        self.index = indices[-1] if indices else None

    def _move_enabled(self, offset: int) -> None:
        indices = self._enabled_indices()
        if not indices:
            self.index = None
            return
        current_index = self.index
        if current_index is None:
            self.index = indices[0] if offset > 0 else indices[-1]
            return
        try:
            position = indices.index(current_index)
        except ValueError:
            self.index = indices[0] if offset > 0 else indices[-1]
            return
        self.index = indices[(position + offset) % len(indices)]

    def _enabled_indices(self) -> list[int]:
        children: Sequence[object] = self.children
        return [index for index, child in enumerate(children) if isinstance(child, ActionItem) and not child.disabled]
