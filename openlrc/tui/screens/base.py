"""Shared page behavior for the v2 page stack."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from textual.binding import Binding
from textual.screen import Screen

from openlrc.tui.navigation import ActionItem, ActionList

if TYPE_CHECKING:
    from openlrc.tui.app import OpenLRCTUI


class OpenLRCScreen(Screen):
    BINDINGS = [Binding("escape", "back", "Back", show=False), Binding("question_mark", "help", "Help", show=False)]

    def action_back(self) -> None:
        self.app.go_back()

    def action_help(self) -> None:
        self.app.action_help()

    def on_screen_resume(self) -> None:
        self.focus_default_action_list()

    def focus_default_action_list(self) -> None:
        action_list = next(iter(self.query(ActionList)), None)
        if action_list is not None:
            self.call_after_refresh(lambda: self.set_focus(action_list))

    def focus_action(self, action_id: str | None = None) -> None:
        """Focus a stable action after a modal closes or a screen recomposes."""

        action_list = next(iter(self.query(ActionList)), None)
        if action_list is None:
            return
        target_index = next(
            (
                index
                for index, child in enumerate(action_list.children)
                if isinstance(child, ActionItem) and not child.disabled and child.action_id == action_id
            ),
            None,
        )
        if target_index is None:
            action_list.action_first_enabled()
        else:
            action_list.index = target_index
        self.set_focus(action_list)

    def recompose_preserving_action(self, action_id: str | None = None) -> None:
        if action_id is None:
            action_list = next(iter(self.query(ActionList)), None)
            action_id = action_list.selected_action() if action_list is not None else None
        self.refresh(recompose=True)
        self.call_after_refresh(self._focus_after_recompose, action_id)

    def _focus_after_recompose(self, action_id: str | None) -> None:
        # Recompose unmounts the old list after the first refresh callback. Defer
        # once more so focus can only land on the newly mounted ActionList.
        self.call_later(self.focus_action, action_id)

    if TYPE_CHECKING:

        @property
        def app(self) -> OpenLRCTUI:
            return cast("OpenLRCTUI", super().app)
