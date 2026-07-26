"""Structured local runtime diagnostics."""

from textual.app import ComposeResult

from openlrc.tui.i18n import tr
from openlrc.tui.modals import DetailModal
from openlrc.tui.navigation import ActionItem, ActionList, action_group
from openlrc.tui.screens.base import OpenLRCScreen
from openlrc.tui.screens.models import ModelsScreen
from openlrc.tui.widgets import PageFooter, PageHeader


class DoctorScreen(OpenLRCScreen):
    def compose(self) -> ComposeResult:
        statuses = self.app.doctor_statuses()
        items = [
            ActionItem(
                f"check:{index}",
                f"{tr('OK') if status.available else tr('CHECK')}  {status.name}",
                status.detail,
            )
            for index, status in enumerate(statuses)
        ]
        yield PageHeader("Doctor", f"{sum(status.available for status in statuses)}/{len(statuses)} checks pass")
        yield ActionList(
            *action_group("Checks", *items),
            *action_group(
                "Actions",
                ActionItem("refresh", "Run checks again", "Refresh diagnostics"),
                ActionItem("setup", "Open suggested setup", "Models & Setup"),
            ),
            id="doctor-checks",
            classes="page-list",
        )
        yield PageFooter("[ENTER] Details/Action   [ESC] Back")

    def on_mount(self) -> None:
        self.call_after_refresh(self.query_one(ActionList).focus)

    def on_action_list_activated(self, event: ActionList.Activated) -> None:
        if event.action_id == "refresh":
            self.app.clear_resource_cache()
            self.refresh(recompose=True)
        elif event.action_id == "setup":
            self.app.push_screen(ModelsScreen())
        else:
            index = int(event.action_id.partition(":")[2])
            status = self.app.doctor_statuses()[index]
            self.app.push_screen(DetailModal(status.name, f"{status.detail}\n\n{status.hint}"))
