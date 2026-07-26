"""Home card presentation built on the shared action row contract."""

from openlrc.tui.i18n import tr
from openlrc.tui.navigation import ActionItem


class HomeCard(ActionItem):
    def __init__(self, action_id: str, label: str, detail: str = "", *, disabled: bool = False) -> None:
        classes = "home-card home-card-disabled" if disabled else "home-card"
        super().__init__(action_id, tr(label).upper(), detail, disabled=disabled, classes=classes)
