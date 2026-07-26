"""Consistent headers and context-specific keyboard hints."""

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from openlrc.tui.i18n import tr


class PageHeader(Horizontal):
    def __init__(self, title: str, context: str = "", *, id: str | None = None) -> None:
        super().__init__(id=id, classes="page-header")
        self.title = title
        self.context = context

    def compose(self) -> ComposeResult:
        yield Static(tr(self.title), classes="page-title", markup=False)
        yield Static(tr(self.context), classes="page-context", markup=False)


class PageFooter(Static):
    def __init__(self, hints: str, *, id: str | None = None) -> None:
        super().__init__(tr(hints), id=id, classes="page-footer", markup=False)


class WorkflowSteps(Static):
    """Non-focusable three-step New Work progress."""

    LABELS = ("Choose work", "Configure", "Preflight")

    def __init__(self, current_step: int) -> None:
        self.current_step = max(1, min(current_step, len(self.LABELS)))
        super().__init__(self._render_steps(), classes="workflow-steps", markup=False)

    def _render_steps(self) -> Text:
        text = Text()
        for index, label in enumerate(self.LABELS, start=1):
            if index > 1:
                text.append("  ───  ", style="#35414d")
            style = (
                "#f0aa4b bold"
                if index == self.current_step
                else "#48c6dc"
                if index < self.current_step
                else "#71808d"
            )
            text.append(f"[{index} {tr(label)}]", style=style)
        return text
