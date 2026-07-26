"""Field rows use the same stable action contract as navigation rows."""

from openlrc.tui.navigation import ActionItem, ActionList

FieldItem = ActionItem
FieldList = ActionList

__all__ = ("FieldItem", "FieldList")
