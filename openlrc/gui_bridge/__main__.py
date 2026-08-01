"""Command-line entry point for the development desktop sidecar."""

from __future__ import annotations

import argparse

from openlrc.gui_bridge.server import serve


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m openlrc.gui_bridge")
    parser.add_argument("command", choices=("serve",))
    parser.add_argument("--stdio", action="store_true", help="Use stdin/stdout JSON Lines transport.")
    args = parser.parse_args()
    if args.command != "serve" or not args.stdio:
        parser.error("GUI bridge v1 requires `serve --stdio`.")
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
