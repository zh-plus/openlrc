# OpenLRC Desktop GUI V1

This directory contains the Electron + React desktop interface for OpenLRC. V1 is a development-mode GUI: it is not a signed application bundle, installer, updater, or Windows/Linux release.

## Runtime baseline

- Node.js `24.18.0` and npm `11.16.0` (see `../.node-version` and `packageManager`)
- Electron `43.2.0`
- React / React DOM `19.2.8`
- React Aria Components `1.20.0`
- Tailwind CSS `4.3.3` with `tailwindcss-react-aria-components 2.2.0`
- TypeScript `6.0.3`
- Python `>=3.11,<3.15`, launched through the repository's existing `uv` environment

All JavaScript versions are exact in `package.json` and `package-lock.json`. Direct packages track the latest compatible stable release selected on 2026-08-01. Three compatibility pins are intentional:

- `@electron/fuses@1.8.0` satisfies Electron Forge 7's `^1` peer contract;
- `@wdio/electron-service@10.0.0` is the newest release that imports correctly with its published native-utils dependency (10.1.0 currently fails during module initialization);
- TypeScript 6 is the newest release accepted by the current `typescript-eslint` peer range.

## Development

From this directory:

```shell
npm ci
npm start
```

Electron starts the Python sidecar as:

```shell
uv run --no-sync python -m openlrc.gui_bridge serve --stdio
```

The root Python environment must already be prepared with `uv sync`. GUI startup uses
`--no-sync` deliberately so opening the application never resolves, downloads, or mutates
Python dependencies. The Renderer never receives Node.js, filesystem, shell, or subprocess
access. It can only call the narrow, validated API exposed by `src/preload/index.ts`.

Interactive controls use React Aria Components behind the OpenLRC `renderer/components/ui` design-system boundary. Tailwind handles ordinary layout, spacing, sizing, color, typography, and responsive rules; scoped custom CSS handles Electron shell behavior, task progress, control indicators, and overlay animation.

React Aria `data-hovered`, `data-pressed`, `data-focused`, and `data-selected` states drive visible interaction feedback. Standard buttons change brightness/surface without translating or scaling their layout box. Theme controls and the variable-width History filter use the same moving selection-indicator language, selected Settings tabs keep their primary treatment while focused, the 42×24 switch keeps its 18px thumb geometrically centered, and the 29px Electron drag strip spans the full window width rather than only the Sidebar.

## Validation

```shell
npm run verify
npm run test:e2e
```

`verify` runs TypeScript, ESLint, Prettier, and Vitest. `test:e2e` compiles the development bundles and drives the real Electron window with WebDriverIO.

The main process also supports an opt-in real-interface smoke run. It captures routes and both themes, checks the Renderer sandbox, executes two short CPU transcription jobs sequentially, exercises pending/active cancellation, verifies that user-cancelled work stays out of History, and switches English/简体中文 through Settings:

```shell
OPENLRC_E2E_OUTPUT_DIR=/private/tmp/openlrc-gui-smoke \
OPENLRC_APP_SUPPORT_DIR=/private/tmp/openlrc-gui-smoke/app-support \
OPENLRC_GUI_REAL_WORKFLOW_INPUT=/absolute/path/to/short.wav \
npm start
```

The JSON result is written to `$OPENLRC_E2E_OUTPUT_DIR/report.json` with PNG evidence beside it.
If another Forge development window already owns ports 3000/9000, set the task-specific
`OPENLRC_FORGE_PORT` and `OPENLRC_FORGE_LOGGER_PORT` variables to unused ports for the smoke run;
normal `npm start` behavior is unchanged.

Python protocol and queue tests are run from the repository root:

```shell
UV_CACHE_DIR=/private/tmp/openlrc-gui-uv-cache \
uv run --with pytest python -m pytest -q tests/test_gui_queue.py tests/test_gui_bridge.py
```

## Security-audit boundary

`npm audit --omit=dev` currently reports the React Router RSC-mode CSRF advisory. This GUI uses a local declarative Hash Router and has no React Server Components, server actions, HTTP application server, cookies, or cross-origin action endpoint, so the vulnerable path is not present. The latest React Router release remains pinned instead of downgrading to versions with older advisories.

The full audit also reports unresolved transitive advisories in Electron Forge's development/build dependencies. V1 neither packages untrusted archives nor ships the development server. Do not apply `npm audit fix --force` or major-version overrides without revalidating Forge and WebDriverIO compatibility.

## V1 scope

Implemented routes are `#/home`, `#/new`, `#/tasks`, `#/history`, `#/history/:jobId`, `#/resources`, `#/settings`, and `#/about`. The conditional Editor route is intentionally absent until the Python handshake advertises an editing capability.

History exposes only `All`, `Completed`, and `Failed` filters. `Failed` groups completed-with-warnings, failed, and interrupted records while preserving each row's exact status label. User-cancelled work is excluded from `jobs.list`, so it appears in neither History nor Home's recent-history summary.

Packaging, code signing, notarization, installers, auto-update, frozen Python, and Windows/Linux release validation are post-V1 work.
