import path from "node:path";
import { mkdirSync } from "node:fs";

import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  nativeTheme,
  screen,
  session,
  shell,
  type IpcMainInvokeEvent,
} from "electron";
import log from "electron-log/main.js";
import { z } from "zod";

import { DesktopStateStore } from "./DesktopStateStore.js";
import { runDevelopmentSmoke } from "./DevelopmentSmoke.js";
import { SidecarSupervisor } from "./sidecar/SidecarSupervisor.js";
import {
  appSettingsSchema,
  appearanceStateSchema,
  backendStatusSchema,
  channels,
  credentialStatusSchema,
  jobDetailSchema,
  jobSummarySchema,
  preflightReportSchema,
  queueEntrySchema,
  queueSnapshotSchema,
  resourceStatusSchema,
  themeSourceSchema,
  workflowDraftSchema,
  workflowEventSchema,
  workflowKindSchema,
} from "../shared/contracts.js";

app.setName("OpenLRC Desktop");
log.initialize();

const smokeOutputDirectory = process.env.OPENLRC_E2E_OUTPUT_DIR;
if (smokeOutputDirectory) {
  mkdirSync(smokeOutputDirectory, { recursive: true });
  app.setPath("userData", path.join(smokeOutputDirectory, "electron-user-data"));
}

const hasSingleInstanceLock = app.requestSingleInstanceLock();
if (!hasSingleInstanceLock) app.quit();

let mainWindow: BrowserWindow | null = null;
let sidecar: SidecarSupervisor | null = null;
let desktopState: DesktopStateStore | null = null;
let quitting = false;
let boundsTimer: NodeJS.Timeout | null = null;

app.on("second-instance", () => {
  if (!mainWindow) return;
  if (mainWindow.isMinimized()) mainWindow.restore();
  mainWindow.show();
  mainWindow.focus();
});

app.whenReady().then(async () => {
  desktopState = new DesktopStateStore(path.join(app.getPath("userData"), "desktop-state.json"));
  await desktopState.load();
  const repositoryRoot = process.env.OPENLRC_REPO_ROOT ?? path.resolve(app.getAppPath(), "..");
  sidecar = new SidecarSupervisor(repositoryRoot, process.env.OPENLRC_UV_PATH ?? "uv");
  registerSecurityPolicy();
  registerIpcHandlers();
  createWindow();
  bindSidecarEvents();
  void sidecar.start().catch((error: unknown) => log.error("Unable to start local service", error));

  nativeTheme.on("updated", sendAppearance);
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
    else mainWindow?.show();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", (event) => {
  if (quitting) return;
  event.preventDefault();
  quitting = true;
  void (sidecar?.stop() ?? Promise.resolve()).finally(() => app.quit());
});

function createWindow(): void {
  if (!desktopState) throw new Error("Desktop state was not initialized.");
  const restored = visibleBounds(desktopState.bounds());
  const effectiveTheme = desktopState.appearance().effectiveTheme;
  mainWindow = new BrowserWindow({
    width: restored?.width ?? 1280,
    height: restored?.height ?? 800,
    ...(restored ? { x: restored.x, y: restored.y } : {}),
    minWidth: 960,
    minHeight: 680,
    show: false,
    backgroundColor: effectiveTheme === "dark" ? "#242424" : "#F6F8FB",
    titleBarStyle: "hiddenInset",
    trafficLightPosition: { x: 18, y: 18 },
    webPreferences: {
      preload: MAIN_WINDOW_PRELOAD_WEBPACK_ENTRY,
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      webviewTag: false,
      backgroundThrottling: !smokeOutputDirectory,
    },
  });
  const window = mainWindow;
  window.once("ready-to-show", () => {
    window.show();
    if (smokeOutputDirectory) {
      void runDevelopmentSmoke(window, smokeOutputDirectory)
        .catch((error: unknown) => log.error("Development GUI smoke failed", error))
        .finally(() => app.quit());
    }
  });
  window.on("closed", () => {
    if (mainWindow === window) mainWindow = null;
  });
  window.on("resize", saveWindowBoundsSoon);
  window.on("move", saveWindowBoundsSoon);
  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  window.webContents.on("console-message", (details) => {
    const write = details.level === "error" ? log.error : details.level === "warning" ? log.warn : log.info;
    write(`[renderer] ${details.message}`, `${details.sourceId}:${details.lineNumber}`);
  });
  window.webContents.on("did-fail-load", (_event, code, description, url) => {
    log.error("Renderer failed to load", { code, description, url });
  });
  window.webContents.on("render-process-gone", (_event, details) => {
    log.error("Renderer process exited", details);
  });
  window.webContents.on("will-navigate", (event, url) => {
    const current = window.webContents.getURL();
    if (current && new URL(url).origin !== new URL(current).origin) event.preventDefault();
  });
  window.webContents.on("will-attach-webview", (event) => event.preventDefault());
  void window.loadURL(MAIN_WINDOW_WEBPACK_ENTRY);
}

function registerSecurityPolicy(): void {
  session.defaultSession.setPermissionRequestHandler((_webContents, _permission, callback) =>
    callback(false),
  );
  session.defaultSession.setPermissionCheckHandler(() => false);
}

function registerIpcHandlers(): void {
  register(channels.appInfo, async () => ({
    name: app.getName(),
    version: app.getVersion(),
    platform: process.platform,
    protocol: 1,
    diagnosticPath: app.getPath("logs"),
  }));
  register(channels.backendStatus, async () => backendStatusSchema.parse(requireSidecar().status));
  register(channels.backendRestart, async () => {
    await requireSidecar().restart();
    return backendStatusSchema.parse(requireSidecar().status);
  });
  register(channels.appOpenExternal, async (raw) => {
    const { url } = z.object({ url: z.string().url().max(2048) }).parse(raw);
    const parsed = new URL(url);
    if (parsed.protocol !== "https:") throw new Error("Only HTTPS links can be opened externally.");
    await shell.openExternal(parsed.toString());
  });
  register(channels.appearanceGet, async () =>
    appearanceStateSchema.parse(requireDesktopState().appearance()),
  );
  register(channels.appearanceSetTheme, async (raw) => {
    const { source } = z.object({ source: themeSourceSchema }).parse(raw);
    const appearance = await requireDesktopState().setTheme(source);
    sendAppearance();
    return appearance;
  });
  register(channels.appearanceSetSidebar, async (raw) => {
    const { collapsed } = z.object({ collapsed: z.boolean() }).parse(raw);
    const appearance = await requireDesktopState().setSidebarCollapsed(collapsed);
    sendAppearance();
    return appearance;
  });
  register(channels.dialogSelectInputs, async (raw) => {
    const { kind } = z.object({ kind: workflowKindSchema }).parse(raw);
    const result = await dialog.showOpenDialog(requireWindow(), {
      properties: ["openFile", "multiSelections"],
      filters: inputFilters(kind),
    });
    return { paths: result.filePaths, cancelled: result.canceled };
  });
  register(channels.workflowPreflight, async (raw) => {
    const { draft } = z.object({ draft: workflowDraftSchema }).parse(raw);
    return preflightReportSchema.parse(await requireSidecar().request("workflow.preflight", { draft }));
  });
  register(channels.workflowActive, async () => requireSidecar().request("operation.active"));
  register(channels.queueSnapshot, async () =>
    queueSnapshotSchema.parse(await requireSidecar().request("queue.snapshot")),
  );
  register(channels.queueEnqueue, async (raw) => {
    const { draft } = z.object({ draft: workflowDraftSchema }).parse(raw);
    const result = await requireSidecar().request("queue.enqueue", { draft });
    return z.object({ queue_id: z.string(), entry: queueEntrySchema }).parse(result);
  });
  register(channels.queueCancel, async (raw) => {
    const { queueId } = z.object({ queueId: z.string().min(1).max(128) }).parse(raw);
    return z
      .object({ queue_id: z.string(), cancelled: z.boolean(), removed: z.boolean() })
      .parse(await requireSidecar().request("queue.cancel", { queue_id: queueId }));
  });
  register(channels.queueReorder, async (raw) => {
    const { orderedQueueIds } = z.object({ orderedQueueIds: z.array(z.string()).max(50) }).parse(raw);
    return queueSnapshotSchema.parse(
      await requireSidecar().request("queue.reorder", { ordered_queue_ids: orderedQueueIds }),
    );
  });
  register(channels.queuePause, async () =>
    queueSnapshotSchema.parse(await requireSidecar().request("queue.pause")),
  );
  register(channels.queueResume, async () =>
    queueSnapshotSchema.parse(await requireSidecar().request("queue.resume")),
  );
  register(channels.jobsList, async () =>
    z.array(jobSummarySchema).parse(await requireSidecar().request("jobs.list")),
  );
  register(channels.jobsGet, async (raw) => {
    const { jobId } = idPayload(raw, "jobId");
    return jobDetailSchema.parse(await requireSidecar().request("jobs.get", { job_id: jobId }));
  });
  register(channels.jobsDelete, async (raw) => {
    const { jobId } = idPayload(raw, "jobId");
    await requireSidecar().request("jobs.delete", { job_id: jobId });
  });
  register(channels.jobsResumeDraft, async (raw) => {
    const { jobId } = idPayload(raw, "jobId");
    return workflowDraftSchema.parse(await requireSidecar().request("jobs.resume_draft", { job_id: jobId }));
  });
  register(channels.resourcesStatus, async () =>
    z.array(resourceStatusSchema).parse(await requireSidecar().request("resources.status", {}, 40_000)),
  );
  register(channels.resourcesRefresh, async () =>
    z.array(resourceStatusSchema).parse(await requireSidecar().request("resources.status", {}, 40_000)),
  );
  register(channels.settingsGet, async () =>
    appSettingsSchema.parse(await requireSidecar().request("settings.get")),
  );
  register(channels.settingsUpdate, async (raw) => {
    const { patch } = z.object({ patch: z.record(z.string(), z.unknown()) }).parse(raw);
    return appSettingsSchema.parse(await requireSidecar().request("settings.update", { patch }));
  });
  register(channels.credentialsStatus, async (raw) => {
    const provider = providerPayload(raw);
    return credentialStatusSchema.parse(await requireSidecar().request("credentials.status", { provider }));
  });
  register(channels.credentialsSet, async (raw) => {
    const { provider, secret } = z
      .object({ provider: providerSchema, secret: z.string().min(1).max(16_384) })
      .parse(raw);
    return credentialStatusSchema.parse(
      await requireSidecar().request("credentials.set", { provider, secret }),
    );
  });
  register(channels.credentialsDelete, async (raw) => {
    const provider = providerPayload(raw);
    await requireSidecar().request("credentials.delete", { provider });
  });
  register(channels.artifactReveal, async (raw) => resolveArtifact(raw, true));
  register(channels.artifactOpen, async (raw) => resolveArtifact(raw, false));
}

function bindSidecarEvents(): void {
  const supervisor = requireSidecar();
  supervisor.on("status", (status) => mainWindow?.webContents.send(channels.backendChanged, status));
  supervisor.on("event", (frame: unknown) => {
    if (typeof frame !== "object" || frame === null || !("event" in frame)) return;
    if (frame.event === "queue.changed" && "payload" in frame) {
      const snapshot = queueSnapshotSchema.safeParse(frame.payload);
      if (snapshot.success) mainWindow?.webContents.send(channels.queueChanged, snapshot.data);
      return;
    }
    const workflowEvent = workflowEventSchema.safeParse(frame);
    if (workflowEvent.success) mainWindow?.webContents.send(channels.workflowEvent, workflowEvent.data);
  });
}

function register(channel: string, handler: (payload?: unknown) => Promise<unknown>): void {
  ipcMain.handle(channel, async (event: IpcMainInvokeEvent, payload?: unknown) => {
    assertTrustedSender(event);
    return handler(payload);
  });
}

function assertTrustedSender(event: IpcMainInvokeEvent): void {
  if (
    !mainWindow ||
    event.sender !== mainWindow.webContents ||
    event.senderFrame !== mainWindow.webContents.mainFrame
  ) {
    throw new Error("Rejected IPC from an untrusted sender.");
  }
}

async function resolveArtifact(raw: unknown, reveal: boolean): Promise<void> {
  const { jobId, artifactId } = z
    .object({ jobId: z.string().min(1).max(128), artifactId: z.string().min(1).max(128) })
    .parse(raw);
  const result = z
    .object({ artifact_id: z.string(), path: z.string() })
    .parse(await requireSidecar().request("artifacts.resolve", { job_id: jobId, artifact_id: artifactId }));
  if (reveal) shell.showItemInFolder(result.path);
  else {
    const error = await shell.openPath(result.path);
    if (error) throw new Error(error);
  }
}

const providerSchema = z.enum(["openai", "anthropic", "google", "litellm", "third_party"]);

function providerPayload(raw: unknown): z.infer<typeof providerSchema> {
  return z.object({ provider: providerSchema }).parse(raw).provider;
}

function idPayload<Key extends string>(raw: unknown, key: Key): Record<Key, string> {
  return z.object({ [key]: z.string().min(1).max(128) }).parse(raw) as Record<Key, string>;
}

function inputFilters(kind: z.infer<typeof workflowKindSchema>): Electron.FileFilter[] {
  if (kind === "translate") return [{ name: "OpenLRC transcription JSON", extensions: ["json"] }];
  return [
    {
      name: "Audio and video",
      extensions: ["wav", "mp3", "m4a", "flac", "ogg", "aac", "mp4", "mov", "mkv", "webm", "avi"],
    },
  ];
}

function visibleBounds(bounds: Electron.Rectangle | null): Electron.Rectangle | null {
  if (!bounds) return null;
  const workArea = screen.getDisplayMatching(bounds).workArea;
  const overlapWidth = Math.max(
    0,
    Math.min(bounds.x + bounds.width, workArea.x + workArea.width) - Math.max(bounds.x, workArea.x),
  );
  const overlapHeight = Math.max(
    0,
    Math.min(bounds.y + bounds.height, workArea.y + workArea.height) - Math.max(bounds.y, workArea.y),
  );
  return overlapWidth >= 160 && overlapHeight >= 120 ? bounds : null;
}

function saveWindowBoundsSoon(): void {
  if (boundsTimer) clearTimeout(boundsTimer);
  boundsTimer = setTimeout(() => {
    if (mainWindow && desktopState && !mainWindow.isMinimized() && !mainWindow.isFullScreen()) {
      void desktopState.setBounds(mainWindow.getBounds());
    }
  }, 250);
}

function sendAppearance(): void {
  if (!desktopState) return;
  mainWindow?.webContents.send(channels.appearanceChanged, desktopState.appearance());
}

function requireWindow(): BrowserWindow {
  if (!mainWindow) throw new Error("Application window is unavailable.");
  return mainWindow;
}

function requireSidecar(): SidecarSupervisor {
  if (!sidecar) throw new Error("Local service supervisor is unavailable.");
  return sidecar;
}

function requireDesktopState(): DesktopStateStore {
  if (!desktopState) throw new Error("Desktop appearance state is unavailable.");
  return desktopState;
}
