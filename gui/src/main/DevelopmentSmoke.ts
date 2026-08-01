import { promises as fs } from "node:fs";
import path from "node:path";

import type { BrowserWindow } from "electron";

interface SmokeReport {
  passed: boolean;
  title: string;
  viewport: { width: number; height: number };
  security: { bridge: boolean; nodeRequire: string; nodeProcess: string };
  backend: { state: string; serviceVersion: string | null };
  navigation: string[];
  routes: Record<string, string>;
  appearance: {
    lightCanvas: string;
    lightText: string;
    darkCanvas: string;
    darkText: string;
    expandedSidebar: number;
    collapsedSidebar: number;
    routePreserved: boolean;
  };
  snapshots: string[];
  localization?: {
    language: string;
    navigation: string[];
    title: string;
    restored: boolean;
  };
  realWorkflow?: {
    inputPaths: string[];
    preflight: string[];
    queuedIds: string[];
    startedIds: string[];
    terminalStatuses: string[];
    eventOrder: string[];
    outputPaths: string[];
    queueDrained: boolean;
    strictlySerial: boolean;
    pendingCancelRemoved: boolean;
    activeCancelAccepted: boolean;
    activeCancelHidden: boolean;
  };
  renderer?: {
    url: string;
    readyState: string;
    bridge: string;
    rootText: string;
    bodyText: string;
  };
  error?: string;
}

export async function runDevelopmentSmoke(window: BrowserWindow, outputDirectory: string): Promise<void> {
  const report: SmokeReport = {
    passed: false,
    title: "",
    viewport: { width: 0, height: 0 },
    security: { bridge: false, nodeRequire: "unknown", nodeProcess: "unknown" },
    backend: { state: "unknown", serviceVersion: null },
    navigation: [],
    routes: {},
    appearance: {
      lightCanvas: "",
      lightText: "",
      darkCanvas: "",
      darkText: "",
      expandedSidebar: 0,
      collapsedSidebar: 0,
      routePreserved: false,
    },
    snapshots: [],
  };
  try {
    await waitFor(window, "document.querySelector('h1')?.textContent === 'Home'");
    const initial = await execute<{
      title: string;
      width: number;
      height: number;
      bridge: boolean;
      nodeRequire: string;
      nodeProcess: string;
      navigation: string[];
      sidebarWidth: number;
    }>(
      window,
      `({
        title: document.title,
        width: innerWidth,
        height: innerHeight,
        bridge: typeof window.openlrc === "object",
        nodeRequire: typeof window.require,
        nodeProcess: typeof window.process,
        navigation: Array.from(document.querySelectorAll(".primary-nav .nav-item")).map((item) => item.textContent?.trim() ?? ""),
        sidebarWidth: document.querySelector(".sidebar")?.getBoundingClientRect().width ?? 0,
      })`,
    );
    report.title = initial.title;
    report.viewport = { width: initial.width, height: initial.height };
    report.security = {
      bridge: initial.bridge,
      nodeRequire: initial.nodeRequire,
      nodeProcess: initial.nodeProcess,
    };
    report.navigation = initial.navigation;
    report.appearance.expandedSidebar = initial.sidebarWidth;
    await waitForResult(
      window,
      `window.openlrc.app.backendStatus().then((status) => status.state === "ready")`,
      45_000,
    );
    report.backend = await execute(
      window,
      `window.openlrc.app.backendStatus().then((status) => ({ state: status.state, serviceVersion: status.serviceVersion }))`,
    );
    if (report.backend.state !== "ready")
      throw new Error(`Backend did not become ready: ${report.backend.state}`);
    await execute(
      window,
      `Promise.all([window.openlrc.queue.snapshot(), window.openlrc.jobs.list(), window.openlrc.settings.get()]).then(([queue, jobs, settings]) => ({ queue, jobs, language: settings.general.language }))`,
    );
    await execute(window, `window.openlrc.appearance.setTheme("light")`);
    await waitFor(window, "document.documentElement.dataset.theme === 'light'");
    const light = await execute<{ canvas: string; text: string }>(
      window,
      `({ canvas: getComputedStyle(document.documentElement).getPropertyValue("--color-canvas").trim(), text: getComputedStyle(document.documentElement).getPropertyValue("--color-text").trim() })`,
    );
    report.appearance.lightCanvas = light.canvas;
    report.appearance.lightText = light.text;
    await capture(window, outputDirectory, "home-light.png", report);

    await execute(window, `window.openlrc.appearance.setTheme("dark")`);
    await waitFor(window, "document.documentElement.dataset.theme === 'dark'");
    const dark = await execute<{ canvas: string; text: string }>(
      window,
      `({ canvas: getComputedStyle(document.documentElement).getPropertyValue("--color-canvas").trim(), text: getComputedStyle(document.documentElement).getPropertyValue("--color-text").trim() })`,
    );
    report.appearance.darkCanvas = dark.canvas;
    report.appearance.darkText = dark.text;
    await capture(window, outputDirectory, "home-dark.png", report);

    await navigate(window, "/tasks", "Tasks", report);
    await capture(window, outputDirectory, "tasks-dark.png", report);
    await execute(window, `document.querySelector('button[aria-label="Collapse sidebar"]')?.click(); true`);
    await waitFor(window, "document.querySelector('.sidebar')?.getBoundingClientRect().width === 80");
    report.appearance.collapsedSidebar = await execute<number>(
      window,
      `document.querySelector(".sidebar")?.getBoundingClientRect().width ?? 0`,
    );
    report.appearance.routePreserved = await execute<boolean>(window, `location.hash === "#/tasks"`);
    await capture(window, outputDirectory, "tasks-collapsed-dark.png", report);
    await execute(window, `window.openlrc.appearance.setSidebarCollapsed(false)`);
    await waitFor(window, "document.querySelector('.sidebar')?.getBoundingClientRect().width === 224");

    await navigate(window, "/new", "New Task", report);
    await capture(window, outputDirectory, "new-task-dark.png", report);
    await navigate(window, "/history", "History", report);
    await capture(window, outputDirectory, "history-dark.png", report);
    await navigate(window, "/resources", "Resources", report, 45_000);
    await capture(window, outputDirectory, "resources-dark.png", report);
    await navigate(window, "/settings", "Settings", report);
    await capture(window, outputDirectory, "settings-dark.png", report);
    const reduceMotionWasEnabled = await execute<boolean>(
      window,
      `(() => { const field = document.querySelector("#setting-reduce-motion"); const input = field?.matches('[role="switch"]') ? field : field?.querySelector('[role="switch"]'); return input?.checked === true; })()`,
    );
    if (!reduceMotionWasEnabled) {
      await execute(
        window,
        `(() => { const field = document.querySelector("#setting-reduce-motion"); const input = field?.matches('[role="switch"]') ? field : field?.querySelector('[role="switch"]'); input?.click(); return true; })()`,
      );
      await waitFor(
        window,
        `(() => { const field = document.querySelector("#setting-reduce-motion"); const input = field?.matches('[role="switch"]') ? field : field?.querySelector('[role="switch"]'); return input?.checked === true; })()`,
      );
    }
    await capture(window, outputDirectory, "settings-switch-on-dark.png", report);
    if (!reduceMotionWasEnabled) {
      await execute(
        window,
        `Array.from(document.querySelectorAll("button")).find((button) => button.textContent?.trim() === "Cancel")?.click(); true`,
      );
      await waitFor(
        window,
        `(() => { const field = document.querySelector("#setting-reduce-motion"); const input = field?.matches('[role="switch"]') ? field : field?.querySelector('[role="switch"]'); return input?.checked === false; })()`,
      );
    }
    window.setSize(960, 680);
    await waitFor(window, "innerWidth === 960 && innerHeight === 680");
    await capture(window, outputDirectory, "settings-minimum-dark.png", report);
    window.setSize(1280, 800);
    await waitFor(window, "innerWidth === 1280 && innerHeight === 800");
    await navigate(window, "/about", "About OpenLRC", report);
    await execute(window, `window.openlrc.appearance.setSidebarCollapsed(false)`);
    await execute(window, `window.openlrc.appearance.setTheme("system")`);

    const realWorkflowInput = process.env.OPENLRC_GUI_REAL_WORKFLOW_INPUT;
    if (realWorkflowInput) {
      await runRealWorkflowSmoke(window, outputDirectory, realWorkflowInput, report);
    }
    await runLocalizationSmoke(window, outputDirectory, report);

    const requiredRoutes = ["/tasks", "/new", "/history", "/resources", "/settings", "/about"];
    report.passed =
      report.security.bridge &&
      report.security.nodeRequire === "undefined" &&
      report.security.nodeProcess === "undefined" &&
      report.backend.state === "ready" &&
      report.navigation.join("|") === "Home|New Task|Tasks|History" &&
      report.appearance.lightCanvas.toLowerCase() === "#f6f8fb" &&
      report.appearance.lightText.toLowerCase() === "#172033" &&
      report.appearance.darkCanvas.toLowerCase() === "#242424" &&
      report.appearance.darkText.toLowerCase() === "#ffffff" &&
      report.appearance.expandedSidebar === 224 &&
      report.appearance.collapsedSidebar === 80 &&
      report.appearance.routePreserved &&
      report.localization?.language === "zh-cn" &&
      report.localization.navigation.join("|") === "主页|新任务|任务列表|历史记录" &&
      report.localization.restored &&
      (!report.realWorkflow ||
        (report.realWorkflow.queueDrained &&
          report.realWorkflow.strictlySerial &&
          report.realWorkflow.pendingCancelRemoved &&
          report.realWorkflow.activeCancelAccepted &&
          report.realWorkflow.activeCancelHidden &&
          report.realWorkflow.terminalStatuses.every((status) => status.startsWith("succeeded")))) &&
      requiredRoutes.every((route) => Boolean(report.routes[route]));
    if (!report.passed) throw new Error("One or more GUI smoke assertions failed.");
  } catch (error) {
    const renderer = await execute<NonNullable<SmokeReport["renderer"]>>(
      window,
      `({
        url: location.href,
        readyState: document.readyState,
        bridge: typeof window.openlrc,
        rootText: document.querySelector("#root")?.textContent?.slice(0, 500) ?? "",
        bodyText: document.body?.innerText?.slice(0, 500) ?? "",
      })`,
    ).catch(() => undefined);
    if (renderer) report.renderer = renderer;
    await capture(window, outputDirectory, "failure.png", report).catch(() => undefined);
    report.error = error instanceof Error ? (error.stack ?? error.message) : String(error);
  } finally {
    await fs.mkdir(outputDirectory, { recursive: true });
    await fs.writeFile(path.join(outputDirectory, "report.json"), JSON.stringify(report, null, 2), "utf8");
  }
  if (!report.passed) throw new Error(report.error ?? "GUI smoke failed.");
}

async function runLocalizationSmoke(
  window: BrowserWindow,
  outputDirectory: string,
  report: SmokeReport,
): Promise<void> {
  await navigate(window, "/settings", "Settings", report, 15_000, false);
  await changeLanguageThroughSettings(window, "zh-cn", "Save changes");
  await waitFor(window, `document.documentElement.lang === "zh-cn"`);
  await navigate(window, "/home", "主页", report, 15_000, false);
  const localized = await execute<{ language: string; navigation: string[]; title: string }>(
    window,
    `({
      language: document.documentElement.lang,
      navigation: Array.from(document.querySelectorAll(".primary-nav .nav-item")).map((item) => item.textContent?.trim() ?? ""),
      title: document.querySelector("h1")?.textContent ?? "",
    })`,
  );
  await capture(window, outputDirectory, "home-zh-cn.png", report);
  await navigate(window, "/settings", "设置", report, 15_000, false);
  await changeLanguageThroughSettings(window, "en", "保存更改");
  await waitFor(window, `document.documentElement.lang === "en"`);
  report.localization = {
    ...localized,
    restored: await execute<boolean>(window, `document.documentElement.lang === "en"`),
  };
}

async function changeLanguageThroughSettings(
  window: BrowserWindow,
  language: "en" | "zh-cn",
  saveLabel: string,
): Promise<void> {
  await execute(
    window,
    `(() => {
      const select = document.querySelector('select[name="general.language"]');
      if (!(select instanceof HTMLSelectElement)) throw new Error("Language select was not found.");
      const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")?.set;
      setter?.call(select, ${JSON.stringify(language)});
      select.dispatchEvent(new Event("change", { bubbles: true }));
      return true;
    })()`,
  );
  await waitFor(
    window,
    `Array.from(document.querySelectorAll("button")).some((button) => button.textContent?.trim() === ${JSON.stringify(saveLabel)} && !button.disabled)`,
  );
  await execute(
    window,
    `(() => {
      const button = Array.from(document.querySelectorAll("button")).find((item) => item.textContent?.trim() === ${JSON.stringify(saveLabel)});
      if (!(button instanceof HTMLButtonElement)) throw new Error("Save button was not found.");
      button.click();
      return true;
    })()`,
  );
}

async function runRealWorkflowSmoke(
  window: BrowserWindow,
  outputDirectory: string,
  sourceInput: string,
  report: SmokeReport,
): Promise<void> {
  const inputDirectory = path.join(outputDirectory, "workflow-inputs");
  await fs.mkdir(inputDirectory, { recursive: true });
  const extension = path.extname(sourceInput) || ".wav";
  const inputPaths = [
    path.join(inputDirectory, `gui-smoke-first${extension}`),
    path.join(inputDirectory, `gui-smoke-second${extension}`),
  ];
  await Promise.all(inputPaths.map((target) => fs.copyFile(sourceInput, target)));

  const setup = await execute<{
    preflight: string[];
    queuedIds: string[];
  }>(
    window,
    `(async () => {
      window.__openlrcSmokeEvents = [];
      window.__openlrcSmokeUnsubscribe = window.openlrc.workflows.onEvent((event) => {
        window.__openlrcSmokeEvents.push(event);
      });
      await window.openlrc.queue.pause();
      const base = {
        task: "transcribe-json",
        workflow: "transcribe",
        source_language: "en",
        whisper_model: "base",
        skip_preprocess: false,
        whisper_use_gpu: false,
        whisper_flash_attn: false,
        translation_backend: "",
        mode: "fast",
        subtitle_optimization: "relaxed",
        clear_temp: true,
        clear_checkpoint: true,
      };
      const drafts = ${JSON.stringify(inputPaths)}.map((input) => ({ ...base, paths: [input] }));
      const preflight = await Promise.all(drafts.map((draft) => window.openlrc.workflows.preflight(draft)));
      if (preflight.some((item) => item.blocked)) {
        throw new Error("Real workflow preflight was blocked: " + JSON.stringify(preflight));
      }
      const queued = [];
      for (const draft of drafts) queued.push(await window.openlrc.queue.enqueue(draft));
      const paused = await window.openlrc.queue.snapshot();
      if (!paused.paused || paused.entries.map((item) => item.queue_id).join("|") !== queued.map((item) => item.queue_id).join("|")) {
        throw new Error("Paused queue did not preserve FIFO insertion order.");
      }
      await window.openlrc.queue.resume();
      return { preflight: preflight.map((item) => item.status), queuedIds: queued.map((item) => item.queue_id) };
    })()`,
  );

  await navigate(window, "/tasks", "Tasks", report);
  await waitForResult(
    window,
    `window.openlrc.queue.snapshot().then((queue) => Boolean(queue.active) && queue.entries.length === 1)`,
    30_000,
  );
  await capture(window, outputDirectory, "tasks-real-active-queued.png", report);
  await waitForResult(window, `window.openlrc.jobs.list().then((jobs) => jobs.length === 2)`, 300_000);

  const result = await execute<{
    jobs: Array<{ status: string; outputs: string[] }>;
    queueDrained: boolean;
    events: Array<{ event: string; queue_id: string }>;
  }>(
    window,
    `(async () => {
      const jobs = await window.openlrc.jobs.list();
      const queue = await window.openlrc.queue.snapshot();
      const events = window.__openlrcSmokeEvents ?? [];
      window.__openlrcSmokeUnsubscribe?.();
      return { jobs, queueDrained: queue.total === 0 && queue.active === null, events };
    })()`,
  );
  const eventOrder = result.events
    .filter((event) =>
      ["workflow.started", "workflow.completed", "workflow.failed", "workflow.cancelled"].includes(
        event.event,
      ),
    )
    .map((event) => `${event.event}:${event.queue_id}`);
  const firstStart = eventOrder.indexOf(`workflow.started:${setup.queuedIds[0]}`);
  const firstTerminal = eventOrder.findIndex(
    (item) => item !== `workflow.started:${setup.queuedIds[0]}` && item.endsWith(`:${setup.queuedIds[0]}`),
  );
  const secondStart = eventOrder.indexOf(`workflow.started:${setup.queuedIds[1]}`);
  const secondTerminal = eventOrder.findIndex(
    (item) => item !== `workflow.started:${setup.queuedIds[1]}` && item.endsWith(`:${setup.queuedIds[1]}`),
  );
  const outputPaths = result.jobs.flatMap((job) => job.outputs);
  await Promise.all(outputPaths.map((output) => fs.access(output)));
  const cancellation = await runRealCancellationSmoke(window, sourceInput, inputDirectory);
  report.realWorkflow = {
    inputPaths,
    preflight: setup.preflight,
    queuedIds: setup.queuedIds,
    startedIds: result.events
      .filter((event) => event.event === "workflow.started")
      .map((event) => event.queue_id),
    terminalStatuses: result.jobs.map((job) => job.status),
    eventOrder,
    outputPaths,
    queueDrained: result.queueDrained,
    strictlySerial:
      firstStart >= 0 &&
      firstTerminal > firstStart &&
      secondStart > firstTerminal &&
      secondTerminal > secondStart,
    ...cancellation,
  };
  await navigate(window, "/history", "History", report);
  await capture(window, outputDirectory, "history-real-completed.png", report);
}

async function runRealCancellationSmoke(
  window: BrowserWindow,
  sourceInput: string,
  inputDirectory: string,
): Promise<{
  pendingCancelRemoved: boolean;
  activeCancelAccepted: boolean;
  activeCancelHidden: boolean;
}> {
  const extension = path.extname(sourceInput) || ".wav";
  const activeInput = path.join(inputDirectory, `gui-smoke-cancel-active${extension}`);
  const pendingInput = path.join(inputDirectory, `gui-smoke-cancel-pending${extension}`);
  await Promise.all([fs.copyFile(sourceInput, activeInput), fs.copyFile(sourceInput, pendingInput)]);
  const queued = await execute<{
    activeQueueId: string;
    pendingCancelRemoved: boolean;
  }>(
    window,
    `(async () => {
      const base = {
        task: "transcribe-json",
        workflow: "transcribe",
        source_language: "en",
        whisper_model: "base",
        skip_preprocess: false,
        whisper_use_gpu: false,
        whisper_flash_attn: false,
        translation_backend: "",
        mode: "fast",
        subtitle_optimization: "relaxed",
        clear_temp: true,
        clear_checkpoint: true,
      };
      await window.openlrc.queue.pause();
      const active = await window.openlrc.queue.enqueue({ ...base, paths: [${JSON.stringify(activeInput)}] });
      const pending = await window.openlrc.queue.enqueue({ ...base, paths: [${JSON.stringify(pendingInput)}] });
      const pendingCancel = await window.openlrc.queue.cancel(pending.queue_id);
      await window.openlrc.queue.resume();
      return { activeQueueId: active.queue_id, pendingCancelRemoved: pendingCancel.removed };
    })()`,
  );
  await waitForResult(
    window,
    `window.openlrc.queue.snapshot().then((queue) => queue.active?.queue_id === ${JSON.stringify(queued.activeQueueId)})`,
    30_000,
  );
  const activeCancel = await execute<{ cancelled: boolean }>(
    window,
    `window.openlrc.queue.cancel(${JSON.stringify(queued.activeQueueId)})`,
  );
  await waitForResult(
    window,
    `window.openlrc.queue.snapshot().then((queue) => queue.total === 0 && queue.active === null)`,
    60_000,
  );
  const activeCancelHidden = await execute<boolean>(
    window,
    `window.openlrc.jobs.list().then((jobs) => !jobs.some((job) => job.input_paths.includes(${JSON.stringify(activeInput)})))`,
  );
  return {
    pendingCancelRemoved: queued.pendingCancelRemoved,
    activeCancelAccepted: activeCancel.cancelled,
    activeCancelHidden,
  };
}

async function navigate(
  window: BrowserWindow,
  route: string,
  expectedTitle: string,
  report: SmokeReport,
  timeout = 15_000,
  record = true,
): Promise<void> {
  await execute(
    window,
    `(() => {
      const link = document.querySelector(${JSON.stringify(`a[href="#${route}"]`)});
      if (!(link instanceof HTMLElement)) throw new Error(${JSON.stringify(`No route link for ${route}`)});
      link.click();
      return true;
    })()`,
  );
  await waitFor(
    window,
    `document.querySelector("h1")?.textContent === ${JSON.stringify(expectedTitle)}`,
    timeout,
  );
  if (record)
    report.routes[route] = await execute<string>(window, `document.querySelector("h1")?.textContent ?? ""`);
}

async function capture(
  window: BrowserWindow,
  outputDirectory: string,
  filename: string,
  report: SmokeReport,
): Promise<void> {
  await execute<boolean>(
    window,
    `new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(() => resolve(true), 80))))`,
  );
  const image = await withTimeout(
    window.webContents.capturePage(),
    30_000,
    `Timed out capturing smoke screenshot: ${filename}`,
  );
  await fs.writeFile(path.join(outputDirectory, filename), image.toPNG());
  report.snapshots.push(filename);
}

async function waitFor(window: BrowserWindow, expression: string, timeout = 15_000): Promise<void> {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await execute<boolean>(window, `Boolean(${expression})`)) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Timed out waiting for Renderer condition: ${expression}`);
}

async function waitForResult(window: BrowserWindow, script: string, timeout: number): Promise<void> {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await execute<boolean>(window, script)) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Timed out waiting for Renderer result: ${script}`);
}

async function execute<T>(window: BrowserWindow, script: string): Promise<T> {
  return withTimeout(
    window.webContents.executeJavaScript(script, true) as Promise<T>,
    30_000,
    `Timed out executing Renderer smoke step: ${script.slice(0, 180)}`,
  );
}

async function withTimeout<T>(operation: Promise<T>, timeout: number, message: string): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      operation,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error(message)), timeout);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}
