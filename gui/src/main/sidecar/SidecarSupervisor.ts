import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { EventEmitter } from "node:events";
import path from "node:path";

import log from "electron-log/main.js";
import { z } from "zod";

import { backendStatusSchema, type BackendStatus } from "../../shared/contracts.js";
import { JsonLineFramer } from "./JsonLineFramer.js";

const readyFrameSchema = z.object({
  type: z.literal("event"),
  protocol: z.literal(1),
  event: z.literal("app.ready"),
  payload: z.object({
    service_version: z.string(),
    min_protocol: z.number(),
    max_protocol: z.number(),
    runtime_mode: z.string(),
    capabilities: z.array(z.string()),
  }),
});

const responseSchema = z.discriminatedUnion("ok", [
  z.object({
    type: z.literal("response"),
    protocol: z.literal(1),
    id: z.string(),
    ok: z.literal(true),
    result: z.unknown(),
  }),
  z.object({
    type: z.literal("response"),
    protocol: z.literal(1),
    id: z.string(),
    ok: z.literal(false),
    error: z.object({
      code: z.string(),
      message: z.string(),
      retryable: z.boolean(),
      hint: z.string().nullable(),
    }),
  }),
]);

interface PendingRequest {
  resolve(value: unknown): void;
  reject(error: Error): void;
  timeout: NodeJS.Timeout;
}

export class SidecarRequestError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly retryable: boolean,
    readonly hint: string | null,
  ) {
    super(message);
  }
}

export class SidecarSupervisor extends EventEmitter {
  private child: ChildProcessWithoutNullStreams | null = null;
  private pending = new Map<string, PendingRequest>();
  private requestSequence = 0;
  private statusValue: BackendStatus = backendStatusSchema.parse({
    state: "stopped",
    message: null,
    serviceVersion: null,
    capabilities: [],
  });
  private readyPromise: Promise<void> | null = null;
  private resolveReady: (() => void) | null = null;
  private rejectReady: ((error: Error) => void) | null = null;

  constructor(
    private readonly repositoryRoot: string,
    private readonly uvExecutable = "uv",
  ) {
    super();
  }

  get status(): BackendStatus {
    return this.statusValue;
  }

  async start(): Promise<void> {
    if (this.child) return this.readyPromise ?? Promise.resolve();
    this.setStatus("starting", "Starting local OpenLRC service…");
    const environment = { ...process.env };
    delete environment.NODE_OPTIONS;
    delete environment.ELECTRON_RUN_AS_NODE;
    delete environment.PYTHONPATH;
    environment.UV_CACHE_DIR = path.join(this.repositoryRoot, ".uv-cache");
    const child = spawn(
      this.uvExecutable,
      ["run", "--no-sync", "python", "-m", "openlrc.gui_bridge", "serve", "--stdio"],
      {
        cwd: this.repositoryRoot,
        env: environment,
        shell: false,
        stdio: ["pipe", "pipe", "pipe"],
      },
    );
    this.child = child;
    const framer = new JsonLineFramer();
    this.readyPromise = new Promise<void>((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    const startupTimeout = setTimeout(() => {
      this.fail(new Error("Local service did not complete its handshake within 20 seconds."));
      child.kill("SIGTERM");
    }, 20_000);
    child.stdout.on("data", (chunk: Buffer) => {
      try {
        for (const frame of framer.push(chunk)) this.acceptFrame(frame, startupTimeout);
      } catch (error) {
        this.fail(error instanceof Error ? error : new Error(String(error)));
        child.kill("SIGTERM");
      }
    });
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => log.warn(redact(chunk.trim()).slice(0, 4000)));
    child.on("error", (error) => this.fail(error));
    child.on("exit", (code, signal) => {
      clearTimeout(startupTimeout);
      try {
        framer.finish();
      } catch (error) {
        log.warn(error);
      }
      const expected = this.statusValue.state === "stopping";
      this.child = null;
      this.readyPromise = null;
      const detail = expected ? null : `Local service exited (${signal ?? code ?? "unknown"}).`;
      this.setStatus(expected ? "stopped" : "failed", detail);
      this.rejectAll(new Error(detail ?? "Local service stopped."));
      if (!expected) this.emit("crash", detail);
    });
    await this.readyPromise;
  }

  async restart(): Promise<void> {
    await this.stop();
    await this.start();
  }

  async request(method: string, params: Record<string, unknown> = {}, timeoutMs = 20_000): Promise<unknown> {
    await this.start();
    const child = this.child;
    if (!child || this.statusValue.state === "failed") throw new Error("Local service is unavailable.");
    const id = `main-${++this.requestSequence}`;
    const frame = JSON.stringify({ type: "request", protocol: 1, id, method, params }) + "\n";
    return new Promise<unknown>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Local service request timed out: ${method}`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timeout });
      const writable = child.stdin.write(frame, "utf8");
      if (!writable) child.stdin.once("drain", () => undefined);
    });
  }

  async stop(): Promise<void> {
    const child = this.child;
    if (!child) {
      this.setStatus("stopped", null);
      return;
    }
    this.setStatus("stopping", "Stopping local OpenLRC service…");
    try {
      await this.request("app.shutdown", {}, 5_000);
    } catch (error) {
      log.warn("Graceful sidecar shutdown request failed", error);
    }
    child.stdin.end();
    await waitForExit(child, 12_000).catch(() => {
      child.kill("SIGTERM");
      return waitForExit(child, 5_000).catch(() => child.kill("SIGKILL"));
    });
  }

  private acceptFrame(frame: unknown, startupTimeout: NodeJS.Timeout): void {
    const ready = readyFrameSchema.safeParse(frame);
    if (ready.success) {
      if (ready.data.payload.min_protocol > 1 || ready.data.payload.max_protocol < 1) {
        this.fail(new Error("GUI and local service protocol versions are incompatible."));
        this.child?.kill("SIGTERM");
        return;
      }
      clearTimeout(startupTimeout);
      this.statusValue = backendStatusSchema.parse({
        state: "ready",
        message: null,
        serviceVersion: ready.data.payload.service_version,
        capabilities: ready.data.payload.capabilities,
      });
      this.emit("status", this.statusValue);
      this.resolveReady?.();
      this.resolveReady = null;
      this.rejectReady = null;
      return;
    }
    const response = responseSchema.safeParse(frame);
    if (response.success) {
      const pending = this.pending.get(response.data.id);
      if (!pending) return;
      clearTimeout(pending.timeout);
      this.pending.delete(response.data.id);
      if (response.data.ok) pending.resolve(response.data.result);
      else {
        pending.reject(
          new SidecarRequestError(
            response.data.error.code,
            response.data.error.message,
            response.data.error.retryable,
            response.data.error.hint,
          ),
        );
      }
      return;
    }
    if (typeof frame === "object" && frame !== null && "type" in frame && frame.type === "event") {
      this.emit("event", frame);
      return;
    }
    throw new Error("Local service emitted an invalid protocol frame.");
  }

  private fail(error: Error): void {
    this.setStatus("failed", error.message);
    this.rejectReady?.(error);
    this.resolveReady = null;
    this.rejectReady = null;
    this.rejectAll(error);
  }

  private rejectAll(error: Error): void {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timeout);
      pending.reject(error);
    }
    this.pending.clear();
  }

  private setStatus(state: BackendStatus["state"], message: string | null): void {
    this.statusValue = { ...this.statusValue, state, message };
    this.emit("status", this.statusValue);
  }
}

function waitForExit(child: ChildProcessWithoutNullStreams, timeoutMs: number): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("Sidecar shutdown timed out.")), timeoutMs);
    child.once("exit", () => {
      clearTimeout(timeout);
      resolve();
    });
  });
}

function redact(value: string): string {
  return value
    .replace(/(api[_-]?key|authorization|token|secret|password)(\s*[:=]\s*)\S+/gi, "$1$2<redacted>")
    .replace(/Bearer\s+[A-Za-z0-9._~+/=-]+/gi, "Bearer <redacted>");
}
