import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";

import { WebpackPlugin } from "@electron-forge/plugin-webpack";
import type { WdioElectronConfig } from "@wdio/native-types";

import { mainConfig } from "./webpack.main.config";
import { rendererConfig } from "./webpack.renderer.config";

let developmentPlugin: WebpackPlugin | undefined;
const silentLogger = {
  createTab: () => ({ log: (_message: string): void => undefined }),
} as unknown as NonNullable<Parameters<WebpackPlugin["compileMain"]>[1]>;
const developmentMainEntry = path.resolve(".webpack/main/index.js");
const electronBinary = path.resolve(
  "node_modules/electron/dist",
  process.platform === "darwin"
    ? "Electron.app/Contents/MacOS/Electron"
    : process.platform === "win32"
      ? "electron.exe"
      : "electron",
);

export const config: WdioElectronConfig = {
  runner: "local",
  specs: ["./e2e/**/*.e2e.ts"],
  maxInstances: 1,
  capabilities: [
    {
      browserName: "electron",
      "wdio:electronServiceOptions": {
        appBinaryPath: electronBinary,
        appArgs: [`--app=${developmentMainEntry}`],
        captureRendererLogs: true,
        rendererLogLevel: "error",
      },
    },
  ],
  services: ["electron"],
  framework: "jasmine",
  reporters: ["spec"],
  logLevel: "warn",
  waitforTimeout: 15_000,
  connectionRetryTimeout: 120_000,
  connectionRetryCount: 1,
  jasmineOpts: {
    defaultTimeoutInterval: 60_000,
  },
  onPrepare: async () => {
    const projectDirectory = process.cwd();
    await fs.rm(path.join(projectDirectory, ".webpack"), { recursive: true, force: true });
    process.env.OPENLRC_REPO_ROOT ??= path.resolve(projectDirectory, "..");
    process.env.OPENLRC_APP_SUPPORT_DIR ??= await fs.mkdtemp(
      path.join(os.tmpdir(), "openlrc-wdio-app-support-"),
    );

    developmentPlugin = new WebpackPlugin({
      port: 3100,
      loggerPort: 9100,
      mainConfig,
      renderer: {
        config: rendererConfig,
        entryPoints: [
          {
            html: "./src/renderer/index.html",
            js: "./src/renderer/main.tsx",
            name: "main_window",
            preload: { js: "./src/preload/index.ts" },
          },
        ],
      },
    });
    developmentPlugin.init(projectDirectory);
    await developmentPlugin.compileMain(true, silentLogger);
    await developmentPlugin.launchRendererDevServers(silentLogger);
  },
  onComplete: () => {
    developmentPlugin?.exitHandler({ cleanup: true });
  },
};
