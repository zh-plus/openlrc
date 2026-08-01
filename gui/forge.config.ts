import { FuseV1Options, FuseVersion } from "@electron/fuses";
import type { ForgeConfig } from "@electron-forge/shared-types";
import { FusesPlugin } from "@electron-forge/plugin-fuses";
import { WebpackPlugin } from "@electron-forge/plugin-webpack";

import { mainConfig } from "./webpack.main.config";
import { rendererConfig } from "./webpack.renderer.config";

const developmentPort = environmentPort("OPENLRC_FORGE_PORT", 3000);
const developmentLoggerPort = environmentPort("OPENLRC_FORGE_LOGGER_PORT", 9000);

const config: ForgeConfig = {
  packagerConfig: {
    asar: true,
    name: "OpenLRC Desktop",
  },
  rebuildConfig: {},
  makers: [],
  plugins: [
    new WebpackPlugin({
      port: developmentPort,
      loggerPort: developmentLoggerPort,
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
    }),
    new FusesPlugin({
      version: FuseVersion.V1,
      [FuseV1Options.RunAsNode]: false,
      [FuseV1Options.EnableCookieEncryption]: true,
      [FuseV1Options.EnableNodeOptionsEnvironmentVariable]: false,
      [FuseV1Options.EnableNodeCliInspectArguments]: false,
      [FuseV1Options.EnableEmbeddedAsarIntegrityValidation]: true,
      [FuseV1Options.OnlyLoadAppFromAsar]: true,
    }),
  ],
};

function environmentPort(name: string, fallback: number): number {
  const value = Number(process.env[name]);
  return Number.isInteger(value) && value > 0 && value <= 65_535 ? value : fallback;
}

export default config;
