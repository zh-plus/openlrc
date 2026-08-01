import type { Configuration } from "webpack";

import { rules } from "./webpack.rules";

export const mainConfig: Configuration = {
  entry: "./src/main/index.ts",
  module: { rules },
  resolve: {
    extensions: [".js", ".ts", ".tsx", ".json"],
    extensionAlias: { ".js": [".js", ".ts", ".tsx"] },
  },
};
