import type { Configuration } from "webpack";

import { rules } from "./webpack.rules";

export const rendererConfig: Configuration = {
  devtool: "source-map",
  module: {
    rules: [
      ...rules,
      {
        test: /\.css$/,
        use: ["style-loader", "css-loader", "postcss-loader"],
      },
    ],
  },
  resolve: {
    extensions: [".js", ".ts", ".tsx", ".json", ".css"],
    extensionAlias: { ".js": [".js", ".ts", ".tsx"] },
  },
};
