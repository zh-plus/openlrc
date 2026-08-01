import { promises as fs } from "node:fs";
import path from "node:path";

import { nativeTheme, type Rectangle } from "electron";
import { z } from "zod";

import { appearanceStateSchema, type AppearanceState, type ThemeSource } from "../shared/contracts.js";

const desktopStateSchema = z.object({
  schemaVersion: z.literal(1),
  themeSource: z.enum(["system", "light", "dark"]),
  sidebarCollapsed: z.boolean(),
  windowBounds: z.object({ x: z.number(), y: z.number(), width: z.number(), height: z.number() }).nullable(),
});

type DesktopState = z.infer<typeof desktopStateSchema>;

export class DesktopStateStore {
  private state: DesktopState = {
    schemaVersion: 1,
    themeSource: "system",
    sidebarCollapsed: false,
    windowBounds: null,
  };

  constructor(private readonly filePath: string) {}

  async load(): Promise<void> {
    try {
      this.state = desktopStateSchema.parse(JSON.parse(await fs.readFile(this.filePath, "utf8")));
    } catch {
      this.state = { schemaVersion: 1, themeSource: "system", sidebarCollapsed: false, windowBounds: null };
    }
    nativeTheme.themeSource = this.state.themeSource;
  }

  appearance(): AppearanceState {
    return appearanceStateSchema.parse({
      themeSource: this.state.themeSource,
      effectiveTheme: nativeTheme.shouldUseDarkColors ? "dark" : "light",
      sidebarCollapsed: this.state.sidebarCollapsed,
    });
  }

  bounds(): Rectangle | null {
    return this.state.windowBounds;
  }

  async setTheme(source: ThemeSource): Promise<AppearanceState> {
    this.state.themeSource = source;
    nativeTheme.themeSource = source;
    await this.save();
    return this.appearance();
  }

  async setSidebarCollapsed(collapsed: boolean): Promise<AppearanceState> {
    this.state.sidebarCollapsed = collapsed;
    await this.save();
    return this.appearance();
  }

  async setBounds(bounds: Rectangle): Promise<void> {
    this.state.windowBounds = bounds;
    await this.save();
  }

  private async save(): Promise<void> {
    await fs.mkdir(path.dirname(this.filePath), { recursive: true });
    const temporary = `${this.filePath}.${process.pid}.tmp`;
    await fs.writeFile(temporary, JSON.stringify(this.state, null, 2), { encoding: "utf8", mode: 0o600 });
    await fs.rename(temporary, this.filePath);
  }
}
