import type { OpenLRCDesktopBridge } from "../shared/contracts.js";

declare global {
  interface Window {
    openlrc: OpenLRCDesktopBridge;
  }
}

export {};
