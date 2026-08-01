import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Link, Outlet, RouterProvider, createMemoryRouter } from "react-router-dom";

import i18n from "../src/renderer/i18n.js";
import { NewTaskPage } from "../src/renderer/routes/NewTaskPage.js";
import type { AppSettings, OpenLRCDesktopBridge } from "../src/shared/contracts.js";

const settings: AppSettings = {
  schema_version: 1,
  general: { theme: "openlrc-dark", language: "en", logo_animation: true, reduce_motion: false },
  providers: {
    openai: { enabled: false, model: "", base_url: "", proxy: "" },
    anthropic: { enabled: false, model: "", base_url: "", proxy: "" },
    google: { enabled: false, model: "", base_url: "", proxy: "" },
    litellm: { enabled: false, model: "", base_url: "", proxy: "" },
    third_party: { enabled: false, model: "", base_url: "", proxy: "" },
  },
  local_models: {
    whisper_model: "base",
    vad_model: "silero-v6.2.0",
    whisper_cli: "",
    qwen_model: "",
    hymt2_profile: "default",
    hymt2_model: "",
    llama_server: "",
    host: "127.0.0.1",
    port: 0,
    context_size: 8192,
    gpu_layers: "auto",
    idle_timeout: 300,
    startup_timeout: 60,
  },
  transcription: { source_language: "", skip_preprocess: false, use_gpu: true, flash_attn: true },
  workflow: {
    target_language: "zh-cn",
    bilingual_subtitle: false,
    subtitle_optimization: "aggressive",
    clear_temp: true,
    clear_checkpoint: true,
    glossary_strict: true,
    force_glossary: false,
    edit_rounds: 1,
    enable_restore: false,
    default_glossary: "",
  },
};

function installBridge(): void {
  const bridge = {
    settings: { get: vi.fn().mockResolvedValue(settings) },
    dialogs: { selectInputs: vi.fn().mockResolvedValue({ paths: [], cancelled: true }) },
    workflows: { preflight: vi.fn() },
    queue: { enqueue: vi.fn() },
  } as unknown as OpenLRCDesktopBridge;
  Object.defineProperty(window, "openlrc", { configurable: true, value: bridge });
}

function TestLayout(): React.JSX.Element {
  return (
    <>
      <Link to="/home">Home</Link>
      <Outlet />
    </>
  );
}

describe("New Task draft navigation", () => {
  beforeEach(async () => {
    installBridge();
    await i18n.changeLanguage("en");
  });

  it("resolves the persisted simplified-Chinese locale key", async () => {
    await i18n.changeLanguage("zh-cn");

    expect(i18n.language).toBe("zh-cn");
    expect(i18n.t("nav.home")).toBe("主页");
    expect(i18n.t("history.title")).toBe("历史记录");
  });

  it("requires an explicit stay or discard decision for a dirty draft", async () => {
    const user = userEvent.setup();
    const router = createMemoryRouter(
      [
        {
          element: <TestLayout />,
          children: [
            { path: "/new", element: <NewTaskPage /> },
            { path: "/home", element: <h1>Home destination</h1> },
          ],
        },
      ],
      { initialEntries: ["/new"] },
    );
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    render(
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
      </QueryClientProvider>,
    );

    await screen.findByRole("heading", { name: "New Task", level: 1 });
    await user.type(screen.getByLabelText(/Source language/), "en");
    await user.click(screen.getByRole("link", { name: "Home" }));
    expect(await screen.findByRole("heading", { name: "Discard this draft?" })).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Stay" }));
    expect(screen.getByRole("heading", { name: "New Task", level: 1 })).toBeTruthy();

    await user.click(screen.getByRole("link", { name: "Home" }));
    await user.click(await screen.findByRole("button", { name: "Discard Draft" }));
    expect(await screen.findByRole("heading", { name: "Home destination" })).toBeTruthy();
  });
});
