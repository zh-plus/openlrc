import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router-dom";

import i18n from "../src/renderer/i18n.js";
import { HistoryPage } from "../src/renderer/routes/HistoryPage.js";
import type { JobSummary, OpenLRCDesktopBridge } from "../src/shared/contracts.js";

const jobs: JobSummary[] = [
  historyJob("succeeded", "Completed task"),
  historyJob("succeeded_with_warnings", "Warning task"),
  historyJob("failed", "Failed task"),
  historyJob("interrupted", "Interrupted task"),
];

describe("History status groups", () => {
  beforeEach(async () => {
    const bridge = {
      jobs: { list: vi.fn().mockResolvedValue(jobs) },
    } as unknown as OpenLRCDesktopBridge;
    Object.defineProperty(window, "openlrc", { configurable: true, value: bridge });
    await i18n.changeLanguage("en");
  });

  it("offers three filters and groups warnings, failures, and interruptions under Failed", async () => {
    const user = userEvent.setup();
    const router = createMemoryRouter([{ path: "/history", element: <HistoryPage /> }], {
      initialEntries: ["/history"],
    });
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    render(
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
      </QueryClientProvider>,
    );

    await screen.findByRole("heading", { name: "History", level: 1 });
    const filters = screen.getByRole("radiogroup", { name: "History status" });
    expect(
      within(filters)
        .getAllByRole("radio")
        .map((option) => option.textContent),
    ).toEqual(["All", "Completed", "Failed"]);

    await user.click(within(filters).getByRole("radio", { name: "Completed" }));
    expect(screen.getByText("Completed task")).toBeTruthy();
    expect(screen.queryByText("Warning task")).toBeNull();

    await user.click(within(filters).getByRole("radio", { name: "Failed" }));
    expect(screen.queryByText("Completed task")).toBeNull();
    expect(screen.getByText("Warning task")).toBeTruthy();
    expect(screen.getByText("Failed task")).toBeTruthy();
    expect(screen.getByText("Interrupted task")).toBeTruthy();
  });
});

function historyJob(status: JobSummary["status"], name: string): JobSummary {
  return {
    job_id: `job-${status}`,
    workflow: "transcribe",
    name,
    status,
    input_paths: [`/${name}.wav`],
    translation_mode: null,
    progress: status === "succeeded" ? 100 : 50,
    started_at: "2026-08-02T00:00:00Z",
    completed_at: "2026-08-02T00:01:00Z",
    outputs: [],
    elapsed_seconds: 60,
    error: null,
    resumed_from: null,
  };
}
