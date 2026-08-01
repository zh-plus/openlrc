import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";

import {
  Button,
  Disclosure,
  ModalDialog,
  ProgressBar,
  RadioCardGroup,
  SegmentedControl,
  SelectField,
  SingleToggleGroup,
  Tabs,
  Tooltip,
} from "../src/renderer/components/ui/index.js";

describe("OpenLRC React Aria design system", () => {
  it("normalizes button press semantics for mouse and keyboard", async () => {
    const user = userEvent.setup();
    const onPress = vi.fn();
    render(<Button onPress={onPress}>Run preflight</Button>);

    const button = screen.getByRole("button", { name: "Run preflight" });
    await user.click(button);
    button.focus();
    await user.keyboard("{Enter}");

    expect(onPress).toHaveBeenCalledTimes(2);
  });

  it("selects list box options and radio cards with semantic roles", async () => {
    const user = userEvent.setup();

    function Harness(): React.JSX.Element {
      const [quality, setQuality] = useState("fast");
      const [backend, setBackend] = useState("qwen");
      return (
        <>
          <SelectField
            label="Quality"
            value={quality}
            onChange={setQuality}
            options={[
              { value: "fast", label: "Fast" },
              { value: "pro", label: "Pro" },
            ]}
          />
          <RadioCardGroup
            label="Backend"
            value={backend}
            onChange={setBackend}
            options={[
              { value: "qwen", label: "Qwen" },
              { value: "hymt2", label: "Hy-MT2", accent: true },
            ]}
          />
        </>
      );
    }

    render(<Harness />);
    await user.click(screen.getByRole("button", { name: /Quality/ }));
    await user.click(await screen.findByRole("option", { name: "Pro" }));
    expect(screen.getByRole("button", { name: /Quality/ }).textContent).toContain("Pro");

    const hyMt2 = screen.getByRole("radio", { name: /Hy-MT2/ });
    await user.click(hyMt2);
    expect((hyMt2 as HTMLInputElement).checked).toBe(true);
  });

  it("supports vertical tab arrow navigation and disclosure state", async () => {
    const user = userEvent.setup();
    render(
      <>
        <Tabs
          label="Settings sections"
          defaultSelectedKey="general"
          tabs={[
            { id: "general", label: "General", content: <p>General panel</p> },
            { id: "models", label: "Models", content: <p>Models panel</p> },
          ]}
        />
        <Disclosure title="Advanced" detail="Optional fields">
          <p>Advanced content</p>
        </Disclosure>
      </>,
    );

    const general = screen.getByRole("tab", { name: "General" });
    act(() => general.focus());
    await user.keyboard("{ArrowDown}");
    expect(screen.getByRole("tab", { name: "Models" }).getAttribute("aria-selected")).toBe("true");

    const disclosure = screen.getByRole("button", { name: /Advanced/ });
    expect(disclosure.getAttribute("aria-expanded")).toBe("false");
    await user.click(disclosure);
    expect(disclosure.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByText("Advanced content")).not.toBeNull();
  });

  it("moves the segmented control indicator with the selected radio", async () => {
    const user = userEvent.setup();

    function Harness(): React.JSX.Element {
      const [theme, setTheme] = useState("system");
      return (
        <SegmentedControl
          label="Theme source"
          value={theme}
          onChange={setTheme}
          options={[
            { value: "system", label: "System" },
            { value: "light", label: "Light" },
            { value: "dark", label: "Dark" },
          ]}
        />
      );
    }

    const { container } = render(<Harness />);
    const group = screen.getByRole("radiogroup", { name: "Theme source" });
    const slider = container.querySelector<HTMLElement>(".segment-slider");
    if (!slider) throw new Error("Segmented control slider did not render.");

    expect(group.style.getPropertyValue("--segment-count")).toBe("3");
    expect(group.style.getPropertyValue("--segment-index")).toBe("0");
    expect(slider.getAttribute("aria-hidden")).toBe("true");

    await user.click(screen.getByRole("radio", { name: "Dark" }));
    expect(group.style.getPropertyValue("--segment-index")).toBe("2");
  });

  it("keeps the History filter single-select and renders its shared sliding indicator", async () => {
    const user = userEvent.setup();

    function Harness(): React.JSX.Element {
      const [status, setStatus] = useState("all");
      return (
        <SingleToggleGroup
          label="History status"
          value={status}
          onChange={setStatus}
          options={[
            { value: "all", label: "All" },
            { value: "completed", label: "Completed" },
            { value: "failed", label: "Failed" },
          ]}
        />
      );
    }

    const { container } = render(<Harness />);
    const group = container.querySelector<HTMLElement>(".sliding-toggle-group");
    const slider = container.querySelector<HTMLElement>(".toggle-group-slider");
    if (!group || !slider) throw new Error("History filter slider did not render.");

    expect(group.getAttribute("data-slider-ready")).toBe("true");
    expect(slider.classList.contains("segment-slider")).toBe(true);
    expect(screen.getByRole("radio", { name: "All" }).getAttribute("aria-checked")).toBe("true");

    await user.click(screen.getByRole("radio", { name: "Failed" }));
    expect(screen.getByRole("radio", { name: "Failed" }).getAttribute("aria-checked")).toBe("true");
    expect(screen.getByRole("radio", { name: "All" }).getAttribute("aria-checked")).toBe("false");
  });

  it("traps dialog focus, closes with Escape, and restores the trigger focus", async () => {
    const user = userEvent.setup();

    function Harness(): React.JSX.Element {
      const [open, setOpen] = useState(false);
      return (
        <>
          <Button onPress={() => setOpen(true)}>Delete history</Button>
          <ModalDialog
            isOpen={open}
            onOpenChange={setOpen}
            title="Delete this record?"
            description="This action cannot be undone."
          >
            <Button autoFocus onPress={() => setOpen(false)}>
              Cancel
            </Button>
            <Button variant="danger">Delete</Button>
          </ModalDialog>
        </>
      );
    }

    render(<Harness />);
    const trigger = screen.getByRole("button", { name: "Delete history" });
    await user.click(trigger);
    expect(await screen.findByRole("dialog", { name: "Delete this record?" })).not.toBeNull();
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Cancel" }));

    await user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    await waitFor(() => expect(document.activeElement).toBe(trigger));
  });

  it("exposes keyboard tooltips and progress values without relying on DOM internals", async () => {
    const user = userEvent.setup();
    render(
      <>
        <Tooltip label="Refresh resources" delay={0}>
          <Button aria-label="Refresh">Refresh</Button>
        </Tooltip>
        <ProgressBar label="Translation progress" value={42} />
      </>,
    );

    await user.tab();
    expect(await screen.findByRole("tooltip", { name: "Refresh resources" })).not.toBeNull();
    const progress = screen.getByRole("progressbar", { name: "Translation progress" });
    expect(progress.getAttribute("aria-valuenow")).toBe("42");
    expect(progress.textContent).toContain("42%");
  });
});
