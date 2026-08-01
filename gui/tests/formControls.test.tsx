import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { Controller, useForm } from "react-hook-form";

import { CheckboxControl, SwitchControl } from "../src/renderer/components/ui/index.js";

function SwitchHarness(): React.JSX.Element {
  const form = useForm<{ enabled: boolean }>({ defaultValues: { enabled: false } });
  return (
    <>
      <label htmlFor="test-switch" id="test-switch-label">
        Reduce motion
      </label>
      <Controller
        control={form.control}
        name="enabled"
        render={({ field }) => (
          <SwitchControl
            id="test-switch"
            name={field.name}
            checked={field.value}
            labelledBy="test-switch-label"
            onBlur={field.onBlur}
            onCheckedChange={field.onChange}
          />
        )}
      />
      <output>{form.formState.isDirty ? "dirty" : "clean"}</output>
      <button type="button" onClick={() => form.reset()}>
        Reset
      </button>
    </>
  );
}

function CheckboxHarness({ disabled = false }: { disabled?: boolean }): React.JSX.Element {
  const [checked, setChecked] = useState(false);
  return (
    <>
      <label htmlFor="test-checkbox" id="test-checkbox-label">
        Bilingual subtitle
      </label>
      <CheckboxControl
        id="test-checkbox"
        name="bilingual"
        checked={checked}
        disabled={disabled}
        labelledBy="test-checkbox-label"
        onBlur={() => undefined}
        onCheckedChange={setChecked}
      />
    </>
  );
}

describe("form controls", () => {
  it("supports keyboard switching and React Hook Form dirty/reset state", async () => {
    const user = userEvent.setup();
    const { container } = render(<SwitchHarness />);

    const control = screen.getByRole("switch", { name: "Reduce motion" });
    expect((control as HTMLInputElement).checked).toBe(false);
    expect(screen.queryByText("clean")).not.toBeNull();

    const visualSwitch = container.querySelector<HTMLElement>(".toggle-switch");
    if (!visualSwitch) throw new Error("Switch visual control did not render.");
    await user.click(visualSwitch);
    expect((control as HTMLInputElement).checked).toBe(true);

    await user.click(screen.getByRole("button", { name: "Reset" }));
    expect((control as HTMLInputElement).checked).toBe(false);

    act(() => control.focus());
    await user.keyboard(" ");
    expect((control as HTMLInputElement).checked).toBe(true);
    expect(screen.queryByText("dirty")).not.toBeNull();

    await user.click(screen.getByRole("button", { name: "Reset" }));
    expect((control as HTMLInputElement).checked).toBe(false);
    expect(screen.queryByText("clean")).not.toBeNull();
  });

  it("uses checkbox semantics and respects disabled state", async () => {
    const user = userEvent.setup();
    const { rerender } = render(<CheckboxHarness />);

    const checkbox = screen.getByRole("checkbox", { name: "Bilingual subtitle" });
    await user.click(checkbox);
    expect((checkbox as HTMLInputElement).checked).toBe(true);

    rerender(<CheckboxHarness key="disabled" disabled />);
    const disabledCheckbox = screen.getByRole("checkbox", { name: "Bilingual subtitle" });
    expect((disabledCheckbox as HTMLInputElement).disabled).toBe(true);
    await user.click(disabledCheckbox);
    expect((disabledCheckbox as HTMLInputElement).checked).toBe(false);
  });
});
