import { $, $$, browser, expect } from "@wdio/globals";

describe("OpenLRC Desktop development window", () => {
  beforeAll(async () => {
    await browser.waitUntil(async () => (await $("h1").getText()) === "Home", {
      timeout: 30_000,
      timeoutMsg: "Home route did not render.",
    });
    await browser.waitUntil(
      async () =>
        (await browser.execute(async () => (await window.openlrc.app.backendStatus()).state)) === "ready",
      { timeout: 45_000, timeoutMsg: "Python GUI service did not become ready." },
    );
  });

  it("keeps Node out of the Renderer and exposes only the typed bridge", async () => {
    const security = await browser.execute(() => ({
      bridge: typeof window.openlrc,
      nodeRequire: typeof window.require,
      nodeProcess: typeof window.process,
    }));
    expect(security).toEqual({ bridge: "object", nodeRequire: "undefined", nodeProcess: "undefined" });
  });

  it("renders the approved navigation and all unconditional v1 routes", async () => {
    const labels = await $$(".primary-nav .nav-item").map((item) => item.getText());
    expect(labels).toEqual(["Home", "New Task", "Tasks", "History"]);

    const routes = [
      ["Tasks", "Tasks"],
      ["History", "History"],
      ["Resources", "Resources"],
      ["Settings", "Settings"],
    ] as const;
    for (const [link, title] of routes) {
      await $(`a=${link}`).click();
      await expect($("h1")).toHaveText(title);
    }
  });

  it("provides a full-width titlebar drag region and visible button hover feedback", async () => {
    const dragRegion = await browser.execute(() => {
      const element = document.querySelector<HTMLElement>(".window-drag-region");
      const rect = element?.getBoundingClientRect();
      const style = element ? getComputedStyle(element) : undefined;
      return {
        left: rect?.left ?? -1,
        top: rect?.top ?? -1,
        width: rect?.width ?? 0,
        viewportWidth: innerWidth,
        appRegion: style?.getPropertyValue("-webkit-app-region") ?? "",
      };
    });
    expect(dragRegion).toEqual({
      left: 0,
      top: 0,
      width: dragRegion.viewportWidth,
      viewportWidth: dragRegion.viewportWidth,
      appRegion: "drag",
    });

    await $(`a=Home`).click();
    await $("h1").moveTo();
    await $(".button.primary").waitForDisplayed();
    const before = await $(".button.primary").getCSSProperty("box-shadow");
    const beforeVisual = await primaryButtonVisual();
    await browser.waitUntil(async () => {
      try {
        await $(".button.primary").moveTo();
        return (await primaryButtonVisual()).filter !== beforeVisual.filter;
      } catch {
        return false;
      }
    });
    const after = await $(".button.primary").getCSSProperty("box-shadow");
    const afterVisual = await primaryButtonVisual();
    expect(after.value).not.toBe(before.value);
    expect(after.value).not.toBe("none");
    expect(afterVisual.filter).not.toBe(beforeVisual.filter);
    expect(afterVisual.transform).toBe("none");
    expect(afterVisual.left).toBe(beforeVisual.left);
    expect(afterVisual.top).toBe(beforeVisual.top);
  });

  it("switches theme and Sidebar state without replacing the current route", async () => {
    await clickSidebarThemeOption("dark");
    await browser.waitUntil(
      async () => (await browser.execute(() => document.documentElement.dataset.theme)) === "dark",
    );
    await browser.waitUntil(() => segmentedIndicatorAligned(".theme-segment"), {
      timeoutMsg: "Sidebar segmented indicator did not settle under the selected option.",
    });
    const tokens = await browser.execute(() => ({
      canvas: getComputedStyle(document.documentElement).getPropertyValue("--color-canvas").trim(),
      text: getComputedStyle(document.documentElement).getPropertyValue("--color-text").trim(),
    }));
    expect(tokens).toEqual({ canvas: "#242424", text: "#ffffff" });

    const themeBounds = await browser.execute(() => {
      const sidebar = document.querySelector<HTMLElement>(".sidebar")?.getBoundingClientRect();
      const themeSegment = document.querySelector<HTMLElement>(".theme-segment");
      const segment = themeSegment?.getBoundingClientRect();
      const slider = themeSegment?.querySelector<HTMLElement>(".segment-slider")?.getBoundingClientRect();
      const selected = themeSegment
        ?.querySelector<HTMLElement>(".segment-option[data-selected]")
        ?.getBoundingClientRect();
      return {
        sidebarLeft: sidebar?.left ?? 0,
        sidebarRight: sidebar?.right ?? 0,
        segmentLeft: segment?.left ?? -1,
        segmentRight: segment?.right ?? -1,
        sliderCenter: slider ? slider.left + slider.width / 2 : 0,
        selectedCenter: selected ? selected.left + selected.width / 2 : 0,
      };
    });
    expect(themeBounds.segmentLeft).toBeGreaterThanOrEqual(themeBounds.sidebarLeft);
    expect(themeBounds.segmentRight).toBeLessThanOrEqual(themeBounds.sidebarRight);
    expect(Math.abs(themeBounds.sliderCenter - themeBounds.selectedCenter)).toBeLessThanOrEqual(0.5);

    const hash = await browser.getUrl();
    await $(`button[aria-label="Collapse sidebar"]`).click();
    await browser.waitUntil(async () => (await $(".sidebar").getSize("width")) === 80);
    expect(await browser.getUrl()).toBe(hash);
    const alignment = await browser.execute(() => {
      const sidebar = document.querySelector<HTMLElement>(".sidebar")?.getBoundingClientRect();
      const items = Array.from(document.querySelectorAll<HTMLElement>(".nav-item")).map((item) => {
        const rect = item.getBoundingClientRect();
        return { center: rect.left + rect.width / 2, width: rect.width, height: rect.height };
      });
      const resource = document.querySelector<HTMLElement>(".sidebar-tools .nav-item");
      const dot = resource?.querySelector<HTMLElement>(".resource-dot");
      const resourceRect = resource?.getBoundingClientRect();
      const dotRect = dot?.getBoundingClientRect();
      return {
        sidebarCenter: sidebar ? sidebar.left + sidebar.width / 2 : 0,
        items,
        resourceDotInside: Boolean(
          resourceRect &&
          dotRect &&
          dotRect.left >= resourceRect.left &&
          dotRect.right <= resourceRect.right &&
          dotRect.top >= resourceRect.top &&
          dotRect.bottom <= resourceRect.bottom,
        ),
      };
    });
    expect(alignment.items.length).toBe(6);
    for (const item of alignment.items) {
      expect(Math.abs(item.center - alignment.sidebarCenter)).toBeLessThanOrEqual(0.5);
      expect(item.width).toBe(44);
      expect(item.height).toBe(44);
    }
    expect(alignment.resourceDotInside).toBe(true);
    await $(`button[aria-label="Expand sidebar"]`).click();
    await browser.waitUntil(async () => (await $(".sidebar").getSize("width")) === 224);
    await clickSidebarThemeOption("system");
  });

  it("keeps theme choices horizontal and preserves Settings switch save/cancel behavior", async () => {
    await $(`a=Settings`).click();
    await expect($("h1")).toHaveText("Settings");
    await browser.waitUntil(() => segmentedIndicatorAligned(".settings-theme-segment"), {
      timeoutMsg: "Settings segmented indicator did not settle under the selected option.",
    });
    const themeLayout = await browser.execute(() => {
      const segment = document.querySelector<HTMLElement>(".settings-theme-segment");
      const options = Array.from(segment?.querySelectorAll<HTMLElement>(".segment-option") ?? []).map(
        (option) => {
          const rect = option.getBoundingClientRect();
          return { top: rect.top, height: rect.height };
        },
      );
      const style = segment ? getComputedStyle(segment) : undefined;
      const heading = document.querySelector<HTMLElement>(".settings-heading h2");
      const slider = segment?.querySelector<HTMLElement>(".segment-slider");
      const selected = segment?.querySelector<HTMLElement>(".segment-option[data-selected]");
      const sliderRect = slider?.getBoundingClientRect();
      const selectedRect = selected?.getBoundingClientRect();
      const segmentRect = segment?.getBoundingClientRect();
      return {
        display: style?.display ?? "",
        width: segment?.getBoundingClientRect().width ?? 0,
        options,
        headingFontSize: heading ? getComputedStyle(heading).fontSize : "",
        sliderTransition: slider ? getComputedStyle(slider).transitionProperty : "",
        sliderCenter: sliderRect ? sliderRect.left + sliderRect.width / 2 : 0,
        selectedCenter: selectedRect ? selectedRect.left + selectedRect.width / 2 : 0,
        segmentRect: segmentRect ? { left: segmentRect.left, width: segmentRect.width } : null,
      };
    });
    expect(themeLayout.display).toBe("grid");
    expect(themeLayout.width).toBe(240);
    expect(themeLayout.options.length).toBe(3);
    expect(new Set(themeLayout.options.map((option) => option.top)).size).toBe(1);
    expect(new Set(themeLayout.options.map((option) => option.height)).size).toBe(1);
    expect(themeLayout.headingFontSize).toBe("14px");
    expect(themeLayout.sliderTransition).toContain("transform");
    expect(Math.abs(themeLayout.sliderCenter - themeLayout.selectedCenter)).toBeLessThanOrEqual(0.5);

    const settingsTabCount = await browser.execute(
      () => document.querySelectorAll(".settings-tab-list [role=tab]").length,
    );
    expect(settingsTabCount).toBe(5);
    for (let index = 0; index < settingsTabCount; index += 1) {
      const tab = (await $$(".settings-tab-list [role=tab]"))[index];
      if (!tab) throw new Error(`Settings tab ${index} did not render.`);
      await tab.click();
      await browser.waitUntil(async () => (await settingsTabVisual(index)).selected);
      const visual = await settingsTabVisual(index);
      expect(visual.focused).toBe(true);
      expect(visual.color).toBe(visual.expectedColor);
      expect(visual.background).toBe(visual.expectedBackground);
    }
    const generalTab = (await $$(".settings-tab-list [role=tab]"))[0];
    if (!generalTab) throw new Error("General settings tab did not render.");
    await generalTab.click();

    const themeOptions = await $$(".settings-theme-segment .segment-option");
    const hoverOption = themeOptions[1];
    if (!hoverOption) throw new Error("Settings theme option did not render.");
    await hoverOption.moveTo();
    await browser.waitUntil(async () => (await hoverOption.getAttribute("data-hovered")) !== null);
    const hoverBackground = await hoverOption.getCSSProperty("background-color");
    expect(hoverBackground.value).not.toBe("rgba(0,0,0,0)");

    const sliderBefore = await $(".settings-theme-segment .segment-slider").getCSSProperty("transform");
    await clickSettingsThemeOption("dark");
    await browser.waitUntil(async () => {
      const selected = await themeOptionChecked("dark");
      const transform = await $(".settings-theme-segment .segment-slider").getCSSProperty("transform");
      return selected && transform.value !== sliderBefore.value;
    });
    await clickSettingsThemeOption("system");
    await browser.waitUntil(() => themeOptionChecked("system"));

    const original = await reduceMotionChecked();
    const switchBefore = await switchGeometry();
    expect(Math.abs(switchBefore.trackCenterY - switchBefore.thumbCenterY)).toBeLessThanOrEqual(0.5);
    await clickReduceMotionSwitch();
    await browser.waitUntil(async () => (await reduceMotionChecked()) === !original);
    await browser.waitUntil(async () => {
      const current = await switchGeometry();
      return Math.abs(current.thumbLeft - switchBefore.thumbLeft) >= 16;
    });
    const switchAfter = await switchGeometry();
    expect(Math.abs(switchAfter.trackCenterY - switchAfter.thumbCenterY)).toBeLessThanOrEqual(0.5);
    await $(`button=Cancel`).click();
    await browser.waitUntil(async () => (await reduceMotionChecked()) === original);

    await clickReduceMotionSwitch();
    await $(`button=Save changes`).click();
    await browser.waitUntil(
      async () =>
        (await browser.execute(async () => (await window.openlrc.settings.get()).general.reduce_motion)) ===
        !original,
    );

    await clickReduceMotionSwitch();
    await $(`button=Save changes`).click();
    await browser.waitUntil(
      async () =>
        (await browser.execute(async () => (await window.openlrc.settings.get()).general.reduce_motion)) ===
        original,
    );
  });

  it("slides the History status indicator and exposes option hover feedback", async () => {
    await $(`a=History`).click();
    await expect($("h1")).toHaveText("History");
    await browser.waitUntil(() => segmentedIndicatorAligned(".filter-tabs"), {
      timeoutMsg: "History filter indicator did not settle under All.",
    });
    const before = await historyFilterVisual();
    expect(before.ready).toBe(true);
    expect(before.optionCount).toBe(3);
    expect(before.transition).toContain("transform");
    expect(before.transition).toContain("width");

    const failed = (await $$(".filter-tabs .filter-option"))[2];
    if (!failed) throw new Error("Failed history filter did not render.");
    await failed.click();
    await browser.waitUntil(async () => {
      const current = await historyFilterVisual();
      return current.selectedIndex === 2 && current.aligned && current.sliderLeft !== before.sliderLeft;
    });

    const completed = (await $$(".filter-tabs .filter-option"))[1];
    if (!completed) throw new Error("Completed history filter did not render.");
    await completed.moveTo();
    await browser.waitUntil(async () => (await completed.getAttribute("data-hovered")) !== null);
    const hoverBackground = await completed.getCSSProperty("background-color");
    expect(hoverBackground.value).not.toBe("rgba(0,0,0,0)");

    const all = (await $$(".filter-tabs .filter-option"))[0];
    if (!all) throw new Error("All history filter did not render.");
    await all.click();
    await browser.waitUntil(() => segmentedIndicatorAligned(".filter-tabs"));
  });

  it("uses the relaxed Resources spacing contract", async () => {
    await $(`a=Resources`).click();
    await expect($("h1")).toHaveText("Resources");
    await $(".resource-card").waitForDisplayed();
    const spacing = await browser.execute(() => {
      const grid = document.querySelector<HTMLElement>(".resource-grid");
      const card = document.querySelector<HTMLElement>(".resource-card");
      const section = document.querySelector<HTMLElement>(".resources-page .section");
      return {
        gridGap: grid ? getComputedStyle(grid).gap : "",
        cardPadding: card ? getComputedStyle(card).paddingTop : "",
        sectionMargin: section ? getComputedStyle(section).marginTop : "",
      };
    });
    expect(spacing).toEqual({ gridGap: "16px", cardPadding: "20px", sectionMargin: "36px" });
  });

  it("requires an explicit decision before leaving a modified Draft", async () => {
    await $(`a=New Task`).click();
    const sourceLabel = $("label=Source language");
    const sourceInputId = await sourceLabel.getAttribute("for");
    if (!sourceInputId) throw new Error("Source language label is not associated with its input.");
    const sourceLanguage = $(`#${sourceInputId}`);
    await sourceLanguage.setValue("en");
    await $(`a=Home`).click();
    await expect($("h2=Discard this draft?")).toBeDisplayed();
    await $(`button=Stay`).click();
    await expect($("h1")).toHaveText("New Task");

    await $(`a=Home`).click();
    await $(`button=Discard Draft`).click();
    await expect($("h1")).toHaveText("Home");
  });
});

async function clickSidebarThemeOption(source: "light" | "system" | "dark"): Promise<void> {
  const clicked = await browser.execute((nextSource) => {
    const index = { light: 0, system: 1, dark: 2 }[nextSource];
    const option = document.querySelectorAll<HTMLElement>(".theme-segment .segment-option")[index];
    option?.click();
    return Boolean(option);
  }, source);
  expect(clicked).toBe(true);
}

async function clickSettingsThemeOption(source: "light" | "system" | "dark"): Promise<void> {
  const clicked = await browser.execute((nextSource) => {
    const index = { system: 0, light: 1, dark: 2 }[nextSource];
    const option = document.querySelectorAll<HTMLElement>(".settings-theme-segment .segment-option")[index];
    option?.click();
    return Boolean(option);
  }, source);
  expect(clicked).toBe(true);
}

async function themeOptionChecked(source: "light" | "system" | "dark"): Promise<boolean> {
  return browser.execute((nextSource) => {
    const index = { system: 0, light: 1, dark: 2 }[nextSource];
    const option = document.querySelectorAll<HTMLElement>(".settings-theme-segment .segment-option")[index];
    return option?.hasAttribute("data-selected") ?? false;
  }, source);
}

async function clickReduceMotionSwitch(): Promise<void> {
  await $(".toggle-switch").click();
}

async function reduceMotionChecked(): Promise<boolean> {
  return browser.execute(() => {
    const field = document.querySelector<HTMLElement>("#setting-reduce-motion");
    const input = field?.matches('[role="switch"]')
      ? field
      : field?.querySelector<HTMLElement>('[role="switch"]');
    return input instanceof HTMLInputElement ? input.checked : false;
  });
}

async function switchGeometry(): Promise<{
  trackCenterY: number;
  thumbCenterY: number;
  thumbLeft: number;
}> {
  return browser.execute(() => {
    const track = document.querySelector<HTMLElement>(".toggle-switch")?.getBoundingClientRect();
    const thumb = document.querySelector<HTMLElement>(".toggle-switch-thumb")?.getBoundingClientRect();
    return {
      trackCenterY: track ? track.top + track.height / 2 : 0,
      thumbCenterY: thumb ? thumb.top + thumb.height / 2 : 0,
      thumbLeft: thumb?.left ?? 0,
    };
  });
}

async function segmentedIndicatorAligned(selector: string): Promise<boolean> {
  return browser.execute((segmentSelector) => {
    const segment = document.querySelector<HTMLElement>(segmentSelector);
    const slider = segment?.querySelector<HTMLElement>(".segment-slider")?.getBoundingClientRect();
    const selected = segment
      ?.querySelector<HTMLElement>(".segment-option[data-selected], .filter-option[data-selected]")
      ?.getBoundingClientRect();
    if (!slider || !selected) return false;
    return Math.abs(slider.left + slider.width / 2 - (selected.left + selected.width / 2)) <= 0.5;
  }, selector);
}

async function primaryButtonVisual(): Promise<{
  filter: string;
  transform: string;
  left: number;
  top: number;
}> {
  return browser.execute(() => {
    const button = document.querySelector<HTMLElement>(".button.primary");
    const rect = button?.getBoundingClientRect();
    const style = button ? getComputedStyle(button) : undefined;
    return {
      filter: style?.filter ?? "",
      transform: style?.transform ?? "",
      left: rect?.left ?? -1,
      top: rect?.top ?? -1,
    };
  });
}

async function settingsTabVisual(index: number): Promise<{
  selected: boolean;
  focused: boolean;
  color: string;
  background: string;
  expectedColor: string;
  expectedBackground: string;
}> {
  return browser.execute((tabIndex) => {
    const tab = document.querySelectorAll<HTMLElement>(".settings-tab-list [role=tab]")[tabIndex];
    const style = tab ? getComputedStyle(tab) : undefined;
    const probe = document.createElement("span");
    probe.style.color = "var(--color-primary)";
    probe.style.backgroundColor = "var(--color-primary-soft)";
    document.body.append(probe);
    const expected = getComputedStyle(probe);
    const result = {
      selected: tab?.hasAttribute("data-selected") ?? false,
      focused: document.activeElement === tab && (tab?.hasAttribute("data-focused") ?? false),
      color: style?.color ?? "",
      background: style?.backgroundColor ?? "",
      expectedColor: expected.color,
      expectedBackground: expected.backgroundColor,
    };
    probe.remove();
    return result;
  }, index);
}

async function historyFilterVisual(): Promise<{
  ready: boolean;
  aligned: boolean;
  optionCount: number;
  selectedIndex: number;
  sliderLeft: number;
  transition: string;
}> {
  return browser.execute(() => {
    const group = document.querySelector<HTMLElement>(".filter-tabs");
    const slider = group?.querySelector<HTMLElement>(".toggle-group-slider");
    const options = Array.from(group?.querySelectorAll<HTMLElement>(".filter-option") ?? []);
    const selectedIndex = options.findIndex((option) => option.hasAttribute("data-selected"));
    const selected = options[selectedIndex];
    const sliderRect = slider?.getBoundingClientRect();
    const selectedRect = selected?.getBoundingClientRect();
    return {
      ready: group?.getAttribute("data-slider-ready") === "true",
      aligned: Boolean(
        sliderRect &&
        selectedRect &&
        Math.abs(sliderRect.left + sliderRect.width / 2 - (selectedRect.left + selectedRect.width / 2)) <=
          0.5,
      ),
      optionCount: options.length,
      selectedIndex,
      sliderLeft: sliderRect?.left ?? -1,
      transition: slider ? getComputedStyle(slider).transitionProperty : "",
    };
  });
}
