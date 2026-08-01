import "@testing-library/jest-dom/vitest";

class ResizeObserverMock implements ResizeObserver {
  disconnect(): void {}

  observe(): void {}

  unobserve(): void {}
}

Object.defineProperty(globalThis, "ResizeObserver", {
  configurable: true,
  value: ResizeObserverMock,
});
