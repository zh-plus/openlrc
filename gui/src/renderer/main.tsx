import React from "react";
import type { ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { createHashRouter, RouterProvider } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { I18nProvider } from "react-aria-components";
import { useTranslation } from "react-i18next";

import "./i18n.js";
import "./styles.css";
import { App } from "./app/App.js";
import { OperationProvider } from "./app/OperationContext.js";

async function bootstrap(): Promise<void> {
  const appearance = await window.openlrc.appearance.get();
  document.documentElement.dataset.theme = appearance.effectiveTheme;
  document.documentElement.dataset.themeSource = appearance.themeSource;
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { staleTime: 15_000, retry: 1 },
      mutations: { retry: false },
    },
  });
  const router = createHashRouter([
    {
      path: "*",
      element: (
        <AriaLocaleProvider>
          <OperationProvider>
            <App />
          </OperationProvider>
        </AriaLocaleProvider>
      ),
    },
  ]);
  createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    </React.StrictMode>,
  );
}

function AriaLocaleProvider({ children }: { children: ReactNode }): React.JSX.Element {
  const { i18n } = useTranslation();
  const locale = i18n.resolvedLanguage?.toLowerCase().startsWith("zh") ? "zh-CN" : "en-US";
  return <I18nProvider locale={locale}>{children}</I18nProvider>;
}

void bootstrap();
