import { useEffect } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";

import { AppShell } from "../components/AppShell.js";
import { AboutPage } from "../routes/AboutPage.js";
import { HistoryDetailPage } from "../routes/HistoryDetailPage.js";
import { HistoryPage } from "../routes/HistoryPage.js";
import { HomePage } from "../routes/HomePage.js";
import { NewTaskPage } from "../routes/NewTaskPage.js";
import { ResourcesPage } from "../routes/ResourcesPage.js";
import { SettingsPage } from "../routes/SettingsPage.js";
import { TasksPage } from "../routes/TasksPage.js";
import { queryKeys } from "./queryKeys.js";

export function App(): React.JSX.Element {
  const queryClient = useQueryClient();
  useEffect(
    () =>
      window.openlrc.app.onBackendStatus((status) => {
        queryClient.setQueryData(queryKeys.backend, status);
      }),
    [queryClient],
  );
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/home" element={<HomePage />} />
        <Route path="/new" element={<NewTaskPage />} />
        <Route path="/tasks" element={<TasksPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/history/:jobId" element={<HistoryDetailPage />} />
        <Route path="/resources" element={<ResourcesPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="*" element={<Navigate to="/home" replace />} />
      </Route>
    </Routes>
  );
}
