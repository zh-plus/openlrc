import { useEffect, type ReactNode } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { Menu, MenuItem, MenuTrigger, Popover } from "react-aria-components";
import {
  CheckSquare2,
  ChevronLeft,
  ChevronRight,
  CirclePlus,
  Clock3,
  Home,
  Laptop,
  Moon,
  PackageSearch,
  RotateCw,
  Settings,
  Sun,
} from "lucide-react";
import clsx from "clsx";

import type { AppearanceState, BackendStatus, ThemeSource } from "../../shared/contracts.js";
import { queryKeys } from "../app/queryKeys.js";
import { AppNavLink, Button, IconButton, SegmentedControl, Tooltip } from "./ui/index.js";

const primaryItems = [
  { to: "/home", icon: Home, key: "nav.home" },
  { to: "/new", icon: CirclePlus, key: "nav.new" },
  { to: "/tasks", icon: CheckSquare2, key: "nav.tasks", badge: true },
  { to: "/history", icon: Clock3, key: "nav.history" },
] as const;

export function AppShell(): React.JSX.Element {
  const { t, i18n } = useTranslation();
  const location = useLocation();
  const queryClient = useQueryClient();
  const appearanceQuery = useQuery({
    queryKey: queryKeys.appearance,
    queryFn: () => window.openlrc.appearance.get(),
  });
  const backendQuery = useQuery({
    queryKey: queryKeys.backend,
    queryFn: () => window.openlrc.app.backendStatus(),
  });
  const queueQuery = useQuery({ queryKey: queryKeys.queue, queryFn: () => window.openlrc.queue.snapshot() });
  const settingsQuery = useQuery({
    queryKey: queryKeys.settings,
    queryFn: () => window.openlrc.settings.get(),
  });
  const appearance = appearanceQuery.data;
  const collapsed = appearance?.sidebarCollapsed ?? false;

  useEffect(() => {
    const unsubscribe = window.openlrc.appearance.onChanged((next) => {
      applyAppearance(next);
      queryClient.setQueryData(queryKeys.appearance, next);
    });
    return unsubscribe;
  }, [queryClient]);

  useEffect(() => {
    const language = settingsQuery.data?.general.language ?? "en";
    void i18n.changeLanguage(language);
    document.documentElement.lang = language;
    document.documentElement.classList.toggle(
      "reduce-motion",
      settingsQuery.data?.general.reduce_motion ?? false,
    );
  }, [i18n, settingsQuery.data]);

  useEffect(() => {
    document.getElementById("main-content")?.scrollTo({ top: 0, left: 0 });
  }, [location.pathname]);

  const setTheme = async (source: ThemeSource): Promise<void> => {
    const next = await window.openlrc.appearance.setTheme(source);
    applyAppearance(next);
    queryClient.setQueryData(queryKeys.appearance, next);
  };

  const toggleSidebar = async (): Promise<void> => {
    const next = await window.openlrc.appearance.setSidebarCollapsed(!collapsed);
    queryClient.setQueryData(queryKeys.appearance, next);
  };

  return (
    <div className={clsx("app-shell", collapsed && "sidebar-collapsed")}>
      <div className="window-drag-region" aria-hidden="true" />
      <aside className="sidebar" aria-label={t("nav.primary")}>
        <div className="brand-row">
          <div className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          {!collapsed && <span className="brand-name">OpenLRC</span>}
          <Tooltip label={collapsed ? t("nav.expand") : t("nav.collapse")}>
            <IconButton
              label={collapsed ? t("nav.expand") : t("nav.collapse")}
              onPress={() => void toggleSidebar()}
              className="collapse-button"
            >
              {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
            </IconButton>
          </Tooltip>
        </div>
        <nav className="primary-nav">
          {primaryItems.map((item) => {
            const count = queueQuery.data?.total ?? 0;
            return (
              <SidebarLink
                key={item.to}
                to={item.to}
                icon={item.icon}
                label={t(item.key)}
                collapsed={collapsed}
                trailing={
                  "badge" in item && item.badge && count > 0 ? (
                    <span className="nav-badge" aria-label={t("nav.unfinished", { count })}>
                      {count}
                    </span>
                  ) : undefined
                }
              />
            );
          })}
        </nav>
        <div className="sidebar-tools">
          <SidebarLink
            to="/resources"
            icon={PackageSearch}
            label={t("nav.resources")}
            collapsed={collapsed}
            trailing={
              <span
                className={clsx("resource-dot", backendQuery.data?.state === "ready" ? "healthy" : "warning")}
              />
            }
          />
          <div className="theme-wrap">
            {collapsed ? (
              <MenuTrigger>
                <IconButton
                  label={t("theme.current", {
                    source: t(`theme.${appearance?.themeSource ?? "system"}`),
                  })}
                  className="nav-icon-button"
                >
                  {renderThemeIcon(appearance?.themeSource ?? "system")}
                </IconButton>
                <Popover className="theme-popup" placement="right bottom">
                  <Menu
                    aria-label={t("theme.source")}
                    selectionMode="single"
                    selectedKeys={new Set([appearance?.themeSource ?? "system"])}
                    onAction={(key) => void setTheme(String(key) as ThemeSource)}
                  >
                    {themeSources.map((source) => (
                      <MenuItem id={source} className="theme-menu-item" key={source}>
                        {renderThemeIcon(source)}
                        <span>{t(`theme.${source}`)}</span>
                        {appearance?.themeSource === source ? <CheckSquare2 size={15} /> : null}
                      </MenuItem>
                    ))}
                  </Menu>
                </Popover>
              </MenuTrigger>
            ) : (
              <SegmentedControl
                label={t("theme.source")}
                value={appearance?.themeSource ?? "system"}
                className="theme-segment w-full"
                optionClassName="theme-option"
                onChange={(source) => void setTheme(source as ThemeSource)}
                options={themeSources.map((source) => ({
                  value: source,
                  label: <span className="sr-only">{t(`theme.${source}`)}</span>,
                  icon: renderThemeIcon(source),
                  ariaLabel: t("theme.option", { source: t(`theme.${source}`) }),
                }))}
              />
            )}
          </div>
          <SidebarLink to="/settings" icon={Settings} label={t("nav.settings")} collapsed={collapsed} />
        </div>
      </aside>
      <main className="content" id="main-content" data-route={location.pathname}>
        <BackendBanner status={backendQuery.data} />
        <Outlet />
      </main>
    </div>
  );
}

function BackendBanner({ status }: { status: BackendStatus | undefined }): React.JSX.Element | null {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  if (!status || status.state === "ready" || status.state === "busy") return null;
  const failed = status.state === "failed" || status.state === "stopped";
  return (
    <div className={clsx("persistent-banner", failed ? "danger" : "info")} role={failed ? "alert" : "status"}>
      <div>
        <strong>{failed ? t("backend.disconnected") : t("backend.starting")}</strong>
        {status.message && <span>{status.message}</span>}
      </div>
      {failed && (
        <Button
          variant="secondary"
          size="small"
          onPress={async () => {
            const next = await window.openlrc.app.restartBackend();
            queryClient.setQueryData(queryKeys.backend, next);
          }}
        >
          <RotateCw size={16} /> {t("backend.restart")}
        </Button>
      )}
    </div>
  );
}

function SidebarLink({
  to,
  icon: Icon,
  label,
  collapsed,
  trailing,
}: {
  to: string;
  icon: typeof Home;
  label: string;
  collapsed: boolean;
  trailing?: ReactNode;
}): React.JSX.Element {
  const link = (
    <AppNavLink to={to} className="nav-item" {...(collapsed ? { "aria-label": label } : {})}>
      <Icon size={20} aria-hidden="true" />
      {!collapsed && <span>{label}</span>}
      {trailing}
    </AppNavLink>
  );
  if (!collapsed) return link;
  return <Tooltip label={label}>{link}</Tooltip>;
}

const themeSources: readonly ThemeSource[] = ["light", "system", "dark"];

function renderThemeIcon(source: ThemeSource): React.JSX.Element {
  if (source === "light") return <Sun size={16} aria-hidden="true" />;
  if (source === "dark") return <Moon size={16} aria-hidden="true" />;
  return <Laptop size={16} aria-hidden="true" />;
}

function applyAppearance(appearance: AppearanceState): void {
  document.documentElement.dataset.theme = appearance.effectiveTheme;
  document.documentElement.dataset.themeSource = appearance.themeSource;
}
