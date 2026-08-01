import type { PropsWithChildren, ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { AlertCircle, Inbox, LoaderCircle } from "lucide-react";
import { Button } from "./ui/index.js";

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}): React.JSX.Element {
  return (
    <header className="page-header">
      <div>
        <h1>{title}</h1>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  );
}

export function Section({
  title,
  meta,
  children,
  className = "",
}: PropsWithChildren<{ title?: string; meta?: ReactNode; className?: string }>): React.JSX.Element {
  return (
    <section className={`section ${className}`}>
      {(title || meta) && (
        <div className="section-heading">
          {title && <h2>{title}</h2>}
          {meta}
        </div>
      )}
      {children}
    </section>
  );
}

export function LoadingState({ label }: { label?: string }): React.JSX.Element {
  const { t } = useTranslation();
  return (
    <div className="state-panel" role="status">
      <LoaderCircle className="spinner" size={24} />
      <span>{label ?? t("common.loading")}</span>
    </div>
  );
}

export function ErrorState({ error, retry }: { error: unknown; retry?: () => void }): React.JSX.Element {
  const { t } = useTranslation();
  return (
    <div className="state-panel error" role="alert">
      <AlertCircle size={24} />
      <strong>{t("common.unableToLoad")}</strong>
      <span>{error instanceof Error ? error.message : String(error)}</span>
      {retry && (
        <Button variant="secondary" onPress={retry}>
          {t("common.retry")}
        </Button>
      )}
    </div>
  );
}

export function EmptyState({
  title,
  detail,
  action,
}: {
  title: string;
  detail: string;
  action?: ReactNode;
}): React.JSX.Element {
  return (
    <div className="state-panel empty">
      <Inbox size={28} />
      <strong>{title}</strong>
      <span>{detail}</span>
      {action}
    </div>
  );
}

export function StatusPill({ status }: { status: string }): React.JSX.Element {
  const { t } = useTranslation();
  const fallback = status.replaceAll("_", " ");
  return (
    <span className={`status-pill status-${status.replaceAll("_", "-")}`}>
      {t(`jobStatus.${status}`, { defaultValue: fallback })}
    </span>
  );
}
