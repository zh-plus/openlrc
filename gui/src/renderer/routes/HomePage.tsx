import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  ArrowRight,
  CheckCircle2,
  CirclePlus,
  Clock3,
  ListChecks,
  PackageCheck,
  TriangleAlert,
} from "lucide-react";

import { PageHeader, LoadingState, ErrorState, EmptyState, Section, StatusPill } from "../components/Page.js";
import { AppLink } from "../components/ui/index.js";
import { queryKeys } from "../app/queryKeys.js";
import { formatDate, formatDuration } from "../utils/format.js";

export function HomePage(): React.JSX.Element {
  const { t } = useTranslation();
  const queue = useQuery({ queryKey: queryKeys.queue, queryFn: () => window.openlrc.queue.snapshot() });
  const jobs = useQuery({ queryKey: queryKeys.jobs, queryFn: () => window.openlrc.jobs.list() });
  const resources = useQuery({
    queryKey: queryKeys.resources,
    queryFn: () => window.openlrc.resources.status(),
  });
  const loading = queue.isLoading || jobs.isLoading || resources.isLoading;
  const error = queue.error ?? jobs.error ?? resources.error;
  const available = resources.data?.filter((item) => item.available).length ?? 0;
  const totalResources = resources.data?.length ?? 0;
  const resourceReady = totalResources > 0 && available === totalResources;
  return (
    <div className="page home-page">
      <PageHeader
        title={t("home.title")}
        subtitle={t("home.subtitle")}
        actions={
          <AppLink className="button primary" to="/new">
            <CirclePlus size={17} />
            {t("common.newTask")}
          </AppLink>
        }
      />
      {loading && <LoadingState />}
      {error && <ErrorState error={error} />}
      {!loading && !error && (
        <>
          <div className="summary-grid">
            <AppLink to="/tasks" className="summary-card accent-blue">
              <div className="summary-icon">
                <ListChecks size={21} />
              </div>
              <div>
                <span>{t("home.tasks")}</span>
                <strong>{queue.data?.active ? t("home.oneActive") : t("home.noActive")}</strong>
                <small>{t("home.queuedCount", { count: queue.data?.pending_count ?? 0 })}</small>
              </div>
              <ArrowRight size={18} />
            </AppLink>
            <AppLink to="/resources" className="summary-card accent-violet">
              <div className="summary-icon">
                {resourceReady ? <PackageCheck size={21} /> : <TriangleAlert size={21} />}
              </div>
              <div>
                <span>{t("home.resources")}</span>
                <strong>
                  {resourceReady
                    ? t("home.allReady")
                    : t("home.attentionCount", { count: totalResources - available })}
                </strong>
                <small>{t("home.availableCount", { available, total: totalResources })}</small>
              </div>
              <ArrowRight size={18} />
            </AppLink>
          </div>
          <Section
            title={t("home.recent")}
            meta={
              <AppLink className="text-link" to="/history">
                {t("home.viewAll")} <ArrowRight size={15} />
              </AppLink>
            }
          >
            {(jobs.data?.length ?? 0) === 0 ? (
              <EmptyState
                title={t("home.historyStarts")}
                detail={t("home.empty")}
                action={
                  <AppLink to="/new" className="button secondary">
                    {t("common.createNewTask")}
                  </AppLink>
                }
              />
            ) : (
              <div className="history-list compact">
                {jobs.data?.slice(0, 5).map((job) => (
                  <AppLink to={`/history/${job.job_id}`} className="history-row" key={job.job_id}>
                    <span className="history-status-icon">
                      {job.status.startsWith("succeeded") ? <CheckCircle2 size={18} /> : <Clock3 size={18} />}
                    </span>
                    <span className="history-name">
                      <strong>{job.name}</strong>
                      <small>
                        {workflowLabel(job.workflow, t)} · {formatDate(job.completed_at)}
                      </small>
                    </span>
                    <StatusPill status={job.status} />
                    <span className="history-duration">{formatDuration(job.elapsed_seconds)}</span>
                    <ArrowRight size={16} />
                  </AppLink>
                ))}
              </div>
            )}
          </Section>
        </>
      )}
    </div>
  );
}

function workflowLabel(workflow: string, t: ReturnType<typeof useTranslation>["t"]): string {
  if (workflow === "transcribe") return t("newTask.transcribe");
  if (workflow === "translate") return t("newTask.translate");
  if (workflow === "run") return t("newTask.fullRun");
  return workflow;
}
