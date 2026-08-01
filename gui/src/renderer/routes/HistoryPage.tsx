import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { ArrowRight } from "lucide-react";

import type { JobSummary } from "../../shared/contracts.js";
import { queryKeys } from "../app/queryKeys.js";
import { EmptyState, ErrorState, LoadingState, PageHeader, StatusPill } from "../components/Page.js";
import { AppLink, SearchField, SingleToggleGroup } from "../components/ui/index.js";
import { basename, formatDate, formatDuration } from "../utils/format.js";

const filters = ["all", "completed", "failed"] as const;
type Filter = (typeof filters)[number];

export function HistoryPage(): React.JSX.Element {
  const { t } = useTranslation();
  const [filter, setFilter] = useState<Filter>("all");
  const [search, setSearch] = useState("");
  const jobs = useQuery({ queryKey: queryKeys.jobs, queryFn: () => window.openlrc.jobs.list() });
  const visible = useMemo(() => {
    const needle = search.trim().toLocaleLowerCase();
    return (jobs.data ?? []).filter((job) => matchesStatus(job, filter) && matchesSearch(job, needle));
  }, [filter, jobs.data, search]);
  return (
    <div className="page history-page">
      <PageHeader title={t("history.title")} subtitle={t("history.subtitle")} />
      <div className="history-toolbar">
        <SearchField
          label={t("history.search")}
          value={search}
          onChange={setSearch}
          placeholder={t("history.search")}
        />
        <SingleToggleGroup
          label={t("history.statusFilter")}
          value={filter}
          className="filter-tabs"
          onChange={(next) => setFilter(next as Filter)}
          options={filters.map((item) => ({ value: item, label: t(`history.${item}`) }))}
        />
      </div>
      {jobs.isLoading && <LoadingState />}
      {jobs.error && <ErrorState error={jobs.error} retry={() => void jobs.refetch()} />}
      {!jobs.isLoading && !jobs.error && visible.length === 0 && (
        <EmptyState
          title={t("history.nothing")}
          detail={t("history.empty")}
          action={
            (jobs.data?.length ?? 0) === 0 ? (
              <AppLink to="/new" className="button primary">
                {t("common.createNewTask")}
              </AppLink>
            ) : undefined
          }
        />
      )}
      {visible.length > 0 && (
        <div className="history-list" role="list">
          <div className="history-columns" aria-hidden="true">
            <span>{t("history.taskColumn")}</span>
            <span>{t("history.statusColumn")}</span>
            <span>{t("history.completedColumn")}</span>
            <span>{t("history.durationColumn")}</span>
            <span />
          </div>
          {visible.map((job) => (
            <div role="listitem" className="history-listitem" key={job.job_id}>
              <AppLink to={`/history/${job.job_id}`} className="history-row">
                <span className="history-name">
                  <strong>{job.name}</strong>
                  <small title={job.input_paths[0]}>
                    {t(
                      job.workflow === "run"
                        ? "newTask.fullRun"
                        : job.workflow === "translate"
                          ? "newTask.translate"
                          : "newTask.transcribe",
                    )}{" "}
                    · {basename(job.input_paths[0] ?? "")}
                  </small>
                </span>
                <StatusPill status={job.status} />
                <span>{formatDate(job.completed_at)}</span>
                <span className="history-duration">{formatDuration(job.elapsed_seconds)}</span>
                <ArrowRight size={17} aria-hidden="true" />
              </AppLink>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function matchesStatus(job: JobSummary, filter: Filter): boolean {
  if (filter === "all") return true;
  if (filter === "completed") return job.status === "succeeded";
  return ["succeeded_with_warnings", "failed", "interrupted"].includes(job.status);
}

function matchesSearch(job: JobSummary, needle: string): boolean {
  if (!needle) return true;
  return [job.name, ...job.input_paths, ...job.outputs].some((value) =>
    value.toLocaleLowerCase().includes(needle),
  );
}
