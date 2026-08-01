import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { ArrowLeft, ExternalLink, File, FolderOpen, RotateCcw, Trash2, TriangleAlert } from "lucide-react";

import { queryKeys } from "../app/queryKeys.js";
import { ErrorState, LoadingState, PageHeader, Section, StatusPill } from "../components/Page.js";
import { AppLink, Button, ModalDialog } from "../components/ui/index.js";
import { formatDate, formatDuration, middleTruncate } from "../utils/format.js";

export function HistoryDetailPage(): React.JSX.Element {
  const { t } = useTranslation();
  const { jobId = "" } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [deleteOpen, setDeleteOpen] = useState(false);
  const job = useQuery({
    queryKey: queryKeys.job(jobId),
    queryFn: () => window.openlrc.jobs.get(jobId),
    enabled: Boolean(jobId),
  });
  const remove = useMutation({
    mutationFn: () => window.openlrc.jobs.delete(jobId),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: queryKeys.jobs });
      navigate("/history");
    },
  });
  const resume = useMutation({
    mutationFn: () => window.openlrc.jobs.resumeDraft(jobId),
    onSuccess: (draft) => navigate("/new", { state: { draft } }),
  });
  if (job.isLoading)
    return (
      <div className="page">
        <LoadingState />
      </div>
    );
  if (job.error || !job.data)
    return (
      <div className="page">
        <ErrorState error={job.error ?? new Error(t("detail.notFound"))} />
      </div>
    );
  const data = job.data;
  const canResume = ["failed", "cancelled", "interrupted", "succeeded_with_warnings"].includes(data.status);
  return (
    <div className="page history-detail-page">
      <AppLink to="/history" className="back-link">
        <ArrowLeft size={16} />
        {t("detail.back")}
      </AppLink>
      <PageHeader
        title={data.name}
        subtitle={`${workflowLabel(data.workflow, t)} · ${t("detail.started", {
          date: formatDate(data.started_at),
        })} · ${formatDuration(data.elapsed_seconds)}`}
        actions={<StatusPill status={data.status} />}
      />
      {data.error != null && (
        <div className="persistent-banner danger">
          <TriangleAlert size={18} />
          <div>
            <strong>{t("detail.failed")}</strong>
            <span>{errorMessage(data.error, t("detail.safeError"))}</span>
          </div>
        </div>
      )}
      <div className="detail-grid">
        <Section title={t("detail.inputs")} className="panel-section">
          <div className="path-list">
            {data.input_paths.map((item) => (
              <div className="path-row" key={item}>
                <File size={17} />
                <code title={item}>{middleTruncate(item)}</code>
              </div>
            ))}
          </div>
        </Section>
        <Section title={t("detail.facts")} className="panel-section">
          <dl className="facts">
            <div>
              <dt>{t("common.status")}</dt>
              <dd>{t(`jobStatus.${data.status}`)}</dd>
            </div>
            <div>
              <dt>{t("common.mode")}</dt>
              <dd>{data.translation_mode ?? "—"}</dd>
            </div>
            <div>
              <dt>{t("detail.apiFee")}</dt>
              <dd>${data.api_fee.toFixed(4)}</dd>
            </div>
            <div>
              <dt>{t("detail.finished")}</dt>
              <dd>{formatDate(data.completed_at)}</dd>
            </div>
          </dl>
        </Section>
      </div>
      <Section title={t("detail.artifacts")}>
        {data.artifacts.length === 0 ? (
          <p className="muted">{t("detail.noArtifacts")}</p>
        ) : (
          <div className="artifact-list">
            {data.artifacts.map((artifact) => (
              <div className="artifact-row" key={artifact.artifact_id}>
                <div>
                  <File size={18} />
                  <span>
                    <strong>{artifact.kind.replaceAll("-", " ")}</strong>
                    <code title={artifact.path}>{middleTruncate(artifact.path, 80)}</code>
                  </span>
                </div>
                <div className="row-actions">
                  <Button
                    variant="secondary"
                    size="small"
                    onPress={() => window.openlrc.artifacts.open(jobId, artifact.artifact_id)}
                  >
                    <ExternalLink size={15} />
                    {t("common.open")}
                  </Button>
                  <Button
                    variant="secondary"
                    size="small"
                    onPress={() => window.openlrc.artifacts.reveal(jobId, artifact.artifact_id)}
                  >
                    <FolderOpen size={15} />
                    {t("common.reveal")}
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </Section>
      {(data.reviews.length > 0 || data.status === "succeeded_with_warnings") && (
        <Section title={t("detail.review")}>
          <pre className="data-block">{JSON.stringify(data.reviews, null, 2)}</pre>
        </Section>
      )}
      <Section title={t("detail.recipe")}>
        <pre className="data-block">{JSON.stringify(data.recipe, null, 2)}</pre>
      </Section>
      <Section title={t("detail.eventLog")}>
        <pre className="data-block log-block">
          {data.event_log.length ? data.event_log.join("\n") : t("detail.noLog")}
        </pre>
      </Section>
      <div className="detail-actions">
        <Button variant="danger" className="secondary" onPress={() => setDeleteOpen(true)}>
          <Trash2 size={16} />
          {t("detail.deleteRecord")}
        </Button>
        {canResume && (
          <Button variant="primary" isPending={resume.isPending} onPress={() => resume.mutate()}>
            <RotateCcw size={16} />
            {t("detail.resume")}
          </Button>
        )}
      </div>
      <ModalDialog
        isOpen={deleteOpen}
        onOpenChange={setDeleteOpen}
        isDismissable={!remove.isPending}
        title={t("detail.deleteRecord")}
        description={t("detail.deleteConfirm", { name: data.name })}
      >
        <div className="dialog-actions">
          <Button variant="secondary" autoFocus onPress={() => setDeleteOpen(false)}>
            {t("common.cancel")}
          </Button>
          <Button
            variant="danger"
            isPending={remove.isPending}
            isDisabled={remove.isPending}
            onPress={() => remove.mutate()}
          >
            <Trash2 size={16} />
            {t("detail.deleteRecord")}
          </Button>
        </div>
      </ModalDialog>
    </div>
  );
}

function errorMessage(error: unknown, fallback: string): string {
  if (typeof error === "object" && error !== null && "message" in error && typeof error.message === "string")
    return error.message;
  return fallback;
}

function workflowLabel(workflow: string, t: ReturnType<typeof useTranslation>["t"]): string {
  if (workflow === "transcribe") return t("newTask.transcribe");
  if (workflow === "translate") return t("newTask.translate");
  if (workflow === "run") return t("newTask.fullRun");
  return workflow;
}
