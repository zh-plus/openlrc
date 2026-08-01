import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  ArrowDown,
  ArrowUp,
  Ban,
  Check,
  CirclePlus,
  Clock3,
  GripVertical,
  LoaderCircle,
  Pause,
  Play,
  Square,
  TriangleAlert,
} from "lucide-react";

import type { QueueEntry, QueueSnapshot } from "../../shared/contracts.js";
import { humanize, useLiveOperation } from "../app/OperationContext.js";
import { queryKeys } from "../app/queryKeys.js";
import { EmptyState, ErrorState, LoadingState, PageHeader, Section } from "../components/Page.js";
import { AppLink, Button, Disclosure, IconButton, ProgressBar, Tooltip } from "../components/ui/index.js";
import { basename } from "../utils/format.js";

export function TasksPage(): React.JSX.Element {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const live = useLiveOperation();
  const queue = useQuery({ queryKey: queryKeys.queue, queryFn: () => window.openlrc.queue.snapshot() });
  const updateSnapshot = (snapshot: QueueSnapshot): void => {
    queryClient.setQueryData(queryKeys.queue, snapshot);
  };
  const pause = useMutation({ mutationFn: () => window.openlrc.queue.pause(), onSuccess: updateSnapshot });
  const resume = useMutation({ mutationFn: () => window.openlrc.queue.resume(), onSuccess: updateSnapshot });
  const cancel = useMutation({
    mutationFn: (queueId: string) => window.openlrc.queue.cancel(queueId),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: queryKeys.queue }),
  });
  const reorder = useMutation({
    mutationFn: (ids: string[]) => window.openlrc.queue.reorder(ids),
    onSuccess: updateSnapshot,
  });
  const move = (index: number, direction: -1 | 1): void => {
    const entries = queue.data?.entries ?? [];
    const target = index + direction;
    if (target < 0 || target >= entries.length) return;
    const ids = entries.map((entry) => entry.queue_id);
    [ids[index], ids[target]] = [ids[target]!, ids[index]!];
    reorder.mutate(ids);
  };
  const active = queue.data?.active;
  const activeLive = active && live.queueId === active.queue_id ? live : null;
  return (
    <div className="page tasks-page">
      <PageHeader
        title={t("tasks.title")}
        subtitle={`${active ? 1 : 0} ${t("common.active")} · ${queue.data?.pending_count ?? 0} ${t("common.queued")}`}
        actions={
          <AppLink to="/new" className="button primary">
            <CirclePlus size={17} />
            {t("common.newTask")}
          </AppLink>
        }
      />
      {queue.isLoading && <LoadingState />}
      {queue.error && <ErrorState error={queue.error} retry={() => void queue.refetch()} />}
      {queue.data?.paused && (
        <div className="persistent-banner warning" role="status">
          <div>
            <strong>{t("tasks.paused")}</strong>
            <span>{t("tasks.pausedDetail")}</span>
          </div>
          <Button variant="secondary" size="small" onPress={() => resume.mutate()}>
            <Play size={15} />
            {t("tasks.resume")}
          </Button>
        </div>
      )}
      {queue.data && (
        <>
          <Section title={t("tasks.active")}>
            {active ? (
              <ActiveTaskCard
                entry={active}
                live={activeLive}
                onCancel={() => cancel.mutate(active.queue_id)}
              />
            ) : (
              <EmptyState title={t("tasks.calm")} detail={t("tasks.noActive")} />
            )}
          </Section>
          <Section
            title={t("tasks.queued")}
            meta={
              <Button
                variant="secondary"
                size="small"
                onPress={() => (queue.data.paused ? resume.mutate() : pause.mutate())}
              >
                {queue.data.paused ? <Play size={15} /> : <Pause size={15} />}
                {queue.data.paused ? t("tasks.resume") : t("tasks.pause")}
              </Button>
            }
          >
            {queue.data.entries.length === 0 ? (
              <EmptyState title={t("tasks.queueEmpty")} detail={t("tasks.empty")} />
            ) : (
              <div className="queue-list" aria-label={t("tasks.listLabel")}>
                {queue.data.entries.map((entry, index) => (
                  <QueuedRow
                    key={entry.queue_id}
                    entry={entry}
                    index={index}
                    total={queue.data.entries.length}
                    onMove={(direction) => move(index, direction)}
                    onRemove={() => cancel.mutate(entry.queue_id)}
                  />
                ))}
              </div>
            )}
            {queue.data.entries.length > 1 && <p className="keyboard-hint">{t("tasks.reorderHint")}</p>}
          </Section>
        </>
      )}
    </div>
  );
}

function ActiveTaskCard({
  entry,
  live,
  onCancel,
}: {
  entry: QueueEntry;
  live: ReturnType<typeof useLiveOperation> | null;
  onCancel(): void;
}): React.JSX.Element {
  const { t } = useTranslation();
  const stages = stagesFor(entry.workflow);
  const completedStages = stages.filter((stage) => live?.stages[stage] === "completed").length;
  const currentStageProgress = Math.max(0, Math.min(100, live?.percent ?? 0));
  const progress = Math.min(
    100,
    Math.round(((completedStages + currentStageProgress / 100) / stages.length) * 100),
  );
  const currentIndex = Math.max(0, stages.indexOf(live?.stage ?? "validate"));
  const inputPaths = Array.isArray(entry.draft_recipe.paths)
    ? entry.draft_recipe.paths.filter((item): item is string => typeof item === "string")
    : [];
  return (
    <article className="active-task-card">
      <ProgressBar
        className="top-progress"
        label={t("tasks.overallProgress")}
        value={progress}
        variant="top"
      />
      <div className="active-card-header">
        <div>
          <div className="eyebrow">
            <span className="live-dot" />
            {t("tasks.runningNow")}
          </div>
          <h3>{entry.display_name}</h3>
          <div className="chip-row">
            <span className="chip blue">{workflowLabel(entry.workflow, t)}</span>
            <span className="chip violet">{t("tasks.localWorkflow")}</span>
          </div>
        </div>
        <div className="progress-copy">
          <strong>{progress}%</strong>
          <span>
            {live?.stage
              ? stageLabel(live.stage, t)
              : entry.state === "dispatching"
                ? t("tasks.starting")
                : t("tasks.preparing")}
          </span>
        </div>
      </div>
      <ol className="stage-stepper" aria-label={t("tasks.stages")}>
        {stages.map((stage, index) => {
          const state =
            live?.stages[stage] ??
            (index < currentIndex ? "completed" : index === currentIndex ? "current" : "pending");
          return (
            <li
              key={stage}
              className={`stage ${state}`}
              aria-current={state === "current" ? "step" : undefined}
            >
              <span className="stage-line" />
              <span className="stage-circle" aria-hidden="true">
                {state === "completed" ? (
                  <Check size={14} />
                ) : state === "current" ? (
                  <LoaderCircle size={18} className="spinner" />
                ) : (
                  index + 1
                )}
              </span>
              <span className="stage-label">
                {stageLabel(stage, t)}
                <small>{t(`stageState.${state}`)}</small>
              </span>
            </li>
          );
        })}
      </ol>
      <div className="active-card-footer">
        <div className="runtime-summary">
          <Clock3 size={16} />
          <span>
            {live?.model ? t("tasks.modelCurrent", { model: live.model }) : t("tasks.modelPreparing")}
          </span>
        </div>
        <Button variant="danger" className="secondary" onPress={onCancel}>
          <Square size={14} />
          {t("tasks.cancelTask")}
        </Button>
      </div>
      {inputPaths.length > 1 && (
        <Disclosure className="task-files" title={`${t("tasks.files")} (${inputPaths.length})`}>
          <div className="task-file-list">
            {inputPaths.map((path) => {
              const exact = live?.items[path];
              const observed =
                exact ??
                Object.entries(live?.items ?? {}).find(([item]) => basename(item) === basename(path))?.[1];
              const current =
                live?.currentItem === path ||
                (live?.currentItem ? basename(live.currentItem) === basename(path) : false);
              return (
                <div key={path}>
                  <span title={path}>{basename(path)}</span>
                  <small>
                    {current ? `${t("tasks.currentFile")} · ` : ""}
                    {observed
                      ? `${stageLabel(observed.stage, t)} · ${Math.round(observed.percent)}%`
                      : t("tasks.waitingFile")}
                  </small>
                </div>
              );
            })}
          </div>
        </Disclosure>
      )}
      {(live?.logs.length ?? 0) > 0 && (
        <Disclosure className="task-log" title={t("tasks.recentLog")}>
          <pre>{live?.logs.join("\n")}</pre>
        </Disclosure>
      )}
    </article>
  );
}

function QueuedRow({
  entry,
  index,
  total,
  onMove,
  onRemove,
}: {
  entry: QueueEntry;
  index: number;
  total: number;
  onMove(direction: -1 | 1): void;
  onRemove(): void;
}): React.JSX.Element {
  const { t, i18n } = useTranslation();
  const addedTime = new Intl.DateTimeFormat(i18n.language, { timeStyle: "short" }).format(
    new Date(entry.enqueued_at),
  );
  return (
    <article className={`queue-row ${entry.state === "blocked" ? "blocked" : ""}`}>
      <GripVertical size={18} aria-hidden="true" className="drag-handle" />
      <span className="queue-number">{index + 1}</span>
      <div className="queue-copy">
        <strong>{entry.display_name}</strong>
        <span>
          {workflowLabel(entry.workflow, t)} · {t("tasks.addedAt", { time: addedTime })}
        </span>
        {entry.blocked_reason && (
          <small className="blocked-reason">
            <TriangleAlert size={14} />
            {entry.blocked_reason}
          </small>
        )}
      </div>
      <span className={`chip ${entry.state === "blocked" ? "warning" : "neutral"}`}>
        {t(`queueState.${entry.state}`)}
      </span>
      <div className="row-actions">
        <Tooltip label={t("tasks.moveUp")}>
          <IconButton
            label={t("tasks.moveItemUp", { name: entry.display_name })}
            isDisabled={index === 0}
            onPress={() => onMove(-1)}
          >
            <ArrowUp size={16} />
          </IconButton>
        </Tooltip>
        <Tooltip label={t("tasks.moveDown")}>
          <IconButton
            label={t("tasks.moveItemDown", { name: entry.display_name })}
            isDisabled={index === total - 1}
            onPress={() => onMove(1)}
          >
            <ArrowDown size={16} />
          </IconButton>
        </Tooltip>
        <Tooltip label={t("tasks.remove")}>
          <IconButton
            label={t("tasks.removeItem", { name: entry.display_name })}
            className="danger-text"
            onPress={onRemove}
          >
            <Ban size={16} />
          </IconButton>
        </Tooltip>
      </div>
    </article>
  );
}

function workflowLabel(workflow: QueueEntry["workflow"], t: ReturnType<typeof useTranslation>["t"]): string {
  if (workflow === "transcribe") return t("newTask.transcribe");
  if (workflow === "translate") return t("newTask.translate");
  return t("newTask.fullRun");
}

function stageLabel(stage: string, t: ReturnType<typeof useTranslation>["t"]): string {
  return t(`stage.${stage.replaceAll("-", "_")}`, { defaultValue: humanize(stage) });
}

function stagesFor(workflow: QueueEntry["workflow"]): string[] {
  if (workflow === "translate") return ["validate", "translate", "target-optimize", "export", "cleanup"];
  if (workflow === "transcribe")
    return ["validate", "preprocess", "transcribe", "source-optimize", "export", "cleanup"];
  return [
    "validate",
    "preprocess",
    "transcribe",
    "brief",
    "translate",
    "target-optimize",
    "export",
    "cleanup",
  ];
}
