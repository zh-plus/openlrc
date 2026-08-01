import { createContext, useContext, useEffect, useMemo, useReducer, type PropsWithChildren } from "react";
import { useQueryClient } from "@tanstack/react-query";

import type { WorkflowEvent } from "../../shared/contracts.js";
import { queryKeys } from "./queryKeys.js";

export interface LiveOperation {
  queueId: string | null;
  jobId: string | null;
  stage: string | null;
  percent: number;
  statusText: string;
  stages: Record<string, "completed" | "current" | "pending" | "failed">;
  model: string | null;
  logs: string[];
  currentItem: string | null;
  items: Record<string, { stage: string; percent: number; statusText: string }>;
}

export const initialOperationState: LiveOperation = {
  queueId: null,
  jobId: null,
  stage: null,
  percent: 0,
  statusText: "Preparing local workflow",
  stages: {},
  model: null,
  logs: [],
  currentItem: null,
  items: {},
};

export function operationReducer(state: LiveOperation, event: WorkflowEvent): LiveOperation {
  const payload = event.payload;
  if (event.event === "workflow.started") {
    return {
      ...initialOperationState,
      queueId: event.queue_id,
      jobId: event.job_id,
      statusText: "Validating task",
    };
  }
  if (state.queueId && event.queue_id !== state.queueId) return state;
  if (event.event === "workflow.stage_started") {
    const stage = typeof payload.stage === "string" ? payload.stage : "working";
    const item = typeof payload.item === "string" ? payload.item : null;
    return {
      ...state,
      queueId: event.queue_id,
      jobId: event.job_id,
      stage,
      percent: 0,
      statusText: humanize(stage),
      stages: { ...state.stages, [stage]: "current" },
      currentItem: item ?? state.currentItem,
      items: item
        ? { ...state.items, [item]: { stage, percent: 0, statusText: humanize(stage) } }
        : state.items,
    };
  }
  if (event.event === "workflow.stage_progress") {
    const stage = typeof payload.stage === "string" ? payload.stage : state.stage;
    const percent = typeof payload.percent === "number" ? payload.percent : state.percent;
    const item = typeof payload.item === "string" ? payload.item : state.currentItem;
    const statusText = typeof payload.message === "string" ? payload.message : humanize(stage);
    return {
      ...state,
      stage,
      percent,
      statusText,
      currentItem: item,
      items: item && stage ? { ...state.items, [item]: { stage, percent, statusText } } : state.items,
    };
  }
  if (event.event === "workflow.stage_completed") {
    const stage = typeof payload.stage === "string" ? payload.stage : state.stage;
    const item = typeof payload.item === "string" ? payload.item : state.currentItem;
    return stage
      ? {
          ...state,
          stages: { ...state.stages, [stage]: "completed" },
          currentItem: item,
          items:
            item && state.items[item]
              ? {
                  ...state.items,
                  [item]: { ...state.items[item], stage, percent: 100, statusText: humanize(stage) },
                }
              : state.items,
        }
      : state;
  }
  if (event.event === "workflow.model_lifecycle") {
    return { ...state, model: typeof payload.model === "string" ? payload.model : state.model };
  }
  if (event.event === "workflow.log") {
    const message = typeof payload.message === "string" ? payload.message : "Workflow update";
    return { ...state, logs: [...state.logs, message].slice(-100) };
  }
  if (["workflow.completed", "workflow.failed", "workflow.cancelled"].includes(event.event)) {
    return {
      ...state,
      percent: event.event === "workflow.completed" ? 100 : state.percent,
      statusText: humanize(event.event.replace("workflow.", "")),
    };
  }
  return state;
}

const OperationContext = createContext<LiveOperation>(initialOperationState);

export function OperationProvider({ children }: PropsWithChildren): React.JSX.Element {
  const [state, dispatch] = useReducer(operationReducer, initialOperationState);
  const queryClient = useQueryClient();
  useEffect(() => {
    const unsubscribeWorkflow = window.openlrc.workflows.onEvent((event) => {
      dispatch(event);
      if (["workflow.completed", "workflow.failed", "workflow.cancelled"].includes(event.event)) {
        void queryClient.invalidateQueries({ queryKey: queryKeys.jobs });
        void queryClient.invalidateQueries({ queryKey: queryKeys.queue });
      }
    });
    const unsubscribeQueue = window.openlrc.queue.onChanged((snapshot) => {
      queryClient.setQueryData(queryKeys.queue, snapshot);
    });
    return () => {
      unsubscribeWorkflow();
      unsubscribeQueue();
    };
  }, [queryClient]);
  const value = useMemo(() => state, [state]);
  return <OperationContext.Provider value={value}>{children}</OperationContext.Provider>;
}

export function useLiveOperation(): LiveOperation {
  return useContext(OperationContext);
}

export function humanize(value: string | null): string {
  if (!value) return "Preparing";
  return value
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}
