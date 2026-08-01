import { z } from "zod";

export const themeSourceSchema = z.enum(["system", "light", "dark"]);
export type ThemeSource = z.infer<typeof themeSourceSchema>;

export const appearanceStateSchema = z.object({
  themeSource: themeSourceSchema,
  effectiveTheme: z.enum(["light", "dark"]),
  sidebarCollapsed: z.boolean(),
});
export type AppearanceState = z.infer<typeof appearanceStateSchema>;

export const backendStatusSchema = z.object({
  state: z.enum(["stopped", "starting", "ready", "busy", "stopping", "failed"]),
  message: z.string().nullable(),
  serviceVersion: z.string().nullable(),
  capabilities: z.array(z.string()),
});
export type BackendStatus = z.infer<typeof backendStatusSchema>;

export const workflowKindSchema = z.enum(["transcribe", "translate", "run"]);
export type WorkflowKind = z.infer<typeof workflowKindSchema>;

export const workflowDraftSchema = z
  .object({
    task: z.string().default(""),
    workflow: workflowKindSchema,
    paths: z.array(z.string()).max(100),
    source_language: z.string().default(""),
    target_language: z.string().default("zh-cn"),
    whisper_model: z.string().default(""),
    vad_model: z.string().default(""),
    skip_preprocess: z.boolean().default(false),
    whisper_use_gpu: z.boolean().default(true),
    whisper_flash_attn: z.boolean().default(true),
    translation_backend: z.enum(["", "none", "online", "local-qwen", "local"]).default(""),
    mode: z.enum(["standard", "fast", "normal", "normal-plus", "pro"]).default("fast"),
    provider: z.string().default("openai"),
    primary_model: z.string().default(""),
    retry_provider: z.string().default(""),
    retry_model: z.string().default(""),
    reviewer_provider: z.string().default(""),
    reviewer_model: z.string().default(""),
    fee_limit: z.number().positive().default(0.8),
    consumer_thread: z.number().int().positive().default(4),
    local_profile: z.string().default(""),
    qwen_model: z.string().default(""),
    hymt2_model: z.string().default(""),
    context_provider: z.string().default("local"),
    context_model: z.string().default(""),
    context_base_url: z.string().default(""),
    context_fee_limit: z.number().positive().default(0.8),
    context_assistance: z.enum(["auto", "off"]).default("auto"),
    glossary_path: z.string().default(""),
    glossary_strict: z.boolean().default(true),
    force_glossary: z.boolean().default(false),
    brief_summary: z.string().default(""),
    brief_characters: z.string().default(""),
    brief_tone_style: z.string().default(""),
    edit_rounds: z.number().int().min(0).default(1),
    enable_restore: z.boolean().default(false),
    bilingual_subtitle: z.boolean().default(false),
    subtitle_optimization: z.enum(["aggressive", "relaxed"]).default("aggressive"),
    clear_temp: z.boolean().default(true),
    clear_checkpoint: z.boolean().default(true),
  })
  .strict();
export type WorkflowDraft = z.infer<typeof workflowDraftSchema>;

export const preflightReportSchema = z.object({
  status: z.enum(["ready", "warning", "blocked"]),
  blocked: z.boolean(),
  issues: z.array(z.object({ severity: z.string(), message: z.string(), field: z.string() })),
  summary: z.record(z.string(), z.string()),
});
export type PreflightReport = z.infer<typeof preflightReportSchema>;

export const queueEntrySchema = z.object({
  queue_id: z.string(),
  state: z.enum(["pending", "dispatching", "active", "blocked"]),
  workflow: workflowKindSchema,
  display_name: z.string(),
  draft_recipe: z.record(z.string(), z.unknown()),
  settings_snapshot: z.record(z.string(), z.unknown()),
  credential_requirements: z.array(z.string()),
  enqueued_at: z.string(),
  operation_id: z.string().nullable(),
  job_id: z.string().nullable(),
  blocked_reason: z.string().nullable(),
});
export type QueueEntry = z.infer<typeof queueEntrySchema>;

export const queueSnapshotSchema = z.object({
  paused: z.boolean(),
  active: queueEntrySchema.nullable(),
  entries: z.array(queueEntrySchema),
  total: z.number().int(),
  pending_count: z.number().int(),
});
export type QueueSnapshot = z.infer<typeof queueSnapshotSchema>;

export const jobStatusSchema = z.enum([
  "running",
  "succeeded",
  "succeeded_with_warnings",
  "failed",
  "cancelled",
  "interrupted",
]);

export const jobSummarySchema = z.object({
  job_id: z.string(),
  workflow: workflowKindSchema,
  name: z.string(),
  status: jobStatusSchema,
  input_paths: z.array(z.string()),
  translation_mode: z.string().nullable(),
  progress: z.number(),
  started_at: z.string(),
  completed_at: z.string().nullable(),
  outputs: z.array(z.string()),
  elapsed_seconds: z.number(),
  error: z.unknown().nullable(),
  resumed_from: z.string().nullable(),
});
export type JobSummary = z.infer<typeof jobSummarySchema>;

export const artifactSchema = z.object({
  artifact_id: z.string(),
  path: z.string(),
  kind: z.string(),
  item: z.string().nullable().optional(),
  primary: z.boolean().optional(),
});

export const jobDetailSchema = jobSummarySchema.extend({
  recipe: z.record(z.string(), z.unknown()),
  current_stage: z.string().nullable(),
  artifacts: z.array(artifactSchema),
  reviews: z.array(z.record(z.string(), z.unknown())),
  models: z.record(z.string(), z.record(z.string(), z.unknown())),
  event_log: z.array(z.string()),
  api_fee: z.number(),
  items: z.record(z.string(), z.record(z.string(), z.unknown())),
});
export type JobDetail = z.infer<typeof jobDetailSchema>;

export const resourceStatusSchema = z.object({
  name: z.string(),
  available: z.boolean(),
  detail: z.string(),
  hint: z.string(),
  role: z.string(),
  group: z.string(),
});
export type ResourceStatus = z.infer<typeof resourceStatusSchema>;

export const appSettingsSchema = z.object({
  schema_version: z.literal(1),
  general: z.object({
    theme: z.string(),
    language: z.enum(["en", "zh-cn"]),
    logo_animation: z.boolean(),
    reduce_motion: z.boolean(),
  }),
  providers: z.record(
    z.string(),
    z.object({ enabled: z.boolean(), model: z.string(), base_url: z.string(), proxy: z.string() }),
  ),
  local_models: z.object({
    whisper_model: z.string(),
    vad_model: z.string(),
    whisper_cli: z.string(),
    qwen_model: z.string(),
    hymt2_profile: z.string(),
    hymt2_model: z.string(),
    llama_server: z.string(),
    host: z.string(),
    port: z.number(),
    context_size: z.number(),
    gpu_layers: z.string(),
    idle_timeout: z.number(),
    startup_timeout: z.number(),
  }),
  transcription: z.object({
    source_language: z.string(),
    skip_preprocess: z.boolean(),
    use_gpu: z.boolean(),
    flash_attn: z.boolean(),
  }),
  workflow: z.object({
    target_language: z.string(),
    bilingual_subtitle: z.boolean(),
    subtitle_optimization: z.string(),
    clear_temp: z.boolean(),
    clear_checkpoint: z.boolean(),
    glossary_strict: z.boolean(),
    force_glossary: z.boolean(),
    edit_rounds: z.number(),
    enable_restore: z.boolean(),
    default_glossary: z.string(),
  }),
});
export type AppSettings = z.infer<typeof appSettingsSchema>;

export const credentialStatusSchema = z.object({
  provider: z.string(),
  present: z.boolean(),
  source: z.enum(["keychain", "environment", "missing"]),
  environment_name: z.string().nullable(),
  warning: z.string().nullable(),
});
export type CredentialStatus = z.infer<typeof credentialStatusSchema>;

export const workflowEventSchema = z.object({
  type: z.literal("event"),
  protocol: z.literal(1),
  event: z.string(),
  queue_id: z.string(),
  operation_id: z.string().nullable(),
  job_id: z.string(),
  sequence: z.number(),
  payload: z.record(z.string(), z.unknown()),
});
export type WorkflowEvent = z.infer<typeof workflowEventSchema>;

export interface AppInfo {
  name: string;
  version: string;
  platform: string;
  protocol: number;
  diagnosticPath: string;
}

export interface OpenLRCDesktopBridge {
  app: {
    info(): Promise<AppInfo>;
    backendStatus(): Promise<BackendStatus>;
    restartBackend(): Promise<BackendStatus>;
    openExternal(url: string): Promise<void>;
    onBackendStatus(listener: (status: BackendStatus) => void): () => void;
  };
  appearance: {
    get(): Promise<AppearanceState>;
    setTheme(source: ThemeSource): Promise<AppearanceState>;
    setSidebarCollapsed(collapsed: boolean): Promise<AppearanceState>;
    onChanged(listener: (state: AppearanceState) => void): () => void;
  };
  dialogs: {
    selectInputs(kind: WorkflowKind): Promise<{ paths: string[]; cancelled: boolean }>;
  };
  workflows: {
    preflight(draft: WorkflowDraft): Promise<PreflightReport>;
    active(): Promise<unknown | null>;
    onEvent(listener: (event: WorkflowEvent) => void): () => void;
  };
  queue: {
    snapshot(): Promise<QueueSnapshot>;
    enqueue(draft: WorkflowDraft): Promise<{ queue_id: string; entry: QueueEntry }>;
    cancel(queueId: string): Promise<{ queue_id: string; cancelled: boolean; removed: boolean }>;
    reorder(orderedQueueIds: string[]): Promise<QueueSnapshot>;
    pause(): Promise<QueueSnapshot>;
    resume(): Promise<QueueSnapshot>;
    onChanged(listener: (snapshot: QueueSnapshot) => void): () => void;
  };
  jobs: {
    list(): Promise<JobSummary[]>;
    get(jobId: string): Promise<JobDetail>;
    delete(jobId: string): Promise<void>;
    resumeDraft(jobId: string): Promise<WorkflowDraft>;
  };
  resources: {
    status(): Promise<ResourceStatus[]>;
    refresh(): Promise<ResourceStatus[]>;
  };
  settings: {
    get(): Promise<AppSettings>;
    update(patch: Record<string, unknown>): Promise<AppSettings>;
  };
  credentials: {
    status(provider: string): Promise<CredentialStatus>;
    set(provider: string, secret: string): Promise<CredentialStatus>;
    delete(provider: string): Promise<void>;
  };
  artifacts: {
    reveal(jobId: string, artifactId: string): Promise<void>;
    open(jobId: string, artifactId: string): Promise<void>;
  };
}

export const channels = {
  appInfo: "app:info",
  backendStatus: "backend:status",
  backendRestart: "backend:restart",
  backendChanged: "backend:changed",
  appOpenExternal: "app:open-external",
  appearanceGet: "appearance:get",
  appearanceSetTheme: "appearance:set-theme",
  appearanceSetSidebar: "appearance:set-sidebar-collapsed",
  appearanceChanged: "appearance:changed",
  dialogSelectInputs: "dialog:select-inputs",
  workflowPreflight: "workflow:preflight",
  workflowActive: "workflow:active",
  workflowEvent: "workflow:event",
  queueSnapshot: "queue:snapshot",
  queueEnqueue: "queue:enqueue",
  queueCancel: "queue:cancel",
  queueReorder: "queue:reorder",
  queuePause: "queue:pause",
  queueResume: "queue:resume",
  queueChanged: "queue:changed",
  jobsList: "jobs:list",
  jobsGet: "jobs:get",
  jobsDelete: "jobs:delete",
  jobsResumeDraft: "jobs:resume-draft",
  resourcesStatus: "resources:status",
  resourcesRefresh: "resources:refresh",
  settingsGet: "settings:get",
  settingsUpdate: "settings:update",
  credentialsStatus: "credentials:status",
  credentialsSet: "credentials:set",
  credentialsDelete: "credentials:delete",
  artifactReveal: "artifact:reveal",
  artifactOpen: "artifact:open",
} as const;
