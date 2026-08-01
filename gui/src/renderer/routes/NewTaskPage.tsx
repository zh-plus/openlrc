import { useEffect, useRef, useState } from "react";
import { useBlocker, useLocation, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Controller, useForm, useWatch, type Control, type FieldPath } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { CirclePlus, FileAudio, FolderOpen, ShieldCheck, TriangleAlert, X } from "lucide-react";

import type { AppSettings, PreflightReport, WorkflowDraft, WorkflowKind } from "../../shared/contracts.js";
import { queryKeys } from "../app/queryKeys.js";
import { ErrorState, LoadingState, PageHeader, Section } from "../components/Page.js";
import {
  Button,
  CheckboxControl,
  Disclosure,
  IconButton,
  ModalDialog,
  RadioCardGroup,
  SelectField,
  TextAreaField,
  TextFieldControl,
  Tooltip,
  type SelectOption,
} from "../components/ui/index.js";
import { basename } from "../utils/format.js";

interface NavigationState {
  draft?: WorkflowDraft;
}

export function NewTaskPage(): React.JSX.Element {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  const settings = useQuery({ queryKey: queryKeys.settings, queryFn: () => window.openlrc.settings.get() });
  const navigationDraft = (location.state as NavigationState | null)?.draft;
  const [report, setReport] = useState<PreflightReport | null>(null);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const issueSummary = useRef<HTMLDivElement>(null);
  const allowNavigation = useRef(false);
  const form = useForm<WorkflowDraft>({ defaultValues: emptyDraft() });
  const workflow = useWatch({ control: form.control, name: "workflow" });
  const backend = useWatch({ control: form.control, name: "translation_backend" });
  const paths = useWatch({ control: form.control, name: "paths" });
  const mode = useWatch({ control: form.control, name: "mode" });
  const navigationBlocker = useBlocker(
    ({ currentLocation, nextLocation }) =>
      !allowNavigation.current &&
      form.formState.isDirty &&
      currentLocation.pathname !== nextLocation.pathname,
  );

  useEffect(() => {
    if (!settings.data) return;
    form.reset(navigationDraft ?? draftFromSettings(settings.data));
  }, [form, navigationDraft, settings.data]);

  useEffect(() => {
    const beforeUnload = (event: BeforeUnloadEvent): void => {
      if (form.formState.isDirty) event.preventDefault();
    };
    window.addEventListener("beforeunload", beforeUnload);
    return () => window.removeEventListener("beforeunload", beforeUnload);
  }, [form.formState.isDirty]);

  const preflightMutation = useMutation({
    mutationFn: (draft: WorkflowDraft) => window.openlrc.workflows.preflight(draft),
    onSuccess: (next) => {
      setReport(next);
      if (next.blocked) requestAnimationFrame(() => issueSummary.current?.focus());
      else setConfirmOpen(true);
    },
  });
  const enqueue = useMutation({
    mutationFn: (draft: WorkflowDraft) => window.openlrc.queue.enqueue(draft),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: queryKeys.queue });
      form.reset(form.getValues());
      allowNavigation.current = true;
      navigate("/tasks");
    },
  });
  const selectInputs = async (): Promise<void> => {
    const result = await window.openlrc.dialogs.selectInputs(workflow);
    if (!result.cancelled) form.setValue("paths", result.paths, { shouldDirty: true, shouldValidate: true });
  };
  const setWorkflow = (next: WorkflowKind): void => {
    form.setValue("workflow", next, { shouldDirty: true });
    if (next === "transcribe") {
      form.setValue("task", "transcribe-json", { shouldDirty: true });
      form.setValue("translation_backend", "", { shouldDirty: true });
    } else if (next === "run") {
      form.setValue("task", "", { shouldDirty: true });
      form.setValue("translation_backend", "none", { shouldDirty: true });
    } else {
      form.setValue("task", "", { shouldDirty: true });
      form.setValue("translation_backend", "local", { shouldDirty: true });
      form.setValue("mode", "fast", { shouldDirty: true });
    }
  };
  const setBackend = (next: WorkflowDraft["translation_backend"]): void => {
    form.setValue("translation_backend", next, { shouldDirty: true });
    form.setValue("mode", next === "local" ? "fast" : "standard", { shouldDirty: true });
  };

  if (settings.isLoading)
    return (
      <div className="page">
        <LoadingState />
      </div>
    );
  if (settings.error)
    return (
      <div className="page">
        <ErrorState error={settings.error} />
      </div>
    );
  return (
    <div className="page new-task-page">
      <PageHeader title={t("newTask.title")} subtitle={t("newTask.subtitle")} />
      {(preflightMutation.error || enqueue.error) && (
        <div className="persistent-banner danger" role="alert">
          <TriangleAlert size={18} />
          <div>
            <strong>{t("newTask.submitFailed")}</strong>
            <span>{errorText(preflightMutation.error ?? enqueue.error)}</span>
          </div>
        </div>
      )}
      {report?.blocked && (
        <div className="validation-summary" role="alert" tabIndex={-1} ref={issueSummary}>
          <TriangleAlert size={19} />
          <div>
            <strong>{t("newTask.preflightFixes")}</strong>
            {report.issues
              .filter((issue) => issue.severity === "blocked")
              .map((issue) => (
                <p key={`${issue.field}-${issue.message}`}>{issue.message}</p>
              ))}
          </div>
        </div>
      )}
      <form onSubmit={form.handleSubmit((draft) => preflightMutation.mutate(normalizeDraft(draft)))}>
        <Section title={t("newTask.inputs")} className="form-section">
          <div className="field-grid three">
            <SelectField
              label={t("newTask.workflow")}
              value={workflow}
              onChange={(next) => setWorkflow(next as WorkflowKind)}
              description={t("newTask.queueOrderHint")}
              options={[
                { value: "transcribe", label: t("newTask.transcribe") },
                { value: "translate", label: t("newTask.translate") },
                { value: "run", label: t("newTask.fullRun") },
              ]}
            />
            {workflow === "transcribe" && (
              <DraftSelectField
                control={form.control}
                name="task"
                label={t("newTask.outputType")}
                description={t("newTask.jsonLater")}
                options={[
                  { value: "transcribe-json", label: t("newTask.transcriptionJson") },
                  { value: "transcribe-subtitle", label: t("newTask.sourceSubtitle") },
                ]}
              />
            )}
            <div className="field file-picker-field">
              <span>{t("newTask.inputFiles")}</span>
              <Button variant="secondary" onPress={() => void selectInputs()}>
                <FolderOpen size={16} />
                {t("newTask.chooseFiles")}
              </Button>
              <small>{workflow === "translate" ? t("newTask.chooseJson") : t("newTask.validateMedia")}</small>
            </div>
          </div>
          {paths.length > 0 ? (
            <div className="selected-files">
              {paths.map((item) => (
                <div key={item}>
                  <FileAudio size={16} />
                  <span title={item}>{basename(item)}</span>
                  <Tooltip label={t("newTask.removeFile", { name: basename(item) })}>
                    <IconButton
                      label={t("newTask.removeFile", { name: basename(item) })}
                      onPress={() =>
                        form.setValue(
                          "paths",
                          paths.filter((path) => path !== item),
                          { shouldDirty: true },
                        )
                      }
                    >
                      <X size={15} />
                    </IconButton>
                  </Tooltip>
                </div>
              ))}
            </div>
          ) : (
            <div className="drop-placeholder">
              <FileAudio size={21} />
              <span>{t("newTask.noInput")}</span>
            </div>
          )}
        </Section>

        <Section title={t("newTask.languages")} className="form-section">
          <div className="field-grid three">
            {workflow !== "translate" && (
              <DraftTextField
                control={form.control}
                name="source_language"
                label={t("newTask.sourceLanguage")}
                placeholder={t("newTask.autoDetect")}
                description={t("newTask.autoDetectHint")}
              />
            )}
            {workflow !== "transcribe" && (
              <DraftTextField
                control={form.control}
                name="target_language"
                label={t("newTask.targetLanguage")}
                placeholder="zh-cn"
                isRequired
              />
            )}
            {workflow !== "transcribe" && (
              <DraftCheckbox
                control={form.control}
                id="draft-bilingual"
                name="bilingual_subtitle"
                label={t("newTask.bilingual")}
                detail={t("newTask.bilingualHint")}
              />
            )}
          </div>
        </Section>

        {workflow !== "translate" && (
          <Section title={t("newTask.transcription")} className="form-section">
            <div className="field-grid three">
              <DraftTextField
                control={form.control}
                name="whisper_model"
                label={t("newTask.whisperModel")}
                placeholder="base"
              />
              <DraftTextField
                control={form.control}
                name="vad_model"
                label={t("newTask.vadModel")}
                placeholder="silero-v6.2.0"
              />
              <div className="check-stack">
                <DraftCheckbox
                  control={form.control}
                  id="draft-whisper-gpu"
                  name="whisper_use_gpu"
                  label={t("newTask.useGpu")}
                  detail={t("newTask.gpuHint")}
                />
                <DraftCheckbox
                  control={form.control}
                  id="draft-flash-attention"
                  name="whisper_flash_attn"
                  label={t("newTask.flashAttention")}
                  detail={t("newTask.flashHint")}
                />
              </div>
            </div>
          </Section>
        )}

        {workflow !== "transcribe" && (
          <Section title={t("newTask.translation")} className="form-section">
            <RadioCardGroup
              label={t("newTask.backendLabel")}
              value={backend}
              onChange={(next) => setBackend(next as WorkflowDraft["translation_backend"])}
              options={[
                ...(workflow === "run"
                  ? [
                      {
                        value: "none",
                        label: t("newTask.noTranslation"),
                        detail: t("newTask.noTranslationDetail"),
                      },
                    ]
                  : []),
                { value: "online", label: t("newTask.online"), detail: t("newTask.onlineDetail") },
                {
                  value: "local-qwen",
                  label: t("newTask.localQwen"),
                  detail: t("newTask.localQwenDetail"),
                },
                {
                  value: "local",
                  label: t("newTask.hymt2"),
                  detail: t("newTask.hymt2Detail"),
                  accent: true,
                },
              ]}
            />
            {backend !== "none" && (
              <div className="field-grid three translation-fields">
                {backend === "online" && (
                  <>
                    <DraftSelectField
                      control={form.control}
                      name="provider"
                      label={t("common.provider")}
                      options={[
                        { value: "openai", label: "OpenAI" },
                        { value: "anthropic", label: "Anthropic" },
                        { value: "google", label: "Google" },
                        { value: "litellm", label: "LiteLLM" },
                        { value: "third_party", label: "Third party" },
                      ]}
                    />
                    <DraftTextField
                      control={form.control}
                      name="primary_model"
                      label={t("newTask.model")}
                      placeholder={t("newTask.providerDefault")}
                    />
                  </>
                )}
                {backend === "local-qwen" && (
                  <DraftTextField
                    control={form.control}
                    name="qwen_model"
                    label={t("newTask.qwenModel")}
                    placeholder={t("newTask.qwenPlaceholder")}
                  />
                )}
                {backend === "local" && (
                  <>
                    <DraftSelectField
                      control={form.control}
                      name="mode"
                      label={t("newTask.hymt2Mode")}
                      options={[
                        { value: "fast", label: "Fast" },
                        { value: "normal", label: "Normal" },
                        { value: "normal-plus", label: "Normal Plus" },
                        { value: "pro", label: "Pro" },
                      ]}
                    />
                    <DraftTextField
                      control={form.control}
                      name="hymt2_model"
                      label={t("newTask.hymt2Model")}
                      placeholder={t("newTask.profileDefault")}
                    />
                  </>
                )}
                <DraftTextField
                  control={form.control}
                  name="glossary_path"
                  label={t("newTask.glossary")}
                  placeholder={t("newTask.glossaryPlaceholder")}
                />
              </div>
            )}
            {backend === "local" && mode !== "fast" && (
              <div className="brief-panel">
                <h3>{t("newTask.brief")}</h3>
                <div className="field-grid">
                  <DraftTextArea
                    control={form.control}
                    name="brief_summary"
                    label={t("newTask.summary")}
                    className="full"
                    rows={3}
                    placeholder={t("newTask.summaryPlaceholder")}
                  />
                  <DraftTextArea
                    control={form.control}
                    name="brief_characters"
                    label={t("newTask.characters")}
                    rows={4}
                    placeholder={t("newTask.characterPlaceholder")}
                  />
                  <DraftTextArea
                    control={form.control}
                    name="brief_tone_style"
                    label={t("newTask.tone")}
                    rows={4}
                    placeholder={t("newTask.tonePlaceholder")}
                  />
                </div>
              </div>
            )}
          </Section>
        )}

        <Section title={t("newTask.output")} className="form-section">
          <div className="output-summary">
            <ShieldCheck size={20} />
            <div>
              <strong>{t("newTask.safeOutput")}</strong>
              <span>{t("newTask.safeOutputDetail")}</span>
            </div>
          </div>
        </Section>

        <Disclosure
          className="advanced-panel"
          title={t("newTask.advanced")}
          detail={t("newTask.advancedDetail")}
        >
          <div className="field-grid three">
            <DraftSelectField
              control={form.control}
              name="subtitle_optimization"
              label={t("newTask.optimization")}
              options={[
                { value: "aggressive", label: t("newTask.aggressive") },
                { value: "relaxed", label: t("newTask.relaxed") },
              ]}
            />
            <DraftCheckbox
              control={form.control}
              id="draft-skip-preprocess"
              name="skip_preprocess"
              label={t("newTask.skipPreprocess")}
              detail={t("newTask.skipPreprocessHint")}
            />
            <DraftCheckbox
              control={form.control}
              id="draft-clear-temp"
              name="clear_temp"
              label={t("newTask.clearTemp")}
              detail={t("newTask.clearTempHint")}
            />
            <DraftCheckbox
              control={form.control}
              id="draft-clear-checkpoint"
              name="clear_checkpoint"
              label={t("newTask.clearCheckpoint")}
              detail={t("newTask.clearCheckpointHint")}
            />
            {backend === "online" && (
              <>
                <DraftNumberField
                  control={form.control}
                  name="fee_limit"
                  label={t("newTask.feeLimit")}
                  min="0.01"
                  step="0.01"
                />
                <DraftNumberField
                  control={form.control}
                  name="consumer_thread"
                  label={t("newTask.consumers")}
                  min="1"
                  max="32"
                />
              </>
            )}
          </div>
        </Disclosure>

        <div className="form-actions">
          <Button variant="secondary" onPress={() => form.reset(draftFromSettings(settings.data!))}>
            {t("common.cancel")}
          </Button>
          <Button
            type="submit"
            variant="primary"
            isPending={preflightMutation.isPending}
            isDisabled={preflightMutation.isPending || paths.length === 0}
          >
            <ShieldCheck size={16} />
            {preflightMutation.isPending ? t("common.checking") : t("newTask.preflight")}
          </Button>
        </div>
      </form>

      <ModalDialog
        isOpen={confirmOpen}
        onOpenChange={setConfirmOpen}
        isDismissable={!enqueue.isPending}
        title={t("newTask.confirmTitle")}
        description={t("newTask.confirmDetail")}
      >
        {report ? (
          <div className="preflight-confirm">
            <div className={`preflight-status ${report.status}`}>
              <ShieldCheck size={20} />
              <strong>{report.status}</strong>
            </div>
            <dl>
              {Object.entries(report.summary).map(([key, value]) => (
                <div key={key}>
                  <dt>{key.replaceAll("_", " ")}</dt>
                  <dd>{value || "—"}</dd>
                </div>
              ))}
            </dl>
            {report.issues.length > 0 ? (
              <div className="preflight-issues">
                {report.issues.map((issue) => (
                  <p key={`${issue.field}-${issue.message}`}>
                    <TriangleAlert size={14} />
                    {issue.message}
                  </p>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}
        <div className="dialog-actions">
          <Button variant="secondary" autoFocus onPress={() => setConfirmOpen(false)}>
            {t("common.back")}
          </Button>
          <Button
            variant="primary"
            isPending={enqueue.isPending}
            isDisabled={enqueue.isPending}
            onPress={() => enqueue.mutate(normalizeDraft(form.getValues()))}
          >
            <CirclePlus size={16} />
            {enqueue.isPending ? t("newTask.adding") : t("newTask.addToQueue")}
          </Button>
        </div>
      </ModalDialog>

      <ModalDialog
        isOpen={navigationBlocker.state === "blocked"}
        onOpenChange={(open: boolean) => {
          if (!open && navigationBlocker.state === "blocked") navigationBlocker.reset();
        }}
        title={t("newTask.discardTitle")}
        description={t("newTask.discardDetail")}
      >
        <div className="dialog-actions">
          <Button
            variant="secondary"
            autoFocus
            onPress={() => {
              if (navigationBlocker.state === "blocked") navigationBlocker.reset();
            }}
          >
            {t("newTask.stay")}
          </Button>
          <Button
            variant="danger"
            onPress={() => {
              allowNavigation.current = true;
              navigationBlocker.proceed?.();
            }}
          >
            {t("newTask.discard")}
          </Button>
        </div>
      </ModalDialog>
    </div>
  );
}

function DraftCheckbox({
  control,
  id,
  name,
  label,
  detail,
}: {
  control: Control<WorkflowDraft>;
  id: string;
  name: FieldPath<WorkflowDraft>;
  label: string;
  detail: string;
}): React.JSX.Element {
  const labelId = `${id}-label`;
  return (
    <div className="check-field">
      <Controller
        control={control}
        name={name}
        render={({ field }) => (
          <CheckboxControl
            id={id}
            name={field.name}
            checked={Boolean(field.value)}
            labelledBy={labelId}
            onBlur={field.onBlur}
            onCheckedChange={field.onChange}
          />
        )}
      />
      <label htmlFor={id} id={labelId}>
        <strong>{label}</strong>
        <small>{detail}</small>
      </label>
    </div>
  );
}

function DraftTextField({
  control,
  name,
  label,
  placeholder = "",
  description,
  isRequired = false,
}: {
  control: Control<WorkflowDraft>;
  name: FieldPath<WorkflowDraft>;
  label: string;
  placeholder?: string;
  description?: string;
  isRequired?: boolean;
}): React.JSX.Element {
  return (
    <Controller
      control={control}
      name={name}
      rules={{ required: isRequired }}
      render={({ field, fieldState }) => (
        <TextFieldControl
          label={label}
          name={field.name}
          value={String(field.value ?? "")}
          onChange={field.onChange}
          onBlur={field.onBlur}
          inputRef={field.ref}
          isRequired={isRequired}
          isInvalid={fieldState.invalid}
          inputProps={{ placeholder }}
          {...(description ? { description } : {})}
        />
      )}
    />
  );
}

function DraftTextArea({
  control,
  name,
  label,
  placeholder,
  rows,
  className,
}: {
  control: Control<WorkflowDraft>;
  name: FieldPath<WorkflowDraft>;
  label: string;
  placeholder: string;
  rows: number;
  className?: string;
}): React.JSX.Element {
  return (
    <Controller
      control={control}
      name={name}
      render={({ field, fieldState }) => (
        <TextAreaField
          label={label}
          name={field.name}
          value={String(field.value ?? "")}
          onChange={field.onChange}
          onBlur={field.onBlur}
          isInvalid={fieldState.invalid}
          rows={rows}
          placeholder={placeholder}
          className={className ?? ""}
        />
      )}
    />
  );
}

function DraftSelectField({
  control,
  name,
  label,
  options,
  description,
}: {
  control: Control<WorkflowDraft>;
  name: FieldPath<WorkflowDraft>;
  label: string;
  options: readonly SelectOption[];
  description?: string;
}): React.JSX.Element {
  return (
    <Controller
      control={control}
      name={name}
      render={({ field, fieldState }) => (
        <SelectField
          label={label}
          name={field.name}
          value={String(field.value ?? "")}
          options={options}
          onChange={field.onChange}
          onBlur={field.onBlur}
          isInvalid={fieldState.invalid}
          {...(description ? { description } : {})}
        />
      )}
    />
  );
}

function DraftNumberField({
  control,
  name,
  label,
  min,
  max,
  step,
}: {
  control: Control<WorkflowDraft>;
  name: FieldPath<WorkflowDraft>;
  label: string;
  min?: string;
  max?: string;
  step?: string;
}): React.JSX.Element {
  return (
    <Controller
      control={control}
      name={name}
      render={({ field, fieldState }) => (
        <TextFieldControl
          label={label}
          name={field.name}
          value={Number.isFinite(Number(field.value)) ? String(field.value) : ""}
          onChange={(value) => field.onChange(Number(value))}
          onBlur={field.onBlur}
          inputRef={field.ref}
          isInvalid={fieldState.invalid}
          inputProps={{
            type: "number",
            ...(min !== undefined ? { min } : {}),
            ...(max !== undefined ? { max } : {}),
            ...(step !== undefined ? { step } : {}),
          }}
        />
      )}
    />
  );
}

function emptyDraft(): WorkflowDraft {
  return {
    task: "transcribe-json",
    workflow: "transcribe",
    paths: [],
    source_language: "",
    target_language: "zh-cn",
    whisper_model: "",
    vad_model: "",
    skip_preprocess: false,
    whisper_use_gpu: true,
    whisper_flash_attn: true,
    translation_backend: "",
    mode: "fast",
    provider: "openai",
    primary_model: "",
    retry_provider: "",
    retry_model: "",
    reviewer_provider: "",
    reviewer_model: "",
    fee_limit: 0.8,
    consumer_thread: 4,
    local_profile: "",
    qwen_model: "",
    hymt2_model: "",
    context_provider: "local",
    context_model: "",
    context_base_url: "",
    context_fee_limit: 0.8,
    context_assistance: "auto",
    glossary_path: "",
    glossary_strict: true,
    force_glossary: false,
    brief_summary: "",
    brief_characters: "",
    brief_tone_style: "",
    edit_rounds: 1,
    enable_restore: false,
    bilingual_subtitle: false,
    subtitle_optimization: "aggressive",
    clear_temp: true,
    clear_checkpoint: true,
  };
}

function draftFromSettings(settings: AppSettings): WorkflowDraft {
  return {
    ...emptyDraft(),
    source_language: settings.transcription.source_language,
    target_language: settings.workflow.target_language,
    whisper_model: settings.local_models.whisper_model,
    vad_model: settings.local_models.vad_model,
    skip_preprocess: settings.transcription.skip_preprocess,
    whisper_use_gpu: settings.transcription.use_gpu,
    whisper_flash_attn: settings.transcription.flash_attn,
    local_profile: settings.local_models.hymt2_profile,
    qwen_model: settings.local_models.qwen_model,
    hymt2_model: settings.local_models.hymt2_model,
    context_model: settings.local_models.qwen_model,
    glossary_path: settings.workflow.default_glossary,
    glossary_strict: settings.workflow.glossary_strict,
    force_glossary: settings.workflow.force_glossary,
    edit_rounds: settings.workflow.edit_rounds,
    enable_restore: settings.workflow.enable_restore,
    bilingual_subtitle: settings.workflow.bilingual_subtitle,
    subtitle_optimization: settings.workflow.subtitle_optimization === "relaxed" ? "relaxed" : "aggressive",
    clear_temp: settings.workflow.clear_temp,
    clear_checkpoint: settings.workflow.clear_checkpoint,
  };
}

function normalizeDraft(draft: WorkflowDraft): WorkflowDraft {
  const next = { ...draft };
  if (next.workflow === "transcribe") next.translation_backend = "";
  if (next.translation_backend === "online" || next.translation_backend === "local-qwen")
    next.mode = "standard";
  if (next.workflow === "run" && next.translation_backend === "none") next.mode = "fast";
  return next;
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
