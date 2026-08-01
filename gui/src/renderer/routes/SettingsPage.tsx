import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Controller, useForm, type Control, type FieldPath } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { Check, ExternalLink, KeyRound, Save, Trash2 } from "lucide-react";

import type { AppSettings, ThemeSource } from "../../shared/contracts.js";
import { queryKeys } from "../app/queryKeys.js";
import { ErrorState, LoadingState, PageHeader } from "../components/Page.js";
import {
  AppLink,
  Button,
  IconButton,
  SegmentedControl,
  SelectField,
  SwitchControl,
  Tabs as SettingsTabs,
  TextFieldControl,
  Tooltip,
} from "../components/ui/index.js";

const providerIds = ["openai", "anthropic", "google", "litellm", "third_party"] as const;
const themeSources: readonly ThemeSource[] = ["system", "light", "dark"];

export function SettingsPage(): React.JSX.Element {
  const { t } = useTranslation();
  const settings = useQuery({ queryKey: queryKeys.settings, queryFn: () => window.openlrc.settings.get() });

  if (settings.isLoading)
    return (
      <div className="page">
        <LoadingState />
      </div>
    );
  if (settings.error || !settings.data)
    return (
      <div className="page">
        <ErrorState error={settings.error ?? new Error(t("settings.unavailable"))} />
      </div>
    );

  return <SettingsForm settingsData={settings.data} />;
}

function SettingsForm({ settingsData }: { settingsData: AppSettings }): React.JSX.Element {
  const { t, i18n } = useTranslation();
  const queryClient = useQueryClient();
  const appearance = useQuery({
    queryKey: queryKeys.appearance,
    queryFn: () => window.openlrc.appearance.get(),
  });
  const form = useForm<AppSettings>({ defaultValues: settingsData });

  const save = useMutation({
    mutationFn: (data: AppSettings) =>
      window.openlrc.settings.update(data as unknown as Record<string, unknown>),
    onSuccess: async (data) => {
      queryClient.setQueryData(queryKeys.settings, data);
      form.reset(data);
      await i18n.changeLanguage(data.general.language);
    },
  });
  const setTheme = async (source: ThemeSource): Promise<void> => {
    const next = await window.openlrc.appearance.setTheme(source);
    queryClient.setQueryData(queryKeys.appearance, next);
  };

  const tabs = [
    {
      id: "general",
      label: t("settings.general"),
      content: (
        <>
          <SettingsHeading title={t("settings.general")} detail={t("settings.generalDetail")} />
          <div className="settings-card">
            <div className="setting-row">
              <div className="setting-copy">
                <strong>{t("theme.source")}</strong>
                <span>{t("settings.themeDetail")}</span>
              </div>
              <SegmentedControl
                label={t("theme.source")}
                value={appearance.data?.themeSource ?? "system"}
                className="settings-theme-segment w-full max-w-60"
                onChange={(source) => void setTheme(source as ThemeSource)}
                options={themeSources.map((source) => ({
                  value: source,
                  label: t(`theme.${source}`),
                  ariaLabel: t("theme.option", { source: t(`theme.${source}`) }),
                }))}
              />
            </div>
            <div className="setting-row">
              <div className="setting-copy">
                <strong>{t("settings.language")}</strong>
                <span>{t("settings.languageDetail")}</span>
              </div>
              <Controller
                control={form.control}
                name="general.language"
                render={({ field }) => (
                  <SelectField
                    label={t("settings.language")}
                    className="setting-control"
                    name={field.name}
                    value={field.value}
                    onChange={field.onChange}
                    onBlur={field.onBlur}
                    options={[
                      { value: "en", label: "English" },
                      { value: "zh-cn", label: "简体中文" },
                    ]}
                  />
                )}
              />
            </div>
            <div className="setting-row">
              <label
                className="setting-copy"
                htmlFor="setting-reduce-motion"
                id="setting-reduce-motion-label"
              >
                <strong>{t("settings.reduceMotion")}</strong>
                <span>{t("settings.reduceMotionDetail")}</span>
              </label>
              <Controller
                control={form.control}
                name="general.reduce_motion"
                render={({ field }) => (
                  <SwitchControl
                    id="setting-reduce-motion"
                    name={field.name}
                    checked={Boolean(field.value)}
                    labelledBy="setting-reduce-motion-label"
                    onBlur={field.onBlur}
                    onCheckedChange={field.onChange}
                  />
                )}
              />
            </div>
          </div>
        </>
      ),
    },
    {
      id: "providers",
      label: t("settings.providers"),
      content: (
        <>
          <SettingsHeading title={t("settings.providers")} detail={t("settings.providersDetail")} />
          <div className="settings-stack">
            {providerIds.map((provider) => (
              <div className="settings-card provider-card" key={provider}>
                <div className="setting-row">
                  <label
                    className="setting-copy"
                    htmlFor={`setting-provider-${provider}`}
                    id={`setting-provider-${provider}-label`}
                  >
                    <strong>{providerLabel(provider)}</strong>
                    <span>{t("settings.enableProvider")}</span>
                  </label>
                  <Controller
                    control={form.control}
                    name={`providers.${provider}.enabled`}
                    render={({ field }) => (
                      <SwitchControl
                        id={`setting-provider-${provider}`}
                        name={field.name}
                        checked={Boolean(field.value)}
                        labelledBy={`setting-provider-${provider}-label`}
                        onBlur={field.onBlur}
                        onCheckedChange={field.onChange}
                      />
                    )}
                  />
                </div>
                <div className="field-grid three">
                  <SettingsTextField
                    control={form.control}
                    name={`providers.${provider}.model`}
                    label={t("settings.defaultModel")}
                  />
                  <SettingsTextField
                    control={form.control}
                    name={`providers.${provider}.base_url`}
                    label={t("settings.baseUrl")}
                    placeholder={t("settings.providerDefault")}
                  />
                  <SettingsTextField
                    control={form.control}
                    name={`providers.${provider}.proxy`}
                    label={t("settings.proxy")}
                    placeholder={t("settings.optional")}
                  />
                </div>
              </div>
            ))}
          </div>
        </>
      ),
    },
    {
      id: "models",
      label: t("settings.models"),
      content: (
        <>
          <SettingsHeading title={t("settings.models")} detail={t("settings.modelsDetail")} />
          <div className="settings-card">
            <div className="field-grid">
              <SettingsTextField
                control={form.control}
                name="local_models.whisper_model"
                label={t("settings.whisperModel")}
              />
              <SettingsTextField
                control={form.control}
                name="local_models.vad_model"
                label={t("settings.whisperVad")}
              />
              <SettingsTextField
                control={form.control}
                name="local_models.qwen_model"
                label={t("settings.qwenModel")}
              />
              <SettingsTextField
                control={form.control}
                name="local_models.hymt2_profile"
                label={t("settings.hymt2Profile")}
              />
              <SettingsTextField
                control={form.control}
                name="local_models.hymt2_model"
                label={t("settings.hymt2Override")}
                placeholder={t("settings.profileDefault")}
              />
              <SettingsTextField
                control={form.control}
                name="local_models.llama_server"
                label={t("settings.llamaOverride")}
                placeholder={t("settings.autoResolve")}
              />
            </div>
          </div>
        </>
      ),
    },
    {
      id: "defaults",
      label: t("settings.defaults"),
      content: (
        <>
          <SettingsHeading title={t("settings.defaults")} detail={t("settings.defaultsDetail")} />
          <div className="settings-card">
            <div className="field-grid">
              <SettingsTextField
                control={form.control}
                name="transcription.source_language"
                label={t("settings.sourceLanguage")}
                placeholder={t("settings.autoDetect")}
              />
              <SettingsTextField
                control={form.control}
                name="workflow.target_language"
                label={t("settings.targetLanguage")}
              />
              <SettingsTextField
                control={form.control}
                name="workflow.default_glossary"
                label={t("settings.defaultGlossary")}
                placeholder={t("settings.optional")}
              />
              <Controller
                control={form.control}
                name="workflow.subtitle_optimization"
                render={({ field }) => (
                  <SelectField
                    label={t("settings.optimization")}
                    name={field.name}
                    value={field.value}
                    onChange={field.onChange}
                    onBlur={field.onBlur}
                    options={[
                      { value: "aggressive", label: t("settings.aggressive") },
                      { value: "relaxed", label: t("settings.relaxed") },
                    ]}
                  />
                )}
              />
            </div>
            <div className="settings-check-grid">
              <SettingsToggle
                control={form.control}
                id="setting-whisper-gpu"
                name="transcription.use_gpu"
                label={t("settings.whisperGpu")}
              />
              <SettingsToggle
                control={form.control}
                id="setting-flash-attention"
                name="transcription.flash_attn"
                label={t("settings.flashAttention")}
              />
              <SettingsToggle
                control={form.control}
                id="setting-bilingual"
                name="workflow.bilingual_subtitle"
                label={t("settings.bilingual")}
              />
              <SettingsToggle
                control={form.control}
                id="setting-clean-temp"
                name="workflow.clear_temp"
                label={t("settings.cleanTemp")}
              />
            </div>
          </div>
        </>
      ),
    },
    {
      id: "credentials",
      label: t("settings.credentials"),
      content: (
        <>
          <SettingsHeading title={t("settings.credentials")} detail={t("settings.credentialsDetail")} />
          <div className="settings-stack">
            {providerIds.map((provider) => (
              <CredentialRow provider={provider} key={provider} />
            ))}
          </div>
        </>
      ),
    },
  ] as const;

  return (
    <div className="page settings-page">
      <PageHeader
        title={t("settings.title")}
        subtitle={t("settings.subtitle")}
        actions={
          <>
            <Button
              variant="secondary"
              isDisabled={!form.formState.isDirty || save.isPending}
              onPress={() => form.reset(settingsData)}
            >
              {t("common.cancel")}
            </Button>
            <Button
              variant="primary"
              isDisabled={!form.formState.isDirty || save.isPending}
              isPending={save.isPending}
              onPress={() => void form.handleSubmit((data) => save.mutate(data))()}
            >
              <Save size={16} />
              {save.isPending ? t("common.saving") : t("common.save")}
            </Button>
          </>
        }
      />
      {save.error ? (
        <div className="persistent-banner danger" role="alert">
          {save.error.message}
        </div>
      ) : null}
      <SettingsTabs
        label={t("settings.sections")}
        tabs={tabs}
        defaultSelectedKey="general"
        className="settings-tabs"
        listClassName="settings-tab-list"
        panelWrapClassName="settings-tab-content"
        afterList={
          <AppLink to="/about" className="settings-about-link">
            {t("settings.about")}
            <ExternalLink size={14} />
          </AppLink>
        }
      />
    </div>
  );
}

function SettingsTextField({
  control,
  name,
  label,
  placeholder = "",
}: {
  control: Control<AppSettings>;
  name: FieldPath<AppSettings>;
  label: string;
  placeholder?: string;
}): React.JSX.Element {
  return (
    <Controller
      control={control}
      name={name}
      render={({ field, fieldState }) => (
        <TextFieldControl
          label={label}
          name={field.name}
          value={String(field.value ?? "")}
          onChange={field.onChange}
          onBlur={field.onBlur}
          inputRef={field.ref}
          isInvalid={fieldState.invalid}
          inputProps={{ placeholder }}
          {...(fieldState.error?.message ? { errorMessage: fieldState.error.message } : {})}
        />
      )}
    />
  );
}

function SettingsToggle({
  control,
  id,
  name,
  label,
}: {
  control: Control<AppSettings>;
  id: string;
  name: FieldPath<AppSettings>;
  label: string;
}): React.JSX.Element {
  const labelId = `${id}-label`;
  return (
    <div className="settings-toggle-field">
      <label htmlFor={id} id={labelId}>
        {label}
      </label>
      <Controller
        control={control}
        name={name}
        render={({ field }) => (
          <SwitchControl
            id={id}
            name={field.name}
            checked={Boolean(field.value)}
            labelledBy={labelId}
            onBlur={field.onBlur}
            onCheckedChange={field.onChange}
          />
        )}
      />
    </div>
  );
}

function CredentialRow({ provider }: { provider: (typeof providerIds)[number] }): React.JSX.Element {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const status = useQuery({
    queryKey: ["credential", provider],
    queryFn: () => window.openlrc.credentials.status(provider),
  });
  const form = useForm<{ secret: string }>({ defaultValues: { secret: "" } });
  const save = useMutation({
    mutationFn: ({ secret }: { secret: string }) => window.openlrc.credentials.set(provider, secret),
    onSuccess: (data) => {
      queryClient.setQueryData(["credential", provider], data);
      form.reset();
    },
  });
  const remove = useMutation({
    mutationFn: () => window.openlrc.credentials.delete(provider),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["credential", provider] }),
  });
  const label = t("settings.newSecret", { provider: providerLabel(provider) });
  return (
    <div className="credential-row">
      <div className="credential-identity">
        <span className={`credential-icon ${status.data?.present ? "present" : ""}`}>
          {status.data?.present ? <Check size={16} /> : <KeyRound size={16} />}
        </span>
        <div>
          <strong>{providerLabel(provider)}</strong>
          <span>
            {status.isLoading
              ? t("settings.checkingCredential")
              : status.data?.present
                ? t("settings.configuredVia", { source: status.data.source })
                : t("settings.notConfigured")}
          </span>
        </div>
      </div>
      <form onSubmit={form.handleSubmit((data) => save.mutate(data))}>
        <Controller
          control={form.control}
          name="secret"
          rules={{ required: true }}
          render={({ field, fieldState }) => (
            <TextFieldControl
              label={label}
              className="credential-field"
              name={field.name}
              value={field.value}
              onChange={field.onChange}
              onBlur={field.onBlur}
              inputRef={field.ref}
              isInvalid={fieldState.invalid}
              inputProps={{
                id: `secret-${provider}`,
                type: "password",
                autoComplete: "new-password",
                placeholder: t("settings.secretPlaceholder"),
              }}
            />
          )}
        />
        <Button variant="secondary" size="small" type="submit" isPending={save.isPending}>
          {t("settings.saveSecret")}
        </Button>
        {status.data?.present ? (
          <Tooltip label={t("settings.deleteCredential", { provider: providerLabel(provider) })}>
            <IconButton
              label={t("settings.deleteCredential", { provider: providerLabel(provider) })}
              className="danger-text"
              isPending={remove.isPending}
              onPress={() => remove.mutate()}
            >
              <Trash2 size={16} />
            </IconButton>
          </Tooltip>
        ) : null}
      </form>
      {save.error || remove.error || status.data?.warning ? (
        <small className="field-error">
          {save.error?.message ?? remove.error?.message ?? status.data?.warning}
        </small>
      ) : null}
    </div>
  );
}

function SettingsHeading({ title, detail }: { title: string; detail: string }): React.JSX.Element {
  return (
    <div className="settings-heading">
      <h2>{title}</h2>
      <p>{detail}</p>
    </div>
  );
}

function providerLabel(provider: string): string {
  return (
    {
      openai: "OpenAI",
      anthropic: "Anthropic",
      google: "Google",
      litellm: "LiteLLM",
      third_party: "Third party",
    }[provider] ?? provider
  );
}
