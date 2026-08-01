import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { ArrowLeft, Code2, ExternalLink, FileText, ShieldCheck } from "lucide-react";

import { queryKeys } from "../app/queryKeys.js";
import { ErrorState, LoadingState, PageHeader, Section } from "../components/Page.js";
import { AppLink, Button } from "../components/ui/index.js";

export function AboutPage(): React.JSX.Element {
  const { t } = useTranslation();
  const info = useQuery({ queryKey: ["app-info"], queryFn: () => window.openlrc.app.info() });
  const backend = useQuery({
    queryKey: queryKeys.backend,
    queryFn: () => window.openlrc.app.backendStatus(),
  });
  if (info.isLoading)
    return (
      <div className="page">
        <LoadingState />
      </div>
    );
  if (info.error || !info.data)
    return (
      <div className="page">
        <ErrorState error={info.error ?? new Error(t("about.appInfoUnavailable"))} />
      </div>
    );
  return (
    <div className="page about-page">
      <AppLink to="/settings" className="back-link">
        <ArrowLeft size={16} />
        {t("about.back")}
      </AppLink>
      <PageHeader title={t("about.title")} subtitle={t("about.subtitle")} />
      <div className="about-hero">
        <div className="about-mark">
          <span />
          <span />
          <span />
        </div>
        <div>
          <h2>{info.data.name}</h2>
          <p>{t("about.version", { version: info.data.version, protocol: info.data.protocol })}</p>
        </div>
      </div>
      <div className="detail-grid">
        <Section title={t("about.runtime")} className="panel-section">
          <dl className="facts">
            <div>
              <dt>{t("about.platform")}</dt>
              <dd>{info.data.platform}</dd>
            </div>
            <div>
              <dt>{t("about.service")}</dt>
              <dd>{backend.data?.serviceVersion ?? t("common.unavailable")}</dd>
            </div>
            <div>
              <dt>{t("about.backendState")}</dt>
              <dd>{backend.data?.state ?? t("common.unknown")}</dd>
            </div>
          </dl>
        </Section>
        <Section title={t("about.privacy")} className="panel-section">
          <div className="about-copy">
            <ShieldCheck size={20} />
            <p>{t("about.privacyDetail")}</p>
          </div>
        </Section>
      </div>
      <Section title={t("about.diagnostics")}>
        <div className="diagnostic-path">
          <FileText size={18} />
          <code>{info.data.diagnosticPath}</code>
        </div>
      </Section>
      <Section title={t("about.project")}>
        <div className="about-links">
          <Button
            variant="secondary"
            onPress={() => window.openlrc.app.openExternal("https://github.com/zh-plus/openlrc")}
          >
            <Code2 size={16} />
            {t("about.upstream")} <ExternalLink size={14} />
          </Button>
          <span>{t("about.license")}</span>
        </div>
      </Section>
    </div>
  );
}
