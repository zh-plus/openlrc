import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { CheckCircle2, RefreshCw, TriangleAlert } from "lucide-react";

import type { ResourceStatus } from "../../shared/contracts.js";
import { queryKeys } from "../app/queryKeys.js";
import { ErrorState, LoadingState, PageHeader, Section } from "../components/Page.js";
import { Button } from "../components/ui/index.js";

export function ResourcesPage(): React.JSX.Element {
  const { t } = useTranslation();
  const resources = useQuery({
    queryKey: queryKeys.resources,
    queryFn: () => window.openlrc.resources.status(),
  });
  const groups = (resources.data ?? []).reduce<Record<string, ResourceStatus[]>>((result, item) => {
    const group = item.role || item.group;
    (result[group] ??= []).push(item);
    return result;
  }, {});
  return (
    <div className="page resources-page">
      <PageHeader
        title={t("resources.title")}
        subtitle={t("resources.subtitle")}
        actions={
          <Button
            variant="primary"
            isDisabled={resources.isFetching}
            onPress={() => void resources.refetch()}
          >
            <RefreshCw size={16} className={resources.isFetching ? "spinner" : ""} />
            {t("common.refresh")}
          </Button>
        }
      />
      {resources.isLoading && <LoadingState label={t("resources.inspecting")} />}
      {resources.error && <ErrorState error={resources.error} retry={() => void resources.refetch()} />}
      {Object.entries(groups).map(([group, items]) => (
        <Section key={group} title={t(`resources.${group}`, { defaultValue: capitalize(group) })}>
          <div className="resource-grid">
            {items?.map((item, index) => (
              <article
                className={`resource-card ${item.available ? "available" : "missing"}`}
                key={`${item.name}-${index}`}
              >
                <div className="resource-card-header">
                  {item.available ? <CheckCircle2 size={19} /> : <TriangleAlert size={19} />}
                  <strong>{item.name}</strong>
                  <span>{item.available ? t("common.ready") : t("common.attention")}</span>
                </div>
                <p title={item.detail}>{item.detail}</p>
                {item.hint && <small>{item.hint}</small>}
              </article>
            ))}
          </div>
        </Section>
      ))}
    </div>
  );
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1).replaceAll("_", " ");
}
