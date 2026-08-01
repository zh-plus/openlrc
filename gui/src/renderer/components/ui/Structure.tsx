import type { ReactNode } from "react";
import {
  Button as AriaButton,
  Disclosure as AriaDisclosure,
  DisclosurePanel,
  Heading,
  Label,
  ProgressBar as AriaProgressBar,
  Tab,
  TabList,
  TabPanel,
  Tabs as AriaTabs,
} from "react-aria-components";
import { ChevronDown } from "lucide-react";
import clsx from "clsx";

export function Disclosure({
  title,
  detail,
  children,
  className,
  defaultExpanded = false,
}: {
  title: ReactNode;
  detail?: ReactNode;
  children: ReactNode;
  className?: string;
  defaultExpanded?: boolean;
}): React.JSX.Element {
  return (
    <AriaDisclosure className={className ?? ""} defaultExpanded={defaultExpanded}>
      <Heading>
        <AriaButton slot="trigger" className="disclosure-trigger">
          <span>
            <strong>{title}</strong>
            {detail ? <small>{detail}</small> : null}
          </span>
          <ChevronDown size={18} aria-hidden="true" />
        </AriaButton>
      </Heading>
      <DisclosurePanel className="disclosure-panel">{children}</DisclosurePanel>
    </AriaDisclosure>
  );
}

export function ProgressBar({
  label,
  value,
  className,
  variant = "standard",
}: {
  label: string;
  value: number;
  className?: string;
  variant?: "standard" | "top";
}): React.JSX.Element {
  return (
    <AriaProgressBar
      value={value}
      minValue={0}
      maxValue={100}
      className={className ?? ""}
      {...(variant === "top" ? { "aria-label": label } : {})}
    >
      {({ percentage, valueText }) =>
        variant === "top" ? (
          <span style={{ width: `${percentage ?? 0}%` }} />
        ) : (
          <>
            <div className="progress-copy">
              <Label>{label}</Label>
              <span>{valueText}</span>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${percentage ?? 0}%` }} />
            </div>
          </>
        )
      }
    </AriaProgressBar>
  );
}

export interface TabDefinition {
  id: string;
  label: ReactNode;
  content: ReactNode;
}

export function Tabs({
  label,
  tabs,
  defaultSelectedKey,
  className,
  listClassName,
  panelWrapClassName,
  afterList,
}: {
  label: string;
  tabs: readonly TabDefinition[];
  defaultSelectedKey: string;
  className?: string;
  listClassName?: string;
  panelWrapClassName?: string;
  afterList?: ReactNode;
}): React.JSX.Element {
  return (
    <AriaTabs defaultSelectedKey={defaultSelectedKey} className={className ?? ""} orientation="vertical">
      <div>
        <TabList aria-label={label} className={listClassName ?? ""}>
          {tabs.map((tab) => (
            <Tab id={tab.id} key={tab.id}>
              {tab.label}
            </Tab>
          ))}
        </TabList>
        {afterList}
      </div>
      <div className={clsx(panelWrapClassName)}>
        {tabs.map((tab) => (
          <TabPanel id={tab.id} key={tab.id}>
            {tab.content}
          </TabPanel>
        ))}
      </div>
    </AriaTabs>
  );
}
