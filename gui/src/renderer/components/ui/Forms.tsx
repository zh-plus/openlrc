import { useLayoutEffect, useRef, type CSSProperties, type ReactNode, type Ref } from "react";
import {
  Button as AriaButton,
  CheckboxButton,
  CheckboxField as AriaCheckboxField,
  FieldError,
  Input,
  Label,
  ListBox,
  ListBoxItem,
  Popover,
  RadioButton,
  RadioField,
  RadioGroup,
  SearchField as AriaSearchField,
  Select,
  SelectValue,
  SwitchButton,
  SwitchField as AriaSwitchField,
  Text,
  TextArea,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  type InputProps,
  type Key,
  type TextFieldProps,
} from "react-aria-components";
import { Check, ChevronDown, Search, X } from "lucide-react";
import clsx from "clsx";

const fieldClass = "field flex min-w-0 flex-col gap-1.5 text-app-text-secondary";
const fieldInputClass =
  "min-h-[38px] w-full rounded-lg border border-app-border bg-app-canvas px-2.5 py-2 text-app-text outline-none transition-[border-color,box-shadow] focus:border-app-primary focus:ring-3 focus:ring-app-primary-soft";

interface FieldChromeProps {
  label: ReactNode;
  description?: ReactNode;
  errorMessage?: ReactNode;
  className?: string;
}

export interface TextFieldControlProps
  extends Omit<TextFieldProps, "children" | "className">, FieldChromeProps {
  inputProps?: Omit<InputProps, "className">;
  inputRef?: Ref<HTMLInputElement>;
}

export function TextFieldControl({
  label,
  description,
  errorMessage,
  className,
  inputProps,
  inputRef,
  ...props
}: TextFieldControlProps): React.JSX.Element {
  return (
    <TextField {...props} validationBehavior="aria" className={clsx(fieldClass, className)}>
      <Label className="font-semibold text-app-text">{label}</Label>
      <Input {...inputProps} ref={inputRef} className={fieldInputClass} />
      {description ? (
        <Text slot="description" className="text-xs text-app-text-secondary">
          {description}
        </Text>
      ) : null}
      {errorMessage ? <FieldError className="text-xs text-app-danger">{errorMessage}</FieldError> : null}
    </TextField>
  );
}

export interface TextAreaFieldProps extends Omit<TextFieldProps, "children" | "className">, FieldChromeProps {
  rows?: number;
  placeholder?: string;
}

export function TextAreaField({
  label,
  description,
  errorMessage,
  className,
  rows,
  placeholder,
  ...props
}: TextAreaFieldProps): React.JSX.Element {
  return (
    <TextField {...props} validationBehavior="aria" className={clsx(fieldClass, className)}>
      <Label className="font-semibold text-app-text">{label}</Label>
      <TextArea className={clsx(fieldInputClass, "resize-y")} rows={rows} placeholder={placeholder} />
      {description ? (
        <Text slot="description" className="text-xs text-app-text-secondary">
          {description}
        </Text>
      ) : null}
      {errorMessage ? <FieldError className="text-xs text-app-danger">{errorMessage}</FieldError> : null}
    </TextField>
  );
}

export interface SelectOption {
  value: string;
  label: ReactNode;
  textValue?: string;
}

interface SelectFieldProps extends FieldChromeProps {
  name?: string;
  value: string;
  options: readonly SelectOption[];
  onChange(value: string): void;
  onBlur?(): void;
  isDisabled?: boolean;
  isRequired?: boolean;
  isInvalid?: boolean;
}

export function SelectField({
  name = "",
  value,
  options,
  onChange,
  onBlur = () => undefined,
  isDisabled = false,
  isRequired = false,
  isInvalid = false,
  label,
  description,
  errorMessage,
  className,
}: SelectFieldProps): React.JSX.Element {
  return (
    <Select
      name={name}
      selectedKey={value}
      onSelectionChange={(key) => onChange(String(key))}
      onBlur={onBlur}
      isDisabled={isDisabled}
      isRequired={isRequired}
      isInvalid={isInvalid}
      validationBehavior="aria"
      className={clsx(fieldClass, "select-field", className)}
    >
      <Label className="font-semibold text-app-text">{label}</Label>
      <AriaButton
        className={clsx(
          fieldInputClass,
          "select-trigger flex items-center justify-between gap-3 text-left rac-hovered:border-app-primary rac-focus-visible:border-app-primary rac-focus-visible:ring-3 rac-focus-visible:ring-app-primary-soft",
        )}
      >
        <SelectValue />
        <ChevronDown size={16} aria-hidden="true" />
      </AriaButton>
      {description ? (
        <Text slot="description" className="text-xs text-app-text-secondary">
          {description}
        </Text>
      ) : null}
      {errorMessage ? <FieldError className="text-xs text-app-danger">{errorMessage}</FieldError> : null}
      <Popover
        className="select-popover min-w-(--trigger-width) overflow-hidden rounded-lg border border-app-border bg-app-surface p-1 text-app-text shadow-app-card rac-entering:animate-[popover-in_120ms_ease-out] rac-exiting:animate-[popover-out_90ms_ease-in]"
        placement="bottom start"
      >
        <ListBox className="select-listbox max-h-64 overflow-auto outline-none">
          {options.map((option) => (
            <ListBoxItem
              id={option.value}
              key={option.value}
              textValue={option.textValue ?? String(option.label)}
              className="select-option flex min-h-9 cursor-default items-center justify-between gap-3 rounded-md px-2.5 py-1.5 outline-none rac-hovered:bg-app-surface-subtle rac-focused:bg-app-primary-soft rac-selected:text-app-primary rac-disabled:opacity-45"
            >
              {({ isSelected }) => (
                <>
                  <span>{option.label}</span>
                  {isSelected ? <Check size={15} aria-hidden="true" /> : null}
                </>
              )}
            </ListBoxItem>
          ))}
        </ListBox>
      </Popover>
    </Select>
  );
}

export interface SearchFieldProps {
  label: string;
  value: string;
  placeholder?: string;
  onChange(value: string): void;
}

export function SearchField({ label, value, placeholder, onChange }: SearchFieldProps): React.JSX.Element {
  return (
    <AriaSearchField
      className="search-field flex min-w-65 max-w-95 flex-1 items-center gap-2 rounded-lg border border-app-border bg-app-surface px-2.5 text-app-text-secondary rac-focus-within:border-app-primary rac-focus-within:ring-3 rac-focus-within:ring-app-primary-soft"
      value={value}
      onChange={onChange}
      aria-label={label}
    >
      <Search size={17} aria-hidden="true" />
      <Input
        className="h-[38px] w-full border-0 bg-transparent p-0 text-app-text outline-none"
        placeholder={placeholder ?? ""}
      />
      <AriaButton
        slot="clear"
        className="search-clear grid size-7 place-items-center rounded-md rac-hovered:bg-app-surface-subtle"
        aria-label={`${label}: clear`}
      >
        <X size={14} aria-hidden="true" />
      </AriaButton>
    </AriaSearchField>
  );
}

interface BooleanControlProps {
  id: string;
  name: string;
  checked: boolean;
  disabled?: boolean;
  labelledBy: string;
  onBlur(): void;
  onCheckedChange(checked: boolean): void;
}

export function SwitchControl({
  id,
  name,
  checked,
  disabled = false,
  labelledBy,
  onBlur,
  onCheckedChange,
}: BooleanControlProps): React.JSX.Element {
  return (
    <AriaSwitchField
      id={id}
      name={name}
      isSelected={checked}
      isDisabled={disabled}
      aria-labelledby={labelledBy}
      onBlur={onBlur}
      onChange={onCheckedChange}
      validationBehavior="aria"
      className="boolean-field inline-flex"
    >
      <SwitchButton className="toggle-switch border-app-border bg-app-surface-subtle rac-hovered:border-app-primary rac-selected:border-app-primary rac-selected:bg-app-primary rac-disabled:opacity-45 rac-focus-visible:outline-2 rac-focus-visible:outline-offset-2 rac-focus-visible:outline-app-primary">
        <span className="toggle-switch-thumb" />
      </SwitchButton>
    </AriaSwitchField>
  );
}

export function CheckboxControl({
  id,
  name,
  checked,
  disabled = false,
  labelledBy,
  onBlur,
  onCheckedChange,
}: BooleanControlProps): React.JSX.Element {
  return (
    <AriaCheckboxField
      id={id}
      name={name}
      isSelected={checked}
      isDisabled={disabled}
      aria-labelledby={labelledBy}
      onBlur={onBlur}
      onChange={onCheckedChange}
      validationBehavior="aria"
      className="boolean-field inline-flex"
    >
      <CheckboxButton className="checkbox-control border-app-border bg-app-surface text-white rac-hovered:border-app-primary rac-hovered:bg-app-primary-soft rac-selected:border-app-primary rac-selected:bg-app-primary rac-disabled:opacity-45 rac-focus-visible:outline-2 rac-focus-visible:outline-offset-2 rac-focus-visible:outline-app-primary">
        {({ isSelected }) => (
          <span className="checkbox-indicator">
            {isSelected ? <Check size={14} strokeWidth={3} aria-hidden="true" /> : null}
          </span>
        )}
      </CheckboxButton>
    </AriaCheckboxField>
  );
}

export interface RadioOption {
  value: string;
  label: ReactNode;
  detail?: ReactNode;
  accent?: boolean;
}

export function RadioCardGroup({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: readonly RadioOption[];
  onChange(value: string): void;
}): React.JSX.Element {
  return (
    <RadioGroup
      className="choice-cards grid grid-cols-4 gap-2.5 max-[1100px]:grid-cols-2"
      aria-label={label}
      value={value}
      onChange={onChange}
    >
      {options.map((option) => (
        <RadioField value={option.value} key={option.value} className="radio-field">
          <RadioButton
            className={clsx(
              "choice-card flex min-h-21.5 w-full items-start gap-2 rounded-[10px] border border-app-border bg-app-canvas p-3.5 text-left text-app-text outline-none rac-hovered:border-app-primary rac-selected:border-app-primary rac-selected:bg-app-primary-soft rac-focus-visible:ring-2 rac-focus-visible:ring-app-primary",
              option.accent && "accent rac-selected:border-app-secondary rac-selected:bg-app-secondary-soft",
            )}
          >
            {({ isSelected }) => (
              <>
                <span className="choice-radio" aria-hidden="true">
                  {isSelected ? <span /> : null}
                </span>
                <span>
                  <strong>{option.label}</strong>
                  {option.detail ? <small>{option.detail}</small> : null}
                </span>
              </>
            )}
          </RadioButton>
        </RadioField>
      ))}
    </RadioGroup>
  );
}

export interface SegmentOption {
  value: string;
  label: ReactNode;
  icon?: ReactNode;
  ariaLabel?: string;
}

export function SegmentedControl({
  label,
  value,
  options,
  onChange,
  className,
  optionClassName,
}: {
  label: string;
  value: string;
  options: readonly SegmentOption[];
  onChange(value: string): void;
  className?: string;
  optionClassName?: string;
}): React.JSX.Element {
  const selectedIndex = Math.max(
    0,
    options.findIndex((option) => option.value === value),
  );
  const segmentStyle = {
    "--segment-count": String(options.length),
    "--segment-index": String(selectedIndex),
  } as CSSProperties;

  return (
    <RadioGroup
      className={clsx(
        "inline-segment relative isolate grid shrink-0 overflow-hidden rounded-[9px] border border-app-border bg-app-surface-subtle p-0.75",
        className,
      )}
      style={segmentStyle}
      aria-label={label}
      orientation="horizontal"
      value={value}
      onChange={onChange}
    >
      <span className="segment-slider" aria-hidden="true" />
      {options.map((option) => (
        <RadioField value={option.value} key={option.value} className="segment-field min-w-0">
          <RadioButton
            aria-label={option.ariaLabel}
            className={clsx(
              "segment-option relative z-1 flex min-h-7.5 w-full items-center justify-center gap-1.5 rounded-md px-2 py-1 text-app-text-secondary outline-none rac-hovered:text-app-text rac-selected:font-semibold rac-selected:text-app-primary rac-focus-visible:ring-2 rac-focus-visible:ring-app-primary",
              optionClassName,
            )}
          >
            {option.icon}
            <span>{option.label}</span>
          </RadioButton>
        </RadioField>
      ))}
    </RadioGroup>
  );
}

export function SingleToggleGroup({
  label,
  value,
  options,
  onChange,
  className,
}: {
  label: string;
  value: string;
  options: readonly SegmentOption[];
  onChange(value: string): void;
  className?: string;
}): React.JSX.Element {
  const groupRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const group = groupRef.current;
    if (!group) return;

    const updateSlider = (): void => {
      const selected = group.querySelector<HTMLElement>(".filter-option[data-selected]");
      if (!selected) {
        group.removeAttribute("data-slider-ready");
        return;
      }
      group.style.setProperty("--toggle-slider-left", `${selected.offsetLeft}px`);
      group.style.setProperty("--toggle-slider-width", `${selected.offsetWidth}px`);
      group.setAttribute("data-slider-ready", "true");
    };

    updateSlider();
    if (typeof ResizeObserver === "undefined") return;

    const observer = new ResizeObserver(updateSlider);
    observer.observe(group);
    group.querySelectorAll<HTMLElement>(".filter-option").forEach((option) => observer.observe(option));
    return () => observer.disconnect();
  }, [options.length, value]);

  return (
    <ToggleButtonGroup
      ref={groupRef}
      aria-label={label}
      selectionMode="single"
      disallowEmptySelection
      selectedKeys={new Set<Key>([value])}
      onSelectionChange={(keys) => {
        const next = [...keys][0];
        if (next !== undefined) onChange(String(next));
      }}
      className={clsx(
        "sliding-toggle-group relative isolate flex gap-0.75 overflow-x-auto rounded-[9px] border border-app-border bg-app-surface-subtle p-0.75",
        className,
      )}
    >
      <span className="segment-slider toggle-group-slider" aria-hidden="true" />
      {options.map((option) => (
        <ToggleButton
          id={option.value}
          key={option.value}
          className="filter-option relative z-1 min-h-7.5 rounded-md px-2.5 py-1 text-app-text-secondary whitespace-nowrap outline-none rac-hovered:text-app-text rac-selected:font-semibold rac-selected:text-app-primary rac-focus-visible:ring-2 rac-focus-visible:ring-app-primary"
        >
          {option.icon}
          {option.label}
        </ToggleButton>
      ))}
    </ToggleButtonGroup>
  );
}
