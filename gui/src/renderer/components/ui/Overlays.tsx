import { useEffect, useRef, type ReactElement, type ReactNode } from "react";
import {
  Dialog as AriaDialog,
  Heading,
  Modal,
  ModalOverlay,
  Text,
  Tooltip as AriaTooltip,
  TooltipTrigger,
} from "react-aria-components";

export function Tooltip({
  children,
  label,
  placement = "right",
  delay = 350,
}: {
  children: ReactElement;
  label: ReactNode;
  placement?: "top" | "bottom" | "left" | "right";
  delay?: number;
}): React.JSX.Element {
  return (
    <TooltipTrigger delay={delay} closeDelay={0}>
      {children}
      <AriaTooltip className="tooltip" placement={placement}>
        {label}
      </AriaTooltip>
    </TooltipTrigger>
  );
}

export interface ModalDialogProps {
  isOpen: boolean;
  onOpenChange(isOpen: boolean): void;
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  isDismissable?: boolean;
}

export function ModalDialog({
  isOpen,
  onOpenChange,
  title,
  description,
  children,
  isDismissable = true,
}: ModalDialogProps): React.JSX.Element {
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const wasOpenRef = useRef(false);

  useEffect(() => {
    if (isOpen) return undefined;
    const rememberFocus = (event: FocusEvent): void => {
      if (event.target instanceof HTMLElement && event.target !== document.body) {
        returnFocusRef.current = event.target;
      }
    };
    if (document.activeElement instanceof HTMLElement && document.activeElement !== document.body) {
      returnFocusRef.current = document.activeElement;
    }
    document.addEventListener("focusin", rememberFocus);
    return () => document.removeEventListener("focusin", rememberFocus);
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen && wasOpenRef.current) {
      const returnTarget = returnFocusRef.current;
      queueMicrotask(() => returnTarget?.focus());
    }
    wasOpenRef.current = isOpen;
  }, [isOpen]);

  return (
    <ModalOverlay
      className="dialog-overlay"
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      isDismissable={isDismissable}
    >
      <Modal className="dialog-content">
        <AriaDialog className="dialog-body">
          <Heading slot="title">{title}</Heading>
          {description ? <Text slot="description">{description}</Text> : null}
          {children}
        </AriaDialog>
      </Modal>
    </ModalOverlay>
  );
}
