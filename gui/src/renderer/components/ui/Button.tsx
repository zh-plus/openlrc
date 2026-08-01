import type { AnchorHTMLAttributes, ReactNode } from "react";
import {
  Button as AriaButton,
  Link as AriaLink,
  type ButtonProps as AriaButtonProps,
  type LinkProps as AriaLinkProps,
} from "react-aria-components";
import { Link as RouterLink, useLocation, type LinkProps as RouterLinkProps } from "react-router-dom";
import clsx from "clsx";

export type ButtonVariant = "primary" | "secondary" | "danger" | "ghost" | "icon";
export type ButtonSize = "small" | "medium";

export interface ButtonProps extends Omit<AriaButtonProps, "className"> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  className?: string;
}

const buttonVariants: Record<ButtonVariant, string> = {
  primary: "button primary border-transparent bg-app-primary text-white",
  secondary: "button secondary border-app-border bg-app-surface text-app-text",
  danger: "button danger border-transparent bg-app-danger text-white",
  ghost: "button ghost border-transparent bg-transparent text-app-text-secondary",
  icon: "icon-button inline-grid size-[34px] shrink-0 place-items-center rounded-[7px] border-0 bg-transparent p-0 text-app-text-secondary",
};

const buttonBase =
  "inline-flex min-h-[38px] items-center justify-center gap-2 rounded-lg border px-3.5 py-2 font-semibold whitespace-nowrap rac-disabled:cursor-not-allowed rac-disabled:opacity-45 rac-focus-visible:outline-2 rac-focus-visible:outline-offset-2 rac-focus-visible:outline-app-primary";

export function Button({
  variant = "secondary",
  size = "medium",
  className,
  ...props
}: ButtonProps): React.JSX.Element {
  return (
    <AriaButton
      {...props}
      className={clsx(
        variant !== "icon" && buttonBase,
        buttonVariants[variant],
        size === "small" && "small min-h-8 px-2.5 py-1.5 text-xs",
        className,
      )}
    />
  );
}

interface AppLinkProps
  extends
    Omit<AriaLinkProps, "className" | "href" | "render">,
    Pick<RouterLinkProps, "to" | "state" | "replace"> {
  className?: string;
}

export function AppLink({ to, state, replace, className, ...props }: AppLinkProps): React.JSX.Element {
  const href = typeof to === "string" ? to : (to.pathname ?? "");
  return (
    <AriaLink
      {...props}
      href={href}
      className={className ?? ""}
      render={(linkProps) => {
        const routerProps = linkProps as AnchorHTMLAttributes<HTMLAnchorElement>;
        return (
          <RouterLink
            {...routerProps}
            to={to}
            {...(state !== undefined ? { state } : {})}
            {...(replace !== undefined ? { replace } : {})}
          />
        );
      }}
    />
  );
}

interface AppNavLinkProps extends AppLinkProps {
  end?: boolean;
  activeClassName?: string;
}

export function AppNavLink({
  to,
  end = false,
  activeClassName = "active",
  className,
  ...props
}: AppNavLinkProps): React.JSX.Element {
  const location = useLocation();
  const pathname = typeof to === "string" ? to : (to.pathname ?? "");
  const isCurrent = end
    ? location.pathname === pathname
    : location.pathname === pathname || location.pathname.startsWith(`${pathname}/`);
  return (
    <AppLink
      {...props}
      to={to}
      className={clsx(className, isCurrent && activeClassName)}
      {...(isCurrent ? { "aria-current": "page" as const } : {})}
    />
  );
}

export interface IconButtonProps extends Omit<ButtonProps, "children" | "variant"> {
  label: string;
  children: ReactNode;
}

export function IconButton({ label, children, ...props }: IconButtonProps): React.JSX.Element {
  return (
    <Button {...props} variant="icon" aria-label={label}>
      {children}
    </Button>
  );
}
