import i18n from "../i18n.js";

export function formatDate(value: string | null): string {
  if (!value) return "—";
  return new Intl.DateTimeFormat(i18n.language, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "—";
  const rounded = Math.round(seconds);
  const minutes = Math.floor(rounded / 60);
  const remainder = rounded % 60;
  const minute = i18n.t("common.minuteShort");
  const second = i18n.t("common.secondShort");
  return minutes ? `${minutes}${minute} ${remainder}${second}` : `${remainder}${second}`;
}

export function basename(value: string): string {
  return value.split(/[\\/]/).at(-1) || value;
}

export function middleTruncate(value: string, length = 60): string {
  if (value.length <= length) return value;
  const side = Math.floor((length - 1) / 2);
  return `${value.slice(0, side)}…${value.slice(-side)}`;
}
