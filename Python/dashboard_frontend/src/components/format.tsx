import type { AnyRow } from "../api/client";

export function cell(value: unknown, key = "") {
  if (value === null || value === undefined || value === "") return "-";
  if (["created_at", "updated_at", "indexed_at", "time"].includes(key)) return formatTimestamp(value);
  if (key.endsWith("_s") || key === "elapsed_s" || key === "epoch_elapsed_s") return formatDuration(value);
  if (key.includes("positions_per_sec")) return formatRate(value);
  if (typeof value === "number") return Number.isInteger(value) ? formatCount(value) : value.toFixed(4);
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "object") return JSON.stringify(value).slice(0, 160);
  return String(value);
}

export function fmt(value: unknown) {
  return typeof value === "number" ? value.toFixed(4) : String(value ?? "-");
}

export function labelFor(key: string) {
  return key.replace(/_/g, " ");
}

export function formatCount(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? new Intl.NumberFormat().format(number) : "-";
}

export function formatRate(value: unknown) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${number.toFixed(number >= 10 ? 1 : 2)}/s`;
}

export function formatDuration(value: unknown) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  if (number < 60) return `${number.toFixed(1)}s`;
  if (number < 3600) return `${Math.floor(number / 60)}m ${Math.round(number % 60)}s`;
  return `${Math.floor(number / 3600)}h ${Math.round((number % 3600) / 60)}m`;
}

export function formatTimestamp(value: unknown) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return String(value ?? "-");
  const ms = number > 10_000_000_000 ? number : number * 1000;
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date(ms));
}

export function compactRows(row: AnyRow | undefined, keys: string[]) {
  return Object.fromEntries(keys.map((key) => [key, row?.[key]]).filter(([, value]) => value !== undefined && value !== null));
}

export function asNumber(value: unknown, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}
