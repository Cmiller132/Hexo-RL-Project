import type { components, paths } from "./schema";

export type AnyRow = Record<string, unknown>;
export type ApiPath = keyof paths & string;
export type SuiteTrialDetail = components["schemas"]["SuiteTrialDetailV2"];

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function apiGet<T>(path: string, fallback?: T): Promise<T> {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" } });
  if (res.status === 404 && fallback !== undefined) return fallback;
  if (!res.ok) throw new ApiError(res.status, await res.text());
  return res.json() as Promise<T>;
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {})
  });
  if (!res.ok) throw new ApiError(res.status, await res.text());
  return res.json() as Promise<T>;
}

export function enc(value: string | number) {
  return encodeURIComponent(String(value));
}
