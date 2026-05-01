import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { apiGet, apiPost, enc, type AnyRow, type SuiteTrialDetail } from "./client";
import { anyRowSchema, anyRowsSchema, familySpaceSchema, gameSchema, parseRows, runSchema, suiteStatusSchema } from "./schemas";

const rows = (path: string, fallback: AnyRow[] = []) => apiGet<unknown>(path, fallback).then(parseRows);
const row = (path: string, fallback: AnyRow = {}) => apiGet<unknown>(path, fallback).then((v) => anyRowSchema.catch(fallback).parse(v));

export const useHealth = () => useQuery({ queryKey: ["health"], queryFn: () => row("/api/health") });
export const useRuns = () => useQuery({ queryKey: ["runs"], queryFn: () => rows("/api/runs").then((v) => v.map((r) => runSchema.parse(r))) });
export const useMetrics = (runId: string) => useQuery({ queryKey: ["metrics", runId], enabled: !!runId, queryFn: () => rows(`/api/metrics/${enc(runId)}`) });
export const useGames = (runId?: string) => useQuery({ queryKey: ["games", runId], queryFn: () => rows(`/api/games?${runId ? `run_id=${enc(runId)}&` : ""}limit=64`).then((v) => v.map((g) => gameSchema.parse(g))) });
export const useCheckpoints = (runId?: string) => useQuery({ queryKey: ["checkpoints", runId], queryFn: () => rows(`/api/checkpoints${runId ? `?run_id=${enc(runId)}` : ""}`) });
export const useReplay = (gameId?: string, runId?: string) => useQuery({ queryKey: ["replay", gameId, runId], enabled: !!gameId, queryFn: () => row(`/api/games/${enc(gameId || "")}/replay${runId ? `?run_id=${enc(runId)}` : ""}`) });
export const usePosition = (gameId?: string, turn = 0, runId?: string) => useQuery({ queryKey: ["position", gameId, turn, runId], enabled: !!gameId, queryFn: () => row(`/api/games/${enc(gameId || "")}/position/${turn}${runId ? `?run_id=${enc(runId)}` : ""}`) });
export const useArena = () => useQuery({ queryKey: ["arena"], queryFn: () => rows("/api/arena/history") });
export const useAxisPrototypes = () => useQuery({ queryKey: ["axis-prototypes"], queryFn: () => rows("/api/axis/prototypes") });
export const useAxisFixtures = () => useQuery({ queryKey: ["axis-fixtures"], queryFn: () => rows("/api/axis/fixtures") });
export const useSuiteTrials = () => useQuery({ queryKey: ["suite-trials"], queryFn: () => rows("/api/suite/trials"), refetchInterval: 15_000 });
export const useSuiteBestCheckpoints = () => useQuery({ queryKey: ["suite-best"], queryFn: () => rows("/api/suite/best-checkpoints", []), refetchInterval: 15_000 });
export const useSuiteGames = () => useQuery({ queryKey: ["suite-games"], queryFn: () => rows("/api/games?limit=32"), refetchInterval: 15_000 });
export const useSuiteEvents = () => useQuery({ queryKey: ["suite-events"], queryFn: () => rows("/api/suite/events?limit=128") });
export const useSuiteTrial = (trialId?: string) => useQuery<SuiteTrialDetail>({ queryKey: ["suite-trial", trialId], enabled: !!trialId, queryFn: () => row(`/api/suite/trials/${enc(trialId || "")}`) as Promise<SuiteTrialDetail> });
export const useTrialScores = (trialId?: string) => useQuery({ queryKey: ["trial-scores", trialId], enabled: !!trialId, queryFn: () => rows(`/api/suite/trials/${enc(trialId || "")}/scores`, []) });
export const useTrialEvents = (trialId?: string) => useQuery({ queryKey: ["trial-events", trialId], enabled: !!trialId, queryFn: () => rows(`/api/suite/trials/${enc(trialId || "")}/events`, []) });
export const useTrialLossCurve = (trialId?: string) => useQuery({ queryKey: ["trial-loss", trialId], enabled: !!trialId, queryFn: () => rows(`/api/suite/trials/${enc(trialId || "")}/loss-curve`, []) });
export const useTrialRuntimeSweep = (trialId?: string) => useQuery({ queryKey: ["trial-runtime-sweep", trialId], enabled: !!trialId, queryFn: async () => sweepRows(await apiGet<unknown>(`/api/suite/trials/${enc(trialId || "")}/runtime-sweep`, {})) });
export const useFamilySpace = () => useQuery({ queryKey: ["family-space"], queryFn: () => apiGet<unknown>("/api/suite/family-space", {}).then((v) => familySpaceSchema.catch({}).parse(v)) });
export const useScheduler = () => useQuery({ queryKey: ["scheduler"], queryFn: () => row("/api/suite/scheduler", {}) });
export const useRuntimeSweep = () => useQuery({ queryKey: ["runtime-sweep"], queryFn: async () => sweepRows(await apiGet<unknown>("/api/suite/runtime-sweep", {})) });

export function useArenaStream() {
  const queryClient = useQueryClient();
  const query = useArena();
  useEffect(() => {
    const source = new EventSource("/api/arena/history/stream");
    source.onmessage = (event) => queryClient.setQueryData(["arena"], anyRowsSchema.catch([]).parse(JSON.parse(event.data)));
    return () => source.close();
  }, [queryClient]);
  return query;
}

export function useSuiteStatusStream() {
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: ["suite-status"],
    queryFn: () => apiGet<unknown>("/api/suite/status", {}).then((v) => suiteStatusSchema.catch({}).parse(v))
  });
  useEffect(() => {
    const source = new EventSource("/api/suite/status/stream");
    source.onmessage = (event) => queryClient.setQueryData(["suite-status"], suiteStatusSchema.catch({}).parse(JSON.parse(event.data)));
    return () => source.close();
  }, [queryClient]);
  return query;
}

export function useSuiteEventsStream() {
  const query = useSuiteEvents();
  const [events, setEvents] = useState<AnyRow[]>([]);
  useEffect(() => setEvents(query.data || []), [query.data]);
  useEffect(() => {
    const source = new EventSource("/api/suite/events/stream");
    source.onmessage = (event) => setEvents((prev) => [anyRowSchema.parse(JSON.parse(event.data)), ...prev].slice(0, 128));
    return () => source.close();
  }, []);
  return useMemo(() => ({ ...query, data: events }), [query, events]);
}

export const mutations = { apiPost };

function sweepRows(payload: unknown): AnyRow[] {
  if (Array.isArray(payload)) return parseRows(payload);
  const row = anyRowSchema.catch({}).parse(payload);
  const probes = Array.isArray(row.probes) ? row.probes as AnyRow[] : [];
  const selected = Array.isArray(row.selected) ? row.selected as AnyRow[] : [];
  const history = Array.isArray(row.history) ? row.history as AnyRow[] : [];
  const results = Array.isArray(row.results) ? row.results as AnyRow[] : [];
  const choice = row.selected && !Array.isArray(row.selected) && typeof row.selected === "object" ? [row.selected as AnyRow] : [];
  return [...probes, ...history, ...results, ...selected, ...choice];
}
