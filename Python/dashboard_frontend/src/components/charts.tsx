import { Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import type { AnyRow } from "../api/client";
import { asNumber } from "./format";

export function LossLines({ rows }: { rows: AnyRow[] }) {
  const data = rows.map((row, i) => ({ epoch: Number(row.epoch ?? i), ...flattenMetrics(row) }));
  const keys = collectKeys(data, "loss_").slice(0, 8);
  return <LinePlot data={data} xKey="epoch" keys={keys} />;
}

export function ScoreLines({ scores }: { scores: AnyRow[] }) {
  const data = scores.map((row, i) => ({ epoch: Number(row.epoch ?? i), score: asNumber(row.score), scheduler_score: asNumber(row.scheduler_score) }));
  return <LinePlot data={data} xKey="epoch" keys={["score", "scheduler_score"]} />;
}

export function RuntimeScatter({ rows }: { rows: AnyRow[] }) {
  const data = rows.map((row) => ({
    workers: asNumber(row.workers ?? row.num_workers),
    batch: asNumber(row.batch ?? row.batch_size ?? row.max_batch_size),
    pps: asNumber(row.positions_per_sec ?? row.positions_per_min) / (row.positions_per_min ? 60 : 1),
    stable: row.stable ?? row.is_stable ?? true
  }));
  return (
    <div className="chartWrap">
      <ResponsiveContainer width="100%" height={240}>
        <ScatterChart data={data}>
          <XAxis dataKey="workers" name="workers" />
          <YAxis dataKey="pps" name="positions/sec" />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
          <Scatter dataKey="pps" fill="#58a6ff" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

export function LinePlot({ data, xKey, keys }: { data: AnyRow[]; xKey: string; keys: string[] }) {
  return (
    <div className="chartWrap">
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data}>
          <XAxis dataKey={xKey} />
          <YAxis />
          <Tooltip />
          {keys.map((key, i) => <Line key={key} dot={false} type="monotone" dataKey={key} stroke={colors[i % colors.length]} />)}
        </LineChart>
      </ResponsiveContainer>
      <div className="chartLegend">{keys.map((key, i) => <span key={key}><i style={{ background: colors[i % colors.length] }} />{key}</span>)}</div>
    </div>
  );
}

function flattenMetrics(row: AnyRow) {
  const source = (row.metrics_json as AnyRow)?.train as AnyRow || row.metrics_json as AnyRow || row;
  return Object.fromEntries(Object.entries(source || {}).filter(([key, value]) => key.startsWith("loss_") && typeof value === "number"));
}

function collectKeys(rows: AnyRow[], prefix: string) {
  const found = new Set<string>();
  rows.forEach((row) => Object.keys(row).forEach((key) => key.startsWith(prefix) && found.add(key)));
  return [...found].sort();
}

const colors = ["#58a6ff", "#3fb950", "#ff7b72", "#d2a8ff", "#f2cc60", "#79c0ff", "#ffa657"];
