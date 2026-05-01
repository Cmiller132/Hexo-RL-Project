import { Link, useSearchParams } from "react-router-dom";
import { Gauge } from "lucide-react";
import { Panel } from "../../components/panel";
import { Table, SortButton } from "../../components/table";
import { fmt, formatCount, formatRate, formatTimestamp } from "../../components/format";
import { useSuiteBestCheckpoints, useSuiteEventsStream, useSuiteGames, useSuiteStatusStream, useSuiteTrials } from "../../api/hooks";
import type { AnyRow } from "../../api/client";

export default function SuiteOverview({ openGame }: { openGame: (gameId: number | string, run?: string) => void }) {
  const [search, setSearch] = useSearchParams();
  const { data: status } = useSuiteStatusStream();
  const { data: trials = [] } = useSuiteTrials();
  const { data: best = [] } = useSuiteBestCheckpoints();
  const { data: events = [] } = useSuiteEventsStream();
  const { data: games = [] } = useSuiteGames();
  const sort = search.get("sort") || "score";
  const filtered = filterTrials(trials, search).sort((a, b) => Number(b[sort] ?? -Infinity) - Number(a[sort] ?? -Infinity));
  const activity = status?.current_activity as AnyRow || {};
  const progress = activity.progress as AnyRow || {};

  const update = (key: string, value: string) => {
    const next = new URLSearchParams(search);
    value ? next.set(key, value) : next.delete(key);
    setSearch(next);
  };

  return (
    <section className="suiteGrid">
      <Panel title="Autotune Suite">
        <div className="suiteHero">
          {[
            ["Stage", status?.current_stage || status?.latest_stage],
            ["Current Trial", status?.current_trial_id || activity.trial_id],
            ["Best Trial", status?.best_trial_id],
            ["Best Score", fmt(status?.best_score)],
            ["Current Model", status?.current_model],
            ["Positions/sec", formatRate(status?.current_positions_per_sec)],
            ["Total Positions", formatCount(status?.total_positions)],
            ["Total Games", formatCount(status?.total_games)],
            ["Workers", progress.workers_total !== undefined ? `${progress.workers_alive}/${progress.workers_total}` : "-"],
            ["Last Event", status?.last_event_name || (status?.last_event as AnyRow)?.event],
            ["Event Time", formatTimestamp(status?.last_event_time || (status?.last_event as AnyRow)?.time)],
            ["Live Trials", `${trials.filter((trial) => !trial.pruned).length}/${status?.trial_count ?? trials.length}`]
          ].map(([label, value]) => <div key={String(label)}><span>{String(label)}</span><strong>{String(value ?? "-")}</strong></div>)}
        </div>
        <div className="activityStrip">
          <Gauge size={15} />
          <span>{String(activity.action || "Waiting for trainer activity")}</span>
          {Boolean(activity.trial_id) && <Link to={`/suite/trials/${String(activity.trial_id)}?${search.toString()}`}>Inspect {String(activity.trial_id)}</Link>}
          <span className="activityMeta">{formatCount(progress.buffer_positions)} buffered positions</span>
        </div>
        <div className="suitePath">{String(status?.run_root || "No suite run root configured")}</div>
      </Panel>
      <div className="suiteSubnav">
        <Link to={`/suite/family-space?${search.toString()}`}>Family Space</Link>
        <Link to={`/suite/scheduler?${search.toString()}`}>Scheduler</Link>
        <Link to={`/suite/runtime-sweep?${search.toString()}`}>Runtime Sweep</Link>
      </div>
      <Panel title="Best Models">
        <Table rows={best} columns={["rank", "trial_id", "score", "scheduler_score", "epoch", "global_step", "is_loadable", "path"]} onRow={(row) => row.trial_id && update("trial", String(row.trial_id))} />
      </Panel>
      <Panel title="Trials">
        <div className="toolbar compact">
          <input placeholder="family" value={search.get("family") || ""} onChange={(e) => update("family", e.target.value)} />
          <input placeholder="architecture" value={search.get("architecture") || ""} onChange={(e) => update("architecture", e.target.value)} />
          <select value={search.get("pruned") || ""} onChange={(e) => update("pruned", e.target.value)}>
            <option value="">all</option><option value="0">active</option><option value="1">pruned</option>
          </select>
          {["score", "scheduler_score", "epoch", "positions_per_sec"].map((id) => <SortButton key={id} id={id} active={sort === id} onClick={(next) => update("sort", next)} />)}
        </div>
        <Table rows={filtered} columns={["trial_id", "family", "architecture", "stage", "epoch", "score", "pruned", "games", "positions", "positions_per_sec", "loss_total"]} onRow={(row) => row.trial_id && window.history.pushState(null, "", `/suite/trials/${row.trial_id}?${search.toString()}`)} />
      </Panel>
      <Panel title="Recent Saved Games">
        <Table rows={games} columns={["game_id", "trial_id", "source", "epoch", "move_count", "terminal_reason", "created_at"]} onRow={(row) => openGame(String(row.game_id), String(row.run_id || ""))} />
      </Panel>
      <Panel title="Recent Suite Events">
        <Table rows={events.slice(0, 96)} columns={["event", "stage", "trial_id", "reason", "score", "selected_positions_per_min", "elapsed_s", "time"]} />
      </Panel>
    </section>
  );
}

function filterTrials(trials: AnyRow[], search: URLSearchParams) {
  return trials.filter((trial) => {
    const family = search.get("family")?.toLowerCase();
    const architecture = search.get("architecture")?.toLowerCase();
    const pruned = search.get("pruned");
    if (family && !String(trial.family || "").toLowerCase().includes(family)) return false;
    if (architecture && !String(trial.architecture || "").toLowerCase().includes(architecture)) return false;
    if (pruned === "0" && trial.pruned) return false;
    if (pruned === "1" && !trial.pruned) return false;
    return true;
  });
}
