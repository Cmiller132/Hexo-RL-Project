import { NavLink, Route, Routes, useParams, useSearchParams } from "react-router-dom";
import { useSuiteTrial } from "../../api/hooks";
import { EmptyState, Panel } from "../../components/panel";
import { fmt, formatRate } from "../../components/format";
import ArchitectureTab from "./trial-tabs/architecture";
import SearchTab from "./trial-tabs/search";
import RuntimeTab from "./trial-tabs/runtime";
import TrainerTab from "./trial-tabs/trainer";
import ScoresTab from "./trial-tabs/scores";
import EventsTab from "./trial-tabs/events";
import CheckpointsTab from "./trial-tabs/checkpoints";

const tabs = [
  ["architecture", "Architecture"],
  ["search", "Search"],
  ["runtime", "Runtime"],
  ["trainer", "Trainer"],
  ["scores", "Scores"],
  ["events", "Events"],
  ["checkpoints", "Checkpoints"]
];

export default function TrialRoute() {
  const { trialId = "" } = useParams();
  const [search] = useSearchParams();
  const { data: detail, isLoading } = useSuiteTrial(trialId);
  if (!trialId) return <EmptyState>No trial selected.</EmptyState>;
  const latest = detail?.latest as Record<string, unknown> || {};
  const train = latest.train as Record<string, unknown> || {};
  const selfplay = latest.selfplay as Record<string, unknown> || {};
  return (
    <section className="suiteGrid">
      <Panel title={`Trial ${trialId}`}>
        <div className="detailHeader">
          <div>
            <h2>{trialId}</h2>
            <p>{String(detail?.architecture_summary || "No architecture metadata yet.")}</p>
          </div>
          <div className="detailCards">
            <Metric label="Epoch" value={latest.epoch || train.epoch} />
            <Metric label="Loss" value={fmt(train.loss_total)} />
            <Metric label="Positions/sec" value={formatRate(Number(selfplay.positions_per_min || 0) / 60)} />
          </div>
        </div>
      </Panel>
      <nav className="tabs subTabs">
        {tabs.map(([id, label]) => <NavLink key={id} to={`${id}?${search.toString()}`}>{label}</NavLink>)}
      </nav>
      {isLoading ? <EmptyState>Loading {trialId}...</EmptyState> : (
        <Routes>
          <Route path="/" element={<ArchitectureTab detail={detail || {}} trialId={trialId} />} />
          <Route path="/architecture" element={<ArchitectureTab detail={detail || {}} trialId={trialId} />} />
          <Route path="/search" element={<SearchTab detail={detail || {}} />} />
          <Route path="/runtime" element={<RuntimeTab detail={detail || {}} trialId={trialId} />} />
          <Route path="/trainer" element={<TrainerTab detail={detail || {}} trialId={trialId} />} />
          <Route path="/scores" element={<ScoresTab trialId={trialId} />} />
          <Route path="/events" element={<EventsTab trialId={trialId} />} />
          <Route path="/checkpoints" element={<CheckpointsTab detail={detail || {}} trialId={trialId} />} />
        </Routes>
      )}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: unknown }) {
  return <div className="metricCard"><span>{label}</span><strong>{String(value ?? "-")}</strong></div>;
}
