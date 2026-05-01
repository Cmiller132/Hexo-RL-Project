import { useTrialEvents, useTrialScores } from "../../../api/hooks";
import { ScoreLines } from "../../../components/charts";
import { Panel } from "../../../components/panel";
import { Table } from "../../../components/table";

export default function ScoresTab({ trialId }: { trialId: string }) {
  const { data: scores = [] } = useTrialScores(trialId);
  const { data: events = [] } = useTrialEvents(trialId);
  return (
    <section className="suiteGrid">
      <Panel title="Score History"><ScoreLines scores={scores} /></Panel>
      <Panel title="Score Rows"><Table rows={scores} columns={["epoch", "score", "scheduler_score", "games", "positions", "time"]} /></Panel>
      <Panel title="Prune And Promote Events"><Table rows={events.filter((event) => /prune|promote/i.test(String(event.event || "")))} columns={["event", "stage", "reason", "score", "time"]} /></Panel>
    </section>
  );
}
