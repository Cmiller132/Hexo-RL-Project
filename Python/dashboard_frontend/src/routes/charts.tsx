import { useMetrics } from "../api/hooks";
import { LossLines } from "../components/charts";
import { Panel } from "../components/panel";
import { Table } from "../components/table";

export default function ChartsRoute({ runId }: { runId: string }) {
  const { data = [] } = useMetrics(runId);
  return (
    <section className="grid two">
      <Panel title="Losses"><LossLines rows={data} /></Panel>
      <Panel title="Recent Metrics">
        <Table rows={data.slice(-24).reverse()} columns={["phase", "epoch", "global_step", "created_at", "metrics_json"]} />
      </Panel>
    </section>
  );
}
