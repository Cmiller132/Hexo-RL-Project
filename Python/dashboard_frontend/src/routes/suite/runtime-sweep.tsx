import { useRuntimeSweep } from "../../api/hooks";
import { RuntimeScatter } from "../../components/charts";
import { Panel } from "../../components/panel";
import { Table } from "../../components/table";

export default function RuntimeSweepRoute() {
  const { data = [] } = useRuntimeSweep();
  return (
    <section className="suiteGrid">
      <Panel title="Runtime Sweep Scatter"><RuntimeScatter rows={data} /></Panel>
      <Panel title="Sweep Results"><Table rows={data} columns={["trial_id", "family", "architecture", "workers", "batch", "positions_per_sec", "stable", "selected"]} /></Panel>
    </section>
  );
}
