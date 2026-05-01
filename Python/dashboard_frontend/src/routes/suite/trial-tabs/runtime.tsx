import type { AnyRow } from "../../../api/client";
import { useTrialRuntimeSweep } from "../../../api/hooks";
import { RuntimeScatter } from "../../../components/charts";
import { Panel } from "../../../components/panel";
import { Table } from "../../../components/table";
import { KeyValue } from "./key-value";

export default function RuntimeTab({ detail, trialId }: { detail: AnyRow; trialId: string }) {
  const cfg = detail.config as AnyRow || {};
  const selfplay = cfg.selfplay as AnyRow || {};
  const inference = cfg.inference as AnyRow || {};
  const runtime = cfg.runtime as AnyRow || {};
  const state = detail.state as AnyRow || {};
  const sweep = state.runtime_sweep as AnyRow || {};
  const { data = [] } = useTrialRuntimeSweep(trialId);
  return (
    <section className="suiteGrid">
      <Panel title="Runtime">
        <KeyValue title="Selected Runtime" rows={{
          num_workers: selfplay.num_workers,
          batch_size_per_worker: selfplay.batch_size_per_worker,
          max_batch_size: inference.max_batch_size,
          max_wait_us: inference.max_wait_us,
          fp16: inference.fp16,
          cpu_threads: runtime.cpu_threads,
          compile_model: runtime.compile_model,
          compile_inference: runtime.compile_inference,
          runtime_sweep_selected: sweep.selected
        }} />
      </Panel>
      <Panel title="Sweep History"><RuntimeScatter rows={data} /><Table rows={data} columns={["workers", "batch", "positions_per_sec", "stable", "selected"]} /></Panel>
    </section>
  );
}
