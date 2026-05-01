import type { AnyRow } from "../../../api/client";
import { useTrialLossCurve } from "../../../api/hooks";
import { LossLines } from "../../../components/charts";
import { Panel } from "../../../components/panel";
import { KeyValue } from "./key-value";

export default function TrainerTab({ detail, trialId }: { detail: AnyRow; trialId: string }) {
  const latest = detail.latest as AnyRow || {};
  const train = latest.train as AnyRow || {};
  const checkpoint = detail.checkpoint_metadata as AnyRow || {};
  const { data = [] } = useTrialLossCurve(trialId);
  return (
    <section className="suiteGrid">
      <Panel title="Latest Trainer">
        <KeyValue title="Metrics" rows={{
          epoch: latest.epoch || train.epoch,
          loss_total: train.loss_total,
          loss_policy: train.loss_policy,
          loss_value: train.loss_value,
          loss_sparse_policy: train.loss_sparse_policy,
          loss_pair_policy: train.loss_pair_policy,
          loss_regret_policy: train.loss_regret_policy,
          loss_regret_value: train.loss_regret_value,
          loss_entropy: train.loss_entropy,
          policy_top1_acc: train.policy_top1_acc,
          sparse_policy_top1_acc: train.sparse_policy_top1_acc,
          pair_policy_top1_acc: train.pair_policy_top1_acc,
          checkpoint_epoch: checkpoint.epoch,
          global_step: checkpoint.global_step
        }} />
      </Panel>
      <Panel title="Loss Curve"><LossLines rows={data} /></Panel>
    </section>
  );
}
