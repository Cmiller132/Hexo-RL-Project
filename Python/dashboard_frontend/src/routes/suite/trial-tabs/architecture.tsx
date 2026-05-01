import type { AnyRow } from "../../../api/client";
import { Panel } from "../../../components/panel";
import { KeyValue } from "./key-value";

export default function ArchitectureTab({ detail }: { detail: AnyRow; trialId: string }) {
  const model = detail.model_metadata as AnyRow || detail.architecture as AnyRow || {};
  const selected = detail.trial as AnyRow || {};
  const family = selected.family as AnyRow || {};
  const fixed = selected.static as AnyRow || {};
  return (
    <Panel title="Architecture">
      <div className="detailGrid">
        <KeyValue title="Model" rows={{
          family: selected.family_id || family.name,
          architecture: model.architecture || family.architecture,
          channels: model.channels,
          blocks: model.blocks,
          heads: Array.isArray(model.heads) ? model.heads.join(", ") : model.heads,
          graph_token_set: model.graph_token_set,
          graph_token_budget: model.graph_token_budget,
          graph_layers: model.graph_layers,
          sparse_policy: model.sparse_policy,
          candidate_budget: model.candidate_budget,
          sparse_prior_stage: model.sparse_prior_stage,
          sparse_prior_mix: model.sparse_prior_mix
        }} />
        <KeyValue title="Family Space Diff" rows={{ sweep_driven: selected.sweep_params, fixed_params: fixed, family_space: family }} />
      </div>
      <details className="configDetails"><summary>Raw config</summary><pre>{JSON.stringify(detail.config || {}, null, 2)}</pre></details>
    </Panel>
  );
}
