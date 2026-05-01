import type { AnyRow } from "../../../api/client";
import { Panel } from "../../../components/panel";
import { KeyValue } from "./key-value";

export default function SearchTab({ detail }: { detail: AnyRow }) {
  const cfg = detail.config as AnyRow || {};
  const selfplay = cfg.selfplay as AnyRow || {};
  const search = cfg.search as AnyRow || {};
  return (
    <Panel title="Search And Self-Play">
      <div className="detailGrid">
        <KeyValue title="MCTS" rows={{
          mcts_simulations: selfplay.mcts_simulations ?? search.mcts_simulations,
          pcr_low_sims: selfplay.pcr_low_sims,
          pcr_low_sim_prob: selfplay.pcr_low_sim_prob,
          c_puct: selfplay.c_puct ?? search.c_puct,
          dirichlet_alpha: selfplay.dirichlet_alpha,
          dirichlet_fraction: selfplay.dirichlet_fraction,
          pair_strategy: selfplay.pair_strategy ?? search.pair_strategy
        }} />
        <KeyValue title="Epoch Budget" rows={{
          max_game_moves: selfplay.max_game_moves,
          states_per_epoch: selfplay.states_per_epoch,
          games_per_epoch: selfplay.games_per_epoch,
          terminal_samples: selfplay.terminal_samples
        }} />
      </div>
    </Panel>
  );
}
