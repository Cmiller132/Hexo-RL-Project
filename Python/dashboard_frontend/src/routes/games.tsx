import { useGames } from "../api/hooks";
import { Panel } from "../components/panel";
import { Table } from "../components/table";

export default function GamesRoute({ runId, openGame }: { runId: string; openGame: (gameId: string | number, run?: string) => void }) {
  const { data = [] } = useGames(runId);
  return (
    <Panel title="Game Browser">
      <Table rows={data} columns={["game_id", "trial_id", "source", "epoch", "move_count", "terminal_reason", "truncated", "outcome", "created_at"]} onRow={(row) => openGame(String(row.game_id), String(row.run_id || runId))} />
    </Panel>
  );
}
