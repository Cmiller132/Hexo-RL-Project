import type { AnyRow } from "../api/client";
import { cell, labelFor } from "./format";

export function Table({ rows, columns, onRow, selected, className = "" }: {
  rows: AnyRow[];
  columns: string[];
  onRow?: (row: AnyRow) => void;
  selected?: (row: AnyRow) => boolean;
  className?: string;
}) {
  return (
    <div className={`tableWrap ${className}`}>
      <table>
        <thead><tr>{columns.map((c) => <th key={c}>{labelFor(c)}</th>)}</tr></thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={String(row.id ?? row.trial_id ?? row.game_id ?? i)} onClick={() => onRow?.(row)} className={selected?.(row) ? "selected" : onRow ? "clickable" : ""}>
              {columns.map((c) => <td key={c}>{cell(row[c], c)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function SortButton({ id, active, onClick }: { id: string; active?: boolean; onClick: (id: string) => void }) {
  return <button className={active ? "active" : ""} onClick={() => onClick(id)}>{id.replace(/_/g, " ")}</button>;
}
