import type { AnyRow } from "../../../api/client";
import { cell, labelFor } from "../../../components/format";

export function KeyValue({ title, rows }: { title: string; rows: AnyRow }) {
  const entries = Object.entries(rows).filter(([, value]) => value !== undefined && value !== null && value !== "");
  return (
    <div className="kvPanel">
      <h3>{title}</h3>
      <dl>
        {entries.map(([key, value]) => (
          <div key={key}>
            <dt>{labelFor(key)}</dt>
            <dd>{cell(value, key)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
