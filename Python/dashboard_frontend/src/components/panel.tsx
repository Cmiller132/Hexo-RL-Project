import { Activity } from "lucide-react";
import type { ReactNode } from "react";

export function Panel({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <section className="panel">
      <div className="panelTitle"><Activity size={14} /> {title}</div>
      {hint && <p className="panelHint">{hint}</p>}
      {children}
    </section>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <div className="emptyState">{children}</div>;
}
