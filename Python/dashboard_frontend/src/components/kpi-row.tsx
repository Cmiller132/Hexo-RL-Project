export function KpiRow({ items }: { items: [string, unknown][] }) {
  return (
    <section className="kpis">
      {items.map(([label, value]) => (
        <div className="kpi" key={label}>
          <span>{label}</span>
          <strong>{String(value)}</strong>
        </div>
      ))}
    </section>
  );
}
