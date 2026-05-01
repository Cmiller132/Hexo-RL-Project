export function ConnectionBanner({ connected }: { connected: boolean }) {
  return (
    <div className={connected ? "connection ok" : "connection bad"}>
      {connected ? "Connected to dashboard API" : "Connection issue: showing cached or fallback data"}
    </div>
  );
}
