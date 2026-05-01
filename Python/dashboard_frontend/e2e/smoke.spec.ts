import { expect, test } from "@playwright/test";

const json = (body: unknown) => ({ status: 200, contentType: "application/json", body: JSON.stringify(body) });

test.beforeEach(async ({ page }) => {
  await page.route("**/api/**", async (route) => {
    const url = route.request().url();
    if (url.includes("/suite/status")) return route.fulfill(json({ enabled: true, current_stage: "seed", best_trial_id: "trial-a", current_activity: { action: "training", trial_id: "trial-a" } }));
    if (url.includes("/suite/trials/trial-a")) return route.fulfill(json({ trial_id: "trial-a", latest: {}, architecture_summary: "CNN residual trunk", config: {}, trial: {}, state: {} }));
    if (url.includes("/suite/trials")) return route.fulfill(json([{ trial_id: "trial-a", family: "cnn", architecture: "cnn", score: 1.2 }]));
    if (url.includes("/suite/family-space")) return route.fulfill(json({ families: [{ name: "cnn" }], recipes: [] }));
    if (url.includes("/suite/scheduler")) return route.fulfill(json({ current_stage: "seed", planned_stages: [], decisions: [], scheduler: {}, budget: {}, state: {} }));
    if (url.includes("/suite/runtime-sweep")) return route.fulfill(json({ probes: [{ trial_id: "trial-a", workers: 1, positions_per_sec: 42 }], selected: [] }));
    if (url.includes("/suite/events")) return route.fulfill(json([]));
    if (url.includes("/suite/best-checkpoints") || url.includes("/checkpoints")) return route.fulfill(json([]));
    if (url.includes("/health")) return route.fulfill(json({ ok: true, db_path: "/tmp/dashboard.sqlite3" }));
    if (url.includes("/runs")) return route.fulfill(json([{ run_id: "trial-a", name: "trial-a" }]));
    if (url.includes("/metrics")) return route.fulfill(json([]));
    if (url.includes("/games")) return route.fulfill(json([]));
    if (url.includes("/arena/history")) return route.fulfill(json([]));
    if (url.includes("/axis/prototypes")) return route.fulfill(json([{ id: "center", label: "Center" }]));
    if (url.includes("/axis/fixtures")) return route.fulfill(json([]));
    return route.fulfill(json({}));
  });
});

for (const [path, text] of [
  ["/suite", "Autotune Suite"],
  ["/suite/trials/trial-a/architecture", "Trial trial-a"],
  ["/suite/family-space", "Family Search Space"],
  ["/suite/scheduler", "Scheduler State"],
  ["/suite/runtime-sweep", "Runtime Sweep Scatter"]
]) {
  test(`route ${path}`, async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => msg.type() === "error" && errors.push(msg.text()));
    await page.goto(path);
    await expect(page.getByText(text)).toBeVisible();
    expect(errors).toEqual([]);
  });
}
