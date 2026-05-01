import { useFamilySpace, useSuiteTrials } from "../../api/hooks";
import { Panel } from "../../components/panel";
import { Table } from "../../components/table";
import type { AnyRow } from "../../api/client";

export default function FamilySpaceRoute() {
  const { data = {} } = useFamilySpace();
  const { data: trials = [] } = useSuiteTrials();
  const families = Array.isArray(data.families) ? data.families as AnyRow[] : [];
  const recipes = Array.isArray(data.recipes) ? data.recipes as AnyRow[] : [];
  const spawned = families.map((family) => ({ ...family, spawned_trials: trials.filter((trial) => trial.family === family.name || trial.family === family.id).length }));
  return (
    <section className="suiteGrid">
      <Panel title="Family Search Space"><Table rows={spawned} columns={["id", "name", "architecture", "spawned_trials", "parameters", "choices", "ranges"]} /></Panel>
      <Panel title="Recipes"><Table rows={recipes} columns={["name", "family", "stage", "budget", "parameters"]} /></Panel>
    </section>
  );
}
