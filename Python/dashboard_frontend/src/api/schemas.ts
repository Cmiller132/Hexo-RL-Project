import { z } from "zod";

export const anyRowSchema = z.object({}).catchall(z.unknown());
export const anyRowsSchema = z.array(anyRowSchema);

export const runSchema = anyRowSchema.extend({
  run_id: z.string(),
  name: z.string().optional()
});

export const gameSchema = anyRowSchema.extend({
  game_id: z.union([z.string(), z.number()]),
  run_id: z.string().optional()
});

export const suiteStatusSchema = anyRowSchema.extend({
  current_positions_per_sec: z.number().optional(),
  best_trial_id: z.string().optional()
});

export const familySpaceSchema = anyRowSchema.extend({
  families: z.array(anyRowSchema).optional(),
  recipes: z.array(anyRowSchema).optional()
});

export function parseRows(value: unknown) {
  return anyRowsSchema.catch([]).parse(value);
}
