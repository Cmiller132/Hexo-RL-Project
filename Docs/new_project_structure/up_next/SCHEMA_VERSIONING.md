# Schema And Versioning

## Purpose

Package boundaries need explicit contracts. The current project has many places
where state rows, replay records, model targets, and runtime configs must agree;
the split should make those agreements visible and reject mismatches clearly.

## Proposal

Version the contracts that cross package boundaries:

- engine state snapshots and legal action identity;
- tactical and rules-derived payloads;
- runner events and replay records;
- model training examples and batch contracts;
- checkpoint metadata and inference adapter contracts;
- evaluation scorecards and performance profiles.

`hexo-utils` can provide small helpers for schema identifiers, compatibility
checks, validation errors, and contract examples. The owning package still owns
the meaning of the schema.

## Simplification Guardrails

Use explicit fail-fast checks instead of compatibility facades in normal
runtime. Keep migrations as offline tooling when needed.

Do not create a central mega-schema that every model must use. Shared contracts
should describe only what actually crosses package boundaries.
