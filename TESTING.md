# Testing

## Quick start
- Run everything: `cargo test`
- Run CLI end-to-end tests only: `cargo test --test cli_integration`
- Run trace end-to-end tests only: `cargo test --test trace`

## Test layers
1. **Unit tests** (in `src/**`)
   - Pure logic: path normalization, query building, scanner filtering, chunking.
2. **CLI end-to-end tests** (in `tests/`)
   - Build tiny temporary repos (with a `.git/` dir) and run the `sx` binary via `assert_cmd`.
   - Prefer `--json` outputs in assertions so tests stay stable as formatting evolves.
   - Include trace-path assertions (`sx trace "<q>" --json`) for graph-specific behavior.

## Adding new end-to-end tests
Guidelines to keep tests deterministic and CI-friendly:
- Use a `tempfile::TempDir` repo fixture instead of indexing the real workspace.
- Create a `.git/` directory so root discovery behaves like a real repo.
- Avoid asserting on BM25 scores; assert on presence/absence of paths, symbols, and kinds.
- Use `sx open --dry-run` instead of launching a real editor.
- When you need isolation between indexes, pass `--db <path>` and reuse it across commands.
