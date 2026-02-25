# Semantic Explorer

`sx` is a local-first semantic and lexical code explorer built in Rust. It indexes
code and configuration files from a project and supports fast search workflows from
the terminal or interactive TUI.

## Features

- Discover repository context and initialize a local `.sx` workspace.
- Build and update indexed content with incremental indexing.
- Run BM25 text search, semantic search, and hybrid query modes.
- Inspect referenced chunks and trace results into working directory guidance.
- Interact from CLI subcommands and default TUI mode.

## Requirements

- Rust 1.75+ for source builds.
- `SQLite` and internet access when using remote embedding providers.

## Installation

### Homebrew tap (recommended)

```sh
brew tap BrainBuzzer/sx
brew install sx
```

### One-line installer

```sh
curl -fsSL https://raw.githubusercontent.com/BrainBuzzer/sx/main/install.sh | sh
```

### Build from source

```sh
cargo install --path .
```

## Onboarding command (complete)

The onboarding entry point is:

```sh
sx onboard
```

This launches the interactive configuration flow for embedding provider/model credentials used by `sx`.

Use non-interactive mode only when needed:

```sh
sx onboard --defaults
sx onboard --skip-check
sx onboard --defaults --skip-check
```

Current available onboarding flags:

- `--defaults` to skip prompts and use current defaults.
- `--skip-check` to skip provider connectivity checks during onboarding.

For provider credentials, use the auth command directly:

```sh
sx auth zhipu --api-key <KEY>
sx auth voyage --from-env
sx auth status
```

## How this repo works right now

`sx` is a terminal-first code exploration workflow built around a project-local index.

1. Initialize workspace and database:

```sh
sx init
```

This sets up `.sx/`, resolves root config (`.sx/config.toml`), and migrates `.sx/index.sqlite`.

2. Build search index:

```sh
sx index
sx index --full
```

Creates chunked, BM25-ready local indexes from project files.

3. Search text:

```sh
sx search "<query>" --limit 20
sx search "<query>" --fts
sx search "<query>" --json
```

4. Build embedding vectors (optional but required for semantic modes):

```sh
sx embed
sx embed --full
sx embed --provider openai --model <model>
```

5. Run semantic / hybrid searches:

```sh
sx vsearch "<query>"
sx query "<query>"
sx query "<query>" --deep
```

6. Navigate findings:

```sh
sx get "<result-id>"
sx open "<result-id>"
sx cd "<query>"
sx guide "<query>"
sx trace "<query>"
```

7. Environment checks:

```sh
sx doctor
```

`sx` opens directly into the interactive TUI when no subcommand is provided:

```sh
sx
```

## Usage

```sh
# initialize workspace, index, and search
sx init
sx index
sx search "semantic search" --limit 20

# open and inspect by reference
sx get "<result-id>"
sx open "<result-id>"

# launch TUI mode
sx
```

## Repository layout

- `src/cli.rs` defines available commands and flags.
- `.sx/config.toml` is the generated project configuration.
- `.sx/index.sqlite` stores local index data.

## License

MIT License. See [LICENSE](LICENSE).
