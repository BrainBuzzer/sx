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
