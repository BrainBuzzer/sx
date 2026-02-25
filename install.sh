#!/usr/bin/env sh
set -eu

if command -v brew >/dev/null 2>&1; then
  echo "Installing sx via Homebrew..."
  brew tap BrainBuzzer/semantic-explorer
  brew install sx
  exit 0
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "Homebrew is not available and Rust/Cargo was not found."
  echo "Install Homebrew or Rust first, or run:"
  echo "  curl -fsSL https://raw.githubusercontent.com/BrainBuzzer/semantic-explorer/main/README.md"
  exit 1
fi

echo "Homebrew is not available. Falling back to cargo install."
cargo install --git https://github.com/BrainBuzzer/semantic-explorer.git --locked --force
echo "Installed sx using cargo."

