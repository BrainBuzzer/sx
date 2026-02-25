#!/usr/bin/env sh
set -eu

if command -v brew >/dev/null 2>&1; then
  echo "Installing sx via Homebrew..."
  brew tap BrainBuzzer/sx
  brew install sx
  exit 0
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "Homebrew is not available and Rust/Cargo was not found."
  echo "Install Homebrew or Rust first, or use one of the alternate methods in README.md."
  exit 1
fi

echo "Homebrew is not available. Falling back to cargo install."
cargo install --git https://github.com/BrainBuzzer/sx.git --locked --force
echo "Installed sx using cargo."
