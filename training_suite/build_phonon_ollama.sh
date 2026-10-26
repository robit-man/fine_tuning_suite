#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${PHONON_OLLAMA_SOURCE:-"$repo_dir/training_suite/vendor/ollama-phonon-hybrid-source"}
output_dir=${PHONON_OLLAMA_OUTPUT:-"$repo_dir/training_suite/outputs/phonon-hybrid/ollama-runtime/bin"}
patch_file="$repo_dir/training_suite/patches/ollama-0.40.11-phonon-hybrid.patch"

if [[ ! -d "$source_dir/.git" && ! -f "$source_dir/.git" ]]; then
  mkdir -p "$(dirname -- "$source_dir")"
  git clone --depth 1 --branch v0.40.11 https://github.com/ollama/ollama.git "$source_dir"
fi
if git -C "$source_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
  printf 'Phonon hybrid patch is already applied.\n'
else
  git -C "$source_dir" apply --check "$patch_file"
  git -C "$source_dir" apply "$patch_file"
fi
mkdir -p "$output_dir"
(
  cd "$source_dir"
  GOMAXPROCS=${GOMAXPROCS:-4} go build -p 4 \
    -ldflags '-X github.com/ollama/ollama/version.Version=0.40.11+phonon-hybrid.1' \
    -o "$output_dir/ollama" .
)
printf 'Built %s\n' "$output_dir/ollama"
