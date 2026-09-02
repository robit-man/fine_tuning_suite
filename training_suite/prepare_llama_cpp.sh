#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE_DIR=${1:-$SCRIPT_DIR/vendor/llama.cpp}
PINNED_COMMIT=458681e1d5d4a29a1463c4732e03226cf384b997
PATCH_FILE=$SCRIPT_DIR/patches/llama.cpp-qwen3tts-pcm-stream.patch
PERSISTENT_PATCH_FILE=$SCRIPT_DIR/patches/llama.cpp-qwen3tts-persistent.patch
STREAM_STATE_PATCH_FILE=$SCRIPT_DIR/patches/llama.cpp-qwen3tts-stream-state.patch
BUILD_DIR=${LLAMA_CPP_BUILD_DIR:-$SOURCE_DIR/build}
BUILD_JOBS=${LLAMA_CPP_BUILD_JOBS:-$(nproc)}

cloned=0
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$SOURCE_DIR"
  cloned=1
fi

if (( cloned )); then
  git -C "$SOURCE_DIR" checkout --detach "$PINNED_COMMIT"
fi

current_commit=$(git -C "$SOURCE_DIR" rev-parse HEAD)
if [[ "$current_commit" != "$PINNED_COMMIT" ]]; then
  printf 'llama.cpp must be at pinned commit %s; found %s\n' \
    "$PINNED_COMMIT" "$current_commit" >&2
  exit 1
fi

if git -C "$SOURCE_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  printf 'Qwen3-TTS PCM stream patch is already applied\n'
elif git -C "$SOURCE_DIR" apply --check "$PATCH_FILE"; then
  git -C "$SOURCE_DIR" apply "$PATCH_FILE"
  printf 'Applied Qwen3-TTS PCM stream patch\n'
else
  printf 'Qwen3-TTS PCM stream patch does not apply cleanly\n' >&2
  exit 1
fi

if git -C "$SOURCE_DIR" apply --reverse --check "$PERSISTENT_PATCH_FILE" >/dev/null 2>&1; then
  printf 'Qwen3-TTS persistent worker patch is already applied\n'
elif git -C "$SOURCE_DIR" apply --check "$PERSISTENT_PATCH_FILE"; then
  git -C "$SOURCE_DIR" apply "$PERSISTENT_PATCH_FILE"
  printf 'Applied Qwen3-TTS persistent worker patch\n'
else
  printf 'Qwen3-TTS persistent worker patch does not apply cleanly\n' >&2
  exit 1
fi

if git -C "$SOURCE_DIR" apply --reverse --check "$STREAM_STATE_PATCH_FILE" >/dev/null 2>&1; then
  printf 'Qwen3-TTS streaming decoder state patch is already applied\n'
elif git -C "$SOURCE_DIR" apply --check "$STREAM_STATE_PATCH_FILE"; then
  git -C "$SOURCE_DIR" apply "$STREAM_STATE_PATCH_FILE"
  printf 'Applied Qwen3-TTS streaming decoder state patch\n'
else
  printf 'Qwen3-TTS streaming decoder state patch does not apply cleanly\n' >&2
  exit 1
fi

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --target llama-server llama-tts --parallel "$BUILD_JOBS"

"$BUILD_DIR/bin/llama-tts" --help 2>&1 | grep -q -- '--tts-stream-frames'
"$BUILD_DIR/bin/llama-tts" --help 2>&1 | grep -q -- '--tts-persistent'
printf 'Built patched llama-server and llama-tts in %s\n' "$BUILD_DIR/bin"
