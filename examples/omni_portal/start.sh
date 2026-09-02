#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
SCRIPT_PATH="$SCRIPT_DIR/start.sh"
PYTHON_BIN=${OMNI_PYTHON_BIN:-$REPO_ROOT/.venv-omni/bin/python}
MODEL=${OMNI_MODEL:-robit/qwen3.8-27b-e03-obliterated-omni:q4km}
LANGUAGE_MODEL=${OMNI_LANGUAGE_MODEL:-robit/qwen3.8-27b-obliterated-e03:27b}
RUNTIME_ROOT=${OMNI_PORTAL_RUNTIME_ROOT:-$REPO_ROOT/training_suite/outputs/omni_portal_runtime}
CACHE_DIR=${OMNI_COMPONENT_CACHE:-$RUNTIME_ROOT/components}
STATE_DIR=${OMNI_PORTAL_STATE_DIR:-$RUNTIME_ROOT/state}
LOG_DIR=${OMNI_PORTAL_LOG_DIR:-$RUNTIME_ROOT/logs}
SESSION_LOG_DIR=${OMNI_PORTAL_SESSION_LOG_DIR:-$RUNTIME_ROOT/session-logs}

COMP_PORT=${OMNI_COMPREHENSION_PORT:-8901}
TTS_PORT=${OMNI_TTS_PORT:-8892}
ADAPTER_PORT=${OMNI_ADAPTER_PORT:-8910}
PORTAL_PORT=${OMNI_PORTAL_PORT:-8920}
METRICS_PORT=${OMNI_CLOUDFLARED_METRICS_PORT:-49312}
COMP_VRAM_MIB=${OMNI_COMPREHENSION_VRAM_MIB:-45000}
COMP_CONTEXT_TOKENS=${OMNI_COMPREHENSION_CONTEXT_TOKENS:-65536}

SUPERVISOR_PID_FILE="$STATE_DIR/supervisor.pid"
ACCESS_URL_FILE="$STATE_DIR/access-url.txt"
TOKEN_FILE="$STATE_DIR/access-token.txt"
GPU_FILE="$STATE_DIR/comprehension-gpu.txt"
GPU_LEASE_FILE="$STATE_DIR/gpu-lease-token.txt"
TTS_ACTIVE_PID_FILE="$STATE_DIR/tts-active.pid"
SUPERVISOR_LOG="$LOG_DIR/supervisor.log"
CACHE_MARKER="$CACHE_DIR/.robit-omni-portal-cache"

COMP_PID=""
TTS_PID=""
ADAPTER_PID=""
PORTAL_PID=""
TUNNEL_PID=""
HEARTBEAT_PID=""
SMOKE_PID=""
LEASE_TOKEN=""
CLEANING_UP=0

log() {
  printf '[omni-portal] %s\n' "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command is missing: $1"
}

verify_language_model() {
  local logical_sources
  local language_sources
  logical_sources=$(ollama show "$MODEL" --modelfile | awk '$1 == "FROM" {print $2}')
  language_sources=$(ollama show "$LANGUAGE_MODEL" --modelfile | awk '$1 == "FROM" {print $2}')
  [[ -n "$logical_sources" && "$logical_sources" == "$language_sources" ]] \
    || die "OMNI_LANGUAGE_MODEL does not reference the combined tag's base/projector blobs"
  log "verified language backend shares the combined tag's standard blobs"
}

wait_http() {
  local url=$1
  local label=$2
  local timeout=${3:-900}
  local child_pid=${4:-}
  local started=$SECONDS
  while (( SECONDS - started < timeout )); do
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      log "$label is ready"
      return 0
    fi
    if [[ -n "$child_pid" ]] && ! kill -0 "$child_pid" 2>/dev/null; then
      die "$label process exited before readiness"
    fi
    sleep 1
  done
  die "$label did not become ready within ${timeout}s"
}

wait_http_child() {
  local url=$1
  local timeout=$2
  local child_pid=$3
  local started=$SECONDS
  while (( SECONDS - started < timeout )); do
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$child_pid" 2>/dev/null; then
      return 1
    fi
    sleep 1
  done
  return 1
}

terminate_child() {
  local pid=$1
  local label=$2
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    log "stopping $label (pid $pid)"
    kill -TERM "$pid" 2>/dev/null || true
    local started=$SECONDS
    while kill -0 "$pid" 2>/dev/null && (( SECONDS - started < 45 )); do
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      log "$label did not stop gracefully; killing exact pid $pid"
      kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  fi
}

terminate_process_group() {
  local pid=$1
  local label=$2
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    log "stopping $label process group (leader $pid)"
    kill -TERM -- "-$pid" 2>/dev/null || true
    local started=$SECONDS
    while kill -0 "$pid" 2>/dev/null && (( SECONDS - started < 15 )); do
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      log "$label did not stop gracefully; killing process group $pid"
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  fi
}

terminate_pid_file() {
  local file=$1
  local label=$2
  if [[ ! -s "$file" ]]; then
    return
  fi
  local pid
  pid=$(sed -n '1p' "$file")
  if [[ "$pid" =~ ^[0-9]+$ ]]; then
    terminate_child "$pid" "$label"
  fi
  if [[ -f "$file" ]]; then unlink "$file"; fi
}

cleanup_cache() {
  if [[ ${OMNI_KEEP_CACHE:-0} == 1 ]] || [[ ! -f "$CACHE_MARKER" ]]; then
    return
  fi
  log "removing portal-owned disposable component cache"
  local file
  for file in \
    "$CACHE_DIR/comprehension-model.gguf" \
    "$CACHE_DIR/comprehension-projector.gguf" \
    "$CACHE_DIR/tts-model.gguf" \
    "$CACHE_DIR/tts-projector.gguf" \
    "$CACHE_MARKER"; do
    if [[ -f "$file" ]]; then
      unlink "$file"
    fi
  done
  rmdir "$CACHE_DIR" 2>/dev/null || true
}

cleanup_session_logs() {
  [[ -d "$SESSION_LOG_DIR" ]] || return
  local file
  shopt -s nullglob
  for file in "$SESSION_LOG_DIR"/*.json "$SESSION_LOG_DIR"/*.tmp; do
    if [[ -f "$file" ]]; then
      unlink "$file"
    fi
  done
  shopt -u nullglob
  rmdir "$SESSION_LOG_DIR" 2>/dev/null || true
}

cleanup() {
  if (( CLEANING_UP )); then
    return
  fi
  CLEANING_UP=1
  trap - EXIT INT TERM
  terminate_child "$TUNNEL_PID" "Cloudflare tunnel"
  terminate_child "$SMOKE_PID" "pre-tunnel smoke gate"
  terminate_child "$PORTAL_PID" "portal"
  terminate_child "$ADAPTER_PID" "adapter"
  terminate_child "$TTS_PID" "TTS wrapper"
  terminate_pid_file "$TTS_ACTIVE_PID_FILE" "TTS CUDA worker"
  terminate_child "$COMP_PID" "broker-scoped comprehension worker"
  terminate_process_group "$HEARTBEAT_PID" "GPU lease heartbeat"
  if [[ -n "$LEASE_TOKEN" ]]; then
    log "releasing scoped GPU lease"
    docker gpu release "$LEASE_TOKEN" >/dev/null 2>&1 || true
    LEASE_TOKEN=""
  fi
  cleanup_session_logs
  cleanup_cache
  if [[ -f "$SUPERVISOR_PID_FILE" ]]; then unlink "$SUPERVISOR_PID_FILE"; fi
  if [[ -f "$ACCESS_URL_FILE" ]]; then unlink "$ACCESS_URL_FILE"; fi
  if [[ -f "$TOKEN_FILE" ]]; then unlink "$TOKEN_FILE"; fi
  if [[ -f "$GPU_FILE" ]]; then unlink "$GPU_FILE"; fi
  if [[ -f "$GPU_LEASE_FILE" ]]; then unlink "$GPU_LEASE_FILE"; fi
  if [[ -f "$TTS_ACTIVE_PID_FILE" ]]; then unlink "$TTS_ACTIVE_PID_FILE"; fi
  log "shutdown complete"
}

service_is_running() {
  [[ -f "$SUPERVISOR_PID_FILE" ]] || return 1
  local pid
  pid=$(sed -n '1p' "$SUPERVISOR_PID_FILE")
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

show_status() {
  if service_is_running; then
    local pid
    pid=$(sed -n '1p' "$SUPERVISOR_PID_FILE")
    log "running (pid $pid)"
    if [[ -s "$ACCESS_URL_FILE" ]]; then sed -n '1p' "$ACCESS_URL_FILE"; fi
    return 0
  fi
  log "not running"
  return 1
}

stop_daemon() {
  if ! service_is_running; then
    log "portal is not running"
    return 0
  fi
  local pid
  pid=$(sed -n '1p' "$SUPERVISOR_PID_FILE")
  log "requesting supervisor shutdown (pid $pid)"
  kill -TERM "$pid"
  local started=$SECONDS
  while kill -0 "$pid" 2>/dev/null && (( SECONDS - started < 60 )); do
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    die "supervisor did not stop within 60s; inspect $SUPERVISOR_LOG"
  fi
  log "portal stopped"
}

start_daemon() {
  mkdir -p "$STATE_DIR" "$LOG_DIR"
  chmod 700 "$RUNTIME_ROOT" "$STATE_DIR" "$LOG_DIR" 2>/dev/null || true
  if service_is_running; then
    show_status
    return
  fi
  if [[ -f "$ACCESS_URL_FILE" ]]; then unlink "$ACCESS_URL_FILE"; fi
  log "starting detached supervisor; progress log: $SUPERVISOR_LOG"
  nohup setsid "$SCRIPT_PATH" --foreground >"$SUPERVISOR_LOG" 2>&1 </dev/null &
  local pid=$!
  printf '%s\n' "$pid" >"$SUPERVISOR_PID_FILE"
  local started=$SECONDS
  while (( SECONDS - started < 1800 )); do
    if [[ -s "$ACCESS_URL_FILE" ]]; then
      log "deployment ready"
      sed -n '1p' "$ACCESS_URL_FILE"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      log "supervisor exited; recent log follows" >&2
      tail -80 "$SUPERVISOR_LOG" >&2 || true
      return 1
    fi
    sleep 1
  done
  die "deployment did not publish a URL within 30 minutes; inspect $SUPERVISOR_LOG"
}

check_port_available() {
  local port=$1
  local label=$2
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$port$"; then
    die "$label port $port is already in use"
  fi
}

prepare_cache() {
  local required=(
    "$CACHE_DIR/comprehension-model.gguf"
    "$CACHE_DIR/comprehension-projector.gguf"
    "$CACHE_DIR/tts-model.gguf"
    "$CACHE_DIR/tts-projector.gguf"
  )
  local missing=0
  local file
  for file in "${required[@]}"; do
    [[ -f "$file" ]] || missing=1
  done
  if (( ! missing )); then
    log "reusing complete component cache: $CACHE_DIR"
    return
  fi
  local free_kib
  free_kib=$(df -Pk "$RUNTIME_ROOT" | awk 'NR==2 {print $4}')
  (( free_kib >= 30 * 1024 * 1024 )) || die "at least 30 GiB free is required to prepare the component cache"
  mkdir -p "$CACHE_DIR"
  log "materializing four verified runtime views from the installed Ollama sidecar"
  "$PYTHON_BIN" -m training_suite omni-prepare "$MODEL" --out "$CACHE_DIR" --overwrite
  : >"$CACHE_MARKER"
}

choose_gpu() {
  log "discovering broker-approved CUDA placement" >&2
  local discovery
  discovery=$(docker gpu discover)
  local broker_status
  broker_status=$(docker gpu status)
  local selected=${OMNI_COMPREHENSION_GPU_UUID:-}
  if [[ -n "$selected" ]]; then
    jq -e --arg uuid "$selected" '.gpus[] | select(.uuid == $uuid and .selected_for_ollama == true)' <<<"$discovery" >/dev/null \
      || die "requested comprehension GPU is not broker-selected: $selected"
  else
    selected=$(jq -nr \
      --argjson discovery "$discovery" \
      --argjson status "$broker_status" \
      --argjson required "$COMP_VRAM_MIB" '
        [$status.leases[]?
          | select(.state == "pending" or .state == "active" or .state == "revoking")
          | .gpu_uuids[]] as $claimed
        | [$discovery.gpus[]
            | select(.selected_for_ollama == true and .total_mib >= $required)
            | .uuid as $uuid
            | select(($claimed | index($uuid)) == null)]
        | sort_by(.free_mib) | reverse | .[0].uuid // empty
      ')
  fi
  [[ -n "$selected" ]] || die "no broker-selected GPU can host the comprehension reservation"
  printf '%s\n' "$selected" >"$GPU_FILE"
  printf '%s' "$selected"
}

run_foreground() {
  cd "$REPO_ROOT"
  mkdir -p "$RUNTIME_ROOT" "$STATE_DIR" "$LOG_DIR"
  chmod 700 "$RUNTIME_ROOT" "$STATE_DIR" "$LOG_DIR" 2>/dev/null || true
  printf '%s\n' "$$" >"$SUPERVISOR_PID_FILE"
  trap cleanup EXIT
  trap 'cleanup; exit 0' INT TERM

  require_command curl
  require_command cloudflared
  require_command docker
  require_command ffmpeg
  require_command jq
  require_command nvidia-smi
  require_command openssl
  require_command setsid
  require_command ss
  [[ -x "$PYTHON_BIN" ]] || die "suite Python is missing: $PYTHON_BIN"
  [[ -x "$REPO_ROOT/training_suite/vendor/llama.cpp/build/bin/llama-server" ]] || die "llama-server is not built"
  [[ -x "$REPO_ROOT/training_suite/vendor/llama.cpp/build/bin/llama-tts" ]] || die "llama-tts is not built"

  check_port_available "$COMP_PORT" "comprehension"
  check_port_available "$TTS_PORT" "TTS"
  check_port_available "$ADAPTER_PORT" "adapter"
  check_port_available "$PORTAL_PORT" "portal"
  check_port_available "$METRICS_PORT" "cloudflared metrics"

  if ! ollama show "$MODEL" >/dev/null 2>&1; then
    log "model is not installed; pulling $MODEL"
    ollama pull "$MODEL"
  fi
  ollama show "$LANGUAGE_MODEL" >/dev/null 2>&1 \
    || die "language backend model is not installed: $LANGUAGE_MODEL"
  verify_language_model
  "$PYTHON_BIN" -m training_suite omni-resolve "$MODEL" >/dev/null
  prepare_cache

  local gpu_uuid
  gpu_uuid=$(choose_gpu)
  log "acquiring ${COMP_VRAM_MIB} MiB scoped CUDA lease on $gpu_uuid"
  LEASE_TOKEN=$(docker gpu acquire \
    --owner robit-omni-phone-portal \
    --vram-mib "$COMP_VRAM_MIB" \
    --gpu "$gpu_uuid" \
    --token-only)
  [[ -n "$LEASE_TOKEN" ]] || die "GPU broker returned an empty lease token"
  printf '%s\n' "$LEASE_TOKEN" >"$GPU_LEASE_FILE"
  setsid docker gpu heartbeat "$LEASE_TOKEN" --watch \
    >"$LOG_DIR/gpu-heartbeat.log" 2>&1 &
  HEARTBEAT_PID=$!

  log "starting CUDA comprehension on exact reservation $gpu_uuid"
  CUDA_VISIBLE_DEVICES="$gpu_uuid" \
  HIP_VISIBLE_DEVICES=-1 \
  ROCR_VISIBLE_DEVICES=-1 \
  "$REPO_ROOT/training_suite/vendor/llama.cpp/build/bin/llama-server" \
    -m "$CACHE_DIR/comprehension-model.gguf" \
    --mmproj "$CACHE_DIR/comprehension-projector.gguf" \
    --host 127.0.0.1 --port "$COMP_PORT" \
    --jinja -ngl 99 -c "$COMP_CONTEXT_TOKENS" \
    >"$LOG_DIR/comprehension.log" 2>&1 &
  COMP_PID=$!
  wait_http "http://127.0.0.1:$COMP_PORT/health" "CUDA comprehension" 1200 "$COMP_PID"
  local residency_started=$SECONDS
  while ! nvidia-smi \
    --query-compute-apps=pid,gpu_uuid,used_memory \
    --format=csv,noheader,nounits \
    | awk -F, -v pid="$COMP_PID" -v uuid="$gpu_uuid" '
        { gsub(/^ +| +$/, "", $1); gsub(/^ +| +$/, "", $2) }
        $1 == pid && $2 == uuid { found=1 }
        END { exit !found }
      '; do
    kill -0 "$COMP_PID" 2>/dev/null \
      || die "comprehension exited before CUDA residency verification"
    (( SECONDS - residency_started < 120 )) \
      || die "comprehension did not become resident on reserved GPU $gpu_uuid"
    sleep 1
  done
  docker gpu ready "$LEASE_TOKEN" >/dev/null
  log "CUDA comprehension is resident and broker-ready"

  log "starting broker-coordinated CUDA TTS wrapper"
  CUDA_VISIBLE_DEVICES="$gpu_uuid" \
  HIP_VISIBLE_DEVICES=-1 \
  ROCR_VISIBLE_DEVICES=-1 \
  OLLAMA_UNIFY_GPU_LEASE="$LEASE_TOKEN" \
  OMNI_OLLAMA_MODEL="$MODEL" \
  OMNI_COMPONENT_CACHE="$CACHE_DIR" \
  OMNI_TTS_MODEL_GGUF="$CACHE_DIR/tts-model.gguf" \
  OMNI_TTS_PROJECTOR_GGUF="$CACHE_DIR/tts-projector.gguf" \
  LLAMA_TTS_BIN="$REPO_ROOT/training_suite/vendor/llama.cpp/build/bin/llama-tts" \
  OMNI_TTS_GPU_LAYERS=-1 \
  OMNI_TTS_STREAM_FRAMES="${OMNI_TTS_STREAM_FRAMES:-2}" \
  OMNI_TTS_PERSISTENT="${OMNI_TTS_PERSISTENT:-1}" \
  OMNI_TTS_WARM_SPEAKER_FILE="${OMNI_TTS_WARM_SPEAKER_FILE:-$SCRIPT_DIR/voices/female_voice.wav}" \
  OMNI_TTS_GPU_UUID="$gpu_uuid" \
  OMNI_TTS_ACTIVE_PID_FILE="$TTS_ACTIVE_PID_FILE" \
  OMNI_TTS_HOST=127.0.0.1 \
  OMNI_TTS_PORT="$TTS_PORT" \
  "$PYTHON_BIN" "$REPO_ROOT/examples/omni_adapter/tts_server.py" \
    >"$LOG_DIR/tts.log" 2>&1 &
  TTS_PID=$!
  wait_http "http://127.0.0.1:$TTS_PORT/healthz" "TTS wrapper" 60 "$TTS_PID"

  log "starting unified Omni adapter"
  OMNI_COMPREHENSION_URL="http://127.0.0.1:$COMP_PORT/v1/chat/completions" \
  OMNI_COMPREHENSION_MODEL=local-qwen3-omni \
  OMNI_COMPREHENSION_CONTEXT_TOKENS="$COMP_CONTEXT_TOKENS" \
  OMNI_LANGUAGE_URL=http://127.0.0.1:11434 \
  OMNI_LANGUAGE_MODEL="$LANGUAGE_MODEL" \
  OMNI_TTS_URL="http://127.0.0.1:$TTS_PORT/synthesize" \
  OMNI_ADAPTER_HOST=127.0.0.1 \
  OMNI_ADAPTER_PORT="$ADAPTER_PORT" \
  "$PYTHON_BIN" "$REPO_ROOT/examples/omni_adapter/server.py" \
    >"$LOG_DIR/adapter.log" 2>&1 &
  ADAPTER_PID=$!
  wait_http "http://127.0.0.1:$ADAPTER_PORT/healthz" "Omni adapter" 60 "$ADAPTER_PID"

  local portal_token=${OMNI_PORTAL_TOKEN:-}
  if [[ -z "$portal_token" ]]; then
    portal_token=$(openssl rand -base64 36 | tr -d '=+/\n' | cut -c1-40)
  fi
  [[ ${#portal_token} -ge 24 ]] || die "portal token must contain at least 24 characters"
  printf '%s\n' "$portal_token" >"$TOKEN_FILE"

  log "starting authenticated phone portal"
  OMNI_MODEL="$MODEL" \
  OMNI_PORTAL_TOKEN="$portal_token" \
  OMNI_ADAPTER_URL="http://127.0.0.1:$ADAPTER_PORT/api/chat" \
  OMNI_ADAPTER_HEALTH_URL="http://127.0.0.1:$ADAPTER_PORT/healthz" \
  OMNI_COMPREHENSION_HEALTH_URL="http://127.0.0.1:$COMP_PORT/health" \
  OMNI_TTS_HEALTH_URL="http://127.0.0.1:$TTS_PORT/healthz" \
  OMNI_PORTAL_SESSION_LOG_DIR="$SESSION_LOG_DIR" \
  OMNI_PORTAL_HOST=127.0.0.1 \
  OMNI_PORTAL_PORT="$PORTAL_PORT" \
  "$PYTHON_BIN" "$REPO_ROOT/examples/omni_portal/app.py" \
    >"$LOG_DIR/portal.log" 2>&1 &
  PORTAL_PID=$!
  wait_http "http://127.0.0.1:$PORTAL_PORT/healthz" "phone portal" 60 "$PORTAL_PID"

  log "running local pre-tunnel smoke gate"
  "$PYTHON_BIN" "$REPO_ROOT/examples/omni_portal/smoke.py" \
    --endpoint "http://127.0.0.1:$PORTAL_PORT" \
    --token-file "$TOKEN_FILE" \
    --model "$MODEL" --text --tts --stream \
    >"$LOG_DIR/pre-tunnel-smoke.log" 2>&1 &
  SMOKE_PID=$!
  wait "$SMOKE_PID"
  SMOKE_PID=""

  log "starting Cloudflare quick tunnel"
  cloudflared tunnel \
    --no-autoupdate \
    --url "http://127.0.0.1:$PORTAL_PORT" \
    --metrics "127.0.0.1:$METRICS_PORT" \
    --loglevel info \
    >"$LOG_DIR/cloudflared.log" 2>&1 &
  TUNNEL_PID=$!

  local public_url=""
  local started=$SECONDS
  while (( SECONDS - started < 120 )); do
    public_url=$(grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' "$LOG_DIR/cloudflared.log" | tail -1 || true)
    if [[ -n "$public_url" ]]; then break; fi
    kill -0 "$TUNNEL_PID" 2>/dev/null || die "cloudflared exited before publishing a URL"
    sleep 1
  done
  [[ -n "$public_url" ]] || die "cloudflared did not publish a quick-tunnel URL"

  local access_url="${public_url}/#access=${portal_token}"
  printf '%s\n' "$access_url" >"$ACCESS_URL_FILE"
  log "public portal ready"
  printf '%s\n' "$access_url"

  while true; do
    local pid label
    for pid in "$COMP_PID" "$TTS_PID" "$ADAPTER_PID" "$PORTAL_PID" "$TUNNEL_PID"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        label="deployment child"
        die "$label pid $pid exited; inspect $LOG_DIR"
      fi
    done
    sleep 5
  done
}

case ${1:---daemon} in
  --daemon)
    start_daemon
    ;;
  --foreground)
    run_foreground
    ;;
  --stop)
    stop_daemon
    ;;
  --status)
    show_status
    ;;
  *)
    die "usage: $0 [--daemon|--foreground|--status|--stop]"
    ;;
esac
