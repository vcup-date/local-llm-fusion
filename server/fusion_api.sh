#!/bin/bash
# Fusion-35b x2 OpenAI API with a fast tier:
#   backend A  :9301  35B Q4_K_M, MTP OFF, --parallel 2, 256K/slot, q8 KV  -> FUSION tier (panelists+judge)
#   backend B  :9302  35B MTP,    MTP ON,  --parallel 1, MTP speculative   -> FAST tier (single call)
#   proxy      :9300  fusion_server.py  (model qwopus-35b-fusion=fusion, qwopus-35b-fast=single/MTP)
#
#   ./fusion_api.sh start | stop | status
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
BIN="${LLAMA_SERVER:-llama-server}"                        # llama.cpp server on PATH (or set LLAMA_SERVER)
PY="${PYTHON:-python3}"                                    # needs `requests`
MODELS="${MODELS_DIR:-$DIR/../models}"
MODEL_A="${FUSION_MODEL:-$MODELS/Qwen3-35B-A3B-Q4_K_M.gguf}"          # fusion backend: any chat GGUF
MODEL_B="${FUSION_MTP_MODEL:-$MODELS/Qwen3-35B-A3B-MTP-Q4_K_M.gguf}"  # fast backend: MTP-enabled GGUF

PROXY_PORT=9300; BACKEND_PORT=9301; FAST_PORT=9302
PARALLEL="${FUSION_PARALLEL:-2}"
CTX_PER_SLOT="${FUSION_CTX:-262144}"; CTX_TOTAL=$(( CTX_PER_SLOT * PARALLEL ))   # fusion: 256K/slot
FAST_CTX="${FUSION_FAST_CTX:-65536}"                                            # fast tier ctx
CRAM="${FUSION_CRAM:-12288}"
KVDIR="${FUSION_KVDIR:-/tmp/fusion-kv}"; mkdir -p "$KVDIR" "$KVDIR-mtp"
CACHE="--slot-save-path $KVDIR --mlock --ctx-checkpoints 128 --checkpoint-min-step 256"

wait_health() { for i in $(seq 1 600); do curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

case "${1:-start}" in
start)
  if lsof -nP -iTCP:$BACKEND_PORT -sTCP:LISTEN >/dev/null 2>&1; then echo "[A] :$BACKEND_PORT already up"; else
    echo "[A fusion] 35B parallel=$PARALLEL ${CTX_PER_SLOT}/slot q8 KV (no MTP)"
    "$BIN" -m "$MODEL_A" -ngl 99 -fa 1 -c "$CTX_TOTAL" --parallel "$PARALLEL" -ctk q8_0 -ctv q8_0 \
      -cram "$CRAM" $CACHE --jinja --host 127.0.0.1 --port $BACKEND_PORT > /tmp/fusion_backend.log 2>&1 &
    echo $! > /tmp/fusion_backend.pid
  fi
  if lsof -nP -iTCP:$FAST_PORT -sTCP:LISTEN >/dev/null 2>&1; then echo "[B] :$FAST_PORT already up"; else
    echo "[B fast/MTP] 35B parallel=1 MTP draft-mtp ${FAST_CTX} ctx q8 KV"
    "$BIN" -m "$MODEL_B" -ngl 99 -fa 1 -c "$FAST_CTX" --parallel 1 --spec-type draft-mtp --spec-draft-n-max 3 \
      -ctk q8_0 -ctv q8_0 -cram 4096 --slot-save-path "$KVDIR-mtp" --mlock --ctx-checkpoints 128 \
      --checkpoint-min-step 256 --jinja --host 127.0.0.1 --port $FAST_PORT > /tmp/fusion_fast.log 2>&1 &
    echo $! > /tmp/fusion_fast.pid
  fi
  echo "[*] waiting for backends..."
  wait_health $BACKEND_PORT && echo "  A ready" || { echo "  A FAILED"; tail -15 /tmp/fusion_backend.log; exit 1; }
  wait_health $FAST_PORT    && echo "  B ready" || { echo "  B FAILED"; tail -15 /tmp/fusion_fast.log; exit 1; }
  echo "[proxy] starting on :$PROXY_PORT"
  BACKEND="http://127.0.0.1:$BACKEND_PORT" SOLO_BACKEND="http://127.0.0.1:$FAST_PORT" PORT=$PROXY_PORT \
    "$PY" "$DIR/fusion_server.py" > /tmp/fusion_proxy.log 2>&1 &
  echo $! > /tmp/fusion_proxy.pid; sleep 2
  curl -sf http://127.0.0.1:$PROXY_PORT/health >/dev/null 2>&1 && echo "[proxy] up" || { echo "[proxy] FAILED"; tail /tmp/fusion_proxy.log; }
  echo; echo "READY -> http://127.0.0.1:$PROXY_PORT/v1  | fusion: qwopus-35b-fusion  fast: qwopus-35b-fast"
  ;;
stop)
  for n in proxy backend fast; do
    [ -f /tmp/fusion_$n.pid ] && kill "$(cat /tmp/fusion_$n.pid)" 2>/dev/null && echo "stopped $n"; rm -f /tmp/fusion_$n.pid
  done ;;
status)
  for pn in "proxy:$PROXY_PORT:/health" "A-fusion:$BACKEND_PORT:/health" "B-fast:$FAST_PORT:/health"; do
    name=${pn%%:*}; rest=${pn#*:}; port=${rest%%:*}; path=${rest#*:}
    printf "%-9s :%s " "$name" "$port"; curl -sf "http://127.0.0.1:$port$path" >/dev/null 2>&1 && echo UP || echo DOWN
  done ;;
*) echo "usage: $0 {start|stop|status}"; exit 1 ;;
esac
