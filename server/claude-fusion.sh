#!/usr/bin/env bash
# Launch Claude Code against the local fusion-35b x2 model.
#
#   Claude Code (Anthropic API) -> CCR :3456 (Anthropic<->OpenAI translate) -> fusion proxy :9300
#                                -> fusion backend :9301 (35B, KV-checkpoints, --jinja)
#
# - Isolated CLAUDE_CONFIG_DIR (~/.claude-fusion) so your normal ~/.claude is untouched.
# - KV cache: --exclude-dynamic-system-prompt-sections keeps Claude Code's huge system prompt a
#   STABLE prefix (cwd/git/env moved to first user msg), so the backend's context checkpoint is
#   reused every turn instead of re-prefilling ~35k tokens.
# - Built-in WebSearch/WebFetch (Anthropic server tools, don't work on a custom backend) are
#   disabled and replaced by the local `search` MCP (ddgs + your own Chrome for page rendering).
# - Your existing CCR config is backed up before being rewritten.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PY="${PYTHON:-python3}"                                   # needs: requests, ddgs, trafilatura
CCR_CFG="$HOME/.claude-code-router/config.json"
CFG_DIR="$HOME/.claude-fusion"; mkdir -p "$CFG_DIR"
MCP_CFG="$CFG_DIR/mcp.json"

# 1) fusion API up?
if ! curl -sf http://127.0.0.1:9300/health >/dev/null 2>&1; then
  echo "[claude-fusion] starting fusion API..."; "$DIR/fusion_api.sh" start
fi

# 2) CCR config -> route everything to the fusion proxy (back up existing first)
if [ -f "$CCR_CFG" ]; then cp "$CCR_CFG" "$CCR_CFG.bak-$(date +%Y%m%d%H%M%S)"; fi
cat > "$CCR_CFG" <<'JSON'
{
  "LOG": true,
  "HOST": "127.0.0.1",
  "PORT": 3456,
  "APIKEY": "ccr-local-key",
  "Providers": [
    {
      "name": "fusion",
      "api_base_url": "http://127.0.0.1:9300/v1/chat/completions",
      "api_key": "no-key-required",
      "models": ["qwopus-35b-fusion"]
    },
    {
      "name": "fast",
      "api_base_url": "http://127.0.0.1:9300/v1/chat/completions",
      "api_key": "no-key-required",
      "models": ["qwopus-35b-fast"]
    }
  ],
  "Router": {
    "default": "fusion,qwopus-35b-fusion",
    "background": "fast,qwopus-35b-fast",
    "think": "fusion,qwopus-35b-fusion",
    "longContext": "fusion,qwopus-35b-fusion",
    "webSearch": "fusion,qwopus-35b-fusion"
  }
}
JSON
echo "[claude-fusion] restarting CCR..."; ccr restart >/dev/null 2>&1 || ccr start >/dev/null 2>&1 || true
for i in $(seq 1 30); do curl -sf http://127.0.0.1:3456/ >/dev/null 2>&1 && break; sleep 1; done

# 3) local search/browse MCP (replaces Anthropic's built-in WebSearch/WebFetch)
cat > "$MCP_CFG" <<JSON
{"mcpServers": {"search": {"command": "$PY", "args": ["$DIR/search_mcp.py"]}}}
JSON

# 4) isolated Claude Code env -> CCR -> fusion
unset ANTHROPIC_API_KEY 2>/dev/null || true
export ANTHROPIC_BASE_URL="http://127.0.0.1:3456"
export ANTHROPIC_AUTH_TOKEN="ccr-local-key"
export ANTHROPIC_MODEL="qwopus-35b-fusion"
export ANTHROPIC_DEFAULT_SONNET_MODEL="qwopus-35b-fusion"
export ANTHROPIC_DEFAULT_OPUS_MODEL="qwopus-35b-fusion"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="qwopus-35b-fusion"
export CLAUDE_CODE_SUBAGENT_MODEL="qwopus-35b-fusion"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000
export CLAUDE_CONFIG_DIR="$CFG_DIR"

echo "[claude-fusion] -> CCR :3456 -> fusion :9300  | config=$CFG_DIR"
exec claude \
  --exclude-dynamic-system-prompt-sections \
  --settings '{"includeGitInstructions":false}' \
  --disallowedTools "WebSearch" "WebFetch" \
  --mcp-config "$MCP_CFG" --strict-mcp-config \
  --dangerously-skip-permissions "$@"
