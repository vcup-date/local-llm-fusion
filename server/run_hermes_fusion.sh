#!/usr/bin/env bash
# Launch Hermes Agent backed by the local fusion-35b x2 API.
# Uses provider: custom + loopback base_url (the pattern Hermes trusts for local
# servers — same as ~/.hermes-ds4). provider: openrouter does NOT work: it
# hardcodes https://openrouter.ai and ignores base_url.
#
#   ./fusion_api.sh start        # bring up the fusion API first
#   ./run_hermes_fusion.sh
set -euo pipefail

BASE_URL="${FUSION_BASE_URL:-http://127.0.0.1:9300/v1}"
MODEL="qwopus-35b-fusion"
HOME_DIR="$HOME/.hermes-fusion"
mkdir -p "$HOME_DIR"

cat > "$HOME_DIR/config.yaml" <<EOF
model:
  default: ${MODEL}
  provider: custom
  base_url: ${BASE_URL}
  context_length: 262144
  max_output_tokens: 8192
providers: {}
fallback_providers: []
credential_pool_strategies: {}
toolsets:
- hermes-cli
agent:
  max_turns: 60
  gateway_timeout: 1800
  api_max_retries: 3
  tool_use_enforcement: auto
  reasoning_effort: medium
EOF

# custom loopback endpoints need no key; give the OpenAI SDK a placeholder so it doesn't complain.
cat > "$HOME_DIR/.env" <<EOF
OPENAI_API_KEY=no-key-required
EOF

if ! curl -sf "${BASE_URL%/v1}/v1/models" >/dev/null 2>&1 && ! curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
  echo "warning: fusion API at ${BASE_URL} did not respond — run ./fusion_api.sh start first" >&2
fi
if ! command -v hermes >/dev/null 2>&1; then
  echo "error: 'hermes' CLI not on PATH." >&2; exit 1
fi

echo "[fusion] Hermes -> ${BASE_URL}  model=${MODEL} provider=custom  (HERMES_HOME=$HOME_DIR)"
export HERMES_HOME="$HOME_DIR"
exec hermes "$@"
