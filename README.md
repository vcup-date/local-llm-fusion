# local-llm-fusion

Local "fusion" inference for llama.cpp models. Run a prompt through **N independent panelists plus
a judge** (the OpenRouter *Fusion* / "fusion-fable" pattern, reproduced locally), exposed as a single
**OpenAI-compatible API** so any client (Hermes, Claude Code via a router, your own scripts) can use it.

Inspired by OpenRouter's Fusion effort: combining multiple models so a panel can reach and even
exceed a single frontier model like Claude Fable 5. This brings that idea to local, open models you
run yourself, and measures how far it actually gets.

Includes the benchmark harness and results behind it: speed (MTP, dense, MoE), and quality on the
**DRACO** deep-research benchmark across fusion configs, model sizes, retrieval, and prompts.

> Tested on an Apple M5 Max (128 GB) with Qwen3-class 35B-A3B / 27B GGUFs and a DeepSeek-V4-Flash
> build. Model files are **not** included; point the scripts at your own GGUFs.

## TL;DR findings

**Speed (35B-A3B MoE, 4-bit, Metal).**
- **MTP (multi-token prediction) is the only method that speeds up *generation*.** It goes from
  ~95 to ~180 tok/s (1.9×) at temp 0, with prefill unaffected. It's sensitive to temperature
  (acceptance drops by ~0.7) and to `--parallel` (gives ~0 gain when batched), so it's best for
  single-stream, low-temperature decoding.
- KV-cache *compression* (TurboQuant turbo3/4) is a **memory** win (3.8 to 4.9× smaller KV), not a
  speed win (roughly speed-neutral after fixes).

**Fusion quality (DRACO, Opus-judged, % of rubric).**

| config | no web | + web (RAG) | time/turn |
|---|---:|---:|---:|
| solo-35b | 36.8 | 43.3 | ~15s |
| solo-27b | 38.3 | 44.0 | ~60s |
| solo-ds4 (DeepSeek-V4-Flash) | n/a | ~45.0 | ~70s |
| **fusion-35b×2** | **39.8** | **46.7** | ~45s |
| fusion-35b+27b | n/a | 45.7 | ~105s |
| fusion-27b×2 (budget panel) | n/a | 43.7 | ~200s |

- **Synthesis helps a little, retrieval helps a lot.** Same-model fusion (35b×2) beats solo by
  about +3, and web retrieval adds about +6 to +7 on top. None reach the deep-research *system* tier
  (Perplexity DR ~70.5, Gemini DR 59.0, o3 52.1); that gap is **retrieval/agentic-loop quality**,
  not the fusion method.
- **The judge matters more than the panel.** A strong judge (35B) beats a weak one (27B) for the same
  panel, and swapping a panelist barely moves the score.
- **Budget panel ≈ frontier solo, but the local economics flip.** 27B×2 matches solo-35B on quality
  yet runs ~13× slower (one GPU runs the weak model 3× sequentially with no per-token saving).
- **Judge prompt and persona barely change the score.** A leaked frontier "persona" prompt and
  elaborate judge prompts change *style* (citations, hedging) but not the number; on convergent
  inputs the outputs were byte-identical.
- **Fusion amplifies the panelists.** With facts in context it reconciles them and catches confident
  hallucinations; without retrieval it can also compound them. Garbage in, garbage out.

Full write-ups: [`docs/DRACO_RESULTS.md`](docs/DRACO_RESULTS.md),
[`docs/MODEL-SPEED-COMPARISON.md`](docs/MODEL-SPEED-COMPARISON.md),
[`docs/mtp-benchmark.md`](docs/mtp-benchmark.md).

## The server (`server/`)

`fusion_server.py` is an OpenAI-compatible proxy. **Every** `/v1/chat/completions` request runs the
same path: 2 panelists (temp 0.7, blind to each other), then a judge that cross-examines and emits
the single best result. Tools are first-class: both panelists and the judge get them, and the judge's
native output (a **tool call** or a **text answer**) is relayed verbatim. Streaming and non-streaming
are both supported.

Two model IDs / tiers, so an agent's main vs. background work can differ:
- `qwopus-35b-fusion`: full fusion (panelists plus judge). Quality.
- `qwopus-35b-fast`: single call on an **MTP** backend, no fusion. Speed, for haiku/background tasks.

```sh
# point these at your GGUFs (any chat model; the fast tier wants an MTP-enabled GGUF)
export LLAMA_SERVER=/path/to/llama-server          # llama.cpp server
export FUSION_MODEL=/path/to/model.gguf            # fusion backend
export FUSION_MTP_MODEL=/path/to/model-mtp.gguf    # fast backend (MTP)

server/fusion_api.sh start      # backend A (fusion, parallel-2) + B (MTP) + proxy on :9300
server/fusion_api.sh status
server/fusion_api.sh stop
```
Then hit `http://127.0.0.1:9300/v1` like any OpenAI endpoint. Knobs: `JUDGE_THINK=0` (faster judge),
`FUSION_CTX`, `FUSION_PARALLEL`, `FUSION_FAST_CTX`.

### Integrations
- `server/run_hermes_fusion.sh`: Hermes agent to fusion (uses `provider: custom` + loopback base_url).
- `server/claude-fusion.sh`: Claude Code to CCR (Anthropic/OpenAI translate) to fusion, with the
  KV-cache fix (`--exclude-dynamic-system-prompt-sections`), isolated config, and routing
  background to the fast tier.
- `server/search_mcp.py`: MCP server giving `web_search` (DuckDuckGo) and `fetch_url` (renders with a
  real Chrome via headless `--dump-dom`), to replace built-in web tools that don't work on a custom backend.

## The benchmark harness (`bench/`)

- `fusion.py`: panelist/judge primitives and the config table.
- `bench.py`: GSM8K / MATH-500 exact-match grading (verifiable, no judge bias).
- `draco_bench.py`: DRACO runner with optional RAG (`--rag`) and reusable queries.
- `ask.py`: interactive single-question fusion. `judge_prompt_test.py`, `paral_test.py`: ablations.

Needs `requests` (plus `datasets` for bench, `ddgs`/`trafilatura` for RAG/search).

## Caveats
- Fusion costs roughly N× a single answer plus the judge, so it's slow for tool-heavy agent loops by
  design. Use the fast/MTP tier for quick work.
- DRACO scores here are holistic Opus-judged estimates on small N. They're directional, not official
  numbers.
- Single GPU: panelists are effectively sequential. "Parallel" here mainly means *independent*, not
  concurrent.

## Credits
Inspired by OpenRouter's Fusion API and its push to combine models to reach and exceed Claude
Fable 5, plus the fusion-fable approach. DRACO benchmark by Perplexity
(`hf.co/datasets/perplexity-ai/draco`).
