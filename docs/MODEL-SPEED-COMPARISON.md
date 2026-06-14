# Model & Method Speed Comparison — qwopus-35b vs dflash vs turboquant

**Date:** 2026-06-14 · **Hardware:** Apple M5 Max, 128 GB · **Model class:** Qwen3.6-35B-A3B (MoE, ~3B active), 4-bit
**Protocol:** updated each repo to latest + rebuilt both llama.cpp forks (Metal). Context allocated at `-c 200000`, short (~270–350 token) prompt, temperature 0, `n_predict=128–256`.

## UPDATE (2026-06-14, later) — migrated qwopus llama.cpp to mainline

MTP merged into mainline `ggml-org/llama.cpp` (PR #22673, 2026-05-16) + a VRAM-leak fix (~May 21). Switched `mtp-bench/src/llama.cpp` from the frozen am17an `mtp-clean` fork to **mainline master (`8ed274ef4`, 2026-06-14)** and rebuilt. Huge gains — the old fork build was the bottleneck:

| Config (`-c 200000`) | OLD fork build | **NEW mainline build** | change |
|---|---:|---:|---|
| 35B-A3B MTP — gen tok/s | 138.8 | **180.0** | +30% |
| 35B-A3B MTP — prefill tok/s | 819 | **2204** | +169% |
| 35B-A3B baseline — prefill | 666 | **2183** | +228% |
| 27B MTP — gen tok/s | (n/a) | **48.2** | 1.80× over 27B base |
| 27B baseline — gen | (n/a) | 26.8 | — |

Mainline `--spec-type` now also offers `draft-eagle3` and ngram variants. The community also ships pre-built MTP GGUFs of this exact model (e.g. `wezzel98765/Qwopus3.6-35B-A3B-v1-oQ4..oQ8-fp16-mtp`), so hand-requantizing is no longer required. Qwopus 35B-A3B-v1 base itself is unchanged (HF last-modified 2026-05-15).

**New leaderboard (mainline build): 35B-A3B MTP = ~180 tok/s gen @ 2204 prefill — fastest on both axes.**

---

## Bottom line (original run, pre-mainline)

- **Fastest *generation*: qwopus MTP** — Qwen3.6-35B-A3B requantized Q4_K_M with `--spec-type draft-mtp` → **~139 tok/s** (1.49× over its 93 tok/s baseline, 90% draft acceptance). It's the *only* method here that actually speeds up decoding.
- **Fastest *prefill*: the MLX 4-bit engine (~1.8k tok/s)** and the **newer turboquant llama.cpp build (~1.5–1.7k tok/s)** — but prefill is governed by the **inference engine/build version, not the method**. None of the three methods targets prefill.
- **TurboQuant is not a speed method** — it's KV-cache *compression* (3.8–4.9×) at only ~6% generation cost. Use it to fit long contexts in memory, not to go faster. (Huge improvement over its old benchmark of 2.4 tok/s / 35× slowdown — the rebuild fixed that.)
- **DFlash does not accelerate on Apple Silicon/MLX** — best case ~130 tok/s vs 136 tok/s plain decode. Speculation *works* (2.3 tok/step) but per-step overhead on MLX eats the gain. Its documented speedups require vLLM/SGLang on CUDA.

## Generation (decode) speed — ranked

| Rank | Stack / method | Gen tok/s | Notes |
|---|---|---:|---|
| 1 | **qwopus MTP** (Q4_K_M requant, llama.cpp) | **138.8** | 90% draft accept; 1.49× over baseline |
| 2 | dflash **MLX base-only** (4bit) | 136.3 | plain MLX decode, no draft |
| 3 | dflash **DFlash** best (block=4) | 130.0 | speculation = net loss vs base-only |
| 4 | qwopus baseline (Q4_K_XL) | 93.4 | llama.cpp, no MTP |
| 4 | turboquant f16 KV | 93.2 | newer llama.cpp build |
| 5 | turboquant q8_0 KV | 92.2 | — |
| 6 | turboquant turbo4 KV | 88.2 | 3.8× KV compression |
| 7 | turboquant turbo3 KV | 87.4 | 4.9× KV compression |

## Prefill (prompt processing) speed

| Stack / build | Prefill tok/s | Caveat |
|---|---:|---|
| dflash MLX (4bit) | ~1740–1870 | 271-tok prompt |
| turboquant llama.cpp (synced to b9190) turbo4 | 1680 | 348-tok prompt |
| turboquant turbo3 / q8_0 / f16 | 1612 / 1613 / 1485 | KV type barely affects prefill |
| qwopus MTP build (am17an mtp-clean) | 819 | older/different llama.cpp build |
| qwopus baseline build | 666 | same build |

> **Prefill caveat:** the 2× gap between the turboquant build (~1.5k) and the qwopus mtp-clean build (~0.7k) is the **same model on different llama.cpp versions** — it's a build-speed difference, not an MTP penalty. MTP/TurboQuant/DFlash all leave prefill essentially unchanged.

## DFlash block-size sweep (matched official base `mlx-community/Qwen3.6-35B-A3B-4bit`)

| block_size | tokens/step | gen tok/s |
|---:|---:|---:|
| 2 | 1.64 | 117.2 |
| 4 | 2.27 | **130.0** |
| 8 | 2.33 | 97.1 |
| 16 | 2.62 | 81.4 |
| — (base-only) | 1.00 | **136.3** |

Even the best block size stays below plain decode → DFlash-on-MLX is net-negative on M5 Max.

## What "fastest" means per goal

- **Want max tokens/sec, single stream:** qwopus **MTP requant Q4_K_M** (~139 tok/s).
- **Want long context (100k–200k) to fit in RAM:** turboquant **turbo3** KV (4.9× smaller cache, ~6% slower).
- **Want fast prefill:** any modern build / MLX — not a method choice; keep prefill in mind only as an engine choice.
- **DFlash:** skip on Mac; revisit only on a CUDA box with vLLM/SGLang.
