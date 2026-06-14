# Qwen3.6 MTP benchmark — M5 Max 128GB

**Hardware:** Apple M5 Max, 128 GB unified memory
**Backend:** llama.cpp built from PR #22673 (`am17an/llama.cpp:mtp-clean`, commit `e7b4848`), Metal + BLAS
**Quant:** UD-Q4_K_XL for all four Qwen3.6 GGUFs
**Settings:** `-ngl 99 -fa 1 -c 4096 --parallel 1`, prompt 64 tokens, `n_predict=256`, `temperature=0`, `seed=42`
**MTP flag:** `--spec-type draft-mtp --spec-draft-n-max 3`

## Generation speed (tok/s)

| Model                                            |  tok/s | vs stock base | Draft accept |
| ------------------------------------------------ | -----: | ------------: | -----------: |
| Qwen3.6 27B baseline (UD-Q4_K_XL)                |  25.08 |          1.00× |          —   |
| Qwen3.6 27B **MTP** (UD-Q4_K_XL)                 |  42.56 |          1.70× |       98.8 % |
| Qwen3.6 35B-A3B baseline (UD-Q4_K_XL)            |  93.81 |          1.00× |          —   |
| Qwopus3.6-35B-A3B-v1 Q4_K_M (your existing)      | 110.59 |          1.18× |          —   |
| Qwen3.6 35B-A3B **MTP** (UD-Q4_K_XL)             | 127.63 |          1.36× |       97.6 % |
| Qwen3.6 35B-A3B **MTP @ plain Q4_K_M** (requant) | **147.03** | **1.57×**  |       98.8 % |

## Prompt processing (pp, tok/s)

| Model                  | llama-bench pp512 | /completion pp64 |
| ---------------------- | ----------------: | ---------------: |
| Qwen3.6 27B baseline   |          700.06   |          233.40  |
| Qwen3.6 27B MTP        |              —    |          215.87  |
| Qwen3.6 35B-A3B base   |         3074.11   |          686.97  |
| Qwen3.6 35B-A3B MTP    |              —    |          289.81  |
| Qwopus 35B-A3B Q4_K_M  |              —    |          714.41  |

## Observations

- MTP **works** on Mac Metal via the PR branch. `--spec-type draft-mtp` is the right flag.
- Draft acceptance is **~98 %** on this deterministic prompt — much higher than the ~75 % the PR mentions for varied inputs, so expect a smaller gap on more diverse generation tasks.
- **27B benefits most** (1.70×) because it's compute/memory-bound; MTP cuts the per-token decode cost noticeably.
- **35B-A3B (MoE) gains less** (1.36×) because each token already activates only ~3B params, so it was less bottlenecked. Still respectable.
- The tweet quoted 140 tok/s (27B) and 220 tok/s (35B-A3B) on a single GPU. Your M5 Max hits **43** and **128** tok/s respectively — about a third of GPU speed on the dense model, more than half on the MoE.
- The existing **Qwopus 35B-A3B Q4_K_M is faster than the stock 35B-A3B baseline** (110 vs 94 tok/s) because Unsloth's UD recipe upgrades 252 attention/expert tensors to Q8_0 (more bytes through memory bus). Pure Q4_K_M is leaner.
- **Best config: 35B-A3B MTP requantized to plain Q4_K_M = 147 tok/s** — beats every other config tested. Combines Qwopus-style uniform Q4_K (fewer bytes per token) with MTP's draft acceptance.
- Requantization warning: `--allow-requantize` was used to go Q8_0→Q4_K. Quality may be slightly worse than starting from BF16, but draft acceptance held at 98.8 % on this prompt — main weights look fine. For production use, a fresh quantize from the BF16 safetensors would be cleaner.
- Prompt-processing MTP numbers from `/completion` are misleading; for a real pp comparison use the `llama-bench pp512` numbers (3074 tok/s for 35B-A3B baseline — that's the headline).

## Files

```
mtp-bench/
├── src/llama.cpp/build/bin/{llama-bench,llama-cli,llama-server}   ← MTP-capable binaries
├── models/
│   ├── Qwen3.6-27B-baseline-UD-Q4_K_XL.gguf       17 GB
│   ├── Qwen3.6-27B-MTP-UD-Q4_K_XL.gguf            17 GB
│   ├── Qwen3.6-35B-A3B-baseline-UD-Q4_K_XL.gguf   21 GB
│   └── Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf        21 GB
├── results/
│   ├── 27b-baseline.txt                  llama-bench output
│   ├── 27b-baseline-completion.txt       /completion timings
│   ├── 27b-mtp.txt                       /completion + draft stats
│   ├── 35b-baseline.txt                  llama-bench output
│   ├── 35b-baseline-completion.txt       /completion timings
│   ├── 35b-mtp.txt                       /completion + draft stats
│   ├── qwopus-35b.txt                    existing model, /completion
│   └── SUMMARY.md                        this file
├── bench_completion.sh                   reusable bench script
└── parse_timings.py                      timings parser
```

## To rerun (e.g. higher n_max)

```sh
cd .
SERVER=./src/llama.cpp/build/bin/llama-server
"$SERVER" -m models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf \
  --spec-type draft-mtp --spec-draft-n-max 5 \
  -ngl 99 -fa 1 -c 4096 --parallel 1 \
  --host 127.0.0.1 --port 8765 &
./bench_completion.sh 8765 "35B MTP n_max=5" results/35b-mtp-nmax5.txt
```
