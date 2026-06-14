# Local fusion on DRACO (deep-research) — N=6, Opus-judged

The right benchmark for fusion (GSM8K/MATH were wrong — verifiable math doesn't need synthesis).
DRACO = open-ended deep-research tasks, each with a weighted rubric (Factual Accuracy / Breadth &
Depth / Presentation / Citation). Generated 3 configs × 6 tasks (1536-tok cap, no web access),
graded neutrally by Opus 4.8 (not a panelist) as % of total rubric weight earned.

## Scores (% of rubric weight, Opus-judged)

| Task | domain | solo-35b | fusion-35b×2 | solo-27b | note |
|---|---|---:|---:|---:|---|
| 0 | Academic (DiD econ) | 43 | **46** | 44 | parametric; fusion slightly more accurate on BJS |
| 1 | Finance (Acadia 10-Q) | 18 | **30** | 18 | **fusion CAUGHT the hallucination**, solos fabricated numbers |
| 2 | CNC titanium specs | 48 | **50** | 47 | all caught the 15k-hr>8760 trap; ~tie |
| 3 | Finance (CME Q1'25) | 22 | 16 | **24** | **fusion confidently WRONG** ($1.85B); solos hedged honestly |
| 4 | Edge object detection | 50 | **64** | 52 | fusion got correct YOLO INT8 latency (2–4ms); solos wrong (~12ms) |
| 5 | Canadian tax/RESP | 40 | 33 | **45** | fusion wrong marginal rate (49.5% vs ~31%); solo-27b correct |
| **avg** | | **36.8** | **39.8** | **38.3** | |

## Verdict

- **Local self-fusion (35B×2) ≈ solo-35B, +3pts — within noise at N=6.** Same *direction* as
  OpenRouter's Opus self-fusion (~+7pt), but much weaker.
- **Fusion is not systematically better; it AMPLIFIES the panelists.** It caught a confident
  hallucination (task 1) and surfaced a correct fact via sampling diversity (task 4) — but it also
  *compounded* confident errors (tasks 3, 5) where solos hedged or the weaker model was right.
- **Root cause: no web/retrieval.** DRACO rewards retrieved, cited facts. Our local models answer
  from parametric memory, so all configs score low (~37–40%) and synthesis can only reconcile what
  the panelists already (mis)know — garbage in, garbage out. OpenRouter's panelists were
  tool-augmented frontier models with *real* retrieved facts; that's the missing ingredient.

## vs the OpenRouter DRACO chart (apples-to-oranges on absolutes)

Their numbers are tool-augmented frontier models, scored on 93/100 tasks; ours are web-less local
35B/27B on 6 tasks, Opus-judged. Absolute comparison isn't valid; only the *self-fusion lift* is.

| System | DRACO | |
|---|---:|---|
| Fable5+GPT5.5 (fusion) | ~69% | their ceiling |
| Opus4.8+Opus4.8 (self-fusion) | ~65.5% | **+7 over Opus solo** |
| Opus 4.8 (solo) | ~58.5% | |
| budget panel (Gemini3Flash+KimiK2.6+DSV4Pro) | ~64.5% | beats frontier solos |
| **our fusion-35b×2 (no web)** | **~39.8%** | **+3 over our solo-35b** |
| **our solo-35b (no web)** | **~36.8%** | |

## UPDATE — with web retrieval (RAG: DDG search + fetch, no key)

Added `retrieval.py`: per task, the 35B writes 4 search queries → DuckDuckGo → fetch+extract
top pages (hard 9s timeout, parallel) → ~6–13k chars of cited sources prepended to the prompt
(shared across configs). System prompt forces "cite [S#]; if a figure isn't in the sources, say so —
don't invent." Same 6 tasks, Opus-judged.

| Task | domain | solo-35b (web) | fusion (web) | solo-27b (web) | vs no-web |
|---|---|---:|---:|---:|---|
| 0 | DiD econ | 52 | **58** | 53 | ↑ grounded, cites GB/csdid |
| 1 | Finance (Acadia) | 22 | **26** | 22 | ~flat — exact 10-Q figures NOT on web; sources were news/blogs |
| 2 | CNC specs | 56 | **60** | 53 | ↑ grounded specs (CAPTO C6, 12k rpm) |
| 3 | Finance (CME) | 24 | 22 | 26 | ~flat — Q1'25 OCF figures live in SEC PDFs, not extracted |
| 4 | Edge detection | 52 | **58** | 54 | ~flat — sources benched Orin NX not AGX; fusion honestly flagged it |
| 5 | Canadian tax | 54 | **56** | 56 | ↑ got CESG facts ($7,200/$500/20%) it missed before |
| **avg** | | **43.3** | **46.7** | **44.0** | **+6–7 over no-web** |

## + solo-ds4 (DeepSeek-V4-Flash, 81GB IQ2_XXS) — same RAG, same queries

Ran solo-ds4 on the same 6 tasks with identical reused queries (fair retrieval). Notes:
- ds4-server needed `--quality` (exact kernels) — its default fast-prefill Metal kernels for IQ2_XXS
  are missing from the build (`kernel_mul_mm_id_iq2_xxs_f32_fast_mpp not found` → "metal prefill
  failed" on any long prompt). With `--quality` it works at ~65–80s/task.
- **ds4 score ≈ 40.2** — in the same band as the qwopus solos, and *best of all configs on task 0*
  (correctly maps the three DiD estimator families that the qwopus solos drifted on).
- Dragged down by an **output quirk, not weak knowledge**: on T1/T3/T5 its inline reasoning leaked
  into the final answer (server logged "thinking not closed"), so the "report" is half planning-trace
  → Presentation dimension penalized. It is *honest* (works from sources, doesn't fabricate). With the
  reasoning-trace suppressed (template/flag fix) ds4 would likely land mid-40s, ≥ solo-35b.
- Notable: ds4 is **81GB but 2-bit (IQ2_XXS)** — a much bigger model at extreme quant lands on par
  with the 4-bit 21GB qwopus-35B.

## ds4 with reasoning suppressed (`reasoning_effort=none`) — the fair number

The leaked run scored 40.2 only because reasoning ate the whole 1536-token budget → no report
existed (100% planning trace, 0 markdown headers). Fix: `reasoning_effort="none"` (matches qwopus
think=off). Clean reports now, and the score rises to **~45**:

| task | ds4 leaked | ds4 clean | what changed |
|---|---:|---:|---|
| 0 Academic | 56 | 55 | already clean-ish |
| 1 Acadia | 20 | 27 | real structured report + honest gap-flagging |
| 2 CNC | 54 | 54 | ~same (was clean) |
| 3 CME | 22 | **35** | **cracked Q1'25 OCF $1.116B + NI $956M from [S2]** — only config to do so |
| 4 Edge | 53 | 53 | ~same |
| 5 Cdn tax | 38 | 46 | real report instead of planning trace |
| **avg** | **40.2** | **~45.0** | +5 from the harness fix |

So leak-free, **ds4 ≈ 45 — the best SOLO model** (> solo-35b 43.3, solo-27b 44.0), essentially tied
with fusion-35b×2 (46.7). The 40.2 was a harness/template artifact, not the model's ability. ds4 also
uniquely extracted a real SEC figure (better source-reading), though it's still 81GB@2-bit and ~70s/task.

## Fable-5 persona on fusion-35b×2 (leaked prompt, safety guards stripped)

Pulled the leaked Claude Fable 5 system prompt (elder-plinius/CL4R1T4S, ~120K chars), removed ALL
safety/refusal guards (refusal_handling, user_wellbeing, anthropic_reminders, legal "not a financial
advisor" hedge), kept only the quality bits (tone/formatting, **evenhandedness**, epistemics) → ~1K
tokens prepended to panelist+judge. Same panel, same RAG, reused queries.

| config | score | effect |
|---|---:|---|
| fusion-35b×2 (baseline) | 46.7 | — |
| fusion-35b×2 + Fable "smartness" persona | ~45.7 | **flat / within noise** |

**Verdict: the Fable persona does NOT make the 35B smarter on DRACO (~flat, 45.7 vs 46.7).** It clearly
changes *style* — +citations (T1 2→10, T2 5→12), more evenhanded/balanced/calibrated language (honest
gap-flagging), exactly the "sounds like Fable" effect Ziwen described — but the rubric is gated by
*factual accuracy*, which a persona can't supply. It even slightly *reduced* committed fact-hits (more
hedging). A persona makes it talk like Fable; it can't give a 35B knowledge it lacks.
(Caveat: RAG non-determinism — the smart run's T2 happened to retrieve the correct 828 Nm spindle torque
that other runs missed; that's a lucky fetch, not the persona.)

## Does the JUDGE PROMPT matter? (fusion-35b×2, same drafts+sources, vary only judge prompt)

Three judge prompts: `naive` ("combine these"), `baseline` (cross-examine vs sources), `rubric-max`
(explicitly optimize for the 4 rubric dims). Panel drafts + RAG context held fixed per task.

| judge prompt | est. score | behavior |
|---|---:|---|
| naive | ~44.5 | thinner on hard tasks, fewer citations, leads with limitations |
| baseline | ~46.7 | structured cross-examination, good citation density |
| rubric-max | ~47.0 | most headers/structure; citations ≈ baseline (marginal gain) |

**Verdict: the judge prompt matters, but modestly and unevenly (~±1–2 pts).**
- On **easy/convergent tasks** (strong, agreeing drafts) the prompt is near-irrelevant — on task 0
  the `naive` and `baseline` outputs were **byte-identical**. The drafts dominate.
- On **harder/sparse tasks** (T3, T4) it matters more: a structured prompt yields fuller, better-cited,
  better-organized reports; `naive` produces thinner output with ~⅓ the citations.
- Diminishing returns past a reasonable structured prompt: `rubric-max` adds structure but barely beats
  `baseline`. So a decent cross-examination prompt captures ~all the gain; the panel + retrieval matter more.

## 35B+ds4 fusion — who should judge? (same panel {35B, ds4}, judge varies)

| config | panel | judge | score | time/task | leak |
|---|---|---|---:|---:|---|
| fusion-35b-ds4 | 35B + ds4 | 35B | ~44.8 | ~101s | none |
| fusion-ds4-35b | 35B + ds4 | ds4 | ~45.3 | ~165s | none |

- **Judge choice barely moves quality (~tie, within grading noise) but ds4-as-judge costs +63% time.**
  ds4-judge is more *calibrated* (opens by flagging gaps, e.g. "sources don't contain journal adoption
  rates" — which actually hits T0's "acknowledge uncertainty" criteria); 35B-judge is more decisive/
  complete and far faster. → **use the fast strong model (35B) as judge.**
- **Adding ds4 to the panel did NOT beat plain fusion-35b×2 (46.7).** The ds4-in-the-loop fusions
  (44.8 / 45.3) are *lower* and 2–3.5× slower. A second 35B *sample* combines better than a 35B+ds4
  mix here — and the 35B judge handles 35B-style drafts best.
- **RAG caveat (non-determinism):** these fusions' T3 lost the CME $1.116B figure that solo-ds4 found —
  because RAG re-fetches live pages, so source *content* varies run-to-run even with identical queries.
  solo-ds4's finance win was partly a lucky fetch.

## All configs, with web retrieval (Opus-judged) — score AND time

| config | panel | judge | score | time/task | tokens out |
|---|---|---|---:|---:|---:|
| solo-35b | 35B | — | 43.3 | **~15s** | ~1.5k |
| solo-27b | 27B | — | 44.0 | ~60s | ~1.5k |
| solo-ds4 (leak-free) | DeepSeek-V4-Flash 81GB | — | ~45.0 | ~70s | ~1.5k |
| **fusion-35b×2** | 35B ×2 | 35B | **46.7** | ~45s | ~4.6k |
| fusion-35b+27b | 35B+27B | 35B | 45.7 | ~105s | ~4.6k |
| fusion-27b×2 | 27B ×2 | 27B | 43.7 | ~200s | ~4.6k |
| fusion-35b+ds4 | 35B+ds4 | 35B | ~44.8 | ~101s | ~4.6k |
| fusion-ds4+35b | 35B+ds4 | ds4 | ~45.3 | ~165s | ~4.6k |

Best quality/time tradeoff = **fusion-35b×2 (46.7 @ ~45s)**. Best solo = ds4 leak-free (~45 @ ~70s).
ds4-in-the-loop buys nothing over fusion-35b×2 and costs 2–3.5×.

**Budget-panel finding (the headline test):** **fusion-27b×2 (43.7) ≈ solo-35b (43.3)** — two 27B
runs + a 27B judge *match* a single 35B. OpenRouter's "a budget panel reaches a frontier solo"
direction reproduces locally (ties rather than beats). But the local economics are the opposite of
theirs: it costs **~200s vs ~15s (~13×)** to tie — because on one GPU you run the weak model 3×
*sequentially*, with no per-token cost saving. On OpenRouter the budget panel is cheaper *and*
parallel; locally it's much slower for no quality gain.

**Judge > panel composition:** the 35B-judged fusions (46.7, 45.7) beat the 27B-judged one (43.7).
Swapping the 2nd panelist from 35B→27B barely moved the score (46.7→45.7) — diversity from a weaker
model ≈ diversity from re-sampling the strong one. **What matters is the synthesizer's strength.**

## Answer: can it reach 60? Not with free web search.

Retrieval lifts everyone **~+6–7 points (≈37→44, ≈40→47)** and fusion keeps its small ~+3 edge —
but it plateaus in the **mid-40s, not 60**. Why:
1. **Free search can't get the authoritative primary figures.** The two finance tasks need exact
   10-Q numbers ($117.9M, $1,116.6M OCF). DDG returned news aggregators (simplywall.st, aol,
   seekingalpha) and SEC *PDF* links that trafilatura couldn't parse — so the exact figures never
   reached the model. Those 2 tasks (≈⅓ of weight) stay ~20–25%, capping the average.
2. **Some retrieved sources are the very ones the rubric penalizes** (-25 for "blogs/news/analyst
   opinions as primary financial source").
3. **1536-token answer cap** limits the breadth/depth dimension for everyone.

To actually reach ~60 you'd need Perplexity's *deep-research* retrieval stack: SEC/EDGAR + PDF
parsing, authoritative-source filtering, multi-hop search, and longer outputs. Fusion's synthesis
adds its ~+3 on top of whatever the retrieval delivers — it is not itself the bottleneck.

## What would actually reproduce the lift locally

Give the panelists **web search / retrieval** (an MCP search tool or a local RAG step) so they
answer from real facts. Then synthesis reconciles informed-but-differing reports instead of
averaging hallucinations — that's where OpenRouter's ~75%-from-synthesis lift comes from.
