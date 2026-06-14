"""Generate answers for DRACO deep-research tasks across fusion configs.

Does NOT auto-grade. Dumps each task's rubric + every config's answer to
results/draco_gen_n<N>.json so a neutral judge (Opus) can score them against
the per-task rubric.

  python draco_bench.py --n 6
  python draco_bench.py --n 6 --configs solo-35b fusion-35b-35b fusion-27b-27b
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import fusion

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

RESEARCH_SYS = (
    "You are an expert research analyst. Answer the research question with a "
    "comprehensive, well-structured report. Be specific and precise: name key "
    "people, works, dates, mechanisms, and numbers. Organize into clear sections, "
    "cover competing viewpoints, and state uncertainty honestly. No fluff."
)
RAG_SYS = (
    "You are an expert research analyst. You are given a research question and a set "
    "of RETRIEVED SOURCES ([S1], [S2], ...). Write a comprehensive, well-structured "
    "report that answers the question. RULES: ground every factual claim, figure, and "
    "date in the sources and cite the source tag like [S3]. If the sources do not "
    "contain a needed figure, say so explicitly — do NOT invent numbers. Be specific."
)
RAG_JUDGE_SYS = (
    "You are a senior research editor. You are given a research question, a set of "
    "RETRIEVED SOURCES ([S1]...), and several independent draft reports (Candidate A, "
    "B, ...). Cross-examine the drafts AGAINST the sources: keep claims supported by "
    "sources (cite [S#]), discard unsupported or contradicted claims, and flag any "
    "figure not found in the sources rather than repeating it. Then write one best, "
    "fully source-grounded final report."
)


def ground(problem, ctx):
    return f"{problem}\n\n=== RETRIEVED SOURCES ===\n{ctx}\n=== END SOURCES ==="
RESEARCH_JUDGE_SYS = (
    "You are a senior research editor. You are given a research question and "
    "several independent draft reports (Candidate A, B, ...) written blind to each "
    "other. Do not assume any is fully correct.\n"
    "1. Note where they AGREE, where they CONTRADICT, what each UNIQUELY adds, and "
    "what they ALL MISSED.\n"
    "2. Resolve conflicts using your own knowledge.\n"
    "3. Write a single best, comprehensive, well-structured final report that is "
    "more accurate and complete than any individual draft. Be specific."
)
MAXTOK = 1536


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--configs", nargs="+", default=fusion.CONFIGS)
    ap.add_argument("--rag", action="store_true", help="retrieve web sources and ground answers")
    ap.add_argument("--reuse-queries-from", default=None,
                    help="path to a prior results json; reuse its per-task queries for fair retrieval")
    ap.add_argument("--sys-prefix", default=None,
                    help="file whose text is prepended to panelist+judge system prompts (e.g. a persona)")
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    args = ap.parse_args()

    sys_prefix = ""
    if args.sys_prefix:
        sys_prefix = Path(args.sys_prefix).read_text().strip() + "\n\n---\n\n"
        print(f"prepending sys-prefix ({len(sys_prefix)} chars) from {args.sys_prefix}")

    reuse_q = {}
    if args.reuse_queries_from:
        prev = json.loads(Path(args.reuse_queries_from).read_text())
        for pt in prev["tasks"]:
            if pt.get("retrieval"):
                reuse_q[pt["id"]] = pt["retrieval"]["queries"]
        print(f"reusing queries for {len(reuse_q)} tasks from {args.reuse_queries_from}")

    h = fusion.health()
    print("server health:", h, flush=True)
    needed = set()
    for c in args.configs:
        members, synth = fusion.PLAN[c]
        needed |= {m for m, _ in members} | ({synth} if synth else set())
    for m in needed:
        if not h.get(m):
            raise SystemExit(f"server '{m}' is DOWN — run ./servers.sh start")

    from datasets import load_dataset
    ds = load_dataset("perplexity-ai/draco", split="test")
    items = [ds[i] for i in range(min(args.n, len(ds)))]
    print(f"DRACO: {len(items)} tasks | configs: {args.configs}\n", flush=True)

    out_tasks = []
    for qi, it in enumerate(items):
        problem = it["problem"]
        try:
            rubric = json.loads(it["answer"])
        except Exception:
            rubric = {"raw": it["answer"]}
        rec = {"id": it["id"], "domain": it.get("domain"), "problem": problem,
               "rubric": rubric, "configs": {}}
        print(f"### task {qi} [{it.get('domain')}] {problem[:90]}...", flush=True)

        # retrieval (shared across configs for fairness)
        q_in, sys_p, judge_p = problem, RESEARCH_SYS, RESEARCH_JUDGE_SYS
        if args.rag:
            import retrieval
            qmodel = "ds4" if any("ds4" in c for c in args.configs) else "35b"
            chat_fn = lambda m: fusion.chat(qmodel, m, temperature=0.2, seed=3, max_tokens=256)[0]
            ctx, sources, queries = retrieval.build_context(
                chat_fn, problem, queries=reuse_q.get(it["id"]))
            q_in, sys_p, judge_p = ground(problem, ctx), sys_prefix + RAG_SYS, sys_prefix + RAG_JUDGE_SYS
            rec["retrieval"] = {"queries": queries, "sources": sources,
                                "ctx_chars": len(ctx)}
            print(f"    [rag] {len(queries)} queries, {len(sources)} sources, "
                  f"{len(ctx)} ctx chars", flush=True)

        for c in args.configs:
            t0 = time.time()
            r = fusion.run_config(c, q_in, max_tokens=MAXTOK, think=False,
                                  system=sys_p, judge_sys=judge_p)
            rec["configs"][c] = {"answer": r["final"], "latency_s": round(r["latency_s"], 1),
                                 "out_tokens": r["out_tokens"]}
            print(f"    {c:16s} {r['latency_s']:6.1f}s  {r['out_tokens']:5d} tok  "
                  f"{len(r['final'])} chars", flush=True)
        out_tasks.append(rec)
        # write incrementally so progress is never lost
        out = RESULTS / f"draco_gen_{'rag_' if args.rag else ''}n{args.n}{args.tag}.json"
        out.write_text(json.dumps({"n": args.n, "configs": args.configs,
                                   "tasks": out_tasks}, indent=2))
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
