"""Does the JUDGE PROMPT affect fusion quality?
Hold panel constant (35B x2, same drafts + same RAG context per task); vary only
the judge/synthesis system prompt. Isolates the judge-prompt effect.

  python judge_prompt_test.py --n 6 --reuse-queries-from results/draco_gen_rag_n6_part1.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import fusion, retrieval

RESULTS = Path(__file__).parent / "results"
RAG_SYS = ("You are an expert research analyst. You are given a research question and a set of "
           "RETRIEVED SOURCES ([S1]...). Write a comprehensive, well-structured report. Ground "
           "every factual claim/figure/date in the sources and cite [S#]; if a needed figure is "
           "not in the sources, say so — do NOT invent. Be specific.")

def ground(p, ctx):
    return f"{p}\n\n=== RETRIEVED SOURCES ===\n{ctx}\n=== END SOURCES ==="

JUDGE_VARIANTS = {
 "naive": ("You are given a question and several candidate answers. Combine them into the single "
           "best answer."),
 "baseline": ("You are a senior research editor. You are given a research question, RETRIEVED "
              "SOURCES ([S1]...), and several independent draft reports (Candidate A, B). "
              "Cross-examine the drafts against the sources: keep claims supported by sources "
              "(cite [S#]), discard unsupported/contradicted claims, flag any figure not in the "
              "sources. Then write one best, fully source-grounded final report."),
 "rubric-max": ("You are a senior research editor producing a publication-grade report from a "
                "question, RETRIEVED SOURCES ([S#]), and candidate drafts. Optimize the final "
                "report for: (1) FACTUAL ACCURACY — state every specific figure, name and date "
                "present in the sources, cite [S#], never invent, and explicitly flag any "
                "requested figure missing from the sources; (2) BREADTH & DEPTH — answer every "
                "sub-question, give competing views, quantify; (3) PRESENTATION — clear sections "
                "with headers, precise domain terminology, lead each section with its key finding; "
                "(4) CITATIONS — cite an authoritative [S#] for each claim. Cross-examine the "
                "drafts: keep source-supported claims, drop the rest. Be comprehensive and specific."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--reuse-queries-from", required=True)
    args = ap.parse_args()

    if not fusion.health().get("35b"):
        raise SystemExit("35b server down")
    reuse = {pt["id"]: pt["retrieval"]["queries"]
             for pt in json.loads(Path(args.reuse_queries_from).read_text())["tasks"]
             if pt.get("retrieval")}

    from datasets import load_dataset
    ds = load_dataset("perplexity-ai/draco", split="test")
    items = [ds[i] for i in range(args.n)]

    out = []
    for qi, it in enumerate(items):
        problem = it["problem"]
        ctx, sources, queries = retrieval.build_context(None, problem, queries=reuse.get(it["id"]))
        gp = ground(problem, ctx)
        # generate the two panelist drafts ONCE, reuse across all judge variants
        A = fusion.panelist("35b", gp, fusion.PANEL_TEMP, 1, max_tokens=1536, system=RAG_SYS)
        B = fusion.panelist("35b", gp, fusion.PANEL_TEMP, 2, max_tokens=1536, system=RAG_SYS)
        rec = {"id": it["id"], "domain": it.get("domain"), "problem": problem,
               "rubric": json.loads(it["answer"]), "judges": {}}
        for name, jsys in JUDGE_VARIANTS.items():
            final, _, _ = fusion.synthesize("35b", gp, [A, B], max_tokens=1536, judge_sys=jsys)
            rec["judges"][name] = final
            print(f"  task {qi} judge={name:10s} {len(final)} chars", flush=True)
        out.append(rec)
        (RESULTS / f"judge_prompt_n{args.n}.json").write_text(
            json.dumps({"tasks": out}, indent=2))
        print(f"### task {qi} done\n", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
