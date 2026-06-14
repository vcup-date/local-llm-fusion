"""Benchmark local fusion vs solo on verifiable tasks (exact-match, no judge bias).

Usage:
  python bench.py --dataset gsm8k --n 15
  python bench.py --dataset math500 --n 15 --configs solo-35b fusion-35b-35b

Reports accuracy, mean output tokens, and mean latency per config, plus a
JSON dump under results/.
"""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
import fusion

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

DATASETS = {
    "gsm8k": dict(load=("openai/gsm8k", "main"), split="test",
                  q=lambda x: x["question"] + "\nReason briefly, then put your final answer within \\boxed{}.",
                  gold=lambda x: x["answer"].split("####")[-1].strip()),
    "math500": dict(load=("HuggingFaceH4/MATH-500",), split="test",
                    q=lambda x: x["problem"] + "\nReason briefly, then put your final answer within \\boxed{}.",
                    gold=lambda x: str(x["answer"]).strip()),
}


def last_boxed(text: str):
    """Return the content of the last \\boxed{...} with brace matching."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        # fallback: last number
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        return nums[-1].replace(",", "") if nums else None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j].strip()
    return None


def norm(s):
    if s is None:
        return None
    s = s.strip().strip("$").replace(",", "").replace(" ", "")
    s = s.replace("\\!", "").replace("\\,", "")
    s = re.sub(r"\\text\{.*?\}", "", s)
    if s.endswith("."):
        s = s[:-1]
    return s


def equal(pred, gold):
    p, g = norm(pred), norm(gold)
    if p is None or g is None:
        return False
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gsm8k", choices=list(DATASETS))
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--configs", nargs="+", default=fusion.CONFIGS)
    args = ap.parse_args()

    h = fusion.health()
    print("server health:", h)
    need = {"solo-35b": ["35b"], "solo-27b": ["27b"],
            "fusion-35b-35b": ["35b"], "fusion-35b-27b": ["35b", "27b"]}
    for c in args.configs:
        for m in need[c]:
            if not h.get(m):
                raise SystemExit(f"server '{m}' (needed by {c}) is DOWN — run ./servers.sh start")

    from datasets import load_dataset
    spec = DATASETS[args.dataset]
    ds = load_dataset(*spec["load"], split=spec["split"])
    items = [ds[i] for i in range(min(args.n, len(ds)))]
    print(f"{args.dataset}: {len(items)} questions | configs: {args.configs}\n")

    agg = {c: dict(correct=0, tok=0, t=0.0, n=0) for c in args.configs}
    per_q = []
    for qi, it in enumerate(items):
        question = spec["q"](it)
        gold = spec["gold"](it)
        row = {"i": qi, "gold": gold, "results": {}}
        for c in args.configs:
            try:
                r = fusion.run_config(c, question)
                pred = last_boxed(r["final"])
                ok = equal(pred, gold)
            except Exception as e:
                pred, ok = f"ERR:{e}", False
                r = {"latency_s": 0, "out_tokens": 0}
            a = agg[c]
            a["correct"] += int(ok); a["tok"] += r["out_tokens"]; a["t"] += r["latency_s"]; a["n"] += 1
            row["results"][c] = {"pred": pred, "ok": ok, "t": round(r["latency_s"], 1),
                                 "tok": r["out_tokens"]}
            print(f"  q{qi:02d} {c:16s} pred={str(pred)[:18]:18s} gold={str(gold)[:12]:12s} "
                  f"{'OK ' if ok else 'x  '} {r['latency_s']:5.1f}s")
        per_q.append(row)
        print()

    print("=" * 64)
    print(f"{'config':16s} {'acc':>7s} {'correct':>9s} {'avg_tok':>8s} {'avg_s':>7s}")
    summary = {}
    for c in args.configs:
        a = agg[c]
        acc = a["correct"] / a["n"] if a["n"] else 0
        summary[c] = dict(acc=acc, correct=a["correct"], n=a["n"],
                          avg_tok=a["tok"] / a["n"], avg_s=a["t"] / a["n"])
        print(f"{c:16s} {acc*100:6.1f}% {a['correct']:>4d}/{a['n']:<4d} "
              f"{a['tok']/a['n']:8.0f} {a['t']/a['n']:7.1f}")

    out = RESULTS / f"{args.dataset}_n{args.n}.json"
    out.write_text(json.dumps({"dataset": args.dataset, "n": args.n,
                               "summary": summary, "per_q": per_q}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
