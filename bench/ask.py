"""Interactive fusion: ask one question, get the synthesized answer.

  python ask.py "your question"                  # default: fusion-35b-35b
  python ask.py --config fusion-35b-27b "..."
  python ask.py --config solo-35b "..."
  python ask.py --show-panel "..."               # also print each panelist answer

Unlike bench.py (verifiable exact-match), this is for open-ended use and prints
the judge's synthesis (consensus / contradictions / what was missed) + answer.
"""
from __future__ import annotations
import argparse, sys
import fusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("--config", default="fusion-35b-35b", choices=fusion.CONFIGS)
    ap.add_argument("--show-panel", action="store_true")
    args = ap.parse_args()
    q = " ".join(args.question)

    h = fusion.health()
    if not all(h.get(m) for m in (["35b", "27b"] if "27b" in args.config else ["35b"])):
        sys.exit(f"server(s) down: {h} — run ./servers.sh start")

    r = fusion.run_config(args.config, q)

    if args.show_panel:
        for i, p in enumerate(r["panel"]):
            if p["model"].startswith("judge"):
                continue
            print(f"\n===== Candidate {chr(ord('A')+i)} ({p['model']}) =====")
            print(p["answer"])
    print("\n" + "=" * 70)
    print(f"FUSED ANSWER  [{args.config}]  ({r['latency_s']:.1f}s, {r['out_tokens']} out tok)")
    print("=" * 70)
    print(r["final"])


if __name__ == "__main__":
    main()
