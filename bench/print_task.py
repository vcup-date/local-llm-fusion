"""Print one DRACO task (rubric criteria + each config's answer) for grading.
  python print_task.py <task_index>
"""
import sys, json
from pathlib import Path

n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
d = json.loads((Path(__file__).parent / "results" / "draco_gen_n6.json").read_text())
t = d["tasks"][n]
print(f"##### TASK {n} | domain={t['domain']} | id={t['id']}")
print("PROBLEM:\n" + t["problem"][:1200])
print("\n===== RUBRIC =====")
total_w = 0
for sec in t["rubric"].get("sections", []):
    print(f"\n[{sec.get('title')}]  (section id={sec.get('id')})")
    for c in sec.get("criteria", []):
        w = c.get("weight", 0); total_w += w
        print(f"  - ({w}) {c.get('id')}: {c.get('requirement')}")
print(f"\nTOTAL WEIGHT = {total_w}")
print("\n===== ANSWERS =====")
for cfg, v in t["configs"].items():
    print(f"\n########## {cfg}  ({v['latency_s']}s, {v['out_tokens']} tok) ##########")
    print(v["answer"])
