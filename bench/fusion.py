"""Local model fusion harness.

Pattern (OpenRouter "Fusion" / Ziwen's fusion-fable, reproduced locally):
  fan out the same prompt to N panelists (blind to each other, sampled with
  diversity) -> a judge model reads all answers ANONYMIZED and synthesizes a
  single best answer. ~75% of the published lift is the synthesis step, ~25%
  is model diversity.

Servers (start with servers.sh):
  35b -> http://127.0.0.1:9001   (Qwen3.6-35B-A3B MTP)
  27b -> http://127.0.0.1:9002   (Qwen3.6-27B MTP)
"""
from __future__ import annotations
import re, time, requests

ENDPOINTS = {
    "35b": "http://127.0.0.1:9001",
    "27b": "http://127.0.0.1:9002",
    "ds4": "http://127.0.0.1:8010",   # DeepSeek-V4-Flash via ds4-server
}
HEALTH_PATH = {"ds4": "/v1/models"}   # ds4-server has no /health

# Diversity temperature for panelists; near-greedy for solo baselines + judge.
PANEL_TEMP = 0.7
SOLO_TEMP = 0.2
JUDGE_TEMP = 0.2


def chat(model: str, messages, temperature=0.7, seed=0, max_tokens=1024, think=False):
    """One /v1/chat/completions call. Returns (text, usage_dict, latency_s)."""
    url = ENDPOINTS[model] + "/v1/chat/completions"
    payload = {
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if model != "ds4":  # Qwen3.6 thinking toggle; ds4-server doesn't take this field
        payload["chat_template_kwargs"] = {"enable_thinking": think}
    else:  # ds4: suppress reasoning so the budget goes to the report (matches qwopus think=off)
        payload["reasoning_effort"] = "none"
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=900)
    r.raise_for_status()
    d = r.json()
    dt = time.time() - t0
    text = d["choices"][0]["message"]["content"]
    usage = d.get("usage", {}) or {}
    return text, usage, dt


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


JUDGE_SYS = (
    "You are a meticulous judge. You are given a question and several independent "
    "candidate answers (A, B, ...) produced blind to each other. Do NOT assume any "
    "is correct.\n"
    "1. Identify where they AGREE (consensus), where they CONTRADICT, what each "
    "UNIQUELY contributes, and what they ALL may have MISSED.\n"
    "2. Resolve contradictions by reasoning and checking the work yourself.\n"
    "3. Then write the single best final answer.\n"
    "Keep it concise. End with the final answer on its own line as \\boxed{...}."
)


def panelist(model, question, temperature, seed, max_tokens=1024, think=False, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": question}]
    text, usage, dt = chat(model, msgs, temperature=temperature, seed=seed,
                           max_tokens=max_tokens, think=think)
    return {"model": model, "answer": strip_think(text), "usage": usage, "t": dt}


def synthesize(judge_model, question, panel, max_tokens=1024, think=False, judge_sys=JUDGE_SYS):
    block = ""
    for i, pa in enumerate(panel):
        label = chr(ord("A") + i)
        block += f"\n--- Candidate {label} ---\n{pa['answer']}\n"
    user = (f"Question:\n{question}\n\nCandidate answers (anonymized):\n{block}\n\n"
            "Produce your synthesis (consensus / contradictions / unique / missed), "
            "then the single best final answer.")
    msgs = [{"role": "system", "content": judge_sys}, {"role": "user", "content": user}]
    text, usage, dt = chat(judge_model, msgs, temperature=JUDGE_TEMP, seed=7,
                           max_tokens=max_tokens, think=think)
    return strip_think(text), usage, dt


# Plan table: config -> (panelist (model,seed) list, synthesizer model or None for solo)
PLAN = {
    "solo-35b":       ([("35b", 1)], None),
    "solo-27b":       ([("27b", 1)], None),
    "solo-ds4":       ([("ds4", 1)], None),
    "fusion-35b-35b": ([("35b", 1), ("35b", 2)], "35b"),
    "fusion-35b-27b": ([("35b", 1), ("27b", 1)], "35b"),
    "fusion-27b-27b": ([("27b", 1), ("27b", 2)], "27b"),  # all-budget panel
    "fusion-35b-ds4": ([("35b", 1), ("ds4", 1)], "35b"),  # panel {35B, ds4}, 35B judge
    "fusion-ds4-35b": ([("ds4", 1), ("35b", 1)], "ds4"),  # panel {35B, ds4}, ds4 judge
}


def run_config(cfg: str, question: str, max_tokens=1024, think=False, system=None,
               judge_sys=JUDGE_SYS) -> dict:
    spec = PLAN.get(cfg)
    if spec is None:
        raise ValueError(f"unknown config {cfg}")
    members, synth = spec
    t0 = time.time()
    if synth is None:  # solo
        pa = panelist(members[0][0], question, SOLO_TEMP, members[0][1],
                      max_tokens=max_tokens, think=think, system=system)
        panel = [pa]; final = pa["answer"]
    else:
        panel = [panelist(m, question, PANEL_TEMP, s, max_tokens=max_tokens,
                          think=think, system=system) for m, s in members]
        final, ju, jt = synthesize(synth, question, panel, max_tokens=max_tokens,
                                   think=think, judge_sys=judge_sys)
        panel = panel + [{"model": f"judge:{synth}", "usage": ju, "t": jt}]
    out_tokens = sum(p["usage"].get("completion_tokens", 0) for p in panel)
    return {"final": final, "latency_s": time.time() - t0,
            "out_tokens": out_tokens, "panel": panel}


CONFIGS = list(PLAN.keys())


def health():
    ok = {}
    for m, url in ENDPOINTS.items():
        try:
            path = HEALTH_PATH.get(m, "/health")
            ok[m] = requests.get(url + path, timeout=5).status_code == 200
        except Exception:
            ok[m] = False
    return ok
