"""Measure whether 2 concurrent 35B requests are faster than 2 sequential ones
on a single GPU (tests the continuous-batching vs GPU-serialization question).

Assumes a llama-server on --port with --parallel >=2.
  python paral_test.py <port> <label>
"""
import sys, time, threading, requests

PORT = sys.argv[1] if len(sys.argv) > 1 else "9101"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "test"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
NPRED = 400
PROMPT = "Explain the transformer architecture and why it scales, in detail."


def one(seed):
    t0 = time.time()
    r = requests.post(URL, json={
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.7, "seed": seed, "max_tokens": NPRED, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=600)
    r.raise_for_status()
    d = r.json()
    dt = time.time() - t0
    tok = d.get("usage", {}).get("completion_tokens", 0)
    return dt, tok


# warmup
one(99)

# --- single request ---
s_dt, s_tok = one(1)
print(f"[{LABEL}] single        : {s_dt:5.1f}s  {s_tok} tok  {s_tok/s_dt:6.1f} tok/s")

# --- two sequential ---
t0 = time.time()
a = one(1); b = one(2)
seq_dt = time.time() - t0
seq_tok = a[1] + b[1]
print(f"[{LABEL}] 2x sequential : {seq_dt:5.1f}s  {seq_tok} tok  {seq_tok/seq_dt:6.1f} tok/s (aggregate)")

# --- two concurrent ---
res = {}
def worker(seed):
    res[seed] = one(seed)
t0 = time.time()
ts = [threading.Thread(target=worker, args=(s,)) for s in (1, 2)]
[t.start() for t in ts]; [t.join() for t in ts]
par_dt = time.time() - t0
par_tok = res[1][1] + res[2][1]
print(f"[{LABEL}] 2x concurrent : {par_dt:5.1f}s  {par_tok} tok  {par_tok/par_dt:6.1f} tok/s (aggregate)")

speedup = seq_dt / par_dt if par_dt else 0
print(f"[{LABEL}] => concurrent is {speedup:.2f}x the throughput of sequential "
      f"({'batching helps' if speedup > 1.15 else 'no real parallel gain (serialized)'})")
