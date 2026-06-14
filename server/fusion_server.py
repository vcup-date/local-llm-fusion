"""OpenAI-compatible proxy that IS fusion-35b x2 — a uniform black box.

EVERY /v1/chat/completions request, regardless of tools, goes through the same path:
  - 2 panelists run concurrently (temp 0.7, seeds 1/2), each given the SAME request + any tools
  - 1 judge call (given the same conversation + both drafts + the tools) decides the single best
    next step and produces it natively
  - whatever the judge emits — a TOOL CALL or a TEXT answer — is relayed verbatim (stream or not)

So tool calling works exactly like a normal model: the judge emits the tool call, we pass it through.

Hermes (or any OpenAI client) points at this as a single model: "qwopus-35b-fusion".

Env: BACKEND (default http://127.0.0.1:9301), PORT (9300), MODEL_ID (qwopus-35b-fusion),
     JUDGE_THINK (1), JUDGE_THINK_BUDGET (4096).
"""
from __future__ import annotations
import json, os, re, time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import requests

BACKEND = os.environ.get("BACKEND", "http://127.0.0.1:9301")          # fusion backend (parallel, no MTP)
SOLO_BACKEND = os.environ.get("SOLO_BACKEND", BACKEND)                # fast tier backend (MTP); falls back to BACKEND
PORT = int(os.environ.get("PORT", "9300"))
MODEL_ID = os.environ.get("MODEL_ID", "qwopus-35b-fusion")
PANEL_TEMP, JUDGE_TEMP, SOLO_TEMP = 0.7, 0.3, 0.3
SOLO_MODEL = "qwopus-35b-fast"   # single-call fast tier (no fusion) for Claude Code's haiku/background
JUDGE_THINK = os.environ.get("JUDGE_THINK", "1") not in ("0", "false", "")  # judge reasons by default
JUDGE_THINK_BUDGET = int(os.environ.get("JUDGE_THINK_BUDGET", "4096"))      # extra tokens for thinking

# Appended as a final user turn (NOT a system swap) so the judge shares the conversation prefix with
# the panelists -> llama.cpp prompt-cache reuse. Covers both tool calls and text answers.
JUDGE_INSTRUCTION = (
    "Two independent assistants each responded to my latest message, blind to each other. Their "
    "drafts are shown below; a draft may be a written answer and/or a proposed tool call. Decide the "
    "single best next step and produce it yourself: if a tool should be called, call it; otherwise "
    "write the best final answer. Keep what is correct, drop what is wrong, and merge the best of "
    "both. Output only the result — no commentary about the drafts."
)


def _backend_chat(messages, temperature, seed, max_tokens, stream=False, think=False,
                  tools=None, tool_choice=None, base=None):
    payload = {
        "messages": messages, "temperature": temperature, "seed": seed,
        "max_tokens": max_tokens, "stream": stream,
        "chat_template_kwargs": {"enable_thinking": think},
    }
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    r = requests.post(f"{base or BACKEND}/v1/chat/completions", json=payload, stream=stream, timeout=3600)
    r.raise_for_status()
    return r


def _strip_think(t):
    return re.sub(r"<think>.*?</think>", "", t or "", flags=re.DOTALL).strip()


def _msg(resp):
    """Full assistant message dict (content + tool_calls) from a non-stream backend response."""
    return resp.json()["choices"][0]["message"]


def _render_draft(m):
    """Render a panelist message (text and/or proposed tool calls) for the judge to cross-examine."""
    parts = []
    c = _strip_think(m.get("content") or "")
    if c:
        parts.append(c)
    for tc in (m.get("tool_calls") or []):
        fn = tc.get("function", {}) or {}
        parts.append(f"[proposed tool call] {fn.get('name')}({fn.get('arguments')})")
    return "\n".join(parts) or "(no output)"


def _panelists_msgs(messages, max_tokens, tools, tool_choice):
    """Two panelists concurrently (backend --parallel>=2 batches them), each with the tools."""
    with ThreadPoolExecutor(max_workers=2) as ex:
        fa = ex.submit(_backend_chat, messages, PANEL_TEMP, 1, max_tokens, False, False, tools, tool_choice)
        fb = ex.submit(_backend_chat, messages, PANEL_TEMP, 2, max_tokens, False, False, tools, tool_choice)
        return _msg(fa.result()), _msg(fb.result())


def _judge_messages(messages, mA, mB):
    graft = (f"{JUDGE_INSTRUCTION}\n\n--- Draft A ---\n{_render_draft(mA)}\n\n"
             f"--- Draft B ---\n{_render_draft(mB)}")
    return list(messages) + [{"role": "user", "content": graft}]


def _judge_max(max_tokens):
    return max_tokens + (JUDGE_THINK_BUDGET if JUDGE_THINK else 0)


def _oai_from_msg(msg, t0):
    """Relay a backend assistant message verbatim (tool_calls and/or content)."""
    tc = msg.get("tool_calls")
    m = {"role": "assistant", "content": _strip_think(msg.get("content") or "") or None}
    if tc:
        m["tool_calls"] = tc
    return {"id": f"fusion-{int(t0)}", "object": "chat.completion", "created": int(t0),
            "model": MODEL_ID, "choices": [{"index": 0,
            "finish_reason": "tool_calls" if tc else "stop", "message": m}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/v1/models") or self.path == "/":
            if "models" in self.path:
                self._json(200, {"object": "list", "data": [
                    {"id": MODEL_ID, "object": "model", "owned_by": "local-fusion"},
                    {"id": SOLO_MODEL, "object": "model", "owned_by": "local-fusion"}]})
            else:
                self._json(200, {"status": "ok", "backend": BACKEND})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        messages = req.get("messages", [])
        max_tokens = int(req.get("max_tokens") or 4096)
        stream = bool(req.get("stream"))
        tools = req.get("tools")
        tool_choice = req.get("tool_choice")
        model = req.get("model", "") or ""
        t0 = time.time()
        try:
            if "fast" in model or "solo" in model:
                # FAST tier (Claude Code haiku/background): single call on the MTP backend, no fusion.
                if stream:
                    self._stream_backend(messages, max_tokens, t0, SOLO_TEMP, 1, False,
                                         tools, tool_choice, base=SOLO_BACKEND)
                else:
                    jmsg = _msg(_backend_chat(messages, SOLO_TEMP, 1, max_tokens, think=False,
                                              tools=tools, tool_choice=tool_choice, base=SOLO_BACKEND))
                    self._json(200, _oai_from_msg(jmsg, t0))
                return
            # FUSION tier: 2 panelists (with tools) -> judge (with tools) -> relay judge output.
            mA, mB = _panelists_msgs(messages, max_tokens, tools, tool_choice)
            jmsgs = _judge_messages(messages, mA, mB)
            if stream:
                self._stream_backend(jmsgs, _judge_max(max_tokens), t0, JUDGE_TEMP, 7, JUDGE_THINK,
                                     tools, tool_choice)
            else:
                jmsg = _msg(_backend_chat(jmsgs, JUDGE_TEMP, 7, _judge_max(max_tokens),
                                          think=JUDGE_THINK, tools=tools, tool_choice=tool_choice))
                self._json(200, _oai_from_msg(jmsg, t0))
        except Exception as e:
            self._json(500, {"error": {"message": f"fusion proxy error: {e}"}})

    def _stream_backend(self, messages, max_tokens, t0, temp, seed, think, tools=None,
                        tool_choice=None, base=None):
        # Connection: close so HTTP/1.1 clients know the SSE body ends on socket close.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "close")
        self.end_headers()
        cid = f"fusion-{int(t0)}"
        def chunk(delta, finish=None):
            o = {"id": cid, "object": "chat.completion.chunk", "created": int(t0),
                 "model": MODEL_ID, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self.wfile.write(f"data: {json.dumps(o)}\n\n".encode()); self.wfile.flush()
        try:
            chunk({"role": "assistant"})
            r = _backend_chat(messages, temp, seed, max_tokens, stream=True,
                              think=think, tools=tools, tool_choice=tool_choice, base=base)
            finish = "stop"
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload.strip() == b"[DONE]":
                    break
                try:
                    ch = json.loads(payload)["choices"][0]
                except Exception:
                    continue
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                delta = ch.get("delta") or {}
                out = {}
                # relay the final answer AND any tool-call deltas; hide judge reasoning_content
                if delta.get("content"):
                    out["content"] = delta["content"]
                if delta.get("tool_calls"):
                    out["tool_calls"] = delta["tool_calls"]
                if out:
                    chunk(out)
            chunk({}, finish=finish)
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client (e.g. Hermes) disconnected mid-stream


if __name__ == "__main__":
    print(f"[fusion-proxy] {MODEL_ID} on :{PORT} -> backend {BACKEND}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
