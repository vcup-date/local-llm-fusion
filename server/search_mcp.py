"""Minimal MCP stdio server providing web search + fetch, backed by DuckDuckGo (ddgs).

Claude Code's built-in WebSearch/WebFetch are Anthropic server-side tools that only work with
the real Anthropic API — with a custom/local backend they fail. This replaces them: register via
--mcp-config and the model gets `web_search` and `fetch_url` tools that actually work locally.

MCP stdio transport = newline-delimited JSON-RPC 2.0 on stdin/stdout.
"""
import json, sys

def _send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def _result(id, result):
    _send({"jsonrpc": "2.0", "id": id, "result": result})

def _error(id, code, msg):
    _send({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": msg}})

TOOLS = [
    {"name": "web_search",
     "description": "Search the web (DuckDuckGo) and return the top results with title, URL and snippet.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "search query"},
         "max_results": {"type": "integer", "description": "default 6"}},
         "required": ["query"]}},
    {"name": "fetch_url",
     "description": "Fetch a web page and return its main text content (cleaned).",
     "inputSchema": {"type": "object", "properties": {
         "url": {"type": "string"}}, "required": ["url"]}},
]

def do_web_search(args):
    from ddgs import DDGS
    q = args.get("query", "")
    n = int(args.get("max_results") or 6)
    out = []
    with DDGS() as d:
        for i, r in enumerate(d.text(q, max_results=n), 1):
            out.append(f"{i}. {r.get('title','')}\n   {r.get('href','')}\n   {r.get('body','')}")
    return "\n\n".join(out) if out else "(no results)"

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

def do_fetch_url(args):
    """Render the page with the user's own Chrome (headless --dump-dom: runs JS like a real
    browser), then extract clean text. Falls back to plain HTTP if Chrome fails."""
    import os, subprocess, requests, trafilatura
    url = args.get("url", "")
    html = None
    if os.path.exists(CHROME):
        try:
            html = subprocess.run(
                [CHROME, "--headless=new", "--disable-gpu", "--no-first-run",
                 "--dump-dom", "--virtual-time-budget=6000", url],
                capture_output=True, text=True, timeout=35).stdout or None
        except Exception:
            html = None
    if not html:
        try:
            html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12).text
        except Exception:
            return "(fetch failed)"
    txt = trafilatura.extract(html, include_comments=False, include_tables=True)
    return (txt or html)[:8000]

HANDLERS = {"web_search": do_web_search, "fetch_url": do_fetch_url}

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if method == "initialize":
            _result(mid, {"protocolVersion": params.get("protocolVersion", "2025-06-18"),
                          "capabilities": {"tools": {}},
                          "serverInfo": {"name": "fusion-search", "version": "1.0.0"}})
        elif method == "tools/list":
            _result(mid, {"tools": TOOLS})
        elif method == "tools/call":
            name = params.get("name"); args = params.get("arguments") or {}
            try:
                text = HANDLERS[name](args)
                _result(mid, {"content": [{"type": "text", "text": text}]})
            except KeyError:
                _error(mid, -32601, f"unknown tool: {name}")
            except Exception as e:
                _result(mid, {"content": [{"type": "text", "text": f"tool error: {e}"}], "isError": True})
        elif mid is not None:
            _result(mid, {})  # ack other requests (ping, etc.)
        # notifications (no id) need no response

if __name__ == "__main__":
    main()
