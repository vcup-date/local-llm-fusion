"""Lightweight RAG for the fusion harness: query-gen -> DDG search -> fetch+extract
-> a compact, cited context block. No API key (DuckDuckGo).

Key robustness: hard per-fetch timeout (trafilatura.fetch_url has none), parallel
fetches, skip PDFs/failures, and always keep the instant DDG snippets as fallback.
"""
from __future__ import annotations
import concurrent.futures as cf
import requests, trafilatura
from ddgs import DDGS

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def gen_queries(chat_fn, problem, n=4):
    """chat_fn(messages)->text. Ask the model for focused search queries."""
    msg = [{"role": "user", "content":
            f"Research question:\n{problem}\n\nOutput exactly {n} concise web-search queries "
            "(one per line, no numbering, no quotes) that would find the specific facts, "
            "figures, named entities, and primary sources needed to answer it."}]
    try:
        txt = chat_fn(msg)
        qs = [l.strip(" -*0123456789.\t") for l in txt.splitlines() if l.strip()]
        qs = [q for q in qs if len(q) > 8][:n]
        return qs or [problem[:200]]
    except Exception:
        return [problem[:200]]


def _fetch(url, timeout=9, max_chars=2200):
    try:
        if url.lower().endswith((".pdf", ".xlsx", ".zip")):
            return None
        r = requests.get(url, headers=UA, timeout=timeout)
        if r.status_code != 200 or "html" not in r.headers.get("content-type", "html"):
            return None
        txt = trafilatura.extract(r.text, include_comments=False, include_tables=True)
        return txt[:max_chars] if txt else None
    except Exception:
        return None


def search(queries, per_query=5):
    """Return deduped list of {title, href, body} across queries."""
    seen, out = set(), []
    with DDGS() as d:
        for q in queries:
            try:
                for r in d.text(q, max_results=per_query):
                    u = r.get("href")
                    if u and u not in seen:
                        seen.add(u)
                        out.append({"title": r.get("title", ""), "href": u,
                                    "body": r.get("body", "")})
            except Exception:
                continue
    return out


def build_context(chat_fn, problem, n_queries=4, n_fetch=6, max_ctx_chars=16000, queries=None):
    """Return (context_string, sources_list, queries). If queries is given, skip
    query-gen (used to reuse the exact queries from a prior run for fair comparison)."""
    if not queries:
        queries = gen_queries(chat_fn, problem, n_queries)
    hits = search(queries, per_query=5)
    # fetch top n_fetch in parallel with hard timeout
    fetched = {}
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_fetch, h["href"]): h["href"] for h in hits[:n_fetch]}
        for fu in cf.as_completed(futs, timeout=40):
            try:
                fetched[futs[fu]] = fu.result()
            except Exception:
                fetched[futs[fu]] = None

    blocks, sources, n = [], [], 0
    for h in hits:
        n += 1
        tag = f"S{n}"
        body = fetched.get(h["href"]) or h.get("body", "")  # extracted text, else snippet
        if not body:
            continue
        blocks.append(f"[{tag}] {h['title']}\nURL: {h['href']}\n{body}")
        sources.append({"tag": tag, "title": h["title"], "url": h["href"]})
        if sum(len(b) for b in blocks) > max_ctx_chars:
            break
    ctx = "\n\n".join(blocks) if blocks else "(no sources retrieved)"
    return ctx, sources, queries
