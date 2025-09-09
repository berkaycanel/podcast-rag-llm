# chat.py
# Grounded Q&A over your Supabase podcast DB using Ollama
# - Exact SQL routing for guest counts / type+keyword
# - Vector RAG over summary or transcript chunks (or both)
# - Smart keyword extraction (DE/EN + Series A/B/C… variants)
# - Numeric funding filter
# - Uses a single default model from env: LLM_NAME (default qwen2.5:14b-instruct-q4_K_M)

import os
import re
import requests
from typing import List, Tuple, Optional

from retriever import (
    search_chunks,
    search_chunks_constrained,
    sql_guest_count,
    sql_type_mentions,
)

# ------------ Config ------------
OLLAMA_URL    = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("LLM_NAME", "qwen2.5:14b-instruct-q4_K_M")

MAX_CHARS_PER_CHUNK = 900
MAX_CHUNKS   = 25
NUM_CTX      = 2048
NUM_PREDICT  = 256
TIMEOUT_S    = 1000

SYSTEM = """You are a precise bilingual (DE/EN) podcast research assistant.
RULES:
- Use ONLY the provided CONTEXT. Do not invent facts. If insufficient, say so.
- When listing episodes, ALWAYS include a working URL (podcast_url, else sharing_url, else audio_url).
- Prefer concise bullet points: Title — Date (ISO) — Type — URL.
- If a COUNT is requested, present the number clearly and then list matches (if few)."""

# ------------ Helpers ------------
def _ctx(hits: List[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        url = h.get("podcast_url") or h.get("sharing_url") or h.get("audio_url") or ""
        txt = (h.get("content") or "")[:MAX_CHARS_PER_CHUNK]
        parts.append(
            f"[{i}] Title: {h.get('title','')}\n"
            f"Type: {h.get('episode_type','')}\n"
            f"Date: {h.get('published_at','')}\n"
            f"URL: {url}\n"
            f"Chunk: {txt}"
        )
    return "\n---\n".join(parts)

_rx_count = re.compile(r"\b(how many|how often|count|wie oft|wie viele)\b", re.I)
_rx_guest = re.compile(r"\bguest|gast\b", re.I)

def _extract_guest(q: str):
    m = re.search(r'["“](.+?)["”]', q)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:guest|gast)\s+([a-zäöüßA-ZÄÖÜ\-'.]+(?:\s+[a-zäöüßA-ZÄÖÜ\-'.]+){0,3})", q, re.I)
    return m.group(1).strip() if m else None

# ---------- Keyword extraction ----------
_STOP = {
    "the","a","an","and","or","of","to","for","in","on","about","über","mit","und","oder",
    "was","what","welche","welcher","welches","wer","wie","oft","viele","are","is","ist",
    "that","die","das","der","den","dem","ein","eine","einen","einem","einer","im","am",
    "podcast","podcasts","episode","episodes","folgen","folge","mentions","erwähnen","erwähnt",
    "funding","investments","investment","investor","investoren","runde","round"
}
_SERIES_RX = re.compile(r"\bseries\s+([a-z])\b", re.I)
_RUNDE_RX  = re.compile(r"\b([abcde])\s*[- ]?\s*runde\b", re.I)
_ROMAN_RX  = re.compile(r"\bseries\s*(i|ii|iii|iv|v)\b", re.I)

def _normalize_token(tok: str) -> str:
    return tok.strip().strip(",.;:!?()[]{}“”\"'").lower()

def _variants_for_token(t: str) -> List[str]:
    alts = {t, t.replace("-", " "), t.replace("-", "")}
    if not t.endswith("s"):  alts.add(t + "s")
    if not t.endswith("es"): alts.add(t + "es")
    return [a for a in sorted(alts) if len(a) >= 2]

def _series_variants(q: str) -> List[str]:
    out = []
    m = _SERIES_RX.search(q)
    if m:
        letter = m.group(1).lower()
        out += [
            f"series {letter}", f"series-{letter}", f"series{letter}",
            f"{letter}-round", f"{letter} round",
            f"{letter}-runde", f"{letter} runde",
            f"{letter}-finanzierungsrunde", f"{letter} finanzierungsrunde",
        ]
    m2 = _ROMAN_RX.search(q)
    if m2:
        out += [f"series {m2.group(1)}", f"series-{m2.group(1)}", f"series{m2.group(1)}"]
    m3 = _RUNDE_RX.search(q)
    if m3:
        letter = m3.group(1).lower()
        out += [
            f"{letter} runde", f"{letter}-runde",
            f"{letter} finanzierungsrunde", f"{letter}-finanzierungsrunde",
            f"series {letter}", f"series-{letter}", f"series{letter}",
        ]
    seen, uniq = set(), []
    for a in out:
        a2 = _normalize_token(a)
        if a2 and a2 not in seen:
            seen.add(a2)
            uniq.append(a2)
    return uniq

def _keywords_from_query(q: str) -> List[str]:
    q0 = q.strip()
    if not q0:
        return []
    qm = re.findall(r'["“](.+?)["”]', q0)
    if qm:
        ph = [" ".join(w for w in _normalize_token(p).split() if w not in _STOP) for p in qm]
        ph = [p for p in ph if p]
        base = []
    else:
        ph = []
        base = [w for w in re.split(r"[\s/|]+", _normalize_token(q0)) if w and w not in _STOP]
    round_alts = _series_variants(q0)
    token_variants = []
    for t in base:
        token_variants += _variants_for_token(t)
    raw = ph + token_variants + round_alts
    seen, out = set(), []
    for r in raw:
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out[:20]

# ---------- Amount parsing / numeric filter ----------
_MORE_RX = re.compile(r"\b(over|more than|greater than|above|>\s*|über|mehr als|größer als)\b", re.I)
_ATLEAST_RX = re.compile(r"\b(at least|min(?:\.|imum)?|≥\s*|>=\s*|mindestens)\b", re.I)
_LESS_RX = re.compile(r"\b(under|less than|below|<\s*|weniger als|unter)\b", re.I)

_Q_AMOUNT_RX = re.compile(
    r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)\s*(milliarden|billionen|billion|mrd\.?|bn|b|millionen|million|mio\.?|mn|m|€|eur|euro)?',
    re.I
)

def _to_float(s: str) -> Optional[float]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        if len(s.split(",")[-1]) == 3:
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def _detect_amount_filter(q: str) -> Optional[Tuple[str, float]]:
    comp = None
    if _MORE_RX.search(q):
        comp = ">"
    elif _ATLEAST_RX.search(q):
        comp = ">="
    elif _LESS_RX.search(q):
        comp = "<"

    m = _Q_AMOUNT_RX.search(q)
    if not m:
        return None
    num_s, unit = m.group(1), (m.group(2) or "").lower().strip()
    val = _to_float(num_s)
    if val is None:
        return None

    if unit in {"milliarden", "billionen", "billion", "mrd.", "mrd", "bn", "b"}:
        eur = val * 1_000_000_000
    elif unit in {"millionen", "million", "mio.", "mio", "mn", "m"}:
        eur = val * 1_000_000
    elif unit in {"€", "eur", "euro"} or unit == "":
        eur = val
    else:
        eur = val

    if comp is None:
        comp = ">="
    return comp, float(eur)

_CHUNK_AMT_PATTERNS = [
    (re.compile(r'(\d{1,3}(?:[.\u202F,]\d{3})*(?:[.,]\d+)?)\s*(mio\.?|millionen|million)\b', re.I), 1_000_000),
    (re.compile(r'(\d{1,3}(?:[.\u202F,]\d{3})*(?:[.,]\d+)?)\s*(mrd\.?|milliarden|billionen|billion)\b', re.I), 1_000_000_000),
    (re.compile(r'[€]\s*(\d{1,3}(?:[.\u202F,]\d{3})*(?:[.,]\d+)?)\s*m\b', re.I), 1_000_000),
    (re.compile(r'(\d{1,3}(?:[.\u202F,]\d{3})*(?:[.,]\d+)?)\s*m\s*[€]?', re.I), 1_000_000),
    (re.compile(r'[€]\s*(\d{1,3}(?:[.\u202F,]\d{3})*(?:[.,]\d+)?)\b', re.I), 1),
    (re.compile(r'(\d{1,3}(?:[.\u202F,]\d{3})*(?:[.,]\d+)?)\s*(?:eur|euro)\b', re.I), 1),
]

def _extract_amount_eur(text: str) -> List[float]:
    vals: List[float] = []
    for rx, mul in _CHUNK_AMT_PATTERNS:
        for m in rx.finditer(text or ""):
            v = _to_float(m.group(1))
            if v is not None:
                vals.append(v * mul)
    return vals

def _passes(amounts: List[float], op: str, thr: float) -> bool:
    if not amounts:
        return False
    if op == ">":
        return any(a > thr for a in amounts)
    if op == ">=":
        return any(a >= thr for a in amounts)
    if op == "<":
        return any(a < thr for a in amounts)
    return False

def _fmt_amount_short(eur: float) -> str:
    if eur >= 1_000_000_000:
        return f"€{eur/1_000_000_000:.1f}B"
    if eur >= 1_000_000:
        return f"€{eur/1_000_000:.1f}M"
    return f"€{eur:,.0f}"

# ------------ Ollama call (fixed model) ------------
def _messages_to_prompt(messages: List[dict]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            lines.append(f"[SYSTEM]\n{m['content']}\n")
        elif role == "user":
            lines.append(f"[USER]\n{m['content']}\n")
        else:
            lines.append(f"[ASSISTANT]\n{m['content']}\n")
    lines.append("[ASSISTANT]\n")
    return "\n".join(lines)

def _ollama_generate(
    prompt: str,
    model_name: str,
    temperature: float = 0.1,
    num_ctx: int = NUM_CTX,
    num_predict: int = NUM_PREDICT,
    timeout_s: int = TIMEOUT_S,
) -> str:
    payload = {
        "model": model_name or DEFAULT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout_s)
        r.raise_for_status()
        return r.json().get("response", "")
    except requests.exceptions.ReadTimeout:
        payload["options"]["num_ctx"] = max(2048, num_ctx // 2)
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout_s)
        r.raise_for_status()
        return r.json().get("response", "")

def _llm_answer(question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",
         "content": f"Question:\n{question}\n\nCONTEXT:\n{context}\n\nAnswer ONLY from the context, in the language of the question."}
    ]
    prompt = _messages_to_prompt(messages)
    return _ollama_generate(prompt)

# ------------ Main entry for Streamlit ------------
def ask(
    question: str,
    top_k: int = 10,
    episode_type: str | None = None,
    source: str = "meta",                 # 'meta' | 'transcript' | 'both'
):
    """
    Returns (answer_text, hits_for_ui)
    """
    q = (question or "").strip()
    if not q:
        return "Bitte eine Frage eingeben. / Please enter a question.", []

    k_ui = max(1, int(top_k or 1))
    top_k = min(MAX_CHUNKS, k_ui)

    # 1) Guest COUNT intent → exact SQL
    if _rx_count.search(q) and _rx_guest.search(q):
        guest = _extract_guest(q) or q
        count, eps = sql_guest_count(guest)
        if count == 0:
            return (f"Keine Treffer für Gast „{guest}“ gefunden. / No matches for guest “{guest}”.", [])
        bullets = [f"- **{e['title']}** — {e['published_at']} — {e['episode_type']}  \n{e['url']}" for e in eps[:50]]
        text = f"**Count: {count}**\n\n" + "\n".join(bullets)
        hits = [{
            "episode_id": e["episode_id"], "chunk_index": 0, "content": f"Guest match: {guest}",
            "title": e["title"], "podcast_url": e["url"], "sharing_url": "", "audio_url": "",
            "episode_type": e["episode_type"], "published_at": e["published_at"],
        } for e in eps[:10]]
        return text, hits

    # 2) Explicit type filter + keyword → exact SQL first
    if episode_type and q and episode_type.lower() != "all":
        rows = sql_type_mentions(episode_type, q, include_transcripts=True, limit=500)
        if rows:
            bullets = [f"- **{r['title']}** — {r['published_at']}  \n{r['url']}" for r in rows[:100]]
            return f"Matches: {len(rows)}\n\n" + "\n".join(bullets), []

    # 3) RAG search (constrained → widen)
    kw = _keywords_from_query(q)
    RESULTS_FLOOR = max(5, top_k // 2)

    def _merge_unique(primary: list[dict], extra: list[dict], limit: int) -> list[dict]:
        seen = set((h["episode_id"], h["chunk_index"]) for h in primary)
        for h in extra:
            key = (h["episode_id"], h["chunk_index"])
            if key not in seen:
                primary.append(h)
                seen.add(key)
                if len(primary) >= limit:
                    break
        return primary[:limit]

    etype = None if (episode_type or "").lower()=="all" else episode_type

    if kw:
        hits = search_chunks_constrained(q, keywords=kw, top_k=top_k, episode_type=etype, source=source)
        if len(hits) < RESULTS_FLOOR:
            more = search_chunks(q, top_k=top_k, episode_type=etype, source=source)
            hits = _merge_unique(hits, more, top_k)
    else:
        hits = search_chunks(q, top_k=top_k, episode_type=etype, source=source)

    # 4) Numeric gating
    num_filter = _detect_amount_filter(q)
    if num_filter:
        op, thr_eur = num_filter

        def enrich(h):
            txt = f"{h.get('title','')}\n{h.get('content','')}"
            vals = _extract_amount_eur(txt)
            return vals

        filtered = []
        for h in hits:
            vals = enrich(h)
            if _passes(vals, op, thr_eur):
                h["_best_amount_eur"] = max(vals) if vals else None
                filtered.append(h)

        if not filtered:
            more = search_chunks(q, top_k=top_k * 2, episode_type=etype, source=source)
            for h in more:
                vals = enrich(h)
                if _passes(vals, op, thr_eur):
                    h["_best_amount_eur"] = max(vals) if vals else None
                    filtered.append(h)

        if filtered:
            filtered = filtered[:top_k]
            bullets = []
            for h in filtered:
                url = h.get("podcast_url") or h.get("sharing_url") or h.get("audio_url") or ""
                amt = h.get("_best_amount_eur")
                amt_s = f" — {_fmt_amount_short(amt)}" if isinstance(amt, (int, float)) else ""
                bullets.append(f"- **{h.get('title','')}** — {h.get('published_at','')} — {h.get('episode_type','')}{amt_s}  \n{url}")
            header = f"Episodes matching {op} { _fmt_amount_short(thr_eur) }:"
            return header + "\n\n" + "\n".join(bullets), filtered

    # 5) Normal LLM answer from context
    hits = hits[:top_k]
    context = _ctx(hits)
    answer = _llm_answer(q, context if context.strip() else "NO CONTEXT")
    return answer, hits
