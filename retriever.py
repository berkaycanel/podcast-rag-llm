# retriever.py
# Vector search + precise SQL helpers (URLs included)
# -------------------------------------------------
# Requires:
#   pip install -U sentence-transformers psycopg[binary] python-dotenv
#
# .env:
#   DB_CONNECTION=postgresql://postgres:<pass>@<host>:5432/postgres?sslmode=require
# -------------------------------------------------

import os
import re
import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
DB_CONNECTION = os.environ["DB_CONNECTION"]
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Tables
META_TABLE  = "podcast_chunks"        # chunks built from metadata/summaries
TRANS_TABLE = "transcription_chunks"  # chunks built from raw transcripts

# ---------- embedding model (lazy) ----------
_model = None
def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

# ---------- postgres connection ----------
def _pg():
    """
    Create a Postgres connection with sane defaults.
    Also tune ivfflat/hnsw search params if available.
    """
    conn = psycopg.connect(
        DB_CONNECTION,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
        options='-c statement_timeout=0'
    )
    with conn.cursor() as cur:
        # IVFFlat: raise probes for better recall (if extension supports it)
        try:
            cur.execute("SET ivfflat.probes = 10;")
        except Exception:
            pass
        # HNSW: higher ef_search improves recall
        try:
            cur.execute("SET hnsw.ef_search = 64;")
        except Exception:
            pass
    return conn

# ============================================================
# Internal helpers for vector search
# ============================================================

def _search_table(query_text: str, table: str, top_k: int, episode_type: str | None):
    """
    ANN search over a chunks table (podcast_chunks or transcription_chunks).
    Returns rows with an internal `score` used for merging when source='both'.
    """
    m = _get_model()
    qvec = m.encode([query_text], normalize_embeddings=True)[0].tolist()

    where = "WHERE 1=1 "
    type_filter = False
    if episode_type:
        where += "AND e.episode_type_openai = %s "
        type_filter = True

    sql = f"""
    SELECT
      c.episode_id,
      c.chunk_index,
      c.content,
      e.title,
      e.podcast_url,
      e.sharing_url,
      e.audio_url,
      e.episode_type_openai,
      e.published_at,
      1 - (c.embedding <#> %s::vector) AS score
    FROM {table} c
    JOIN episodes e ON e.id = c.episode_id
    {where}
    ORDER BY c.embedding <#> %s::vector
    LIMIT %s
    """

    # Build args strictly in placeholder order:
    #   1) qvec (SELECT score)
    #   2) [episode_type]
    #   3) qvec (ORDER BY)
    #   4) top_k
    args = [qvec]
    if type_filter:
        args.append(episode_type)
    args.extend([qvec, top_k])

    with _pg() as conn, conn.cursor() as cur:
        cur.execute(sql, args)
        rows = cur.fetchall()

    out = []
    for (eid, cidx, content, title, purl, surl, aurl, etype, published, _score) in rows:
        out.append({
            "episode_id": eid,
            "chunk_index": cidx,
            "content": content,
            "title": title or "",
            "podcast_url": purl or "",
            "sharing_url": surl or "",
            "audio_url": aurl or "",
            "episode_type": etype or "",
            "published_at": published.isoformat() if hasattr(published, "isoformat") else (published or ""),
            "_score": float(_score) if _score is not None else None,
        })
    return out

def _search_table_constrained(
    query_text: str,
    table: str,
    keywords: list[str],
    top_k: int,
    episode_type: str | None
):
    """
    ANN search but require content ILIKE ANY of the keywords (ANY-of).
    """
    m = _get_model()
    qvec = m.encode([query_text], normalize_embeddings=True)[0].tolist()

    where = "WHERE 1=1 "
    args: list = []

    # optional type filter
    if episode_type and episode_type.strip():
        where += "AND coalesce(e.episode_type_openai,'') ILIKE %s "
        args.append(f"%{episode_type.strip()}%")

    # keyword constraints – build OR list
    patterns = [f"%{k}%" for k in (keywords or []) if k]
    if patterns:
        where += "AND (" + " OR ".join(["c.content ILIKE %s"] * len(patterns)) + ") "
        args.extend(patterns)

    sql = f"""
    SELECT
      c.episode_id,
      c.chunk_index,
      c.content,
      e.title,
      e.podcast_url,
      e.sharing_url,
      e.audio_url,
      e.episode_type_openai,
      e.published_at,
      1 - (c.embedding <#> %s::vector) AS score
    FROM {table} c
    JOIN episodes e ON e.id = c.episode_id
    {where}
    ORDER BY c.embedding <#> %s::vector
    LIMIT %s
    """

    # Final args in placeholder order:
    #   1) qvec (SELECT score)
    #   2) [episode_type?] + [patterns...]
    #   3) qvec (ORDER BY)
    #   4) top_k
    exec_args = [qvec] + args + [qvec, top_k]

    with _pg() as conn, conn.cursor() as cur:
        cur.execute(sql, exec_args)
        rows = cur.fetchall()

    out = []
    for (eid, cidx, content, title, purl, surl, aurl, etype, published, _score) in rows:
        out.append({
            "episode_id": eid,
            "chunk_index": cidx,
            "content": content,
            "title": title or "",
            "podcast_url": purl or "",
            "sharing_url": surl or "",
            "audio_url": aurl or "",
            "episode_type": etype or "",
            "published_at": published.isoformat() if hasattr(published, "isoformat") else (published or ""),
            "_score": float(_score) if _score is not None else None,
        })
    return out

def _merge_by_score(a: list[dict], b: list[dict], k: int) -> list[dict]:
    combined = [*a, *b]
    combined.sort(key=lambda r: (r.get("_score") or 0.0), reverse=True)
    for r in combined:
        r.pop("_score", None)
    return combined[:k]

# ============================================================
# Public vector search APIs
# ============================================================

def search_chunks(
    query_text: str,
    top_k: int = 10,
    episode_type: str | None = None,
    source: str = "meta",                   # 'meta' | 'transcript' | 'both'
):
    """
    ANN search returning content + episode metadata incl. URLs.
    """
    source = (source or "meta").lower()
    if source == "meta":
        return _search_table(query_text, META_TABLE, top_k, episode_type)
    if source == "transcript":
        return _search_table(query_text, TRANS_TABLE, top_k, episode_type)

    # both → split and merge by similarity score
    k_each = max(1, top_k // 2)
    a = _search_table(query_text, META_TABLE, k_each, episode_type)
    b = _search_table(query_text, TRANS_TABLE, k_each, episode_type)
    return _merge_by_score(a, b, top_k)

def search_chunks_constrained(
    query_text: str,
    keywords: list[str],
    top_k: int = 12,
    episode_type: str | None = None,
    source: str = "meta",                   # 'meta' | 'transcript' | 'both'
):
    """
    Vector search over chunks with ANY-keyword ILIKE constraint on the CHUNK content.
    """
    source = (source or "meta").lower()
    if source == "meta":
        return _search_table_constrained(query_text, META_TABLE, keywords, top_k, episode_type)
    if source == "transcript":
        return _search_table_constrained(query_text, TRANS_TABLE, keywords, top_k, episode_type)

    k_each = max(1, top_k // 2)
    a = _search_table_constrained(query_text, META_TABLE, keywords, k_each, episode_type)
    b = _search_table_constrained(query_text, TRANS_TABLE, keywords, k_each, episode_type)
    return _merge_by_score(a, b, top_k)

# ============================================================
# Exact SQL helpers
# ============================================================

def sql_guest_count(guest_substring: str) -> tuple[int, list[dict]]:
    """
    Case-insensitive substring match on episode_guests_openai.
    Returns (count, episodes[]) with URLs.
    """
    q = """
    SELECT id, title, podcast_url, sharing_url, audio_url, published_at, episode_type_openai
    FROM episodes
    WHERE episode_guests_openai ILIKE '%%' || %s || '%%'
    ORDER BY published_at DESC NULLS LAST
    """
    with _pg() as conn, conn.cursor() as cur:
        cur.execute(q, (guest_substring,))
        rows = cur.fetchall()

    eps = []
    for (eid, title, purl, surl, aurl, pub, etype) in rows:
        eps.append({
            "episode_id": eid,
            "title": title or "",
            "url": purl or surl or aurl or "",
            "published_at": pub.isoformat() if hasattr(pub, "isoformat") else (pub or ""),
            "episode_type": etype or ""
        })
    return len(eps), eps

def list_episode_types() -> list[str]:
    """Return distinct episode_type_openai values (sorted, non-empty)."""
    with _pg() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT TRIM(episode_type_openai)
            FROM episodes
            WHERE COALESCE(TRIM(episode_type_openai), '') <> ''
            ORDER BY 1
        """)
        rows = cur.fetchall()
    return [r[0] for r in rows]

# ---------- strict & accurate type+keyword matching ----------
def sql_type_mentions(
    episode_type: str | None,
    keyword: str,
    include_transcripts: bool = True,
    limit: int = 500,
) -> list[dict]:
    """
    Strict matching:
      - If `episode_type` is None/'All' → no type filter; otherwise partial ILIKE on episode_type_openai.
      - Tokenizes `keyword`. Quoted text is treated as a single phrase/token.
      - For EACH token-group, at least one variant must be present (dash/space/plural variants).
        ALL groups must match (AND).
      - Match can occur in metadata OR (optionally) in transcript chunks.
    """

    def _is_quoted(s: str) -> bool:
        s = s.strip()
        return len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', '“', '”'}

    def _strip_quotes(s: str) -> str:
        return s.strip().strip('"').strip("“").strip("”").strip()

    def _normalize_token(tok: str) -> str:
        return tok.strip().strip(",.;:!?()[]{}").lower()

    def _token_groups(q: str) -> list[list[str]]:
        q = q.strip()
        groups: list[list[str]] = []
        if _is_quoted(q):
            q0 = _strip_quotes(q)
            if q0:
                groups.append([q0])
            return groups
        raw = re.split(r"[\s/|]+", q)
        for t in raw:
            t = _normalize_token(t)
            if not t:
                continue
            alts = {t, t.replace("-", " "), t.replace("-", "")}
            if not t.endswith("s"):
                alts.add(t + "s")
            if not t.endswith("es"):
                alts.add(t + "es")
            if t.startswith("startup"):
                alts.update({"start-up", "startups"})
            alts = {a for a in alts if len(a) >= 2}
            if alts:
                groups.append(sorted(alts))
        return groups

    groups = _token_groups(keyword)
    if not groups:
        return []

    where_type = ""
    params: list = []
    if episode_type and episode_type.lower() != "all":
        where_type = "WHERE COALESCE(e.episode_type_openai,'') ILIKE %s"
        params.append(f"%{episode_type}%")

    # Build strict AND over token-groups:
    group_clauses: list[str] = []
    for variants in groups:
        meta_or = " OR ".join(["meta ILIKE %s"] * len(variants))
        if include_transcripts:
            tx_or  = " OR ".join(["t.content ILIKE %s"] * len(variants))
            tx_sub = f"EXISTS (SELECT 1 FROM transcription_chunks t WHERE t.episode_id = e.id AND ({tx_or}))"
            clause = f"(({meta_or}) OR {tx_sub})"
        else:
            clause = f"(({meta_or}))"
        group_clauses.append(clause)
        params.extend([f"%{v}%" for v in variants])
        if include_transcripts:
            params.extend([f"%{v}%" for v in variants])

    groups_and = " AND ".join(group_clauses)

    sql = f"""
    WITH docs AS (
      SELECT
        e.id,
        e.title,
        e.podcast_url,
        e.sharing_url,
        e.audio_url,
        e.published_at,
        COALESCE(e.title,'') || ' ' ||
        COALESCE(e.description,'') || ' ' ||
        COALESCE(e.episode_summary,'') || ' ' ||
        COALESCE(e.search_keywords,'') || ' ' ||
        COALESCE(e.episode_tags,'') || ' ' ||
        COALESCE(e.episode_guests_openai,'') AS meta
      FROM episodes e
      {where_type}
    )
    SELECT d.id, d.title, d.podcast_url, d.sharing_url, d.audio_url, d.published_at
    FROM docs d
    JOIN episodes e ON e.id = d.id
    WHERE {groups_and}
    ORDER BY d.published_at DESC NULLS LAST
    LIMIT %s
    """

    params.append(limit)

    with _pg() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    out = []
    for (eid, title, purl, surl, aurl, pub) in rows:
        out.append({
            "episode_id": eid,
            "title": title or "",
            "url": purl or surl or aurl or "",
            "published_at": pub.isoformat() if hasattr(pub, "isoformat") else (pub or "")
        })
    return out