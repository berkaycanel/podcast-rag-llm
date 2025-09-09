# ingest.py — pure psycopg (robust inserts + retries; multi-CSV support)
# ---------------------------------------------------------
# 1) Upsert rows into episodes (BIGINT id, plus original_id text)
# 2) Chunk + embed + insert into podcast_chunks (summary/description/metadata)
# 3) Chunk + embed + insert into transcription_chunks (transcription only)
#
# Usage:
#   python3 ingest.py Second1300_DATA_EVERYTHING.csv
#   # or multiple:
#   python3 ingest.py 1000_DATA_EVERYTHING_cleaned.csv Second1300_DATA_EVERYTHING.csv
#
# Requirements:
#   pip install -U pandas numpy tqdm python-dotenv "psycopg[binary]" sentence-transformers
#
# .env must contain:
#   DB_CONNECTION=postgresql://postgres:<pass>@<host>:5432/postgres?sslmode=require
# ---------------------------------------------------------

import os, re, math, uuid, time, sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import psycopg
from psycopg import OperationalError
from sentence_transformers import SentenceTransformer

# ----------------------- Config --------------------------
DEFAULT_CSV = "1000_DATA_EVERYTHING_cleaned.csv"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 384-dim, multilingual

# chunking
CHARS_PER_CHUNK_META = 900
CHARS_OVERLAP_META   = 200

CHARS_PER_CHUNK_TXT  = 900
CHARS_OVERLAP_TXT    = 200

EPISODE_UPSERT_BATCH = 200

# Smaller vector batches to avoid SSL frame issues
CHUNK_INSERT_BATCH   = 40
MAX_RETRIES          = 5

# Target tables
META_TABLE = "podcast_chunks"
TRANS_TABLE = "transcription_chunks"

# ----------------------- Env -----------------------------
load_dotenv()
DB_CONNECTION = os.environ["DB_CONNECTION"]

def pg_connect():
    return psycopg.connect(
        DB_CONNECTION,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
        options='-c statement_timeout=0'
    )

# -------------------- Helpers ----------------------------
INT63_MAX = (1 << 63) - 1

def to_bigint_id(v) -> int | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        i = int(v)
        return i & INT63_MAX
    except Exception:
        pass
    try:
        u = uuid.UUID(str(v))
    except Exception:
        u = uuid.uuid5(uuid.NAMESPACE_DNS, str(v))
    return u.int & INT63_MAX

def to_int_or_none(x):
    if x is None:
        return None
    try:
        if isinstance(x, str):
            x = x.replace(",", "").replace("_", "").strip()
            if x == "":
                return None
        return int(float(x))
    except Exception:
        return None

def clamp_bigint(x):
    v = to_int_or_none(x)
    if v is None:
        return None
    if v > INT63_MAX: return INT63_MAX
    if v < -INT63_MAX - 1: return -INT63_MAX - 1
    return v

def as_text(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    if isinstance(x, (list, dict)):
        return str(x)
    return str(x)

def normalize_guests(val):
    if val is None or (isinstance(val, float) and math.isnan(val)): return ""
    items = val if isinstance(val, list) else re.split(r"[;,/|]", str(val))
    out = []
    for g in items:
        g = re.sub(r"\(.*?\)", "", g)
        g = re.sub(r"\s+", " ", g).strip(" .,-")
        if g: out.append(g)
    return ", ".join(out)

def build_doc_meta(row: pd.Series) -> str:
    guests_val = row.get("episode_guests_openai") or row.get("episode_guests_openAI")
    parts = [
        f"Title: {as_text(row.get('title'))}",
        f"Type: {as_text(row.get('episode_type_openai'))}",
        f"Guests: {normalize_guests(guests_val)}",
        f"Tags: {as_text(row.get('episode_tags'))}",
        f"Search keywords: {as_text(row.get('search_keywords'))}",
        f"Description: {as_text(row.get('description'))}",
        f"Summary: {as_text(row.get('episode_summary'))}",
        f"Podcast URL: {as_text(row.get('podcast_url'))}",
        f"Sharing URL: {as_text(row.get('sharing_url'))}",
        f"Audio URL: {as_text(row.get('audio_url'))}",
    ]
    return "\n".join([p for p in parts if p])

def build_doc_transcript(row: pd.Series) -> str:
    t = as_text(row.get("transcription"))
    if not t.strip(): return ""
    title = as_text(row.get("title"))
    return f"Title: {title}\nTranscript:\n{t}" if title else t

def sentence_chunk(text: str, target_chars=900, overlap=200) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+\n?|\n{2,}", text)
    buf, chunks = "", []
    for s in sents:
        if not s: continue
        if len(buf) + len(s) + 1 <= target_chars:
            buf = (buf + " " + s).strip()
        else:
            if buf: chunks.append(buf)
            buf = s.strip()
    if buf: chunks.append(buf)
    if overlap > 0 and len(chunks) > 1:
        out = []
        for i,c in enumerate(chunks):
            if i == 0: out.append(c); continue
            out.append((chunks[i-1][-overlap:] + " " + c).strip())
        chunks = out
    return [c for c in chunks if c.strip()]

def ts_to_pg(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return None
    try:
        dt = pd.to_datetime(value, errors="coerce", utc=True)
        return None if pd.isna(dt) else dt.isoformat()
    except Exception:
        return None

def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notnull(df), None)
    return df

# ---------------- Episodes upsert (psycopg) ---------------
def upsert_episodes_psycopg(df: pd.DataFrame):
    cols = ['id','number','title','status','published_at','duration_in_seconds','total_downloads',
            'description','audio_url','sharing_url','cover_artwork_url','episode_summary',
            'episode_contributors','podcast_url','search_keywords','episode_type_openai',
            'episode_guests_openai','episode_anlass','episode_tags','podcast_host',
            'transcription','original_id']
    for c in cols:
        if c not in df.columns: df[c] = None

    with pg_connect() as conn, conn.cursor() as cur:
        batch: List[Tuple] = []
        for _, r in tqdm(df.iterrows(), total=len(df), desc="Upserting episodes (psycopg)"):
            original = r.get("id")
            bid = to_bigint_id(original)
            if bid is None: continue

            num   = clamp_bigint(r["number"])
            dur   = clamp_bigint(r["duration_in_seconds"])
            downs = clamp_bigint(r["total_downloads"])

            row = (
                bid,
                num, r["title"], r["status"],
                ts_to_pg(r["published_at"]),
                dur, downs,
                r["description"], r["audio_url"], r["sharing_url"], r["cover_artwork_url"],
                r["episode_summary"], r["episode_contributors"], r["podcast_url"],
                r["search_keywords"], r["episode_type_openai"], r["episode_guests_openai"],
                r["episode_anlass"], r["episode_tags"], r["podcast_host"],
                r["transcription"], str(original) if original is not None else None
            )
            batch.append(row)
            if len(batch) >= EPISODE_UPSERT_BATCH:
                _exec_upsert_episodes(cur, batch); batch = []
        if batch:
            _exec_upsert_episodes(cur, batch)
        conn.commit()

def _exec_upsert_episodes(cur, rows: List[Tuple]):
    cur.executemany("""
        insert into episodes (
          id, number, title, status, published_at, duration_in_seconds, total_downloads,
          description, audio_url, sharing_url, cover_artwork_url, episode_summary,
          episode_contributors, podcast_url, search_keywords, episode_type_openai,
          episode_guests_openai, episode_anlass, episode_tags, podcast_host,
          transcription, original_id
        ) values (
          %s,%s,%s,%s,%s,%s,%s,
          %s,%s,%s,%s,%s,
          %s,%s,%s,%s,
          %s,%s,%s,%s,
          %s,%s
        )
        on conflict (id) do update set
          number=excluded.number,
          title=excluded.title,
          status=excluded.status,
          published_at=excluded.published_at,
          duration_in_seconds=excluded.duration_in_seconds,
          total_downloads=excluded.total_downloads,
          description=excluded.description,
          audio_url=excluded.audio_url,
          sharing_url=excluded.sharing_url,
          cover_artwork_url=excluded.cover_artwork_url,
          episode_summary=excluded.episode_summary,
          episode_contributors=excluded.episode_contributors,
          podcast_url=excluded.podcast_url,
          search_keywords=excluded.search_keywords,
          episode_type_openai=excluded.episode_type_openai,
          episode_guests_openai=excluded.episode_guests_openai,
          episode_anlass=excluded.episode_anlass,
          episode_tags=excluded.episode_tags,
          podcast_host=excluded.podcast_host,
          transcription=excluded.transcription,
          original_id=excluded.original_id
    """, rows)

# --------- Robust inserts (generic) -----------------------
def _insert_chunks_with_retry(primary_conn, rows: List[Tuple], table: str):
    SQL = f"""
        insert into {table}
          (episode_id, chunk_index, content, embedding, episode_type, title, published_at)
        values
          (%s,%s,%s,%s::vector,%s,%s,%s)
    """
    try:
        with primary_conn.cursor() as cur:
            cur.executemany(SQL, rows)
        primary_conn.commit()
        return
    except OperationalError:
        pass
    except Exception:
        try: primary_conn.rollback()
        except Exception: pass
        try:
            with primary_conn.cursor() as cur:
                for r in rows:
                    try: cur.execute(SQL, r)
                    except Exception: pass
            primary_conn.commit()
            return
        except Exception:
            pass

    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            with pg_connect() as conn:
                with conn.cursor() as cur:
                    cur.executemany(SQL, rows)
                conn.commit()
                return
        except OperationalError:
            time.sleep(min(2 ** attempt, 10))
            attempt += 1
        except Exception:
            try:
                with pg_connect() as conn:
                    with conn.cursor() as cur:
                        for r in rows:
                            try: cur.execute(SQL, r)
                            except Exception: pass
                    conn.commit()
                return
            except Exception:
                time.sleep(min(2 ** attempt, 10))
                attempt += 1

    for r in rows:
        try:
            with pg_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(SQL, r)
                conn.commit()
        except Exception:
            pass

def _insert_chunks(conn, rows: List[Tuple], table: str):
    _insert_chunks_with_retry(conn, rows, table)

# ----------------- Embedding + chunks ---------------------
def embed_and_write(df: pd.DataFrame, model: SentenceTransformer):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df["id"] = df["id"].apply(to_bigint_id)
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)

    with pg_connect() as conn:
        with conn.cursor() as cur:
            ids = [int(x) for x in df["id"].dropna().tolist()]
            if ids:
                # delete existing chunks for these ids so re-runs are idempotent
                cur.execute(f"delete from {META_TABLE}  where episode_id = any(%s)", (ids,))
                cur.execute(f"delete from {TRANS_TABLE} where episode_id = any(%s)", (ids,))
                conn.commit()

        batch_meta: List[Tuple] = []
        batch_trans: List[Tuple] = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking & embedding"):
            if row["id"] is None or (isinstance(row["id"], float) and math.isnan(row["id"])): continue
            episode_id = int(row["id"])

            meta_type  = as_text(row.get("episode_type_openai"))
            meta_title = as_text(row.get("title"))
            meta_pub   = ts_to_pg(row.get("published_at"))

            # META
            meta_doc = build_doc_meta(row)
            if meta_doc.strip():
                meta_chunks = sentence_chunk(meta_doc, CHARS_PER_CHUNK_META, CHARS_OVERLAP_META)
                if meta_chunks:
                    meta_vecs = model.encode(meta_chunks, normalize_embeddings=True)
                    for idx, (content, vec) in enumerate(zip(meta_chunks, meta_vecs)):
                        batch_meta.append((episode_id, idx, content, vec.tolist(), meta_type, meta_title, meta_pub))
                    if len(batch_meta) >= CHUNK_INSERT_BATCH:
                        _insert_chunks(conn, batch_meta, META_TABLE); batch_meta = []

            # TRANSCRIPTION
            tr_doc = build_doc_transcript(row)
            if tr_doc.strip():
                tr_chunks = sentence_chunk(tr_doc, CHARS_PER_CHUNK_TXT, CHARS_OVERLAP_TXT)
                if tr_chunks:
                    tr_vecs = model.encode(tr_chunks, normalize_embeddings=True)
                    for idx, (content, vec) in enumerate(zip(tr_chunks, tr_vecs)):
                        batch_trans.append((episode_id, idx, content, vec.tolist(), meta_type, meta_title, meta_pub))
                    if len(batch_trans) >= CHUNK_INSERT_BATCH:
                        _insert_chunks(conn, batch_trans, TRANS_TABLE); batch_trans = []

        if batch_meta:  _insert_chunks(conn, batch_meta, META_TABLE)
        if batch_trans: _insert_chunks(conn, batch_trans, TRANS_TABLE)

    with pg_connect() as c2, c2.cursor() as cur2:
        cur2.execute(f"analyze {META_TABLE};")
        cur2.execute(f"analyze {TRANS_TABLE};")
        c2.commit()

# ----------------- Counts (for visibility) ----------------
def _print_counts(label: str):
    try:
        with pg_connect() as c, c.cursor() as cur:
            cur.execute("""
                SELECT 'episodes' AS t, COUNT(*) FROM episodes
                UNION ALL
                SELECT 'podcast_chunks', COUNT(*) FROM podcast_chunks
                UNION ALL
                SELECT 'transcription_chunks', COUNT(*) FROM transcription_chunks
                ORDER BY 1;
            """)
            rows = cur.fetchall()
        print(f"\n[{label}] table counts:")
        for t, n in rows:
            print(f"  {t:20s} {n}")
    except Exception as e:
        print(f"[{label}] could not fetch counts: {e}")

# ----------------------- Main -----------------------------
if __name__ == "__main__":
    files = sys.argv[1:] or [DEFAULT_CSV]

    # Load embed model once
    print("Loading embedding model…")
    model = SentenceTransformer(EMBED_MODEL)

    _print_counts("BEFORE")

    for path in files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")

        print(f"\n📥 Processing CSV: {path}")
        df = pd.read_csv(path)
        df = sanitize_df(df)

        print(f"⬆️ Upserting {len(df)} rows into episodes (psycopg)…")
        upsert_episodes_psycopg(df)

        print("🧩 Chunking + embedding + writing vectors…")
        embed_and_write(df, model)

    _print_counts("AFTER")
    print("\n✅ Done. Data is in `episodes`, `podcast_chunks`, and `transcription_chunks`.")