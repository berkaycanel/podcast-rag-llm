# app.py
import os
import streamlit as st
from chat import ask
from retriever import (
    sql_guest_count,
    sql_type_mentions,
    list_episode_types,
)

st.set_page_config(page_title="🎧 Podcast RAG", layout="wide")
st.title("🎧 Podcast RAG (DE/EN)")

MODEL_DISPLAY = os.environ.get("LLM_NAME", "qwen2.5:14b-instruct-q4_K_M")

tab1, tab2 = st.tabs(["Chat (RAG)", "Precise tools (SQL)"])

# ---------------- Chat (RAG) ----------------
with tab1:
    q = st.text_input('Ask a question… z.B. „Welche Folgen erwähnen AI?“, „Wie oft war Gast X dabei?“')

    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        top_k = st.slider("Top-K", 3, 25, 10)
    with col3:
        source_ui = st.selectbox(
            "Search in",
            [
                "Summaries & metadata (fast)",
                "Full transcripts (slower)",
                "Both (merge)",
            ],
            index=0,
            help="Choose which chunks table(s) to search",
        )
        source_map = {
            "Summaries & metadata (fast)": "meta",
            "Full transcripts (slower)": "transcript",
            "Both (merge)": "both",
        }
        source = source_map[source_ui]

    # Episode type dropdown (+ All)
    types = ["All"] + list_episode_types()
    sel = st.selectbox("Filter episode_type_openai (optional)", options=types, index=0)
    ep_type = None if sel == "All" else sel

    if st.button("Ask", type="primary"):
        with st.spinner("Thinking…"):
            answer, hits = ask(
                q,
                top_k=top_k,
                episode_type=ep_type,
                source=source,
            )

        st.caption(f"Model: `{MODEL_DISPLAY}` • Top-K: {top_k} • Source: {source_ui}")
        st.markdown("### Answer")
        st.write(answer)
        st.markdown("### Sources")
        if not hits:
            st.info("No retrieved chunks.")
        for i, h in enumerate(hits, 1):
            url = h["podcast_url"] or h["sharing_url"] or h["audio_url"] or ""
            st.markdown(f"**[{i}] {h['title']}** — {h['episode_type']} • {h['published_at']}  \n{url}")
            with st.expander("Matched chunk"):
                st.write(h["content"])

# ---------------- Precise tools (SQL) ----------------
with tab2:
    st.subheader("Guest count (exact)")
    guest = st.text_input("Guest substring (case-insensitive)", key="guest")
    if st.button("Count guest appearances"):
        with st.spinner("Querying…"):
            cnt, eps = sql_guest_count(guest)
        st.success(f"Count: {cnt}")
        for e in eps[:50]:
            st.markdown(f"- **{e['title']}** — {e['published_at']}  \n{e['url']}")

    st.markdown("---")
    st.subheader("Type mentions (exact)")

    types2 = ["All"] + list_episode_types()
    colA, colB, colC = st.columns([1.2, 1.2, 0.6])
    with colA:
        t = st.selectbox("episode_type_openai", options=types2, index=0)
    with colB:
        kw = st.text_input("Keyword/phrase", value="Project A")
    with colC:
        incl_tx = st.checkbox("Search transcripts", value=True)

    if st.button("Find"):
        if not kw.strip():
            st.warning("Enter a keyword/phrase.")
        else:
            with st.spinner("Querying…"):
                rows = sql_type_mentions(
                    None if t == "All" else t,
                    kw,
                    include_transcripts=incl_tx,
                    limit=500,
                )
            st.info(f"Matches: {len(rows)}")
            for r in rows[:200]:
                st.markdown(f"- **{r['title']}** — {r['published_at']}  \n{r['url']}")