import streamlit as st
import requests
import time

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = "http://localhost:8000"   # ← change if your FastAPI runs elsewhere

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* ---------- palette ---------- */
:root {
    --bg:        #0d0f14;
    --surface:   #161921;
    --border:    #252a35;
    --accent:    #5af0b0;
    --accent2:   #8b6dff;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --danger:    #ff5f6d;
}

/* dark background */
.stApp { background: var(--bg); color: var(--text); }

/* sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* headings */
h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

/* chat bubbles */
.bubble-user {
    background: linear-gradient(135deg, #1e2130, #252a3a);
    border: 1px solid var(--accent2);
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px;
    margin: 8px 0 8px 60px;
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text);
}
.bubble-bot {
    background: linear-gradient(135deg, #161c24, #1a2030);
    border: 1px solid var(--accent);
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px;
    margin: 8px 60px 8px 0;
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text);
}
.label-user { text-align: right; font-size: 0.72rem; color: var(--accent2); font-family: 'Space Mono', monospace; margin-right: 6px; }
.label-bot  { font-size: 0.72rem; color: var(--accent); font-family: 'Space Mono', monospace; margin-left: 6px; }

/* file tag */
.file-tag {
    display: inline-block;
    background: #1a2030;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    color: var(--muted);
    margin: 2px 3px;
}
.file-tag:hover { border-color: var(--accent); color: var(--accent); cursor: default; }

/* status badge */
.badge-ok   { background: #0e2f1e; color: var(--accent); border: 1px solid var(--accent); border-radius: 20px; padding: 2px 12px; font-size: 0.78rem; font-family:'Space Mono',monospace; }
.badge-err  { background: #2f0e0e; color: var(--danger); border: 1px solid var(--danger); border-radius: 20px; padding: 2px 12px; font-size: 0.78rem; font-family:'Space Mono',monospace; }

/* buttons */
.stButton > button {
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    padding: 6px 18px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: var(--accent);
    color: #0d0f14;
}

/* delete button red */
.stButton.delete > button { border-color: var(--danger); color: var(--danger); }
.stButton.delete > button:hover { background: var(--danger); color: #0d0f14; }

/* text / select inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
}

/* file uploader */
.stFileUploader { border: 1px dashed var(--border); border-radius: 10px; padding: 8px; }

/* divider */
hr { border-color: var(--border); }

/* scrollable chat pane */
.chat-scroll {
    max-height: 62vh;
    overflow-y: auto;
    padding-right: 6px;
}
.chat-scroll::-webkit-scrollbar { width: 4px; }
.chat-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

/* pulse dot */
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--accent); animation: pulse 1.2s infinite; margin-right:5px; }

/* source cards */
.sources-wrap {
    margin: 4px 60px 14px 0;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.src-card {
    background: #10141c;
    border: 1px solid #252a35;
    border-left: 3px solid var(--accent2);
    border-radius: 8px;
    padding: 10px 14px;
    flex: 1 1 300px;
    min-width: 220px;
    max-width: 480px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    transition: border-color 0.2s;
}
.src-card:hover { border-left-color: var(--accent); }
.src-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    gap: 8px;
    flex-wrap: wrap;
}
.src-rank {
    background: var(--accent2);
    color: #0d0f14;
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 0.7rem;
    font-weight: 700;
    white-space: nowrap;
}
.src-file {
    color: var(--accent);
    font-size: 0.72rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 180px;
}
.src-score {
    color: var(--muted);
    font-size: 0.7rem;
    white-space: nowrap;
}
.src-domain {
    display: inline-block;
    background: #1a2030;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 0.68rem;
    color: var(--muted);
    margin-bottom: 6px;
}
.src-text {
    color: #9aa3b5;
    font-size: 0.75rem;
    line-height: 1.5;
    font-family: 'Syne', sans-serif;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.sources-label {
    font-size: 0.7rem;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    margin: 0 60px 4px 2px;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    # each item: {"role": "user"|"bot", "text": "...", "sources": [...]}
    st.session_state.messages = []
if "files_cache" not in st.session_state:
    st.session_state.files_cache = {}


# ── Helpers ───────────────────────────────────────────────────────────────────
def api(method, path, **kwargs):
    try:
        r = getattr(requests, method)(f"{BASE_URL}{path}", timeout=60, **kwargs)
        return r
    except requests.exceptions.ConnectionError:
        st.error("⚠️  Cannot reach the FastAPI server. Is it running?")
        return None


def fetch_files():
    r = api("get", "/files")
    if r and r.ok:
        st.session_state.files_cache = r.json()
    return st.session_state.files_cache


def render_sources_html(sources: list) -> str:
    if not sources:
        return ""
    cards = ""
    for s in sources:
        rank   = s.get("rank", "?")
        score  = s.get("score", 0)
        source = s.get("source", "unknown")
        domain = s.get("domain", "")
        text   = s.get("text", "").replace("<", "&lt;").replace(">", "&gt;")
        score_pct = f"{score * 100:.1f}%" if isinstance(score, float) else str(score)
        cards += f"""
        <div class="src-card">
            <div class="src-header">
                <span class="src-rank">#{rank}</span>
                <span class="src-file">📄 {source}</span>
                <span class="src-score">sim {score_pct}</span>
            </div>
            {"" if not domain else f'<div class="src-domain">🏷 {domain}</div>'}
            <div class="src-text">{text}</div>
        </div>"""
    return f'<div class="sources-label">SOURCES</div><div class="sources-wrap">{cards}</div>'


def render_chat():
    html = '<div class="chat-scroll">'
    for m in st.session_state.messages:
        if m["role"] == "user":
            html += f'<div class="label-user">you</div><div class="bubble-user">{m["text"]}</div>'
        else:
            # escape answer text for HTML safety
            answer_html = m["text"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            html += f'<div class="label-bot"><span class="dot"></span>rag agent</div><div class="bubble-bot">{answer_html}</div>'
            sources = m.get("sources") or []
            if sources:
                html += render_sources_html(sources)
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Agentic RAG")
    st.markdown("<span style='font-size:0.78rem;color:#6b7280;font-family:Space Mono'>document intelligence system</span>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("### Upload Document")
    domain_input = st.text_input("Domain tag", placeholder="e.g. finance, legal, medical")
    uploaded = st.file_uploader("Drop a file", type=["pdf", "docx", "txt"], label_visibility="collapsed")

    if st.button("Upload & Ingest", use_container_width=True):
        if not uploaded:
            st.warning("Pick a file first.")
        elif not domain_input.strip():
            st.warning("Enter a domain tag.")
        else:
            with st.spinner("Uploading…"):
                r = api(
                    "post", "/upload",
                    params={"domain": domain_input.strip()},
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                )
            if r and r.status_code == 201:
                st.markdown('<span class="badge-ok">✓ uploaded — ingestion queued</span>', unsafe_allow_html=True)
                fetch_files()
            elif r:
                st.markdown(f'<span class="badge-err">✗ {r.json().get("detail","error")}</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── File browser ──────────────────────────────────────────────────────────
    st.markdown("### Indexed Files")
    all_files = fetch_files()
    if st.button("↻ Refresh", use_container_width=True):
        all_files = fetch_files()

    if all_files:
        for ext, names in all_files.items():
            if names:
                st.markdown(f"**{ext.upper()}**")
                for name in names:
                    cols = st.columns([5, 1])
                    cols[0].markdown(f'<span class="file-tag">📄 {name}</span>', unsafe_allow_html=True)
                    with cols[1]:
                        if st.button("✕", key=f"del_{name}", help=f"Delete {name}"):
                            r = api("delete", f"/files/{name}")
                            if r and r.ok:
                                st.success(f"Deleted {name}")
                                fetch_files()
                                st.rerun()
                            elif r:
                                st.error(r.json().get("detail", "delete failed"))
    else:
        st.markdown('<span style="color:#6b7280;font-size:0.82rem">No files indexed yet.</span>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# Ask Your Documents")
st.markdown('<span style="color:#6b7280;font-size:0.88rem;font-family:Space Mono">Retrieval-augmented answers from your uploaded corpus</span>', unsafe_allow_html=True)
st.markdown("---")

render_chat()

# ── Input row ─────────────────────────────────────────────────────────────────
col_q, col_btn = st.columns([8, 1])
with col_q:
    query = st.text_input(
        "query",
        placeholder="Ask something about your documents…",
        label_visibility="collapsed",
        key="query_input",
    )
with col_btn:
    send = st.button("Send", use_container_width=True)

if send and query.strip():
    st.session_state.messages.append({"role": "user", "text": query.strip()})
    with st.spinner("Thinking…"):
        r = api("post", "/chat", params={"message": query.strip()})
    if r and r.ok:
        data    = r.json()
        results = data.get("results", {})
        # results can be a dict with "answer"/"sources" OR a plain string
        if isinstance(results, dict):
            answer  = results.get("answer", "(no answer returned)")
            sources = results.get("sources", [])
        else:
            answer  = str(results)
            sources = []
        st.session_state.messages.append({"role": "bot", "text": answer, "sources": sources})
    elif r:
        st.session_state.messages.append({"role": "bot", "text": f"⚠️ Error {r.status_code}: {r.text}", "sources": []})
    else:
        st.session_state.messages.append({"role": "bot", "text": "⚠️ Could not reach the server.", "sources": []})
    st.rerun()