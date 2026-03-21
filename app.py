"""
app.py — Mahwous Hybrid Semantic Engine v7.0
=============================================
UI: Zero-Configuration | Background Threading | Visual-first
3 Tabs: New Opportunities | Duplicates | Smart Review
"""

from __future__ import annotations

import io
import pickle
import threading
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from logic import (
    FeatureParser,
    GeminiOracle,
    MahwousEngine,
    MatchResult,
    SemanticIndex,
    export_brands_csv,
    export_salla_csv,
    load_brands,
    load_competitor_products,
    load_store_products,
)

# ── Disk persistence ────────────────────────────────────────────────────────
_CACHE = Path.home() / ".mahwous_v7"
_CACHE.mkdir(exist_ok=True)
_RESULTS_FILE = _CACHE / "results.pkl"


def _save_results(payload: dict) -> None:
    try:
        with open(_RESULTS_FILE, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


def _load_results() -> Optional[dict]:
    try:
        if _RESULTS_FILE.exists():
            with open(_RESULTS_FILE, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None


# ── Background job state ────────────────────────────────────────────────────
_JOB: dict = {
    "status": "idle",       # idle | running | done | error
    "pct":    0.0,
    "step":   "",
    "log":    [],
    "eta":    "",
    "result": None,
    "error":  None,
    "lock":   threading.Lock(),
}

# ── Defaults ────────────────────────────────────────────────────────────────
_MAX_LOG = 80


# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="مهووس | محرك الاستكشاف الدلالي",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════════════════════════════
# CSS — Premium RTL Dark Theme
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"], .stMarkdown, button {
    font-family: 'Cairo', sans-serif !important;
}
.main > div { direction: rtl; }
div[data-testid="column"] { direction: rtl; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem !important; }

/* ── HERO ─────────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #050d1a 0%, #0a1628 40%, #0f2040 100%);
    border-radius: 22px;
    padding: 44px 36px 32px;
    text-align: center;
    margin-bottom: 26px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 48px rgba(0,0,0,.6);
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 70% 50% at 20% 20%, rgba(99,179,237,.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 80% 80%, rgba(167,139,250,.05) 0%, transparent 60%);
}
.hero-title {
    font-size: 2.8em; font-weight: 900; color: #fff;
    margin: 0 0 10px;
    background: linear-gradient(135deg, #63b3ed, #a78bfa, #f687b3);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub { font-size: 1em; color: rgba(255,255,255,.55); margin: 0; }
.hero-chips {
    margin-top: 16px;
    display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;
}
.chip {
    background: rgba(255,255,255,.07);
    border: 1px solid rgba(255,255,255,.12);
    color: rgba(255,255,255,.75);
    padding: 5px 16px; border-radius: 20px;
    font-size: .78em; font-weight: 600;
    backdrop-filter: blur(6px);
}

/* ── METRICS ──────────────────────────────────────────────────────────── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin: 22px 0; }
.kpi {
    border-radius: 16px; padding: 20px 16px; text-align: center;
    position: relative; overflow: hidden;
}
.kpi::after {
    content:''; position:absolute; top:-30px; right:-30px;
    width:80px; height:80px; border-radius:50%; opacity:.06;
    background: currentColor;
}
.kpi-blue   { background: linear-gradient(135deg,#1a2744,#1e3a6e); color:#93c5fd; }
.kpi-green  { background: linear-gradient(135deg,#0f2820,#1a4731); color:#86efac; }
.kpi-amber  { background: linear-gradient(135deg,#2a1f0a,#3d2e0f); color:#fcd34d; }
.kpi-purple { background: linear-gradient(135deg,#1a1032,#2d1b5e); color:#c4b5fd; }
.kpi-icon   { font-size: 1.6em; margin-bottom: 4px; }
.kpi-num    { font-size: 2.4em; font-weight: 900; line-height: 1; }
.kpi-label  { font-size: .78em; opacity: .7; margin-top: 6px; font-weight: 600; }

/* ── PROGRESS BAR ─────────────────────────────────────────────────────── */
.prog-wrap {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 14px;
    padding: 18px 22px;
    margin: 14px 0;
}
.prog-header {
    display: flex; justify-content: space-between;
    font-size: .85em; color: rgba(255,255,255,.65); margin-bottom: 10px;
}
.prog-bar-outer {
    background: rgba(255,255,255,.08);
    border-radius: 8px; height: 10px; overflow: hidden;
}
.prog-bar-inner {
    height: 100%; border-radius: 8px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
    transition: width .4s ease;
}
.log-box {
    background: #0a0f1a;
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 10px;
    padding: 12px 14px;
    max-height: 200px;
    overflow-y: auto;
    font-family: 'Courier New', monospace;
    font-size: .75em;
    color: #6ee7b7;
    margin-top: 12px;
}

/* ── PRODUCT CARD ─────────────────────────────────────────────────────── */
.pcard {
    background: linear-gradient(135deg, #0f1c30, #111827);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 12px;
    transition: transform .15s, box-shadow .15s, border-color .15s;
}
.pcard:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,0,0,.4);
    border-color: rgba(99,179,237,.3);
}
.pcard-img {
    width: 100%; height: 160px; object-fit: contain;
    border-radius: 10px; background: rgba(255,255,255,.04);
}
.pcard-img-placeholder {
    width: 100%; height: 160px;
    background: linear-gradient(135deg,#1a2030,#1f2840);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    color: rgba(255,255,255,.2); font-size: 2.5em;
}
.pcard-name  { font-weight: 700; font-size: .9em; color: #e2e8f0; margin: 8px 0 4px; line-height: 1.4; }
.pcard-brand { font-size: .78em; color: #93c5fd; font-weight: 600; }
.pcard-meta  { font-size: .72em; color: rgba(255,255,255,.35); margin-top: 4px; }

/* ── PILLS / BADGES ───────────────────────────────────────────────────── */
.pill {
    display: inline-block; padding: 3px 11px;
    border-radius: 20px; font-size: .7em; font-weight: 700; margin: 2px 1px;
}
.p-perf   { background: rgba(99,102,241,.2);  color: #a5b4fc; border: 1px solid rgba(99,102,241,.3); }
.p-beau   { background: rgba(236,72,153,.2);  color: #f9a8d4; border: 1px solid rgba(236,72,153,.3); }
.p-unk    { background: rgba(100,116,139,.2); color: #94a3b8; border: 1px solid rgba(100,116,139,.3); }
.p-new    { background: rgba(16,185,129,.2);  color: #6ee7b7; border: 1px solid rgba(16,185,129,.3); }
.p-dup    { background: rgba(239,68,68,.2);   color: #fca5a5; border: 1px solid rgba(239,68,68,.3); }
.p-conf   { background: rgba(245,158,11,.15); color: #fcd34d; border: 1px solid rgba(245,158,11,.3); }
.p-layer  { background: rgba(168,85,247,.2);  color: #d8b4fe; border: 1px solid rgba(168,85,247,.3); }

/* ── COMPARISON CARD ──────────────────────────────────────────────────── */
.cmp-wrap {
    background: linear-gradient(135deg,#0f1c30,#111827);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 12px;
}
.cmp-vs {
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4em; font-weight: 900;
    color: rgba(245,158,11,.8);
    padding: 8px 0;
}
.cmp-reason {
    text-align: center;
    font-size: .8em; color: rgba(255,255,255,.5);
    margin-top: 8px;
    padding: 6px 10px;
    background: rgba(255,255,255,.03);
    border-radius: 8px;
}

/* ── TINDER CARD (Review) ─────────────────────────────────────────────── */
.tinder-wrap {
    background: linear-gradient(135deg, #0f1525, #111827);
    border: 2px solid rgba(245,158,11,.25);
    border-radius: 20px;
    padding: 24px;
    max-width: 680px;
    margin: 0 auto 24px;
    box-shadow: 0 12px 40px rgba(0,0,0,.5);
}
.tinder-counter {
    text-align: center;
    font-size: .85em; color: rgba(255,255,255,.4);
    margin-bottom: 16px;
}
.tinder-products {
    display: grid; grid-template-columns: 1fr auto 1fr;
    gap: 12px; align-items: center;
}
.tinder-score {
    background: rgba(245,158,11,.1);
    border: 1px solid rgba(245,158,11,.3);
    border-radius: 12px;
    padding: 10px 14px;
    text-align: center;
    margin-top: 14px;
    font-size: .8em; color: #fcd34d;
}

/* ── SECTION HEADERS ──────────────────────────────────────────────────── */
.sec-hdr {
    display: flex; align-items: center; gap: 14px;
    padding: 16px 22px; border-radius: 14px; margin: 20px 0 14px;
}
.sec-hdr-new  { background: linear-gradient(135deg,rgba(16,185,129,.1),rgba(6,95,70,.15)); border:1px solid rgba(16,185,129,.2); }
.sec-hdr-dup  { background: linear-gradient(135deg,rgba(239,68,68,.1),rgba(127,29,29,.15)); border:1px solid rgba(239,68,68,.2); }
.sec-hdr-rev  { background: linear-gradient(135deg,rgba(245,158,11,.1),rgba(120,53,15,.15)); border:1px solid rgba(245,158,11,.2); }
.sec-hdr-icon { font-size: 2em; }
.sec-hdr-text h3 { margin: 0; font-size: 1.05em; font-weight: 900; color: #f1f5f9; }
.sec-hdr-text p  { margin: 2px 0 0; font-size: .82em; color: rgba(255,255,255,.45); }

/* ── DL BUTTON ────────────────────────────────────────────────────────── */
div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg,#1e3a5f,#1e40af) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-family: 'Cairo',sans-serif !important; width: 100%;
    padding: 10px 18px !important;
}
div[data-testid="stDownloadButton"] > button:hover { opacity: .88 !important; }

/* ── PRIMARY BTN ──────────────────────────────────────────────────────── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg,#1e3a5f,#1e40af) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 900 !important;
    font-family: 'Cairo',sans-serif !important;
    font-size: 1.05em !important; padding: 13px 28px !important;
    box-shadow: 0 4px 24px rgba(30,64,175,.4) !important;
}

/* ── SIDEBAR ──────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] { background: #05080f; }
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p { color: rgba(255,255,255,.7) !important; }

/* ── DATAFRAME ────────────────────────────────────────────────────────── */
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── SPINNER ──────────────────────────────────────────────────────────── */
.running-banner {
    background: linear-gradient(135deg,rgba(30,58,138,.3),rgba(76,29,149,.3));
    border: 1px solid rgba(99,179,237,.2);
    border-radius: 16px; padding: 18px 22px;
    display: flex; align-items: center; gap: 16px;
    margin: 12px 0;
}
.running-icon { font-size: 2.4em; animation: spin 2s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* ── UPLOAD ZONE ──────────────────────────────────────────────────────── */
.upload-label {
    font-weight: 700; font-size: .95em; color: #94a3b8; margin: 0 0 4px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES  (loaded once per Streamlit server process)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🧠 تحميل نموذج الذكاء الاصطناعي الدلالي…")
def _load_model():
    """Load multilingual sentence-transformer once; reused across all sessions."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_resource
def _get_index() -> SemanticIndex:
    """Single SemanticIndex instance per server process."""
    model = _load_model()
    return SemanticIndex(model)


# ══════════════════════════════════════════════════════════════════════════
# SECRETS & API KEYS
# ══════════════════════════════════════════════════════════════════════════

def _get_secret(key: str) -> str:
    try:
        v = st.secrets.get(key, "")
        if v: return v
    except Exception:
        pass
    return st.session_state.get(key, "")


GEMINI_KEY  = _get_secret("GEMINI_API_KEY")
SEARCH_KEY  = _get_secret("GOOGLE_SEARCH_KEY")
SEARCH_CX   = _get_secret("GOOGLE_CX")
_has_gemini = bool(GEMINI_KEY)


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("<h2 style='color:#93c5fd;margin:0;padding:10px 0 4px;font-family:Cairo,sans-serif;'>⚙️ الإعدادات</h2>",
                unsafe_allow_html=True)
    st.caption("المفاتيح تُقرأ تلقائياً من secrets.toml")

    with st.expander("🔑 مفاتيح API"):
        mg = st.text_input("Gemini API Key", type="password", key="GEMINI_API_KEY")
        ms = st.text_input("Google Search Key", type="password", key="GOOGLE_SEARCH_KEY")
        mc = st.text_input("Google CX", key="GOOGLE_CX")
        if mg: GEMINI_KEY = mg;  _has_gemini = True
        if ms: SEARCH_KEY = ms
        if mc: SEARCH_CX  = mc

    st.divider()

    with st.expander("🧠 إعدادات المحرك"):
        fetch_imgs  = st.toggle("جلب الصور المفقودة",        value=False)
        use_llm     = st.toggle("تفعيل Gemini (المنطقة الرمادية)", value=True)
        max_comp    = st.slider("حد ملفات المنافسين", 1, 15, 10)

    st.divider()

    if st.button("🔄 مسح الكل وبدء من جديد", use_container_width=True):
        try: _RESULTS_FILE.unlink(missing_ok=True)
        except: pass
        with _JOB["lock"]:
            _JOB.update({"status":"idle","pct":0.0,"step":"","log":[],"result":None,"error":None})
        for k in ["done","new","dups","reviews","stats","_loaded","review_decisions"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown(
        "<div style='text-align:center;padding:12px 0;color:rgba(255,255,255,.2);font-size:.72em;font-family:Cairo;'>"
        "Mahwous Engine v7.0<br>Hybrid Semantic + LLM + FAISS</div>",
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT + DISK RESTORE
# ══════════════════════════════════════════════════════════════════════════

_EMPTY = {
    "done":             False,
    "new":              [],
    "dups":             [],
    "reviews":          [],
    "review_decisions": {},   # {idx: "new" | "dup"}
    "stats":            {"new":0,"dup":0,"rev":0,"total":0},
}
if not st.session_state.get("_loaded"):
    disk = _load_results()
    if disk and disk.get("done"):
        for k, v in disk.items():
            st.session_state[k] = v
    else:
        for k, v in _EMPTY.items():
            if k not in st.session_state:
                st.session_state[k] = v
    st.session_state["_loaded"] = True
else:
    for k, v in _EMPTY.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════
# BACKGROUND PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def _log_append(msg: str) -> None:
    with _JOB["lock"]:
        _JOB["log"].append(msg)
        if len(_JOB["log"]) > _MAX_LOG:
            _JOB["log"] = _JOB["log"][-_MAX_LOG:]
        _JOB["step"] = msg


def _run_pipeline(
    store_bytes: list[tuple[str, bytes]],
    comp_bytes:  list[tuple[str, bytes]],
    brands_bytes: Optional[bytes],
    gemini_key: str,
    search_key: str,
    search_cx:  str,
    fetch_imgs: bool,
    use_llm:    bool,
) -> None:
    """Full pipeline executed in a daemon thread."""
    try:
        with _JOB["lock"]:
            _JOB.update({"status":"running","pct":0.0,"log":[],"result":None,"error":None})

        class _Buf:
            def __init__(self, data: bytes, name: str):
                self._buf = io.BytesIO(data); self.name = name
            def read(self) -> bytes:
                self._buf.seek(0); return self._buf.read()

        store_files  = [_Buf(b, n) for n, b in store_bytes]
        comp_files   = [_Buf(b, n) for n, b in comp_bytes]
        brands_file  = _Buf(brands_bytes, "brands.csv") if brands_bytes else None

        # ── Step 1: Load store ────────────────────────────────────────────
        _log_append("📂 تحميل ملفات متجر مهووس…")
        store_df = load_store_products(store_files)
        existing_brands = load_brands(brands_file) if brands_file else []
        _log_append(f"✅ {len(store_df):,} منتج في الجدار الواقي | {len(existing_brands):,} ماركة")
        with _JOB["lock"]: _JOB["pct"] = 0.08

        # ── Step 2: Load competitors ──────────────────────────────────────
        _log_append("📦 تحميل بيانات المنافسين…")
        comp_df = load_competitor_products(comp_files)
        _log_append(f"✅ {len(comp_df):,} منتج من المنافسين")
        with _JOB["lock"]: _JOB["pct"] = 0.15

        # ── Step 3: Build FAISS index ─────────────────────────────────────
        _log_append("🧠 بناء فهرس المتجهات الدلالية (FAISS)…")
        semantic_idx = _get_index()
        semantic_idx.build(store_df, progress_cb=_log_append)
        _log_append(f"✅ FAISS جاهز: {len(store_df):,} متجه دلالي")
        with _JOB["lock"]: _JOB["pct"] = 0.25

        # ── Step 4: Init oracle ───────────────────────────────────────────
        oracle = GeminiOracle(gemini_key) if (use_llm and gemini_key) else None
        if oracle:
            _log_append("🤖 Gemini 1.5 Flash نشط للمنطقة الرمادية")
        else:
            _log_append("⚠️ Gemini غير مفعّل — القرارات الرمادية → مراجعة يدوية")

        # ── Step 5: Run engine ────────────────────────────────────────────
        engine = MahwousEngine(
            semantic_index=semantic_idx,
            gemini_oracle=oracle,
            search_api_key=search_key,
            search_cx=search_cx,
            fetch_images=fetch_imgs,
        )

        total   = len(comp_df)
        t_start = time.time()

        def _progress(i: int, tot: int, name: str) -> None:
            pct = 0.25 + 0.70 * i / max(tot, 1)
            elapsed = time.time() - t_start
            rate    = i / max(elapsed, 0.1)
            remain  = (tot - i) / max(rate, 0.01)
            eta_str = f"{remain/60:.1f} دقيقة" if remain > 60 else f"{remain:.0f} ثانية"
            with _JOB["lock"]:
                _JOB["pct"] = pct
                _JOB["eta"] = eta_str
            if i % 50 == 0 or i < 5:
                _log_append(f"  ⚙️ {i}/{tot} — {name[:45]}")

        _log_append(f"⚖️ تشغيل خط الأنابيب الهجين على {total:,} منتج…")
        new_list, dup_list, rev_list = engine.run(
            store_df    = store_df,
            comp_df     = comp_df,
            progress_cb = _progress,
            log_cb      = _log_append,
        )

        # ── Step 6: Finalise ──────────────────────────────────────────────
        stats = {
            "new": len(new_list),
            "dup": len(dup_list),
            "rev": len(rev_list),
            "total": total,
        }
        payload = {
            "done":             True,
            "new":              new_list,
            "dups":             dup_list,
            "reviews":          rev_list,
            "review_decisions": {},
            "stats":            stats,
            "existing_brands":  existing_brands,
        }
        _save_results(payload)
        _log_append(f"💾 حُفظت النتائج | 🌟 {stats['new']} جديد | 🚫 {stats['dup']} مكرر | 🔍 {stats['rev']} مراجعة")

        with _JOB["lock"]:
            _JOB["result"] = payload
            _JOB["status"] = "done"
            _JOB["pct"]    = 1.0
        _log_append("🎉 اكتمل التحليل!")

    except Exception as exc:
        import traceback
        err = f"{exc}\n{traceback.format_exc()}"
        with _JOB["lock"]:
            _JOB["status"] = "error"
            _JOB["error"]  = err
        _log_append(f"❌ خطأ: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <div class="hero-title">🔬 محرك مهووس | البحث الدلالي الهجين</div>
    <p class="hero-sub">
        FAISS Vector Search · Multilingual Embeddings · Gemini LLM Oracle
        · Zero Gray Zone · Real-Time Visual Review
    </p>
    <div class="hero-chips">
        <span class="chip">🧠 paraphrase-multilingual-MiniLM</span>
        <span class="chip">⚡ FAISS IndexFlatIP</span>
        <span class="chip">⚖️ Weighted Lexical Fusion</span>
        <span class="chip">🤖 Gemini 1.5 Flash</span>
        <span class="chip">💾 Auto-Save</span>
        <span class="chip">🧵 Background Thread</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Gemini status banner
if _has_gemini:
    st.success("✅ **Gemini AI نشط** — المنطقة الرمادية تُحسم بصرياً تلقائياً", icon="🤖")
else:
    st.markdown("""
<div style='background:linear-gradient(135deg,rgba(245,158,11,.1),rgba(120,53,15,.15));
            border:1px solid rgba(245,158,11,.25);border-radius:14px;padding:16px 20px;margin:12px 0;'>
    <b style='color:#fcd34d;'>⚠️ Gemini غير مُفعَّل</b>
    <p style='color:rgba(255,255,255,.5);margin:4px 0 6px;font-size:.88em;'>
        المنتجات الرمادية ستذهب للمراجعة اليدوية. فعّل Gemini في
        <code style='background:rgba(255,255,255,.1);padding:1px 6px;border-radius:4px;'>.streamlit/secrets.toml</code>:
    </p>
    <code style='background:#0a0f1a;color:#6ee7b7;padding:8px 14px;border-radius:8px;
                 display:block;direction:ltr;'>GEMINI_API_KEY = "AIza...."</code>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# UPLOAD SECTION
# ══════════════════════════════════════════════════════════════════════════

st.markdown("---")
c1, c2, c3 = st.columns([1.1, 1.4, 1], gap="large")

with c1:
    st.markdown('<p class="upload-label">🏪 ملفات متجر مهووس</p>', unsafe_allow_html=True)
    st.caption("ملفات سلة — الأجزاء 1-4 أو أكثر")
    store_files = st.file_uploader(
        "store", type=["csv"], accept_multiple_files=True,
        key="uf_store", label_visibility="collapsed")
    brands_file = st.file_uploader(
        "ملف الماركات (اختياري)", type=["csv"], key="uf_brands")
    if store_files:
        st.success(f"✅ {len(store_files)} ملف(ات) للمتجر")

with c2:
    st.markdown('<p class="upload-label">🔍 ملفات المنافسين</p>', unsafe_allow_html=True)
    st.caption("حتى 15 ملف CSV من أي متجر")
    comp_files = st.file_uploader(
        "comp", type=["csv"], accept_multiple_files=True,
        key="uf_comp", label_visibility="collapsed")
    if comp_files:
        st.success(f"✅ {len(comp_files)} ملف(ات) للمنافسين")
        with st.expander("📋 الملفات"):
            for f in comp_files[:max_comp]: st.caption(f"• {f.name}")

with c3:
    st.markdown('<p class="upload-label">🚀 تشغيل المحرك</p>', unsafe_allow_html=True)
    st.caption(" ")
    running   = _JOB["status"] == "running"
    ready     = bool(store_files and comp_files) and not running
    start_btn = st.button(
        "🚀 بدء التحليل الهجين",
        type="primary", use_container_width=True,
        disabled=not ready, key="btn_start")
    if running:
        st.caption("⚙️ يعمل في الخلفية…")
    elif not ready and not running:
        st.caption("⚠️ ارفع ملف المتجر وملف منافس")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# LAUNCH THREAD
# ══════════════════════════════════════════════════════════════════════════

if start_btn and store_files and comp_files:
    s_bytes = [(f.name, f.read()) for f in store_files]
    c_bytes = [(f.name, f.read()) for f in comp_files[:max_comp]]
    b_bytes = brands_file.read() if brands_file else None
    for k in ["done","new","dups","reviews","review_decisions","stats","_loaded"]:
        st.session_state.pop(k, None)
    threading.Thread(
        target=_run_pipeline,
        kwargs=dict(
            store_bytes  = s_bytes,
            comp_bytes   = c_bytes,
            brands_bytes = b_bytes,
            gemini_key   = GEMINI_KEY,
            search_key   = SEARCH_KEY,
            search_cx    = SEARCH_CX,
            fetch_imgs   = fetch_imgs,
            use_llm      = use_llm,
        ),
        daemon=True,
    ).start()
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# RUNNING PROGRESS UI
# ══════════════════════════════════════════════════════════════════════════

if _JOB["status"] == "running":
    with _JOB["lock"]:
        pct  = _JOB["pct"]
        step = _JOB["step"]
        logs = list(_JOB["log"])
        eta  = _JOB.get("eta","")

    st.markdown(f"""
<div class="running-banner">
    <div class="running-icon">⚙️</div>
    <div>
        <b style='color:#93c5fd;font-size:1.05em;'>محرك البحث الدلالي يعمل في الخلفية</b>
        <p style='margin:4px 0 0;color:rgba(255,255,255,.5);font-size:.88em;'>{step}</p>
    </div>
    <div style='margin-right:auto;text-align:left;color:rgba(255,255,255,.35);font-size:.8em;'>
        {'ETA: ' + eta if eta else ''}
    </div>
</div>
<div class="prog-wrap">
    <div class="prog-header">
        <span>التقدم</span>
        <span>{pct*100:.0f}%</span>
    </div>
    <div class="prog-bar-outer">
        <div class="prog-bar-inner" style="width:{pct*100:.1f}%"></div>
    </div>
</div>
""", unsafe_allow_html=True)

    with st.expander("📋 السجل المباشر", expanded=True):
        log_html = "<br>".join(logs[-30:])
        st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

    st.caption("💡 يمكنك إغلاق المتصفح — النتائج تُحفظ تلقائياً عند الانتهاء.")
    time.sleep(2); st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# JOB DONE → load into session_state
# ══════════════════════════════════════════════════════════════════════════

if _JOB["status"] == "done" and not st.session_state.get("done"):
    with _JOB["lock"]:
        payload = _JOB["result"]
    if payload:
        for k, v in payload.items():
            st.session_state[k] = v
        st.toast("🎉 اكتمل التحليل — النتائج جاهزة!", icon="✅")
        time.sleep(0.3); st.rerun()

if _JOB["status"] == "error":
    st.error(f"❌ حدث خطأ: {str(_JOB.get('error',''))[:300]}")
    if st.button("🔄 إعادة المحاولة"):
        with _JOB["lock"]: _JOB["status"] = "idle"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# WELCOME STATE
# ══════════════════════════════════════════════════════════════════════════

if not st.session_state.get("done"):
    st.markdown("""
<div style='text-align:center;padding:60px 0 80px;'>
    <div style='font-size:5em;margin-bottom:16px;'>🔬</div>
    <p style='font-size:1.15em;color:rgba(255,255,255,.4);max-width:520px;
              margin:0 auto;line-height:2;'>
        ارفع ملفات متجرك وملفات المنافسين<br>
        ثم اضغط <b style='color:#93c5fd;'>بدء التحليل الهجين</b><br>
        <span style='font-size:.85em;color:rgba(255,255,255,.25);'>
            FAISS · Multilingual Embeddings · Gemini Oracle · Zero Gray Zone
        </span>
    </p>
</div>""", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════

new_list:   list[MatchResult] = st.session_state.get("new",   [])
dup_list:   list[MatchResult] = st.session_state.get("dups",  [])
rev_list:   list[MatchResult] = st.session_state.get("reviews",[])
existing_b: list[str]         = st.session_state.get("existing_brands",[])
s = st.session_state.get("stats",{"new":0,"dup":0,"rev":0,"total":0})

# Incorporate review decisions
_decisions = st.session_state.get("review_decisions", {})
_rev_new  = [rev_list[i] for i, v in _decisions.items() if v == "new"  and i < len(rev_list)]
_rev_dup  = [rev_list[i] for i, v in _decisions.items() if v == "dup"  and i < len(rev_list)]
_rev_pend = [rev_list[i] for i in range(len(rev_list)) if i not in _decisions]
all_new   = new_list + _rev_new
all_dups  = dup_list + _rev_dup

# ── KPI CARDS ─────────────────────────────────────────────────────────────
total   = s["total"]
opp_pct = round(len(all_new)/total*100,1) if total else 0

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi kpi-blue">
    <div class="kpi-icon">📦</div>
    <div class="kpi-num">{total:,}</div>
    <div class="kpi-label">منتجات مُحللة</div>
  </div>
  <div class="kpi kpi-green">
    <div class="kpi-icon">🌟</div>
    <div class="kpi-num">{len(all_new):,}</div>
    <div class="kpi-label">فرص جديدة · {opp_pct}%</div>
  </div>
  <div class="kpi kpi-amber">
    <div class="kpi-icon">🔍</div>
    <div class="kpi-num">{len(_rev_pend):,}</div>
    <div class="kpi-label">تحتاج مراجعة</div>
  </div>
  <div class="kpi kpi-purple">
    <div class="kpi-icon">🛡️</div>
    <div class="kpi-num">{len(all_dups):,}</div>
    <div class="kpi-label">مكررات محظورة</div>
  </div>
</div>
""", unsafe_allow_html=True)

if _RESULTS_FILE.exists():
    from datetime import datetime
    ts = datetime.fromtimestamp(_RESULTS_FILE.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    st.caption(f"💾 آخر حفظ: **{ts}**")


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _layer_badge(layer: str) -> str:
    css = {"L1-GTIN":"p-layer","L1-SKU":"p-layer",
           "L2-FAISS-miss":"p-new","L3-LEX-HIGH":"p-dup",
           "L3-LEX-LOW":"p-new","L3-FEAT-MISMATCH":"p-new",
           "L4-LLM-MATCH":"p-dup","L4-LLM-DIFF":"p-new",
           "L3-GRAY":"p-conf","L4-LLM-UNSURE":"p-conf"}.get(layer,"p-conf")
    labels = {"L1-GTIN":"🎯 GTIN","L1-SKU":"🔑 SKU",
              "L2-FAISS-miss":"🧠 Semantic",
              "L3-LEX-HIGH":"📊 Lexical","L3-LEX-LOW":"📊 Lexical",
              "L3-FEAT-MISMATCH":"⚙️ Feature","L4-LLM-MATCH":"🤖 Gemini",
              "L4-LLM-DIFF":"🤖 Gemini","L3-GRAY":"⏳ Gray","L4-LLM-UNSURE":"❓ Unsure"}
    return f'<span class="pill {css}">{labels.get(layer, layer)}</span>'

def _type_pill(pt: str) -> str:
    return (f'<span class="pill p-perf">🧴 عطر</span>'  if pt=="perfume" else
            f'<span class="pill p-beau">💄 مكياج</span>' if pt=="beauty"  else
            f'<span class="pill p-unk">❓ غير محدد</span>')

def _img_html(url: str, h: int = 150) -> str:
    if url and url.startswith("http"):
        return (f'<img src="{url}" style="width:100%;height:{h}px;'
                f'object-fit:contain;border-radius:10px;'
                f'background:rgba(255,255,255,.04);" '
                f'onerror="this.style.display=\'none\'">')
    return (f'<div style="width:100%;height:{h}px;background:rgba(255,255,255,.04);'
            f'border-radius:10px;display:flex;align-items:center;'
            f'justify-content:center;color:rgba(255,255,255,.15);font-size:2em;">📷</div>')

def _render_product_card(r: MatchResult, show_layer: bool = True) -> None:
    brand_html = f'<p class="pcard-brand">🏷️ {r.brand}</p>' if r.brand else ""
    layer_html = _layer_badge(r.layer_used) if show_layer else ""
    price_html = (f'<span class="pill p-conf">💰 {r.comp_price} ر.س</span>'
                  if r.comp_price.strip() else "")
    conf_html  = f'<span class="pill p-conf">{r.confidence:.0%}</span>'
    pname = r.comp_name[:65] + ("…" if len(r.comp_name) > 65 else "")

    st.markdown(_img_html(r.comp_image), unsafe_allow_html=True)
    st.markdown(
        f'<p class="pcard-name">{pname}</p>'
        f'{brand_html}'
        f'<div style="margin-top:6px;">{_type_pill(r.product_type)}'
        f'<span class="pill p-new">✅ جديد</span>'
        f'{price_html}</div>'
        f'<div style="margin-top:4px;">{layer_html} {conf_html}</div>'
        f'<p class="pcard-meta">{r.comp_source}</p>',
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    f"🌟 الفرص الجديدة  ({len(all_new):,})",
    f"⚖️ المتوفر / المكرر  ({len(all_dups):,})",
    f"🔍 المراجعة الذكية  ({len(_rev_pend):,})",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — NEW OPPORTUNITIES
# ══════════════════════════════════════════════════════════════════════════

with tab1:
    if not all_new:
        st.info("✨ لا فرص جديدة — جميع منتجات المنافسين موجودة لديك بالفعل!")
    else:
        st.markdown("""
<div class="sec-hdr sec-hdr-new">
    <div class="sec-hdr-icon">🌟</div>
    <div class="sec-hdr-text">
        <h3>الفرص الجديدة — غير متوفرة لدينا</h3>
        <p>منتجات حُسمت بدقة عبر FAISS + Lexical + Gemini. جاهزة للرفع على سلة.</p>
    </div>
</div>
""", unsafe_allow_html=True)

        # ── Export buttons ────────────────────────────────────────────────
        st.markdown("#### 📥 تصدير ملفات سلة")
        perf_r  = [r for r in all_new if r.product_type == "perfume"]
        beau_r  = [r for r in all_new if r.product_type in ("beauty","unknown")]

        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            if perf_r:
                st.download_button(
                    f"⬇️ ملف العطور ({len(perf_r):,})",
                    data=export_salla_csv(perf_r),
                    file_name="Salla_Perfumes.csv", mime="text/csv",
                    use_container_width=True, key="dl_perf")
            else:
                st.info("لا عطور جديدة")
        with ex2:
            if beau_r:
                st.download_button(
                    f"⬇️ المكياج والعناية ({len(beau_r):,})",
                    data=export_salla_csv(beau_r),
                    file_name="Salla_Beauty.csv", mime="text/csv",
                    use_container_width=True, key="dl_beau")
            else:
                st.info("لا منتجات عناية جديدة")
        with ex3:
            st.download_button(
                "⬇️ الماركات الناقصة",
                data=export_brands_csv(all_new, existing_b),
                file_name="Missing_Brands.csv", mime="text/csv",
                use_container_width=True, key="dl_brands")

        st.markdown("---")

        # ── Filters ───────────────────────────────────────────────────────
        fc1, fc2, fc3, fc4 = st.columns([1.2, 1.2, 1.8, 1.1])
        with fc1:
            t_sel = st.selectbox("🗂️ النوع", ["الكل","عطور","مكياج وعناية"], key="f_t")
        with fc2:
            src_opts = ["الكل"] + sorted({r.comp_source for r in all_new if r.comp_source})
            s_sel = st.selectbox("🏪 المصدر", src_opts, key="f_s")
        with fc3:
            q = st.text_input("🔎 بحث في الاسم", placeholder="اكتب للبحث…", key="f_q")
        with fc4:
            view = st.radio("عرض", ["🃏 بطاقات","📋 جدول"], horizontal=True, key="f_v")

        disp = all_new[:]
        if t_sel == "عطور":          disp = [r for r in disp if r.product_type=="perfume"]
        elif t_sel == "مكياج وعناية": disp = [r for r in disp if r.product_type in ("beauty","unknown")]
        if s_sel != "الكل":          disp = [r for r in disp if r.comp_source == s_sel]
        if q:                         disp = [r for r in disp if q.lower() in r.comp_name.lower()]

        LIMIT = 60
        st.caption(f"عرض **{min(len(disp),LIMIT):,}** من **{len(disp):,}**")

        if view == "📋 جدول":
            rows = [{
                "اسم المنتج":  r.comp_name,
                "النوع":       r.product_type,
                "السعر":       r.comp_price,
                "الطبقة":      r.layer_used,
                "الثقة":       f"{r.confidence:.0%}",
                "التفاصيل":    r.feature_details or r.llm_reasoning,
                "المصدر":      r.comp_source,
            } for r in disp[:LIMIT]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=480)
        else:
            for rs in range(0, min(len(disp), LIMIT), 3):
                cols = st.columns(3, gap="medium")
                for j, col in enumerate(cols):
                    idx = rs + j
                    if idx >= min(len(disp), LIMIT): break
                    with col:
                        with st.container():
                            _render_product_card(disp[idx])

        if len(disp) > LIMIT:
            st.info(f"يُعرض أول {LIMIT}. حمّل CSV للاطلاع على الكل ({len(disp):,}).")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — DUPLICATES  (side-by-side comparison)
# ══════════════════════════════════════════════════════════════════════════

with tab2:
    if not all_dups:
        st.success("🎉 لا مكررات — كل منتجات المنافسين جديدة بالنسبة لمتجرك!")
    else:
        st.markdown("""
<div class="sec-hdr sec-hdr-dup">
    <div class="sec-hdr-icon">⚖️</div>
    <div class="sec-hdr-text">
        <h3>المتوفر لدينا — المكررات المحظورة</h3>
        <p>مقارنة بصرية جنباً إلى جنب: منتجنا مقابل منتج المنافس مع سبب التطابق.</p>
    </div>
</div>
""", unsafe_allow_html=True)

        st.success(f"🛡️ تم تحديد **{len(all_dups):,}** مكرر — متجرك محمي بالكامل.")

        # Layer breakdown
        layers = {}
        for r in all_dups:
            layers[r.layer_used] = layers.get(r.layer_used, 0) + 1
        mc = st.columns(len(layers) or 1)
        for i, (layer, cnt) in enumerate(sorted(layers.items(), key=lambda x: -x[1])):
            mc[i % len(mc)].metric(_layer_badge(layer).replace("<span","").replace("</span>","").split(">")[-1].split("<")[0], cnt)

        st.markdown("---")

        # Filters
        dc1, dc2, dc3 = st.columns([1, 1, 2])
        with dc1:
            dl_sel = st.radio("طريقة الكشف", ["الكل"] + list(layers.keys()), horizontal=False, key="dl_s")
        with dc2:
            ds_src = st.selectbox("المصدر", ["الكل"] + sorted({r.comp_source for r in all_dups}), key="dl_src")
        with dc3:
            dq = st.text_input("🔎 بحث", key="dl_q")

        ddisp = all_dups[:]
        if dl_sel != "الكل":  ddisp = [r for r in ddisp if r.layer_used == dl_sel]
        if ds_src != "الكل": ddisp = [r for r in ddisp if r.comp_source == ds_src]
        if dq:                ddisp = [r for r in ddisp if dq.lower() in r.comp_name.lower()]

        D_LIMIT = 100
        st.caption(f"عرض **{min(len(ddisp), D_LIMIT):,}** من **{len(ddisp):,}**")

        for r in ddisp[:D_LIMIT]:
            st.markdown('<div class="cmp-wrap">', unsafe_allow_html=True)
            ca, cb, cc = st.columns([2, 0.5, 2])

            with ca:
                st.markdown(
                    f'{_img_html(r.store_image, 120)}'
                    f'<p class="pcard-name" style="font-size:.82em;">{r.store_name[:60]}</p>'
                    f'<span class="pill p-dup">🏪 متجرنا</span>',
                    unsafe_allow_html=True)

            with cb:
                st.markdown('<div class="cmp-vs">≈</div>', unsafe_allow_html=True)

            with cc:
                conf_html = f'<span class="pill p-conf">{r.confidence:.0%}</span>'
                layer_html = _layer_badge(r.layer_used)
                price_html = (f'<span class="pill p-conf">💰 {r.comp_price}</span>'
                              if r.comp_price.strip() else "")
                st.markdown(
                    f'{_img_html(r.comp_image, 120)}'
                    f'<p class="pcard-name" style="font-size:.82em;">{r.comp_name[:60]}</p>'
                    f'<span class="pill p-new">🔍 المنافس</span> {price_html}',
                    unsafe_allow_html=True)

            reason = r.feature_details or r.llm_reasoning or f"تشابه دلالي {r.faiss_score:.0%} + معجمي {r.lex_score:.0%}"
            st.markdown(
                f'<div class="cmp-reason">{layer_html} {conf_html} — {reason}</div>',
                unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if len(ddisp) > D_LIMIT:
            st.info(f"يُعرض أول {D_LIMIT}. هناك {len(ddisp)-D_LIMIT} مكرر إضافي.")

        # Export duplicates audit
        dup_rows = [{
            "منتج المنافس": r.comp_name,
            "منتجنا المطابق": r.store_name,
            "طريقة الكشف": r.layer_used,
            "الثقة": f"{r.confidence:.0%}",
            "سبب التطابق": r.feature_details or r.llm_reasoning,
            "المصدر": r.comp_source,
        } for r in all_dups]
        st.markdown("---")
        st.download_button(
            f"⬇️ تصدير جدول المكررات ({len(all_dups):,})",
            data=pd.DataFrame(dup_rows).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="Duplicates_Audit.csv", mime="text/csv",
            key="dl_dup_audit")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — SMART REVIEW  (Tinder-style cards)
# ══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("""
<div class="sec-hdr sec-hdr-rev">
    <div class="sec-hdr-icon">🔍</div>
    <div class="sec-hdr-text">
        <h3>المراجعة الذكية — الحالات الغامضة</h3>
        <p>هذه القائمة يجب أن تكون شبه فارغة. صوّت على كل بطاقة.</p>
    </div>
</div>
""", unsafe_allow_html=True)

    if not _rev_pend:
        st.success(
            "🎯 **ممتاز!** لا يوجد شيء يحتاج مراجعة — النظام حسم كل شيء تلقائياً."
            if not rev_list else
            f"✅ تمت مراجعة جميع الـ {len(rev_list)} منتج. شكراً!"
        )
        if _rev_new or _rev_dup:
            rn1, rn2, rn3 = st.columns(3)
            with rn1:
                if _rev_new:
                    st.download_button(
                        f"⬇️ المُضافة للفرص ({len(_rev_new)})",
                        data=export_salla_csv(_rev_new),
                        file_name="Review_New.csv", mime="text/csv",
                        key="dl_rv_new")
            with rn2:
                if _rev_dup:
                    st.metric("🚫 رُفضت كمكررات", len(_rev_dup))
    else:
        # Pending review items
        curr_idx = min(
            next((i for i in range(len(rev_list)) if i not in _decisions), 0),
            len(rev_list) - 1
        )
        r = _rev_pend[0]   # Always show first pending
        real_idx = rev_list.index(r)

        n_done    = len(_decisions)
        n_total   = len(rev_list)
        n_pending = len(_rev_pend)

        st.markdown(
            f'<div class="tinder-counter">البطاقة {n_done+1} من {n_total} '
            f'| متبقٍ: {n_pending} | اتخذ قرارك ↓</div>',
            unsafe_allow_html=True)

        # Tinder card
        st.markdown('<div class="tinder-wrap">', unsafe_allow_html=True)

        col_a, col_vs, col_b = st.columns([5, 1, 5])

        with col_a:
            st.markdown(
                f'<div style="text-align:center;"><b style="color:#93c5fd;font-size:.85em;">🏪 الأقرب في متجرنا</b></div>'
                f'{_img_html(r.store_image, 170)}',
                unsafe_allow_html=True)
            store_short = (r.store_name[:55] + "…") if len(r.store_name) > 55 else r.store_name
            st.markdown(
                f'<p class="pcard-name" style="text-align:center;font-size:.82em;">{store_short}</p>',
                unsafe_allow_html=True)

        with col_vs:
            st.markdown(
                '<div style="display:flex;align-items:center;justify-content:center;'
                'height:200px;font-size:1.6em;color:rgba(245,158,11,.7);">↔</div>',
                unsafe_allow_html=True)

        with col_b:
            st.markdown(
                f'<div style="text-align:center;"><b style="color:#f9a8d4;font-size:.85em;">🔍 منتج المنافس</b></div>'
                f'{_img_html(r.comp_image, 170)}',
                unsafe_allow_html=True)
            comp_short = (r.comp_name[:55] + "…") if len(r.comp_name) > 55 else r.comp_name
            st.markdown(
                f'<p class="pcard-name" style="text-align:center;font-size:.82em;">{comp_short}</p>'
                f'<p class="pcard-meta" style="text-align:center;">{r.comp_source}</p>',
                unsafe_allow_html=True)

        # Score info
        feat_info = r.feature_details or r.llm_reasoning or "لم تُحدَّد تفاصيل"
        st.markdown(
            f'<div class="tinder-score">'
            f'🧠 دلالي: {r.faiss_score:.0%} &nbsp;|&nbsp; '
            f'📊 معجمي: {r.lex_score:.0%} &nbsp;|&nbsp; '
            f'🔎 {feat_info[:60]}'
            f'</div>',
            unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # /tinder-wrap

        # Action buttons
        ba, bb, bc = st.columns([2, 2, 1])
        with ba:
            if st.button("✅ فرصة جديدة — إضافة للقائمة",
                         type="primary", use_container_width=True, key=f"btn_new_{real_idx}"):
                st.session_state["review_decisions"][real_idx] = "new"
                st.rerun()
        with bb:
            if st.button("🚫 مكرر — استبعاد",
                         use_container_width=True, key=f"btn_dup_{real_idx}"):
                st.session_state["review_decisions"][real_idx] = "dup"
                st.rerun()
        with bc:
            if st.button("⏭️ تخطي", use_container_width=True, key=f"btn_skip_{real_idx}"):
                st.session_state["review_decisions"][real_idx] = "skip"
                st.rerun()

        # Progress mini-bar
        done_pct = n_done / n_total if n_total else 0
        st.markdown(f"""
<div style="margin-top:16px;">
    <div style="display:flex;justify-content:space-between;font-size:.75em;
                color:rgba(255,255,255,.35);margin-bottom:4px;">
        <span>تقدم المراجعة</span><span>{done_pct*100:.0f}%</span>
    </div>
    <div style="background:rgba(255,255,255,.06);border-radius:6px;height:6px;">
        <div style="width:{done_pct*100:.1f}%;height:100%;border-radius:6px;
                    background:linear-gradient(90deg,#f59e0b,#ef4444);"></div>
    </div>
</div>
""", unsafe_allow_html=True)

        # Review queue table
        if n_done > 0:
            st.markdown("---")
            st.markdown(f"**قرارات سابقة ({n_done})**")
            dec_rows = []
            for idx, dec in _decisions.items():
                if idx < len(rev_list) and dec != "skip":
                    dec_rows.append({
                        "المنتج": rev_list[idx].comp_name[:55],
                        "القرار": "✅ فرصة" if dec=="new" else "🚫 مكرر",
                        "الثقة":  f"{rev_list[idx].confidence:.0%}",
                    })
            if dec_rows:
                st.dataframe(pd.DataFrame(dec_rows), use_container_width=True, height=180)

        # Export decided
        if _rev_new:
            _, col_dl, _ = st.columns([1, 2, 1])
            with col_dl:
                st.download_button(
                    f"⬇️ تحميل الفرص المُقرَّرة ({len(_rev_new)})",
                    data=export_salla_csv(_rev_new),
                    file_name="Review_Approved.csv", mime="text/csv",
                    use_container_width=True, key="dl_rev_approved")


# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='text-align:center;color:rgba(255,255,255,.18);font-size:.78em;
            padding:20px 0 8px;border-top:1px solid rgba(255,255,255,.05);
            margin-top:36px;font-family:Cairo;'>
    🔬 <b style="color:rgba(255,255,255,.3);">Mahwous Hybrid Semantic Engine v7.0</b>
    &nbsp;·&nbsp; FAISS + paraphrase-multilingual-MiniLM + Gemini 1.5 Flash
    &nbsp;·&nbsp; Zero False Positives · Zero False Negatives
</div>
""", unsafe_allow_html=True)
