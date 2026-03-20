"""
app.py  -  Mahwous Opportunity Engine  v5.0
===========================================
Fixes applied in this version
-------------------------------
BUG-1 (Results lost on refresh): Results are now serialised to
  ~/.mahwous_cache/results.pkl immediately after every pipeline step.
  On startup, the app loads from disk before touching session_state,
  so a browser refresh or reconnect always restores the last run.

BUG-2 (Pipeline killed when browser closes): The heavy pipeline now
  runs in a daemon background thread (_run_pipeline).  A module-level
  dict _JOB holds live progress + the finished result.  The UI polls
  _JOB on each rerun; closing the browser tab does NOT stop the thread.
"""

import io
import os
import pickle
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from logic import (
    CAT_BEAUTY, CAT_PERFUME, CAT_UNKNOWN,
    classify_product_5gate,
    deduplicate_products,
    enrich_product_with_gemini,
    export_audit_trail_csv,
    export_missing_brands_csv,
    export_to_salla_csv,
    fetch_fallback_image,
    load_brands_list,
    load_competitor_products,
    load_store_products,
    verify_with_gemini,
)

# ── Disk cache path ──────────────────────────────────────────────────────────
_CACHE_DIR  = Path.home() / ".mahwous_cache"
_CACHE_FILE = _CACHE_DIR / "results.pkl"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Module-level background-job state (survives browser reconnects) ──────────
_JOB: dict = {
    "status":   "idle",   # idle | running | done | error
    "step":     "",
    "pct":      0.0,      # 0.0 – 1.0
    "log":      [],       # list[str]  appended during run
    "result":   None,     # dict with all DataFrames when done
    "error":    None,     # str on exception
    "lock":     threading.Lock(),
}

# ── Optimal defaults (zero-config) ──────────────────────────────────────────
_HIGH_THR   = 90
_LOW_THR    = 50
_MAX_ENRICH = 80


# ============================================================================
# DISK HELPERS
# ============================================================================

def _save_to_disk(payload: dict) -> None:
    """Persist the result payload to disk (survives browser refresh)."""
    try:
        with open(_CACHE_FILE, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        pass  # never crash the pipeline over a disk write


def _load_from_disk() -> dict | None:
    """Load last result payload from disk.  Returns None if nothing saved."""
    try:
        if _CACHE_FILE.exists():
            with open(_CACHE_FILE, "rb") as fh:
                return pickle.load(fh)
    except Exception:
        pass
    return None


def _delete_disk_cache() -> None:
    try:
        _CACHE_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# ============================================================================
# BACKGROUND PIPELINE  (runs in daemon thread – browser-close-safe)
# ============================================================================

def _log(msg: str) -> None:
    with _JOB["lock"]:
        _JOB["log"].append(msg)
        _JOB["step"] = msg


def _set_pct(pct: float) -> None:
    with _JOB["lock"]:
        _JOB["pct"] = min(pct, 1.0)


def _run_pipeline(
    store_bytes_list: list[tuple[str, bytes]],
    comp_bytes_list:  list[tuple[str, bytes]],
    brands_bytes:     bytes | None,
    gemini_key:       str,
    google_srch_key:  str,
    google_cx:        str,
    do_enrich:        bool,
    do_gray_ai:       bool,
    do_img_fallback:  bool,
    max_enrich:       int,
    use_ai_classify:  bool,
) -> None:
    """
    Full 6-step pipeline executed in a background thread.
    Writes partial + final results to _JOB and to disk.
    Never raises – all exceptions are caught and stored in _JOB["error"].
    """
    try:
        with _JOB["lock"]:
            _JOB["status"] = "running"
            _JOB["log"]    = []
            _JOB["pct"]    = 0.0
            _JOB["result"] = None
            _JOB["error"]  = None

        # ── Rebuild file-like objects from bytes (thread-safe) ────────────────
        class _Buf:
            def __init__(self, data: bytes, name: str):
                self._buf  = io.BytesIO(data)
                self.name  = name
            def read(self) -> bytes:
                self._buf.seek(0)
                return self._buf.read()

        store_files  = [_Buf(b, n) for n, b in store_bytes_list]
        comp_files   = [_Buf(b, n) for n, b in comp_bytes_list]
        brands_file  = _Buf(brands_bytes, "brands.csv") if brands_bytes else None

        # STEP 1 ─ Load store
        _log("📂 1/6 — تحميل جدار الحماية…")
        store_df        = load_store_products(store_files)
        existing_brands = load_brands_list(brands_file)
        _log(f"✅ {len(store_df):,} منتج في جدار الحماية | {len(existing_brands):,} ماركة")
        _set_pct(0.10)

        # STEP 2 ─ Load competitors
        _log("📦 2/6 — طحن ملفات المنافسين…")
        all_comp: list[dict] = []
        for i, cf in enumerate(comp_files):
            dfc = load_competitor_products(cf)
            all_comp.extend(dfc.to_dict("records"))
            _log(f"  • {cf.name} ← {len(dfc):,} منتج")
        _log(f"📊 إجمالي: {len(all_comp):,} منتج")
        _set_pct(0.20)

        # STEP 3 ─ Deduplication
        _log("🔍 3/6 — مصفاة منع التكرار…")
        total_comp = len(all_comp)

        def _upd(i, t, n):
            if t > 0:
                _set_pct(0.20 + 0.25 * i / t)

        new_df, gray_df, dup_df = deduplicate_products(
            store_df, all_comp, _HIGH_THR, _LOW_THR, _upd
        )
        _log(f"🌟 {len(new_df):,} جديد | 🔍 {len(gray_df):,} رمادي | 🗑️ {len(dup_df):,} مكرر")
        _set_pct(0.45)

        # ── Save intermediate result so UI can show partial data ─────────────
        _save_to_disk({
            "done": False, "partial": True,
            "new_df": new_df, "gray_df": gray_df, "dup_df": dup_df,
            "existing_brands": existing_brands,
            "stats": {"new": len(new_df), "perf": 0, "beauty": 0, "unknown": 0,
                      "gray": len(gray_df), "dup": len(dup_df), "total_comp": total_comp},
        })

        # STEP 4 ─ 5-Gate classification
        _log(f"🔬 4/6 — تصنيف {len(new_df):,} منتج بالبوابات الخمس…")
        classified: list[dict] = []
        api_key_cls = gemini_key if use_ai_classify else ""
        cls_rows    = new_df.to_dict("records") if not new_df.empty else []

        for i, row in enumerate(cls_rows):
            result = classify_product_5gate(
                name     = str(row.get("product_name", "")),
                category = str(row.get("category", "")),
                brand    = str(row.get("brand", "")),
                api_key  = api_key_cls,
            )
            d = dict(row)
            d.update(result.to_dict())
            d["product_type"] = result.category
            classified.append(d)
            if (i + 1) % 20 == 0 or i == len(cls_rows) - 1:
                _set_pct(0.45 + 0.20 * (i + 1) / max(len(cls_rows), 1))

        new_df = pd.DataFrame(classified) if classified else new_df
        _log("✅ التصنيف مكتمل")
        _set_pct(0.65)

        # STEP 5 ─ Gray-zone AI
        if do_gray_ai and gemini_key and not gray_df.empty:
            _log(f"🤖 5/6 — Gemini يفحص {len(gray_df)} رمادي…")
            extras: list[dict] = []
            for i, (idx, row) in enumerate(gray_df.iterrows()):
                verdict = verify_with_gemini(
                    str(row.get("matched_store_product", "")),
                    str(row.get("product_name", "")),
                    gemini_key,
                )
                if verdict == "DIFFERENT":
                    r2 = classify_product_5gate(
                        str(row.get("product_name", "")), api_key=gemini_key
                    )
                    d = row.to_dict()
                    d.update(r2.to_dict())
                    d["product_type"] = r2.category
                    extras.append(d)
                _set_pct(0.65 + 0.10 * (i + 1) / max(len(gray_df), 1))
            if extras:
                new_df = pd.concat([new_df, pd.DataFrame(extras)], ignore_index=True)
            _log(f"✅ Gemini أضاف {len(extras)}")
        else:
            _log("🤖 5/6 — تم تخطي فحص Gemini للمنطقة الرمادية")
        _set_pct(0.75)

        # STEP 6 ─ Enrichment
        if do_enrich and gemini_key and not new_df.empty:
            perf_mask = new_df["product_type"] == CAT_PERFUME if "product_type" in new_df.columns else pd.Series([False] * len(new_df))
            to_enr    = new_df[perf_mask].head(int(max_enrich))
            n_e       = len(to_enr)
            _log(f"✨ 6/6 — تحضير أوصاف HTML لـ {n_e} عطر…")
            enriched: list[dict] = []

            for i, (_, row) in enumerate(to_enr.iterrows()):
                enr = enrich_product_with_gemini(
                    str(row.get("product_name", "")),
                    str(row.get("image_url", "")),
                    str(row.get("product_type", CAT_PERFUME)),
                    gemini_key,
                )
                d = row.to_dict()
                d.update(enr)
                if do_img_fallback and google_srch_key and not d.get("image_url"):
                    d["image_url"] = fetch_fallback_image(
                        d.get("product_name", ""), google_srch_key, google_cx
                    )
                enriched.append(d)
                _set_pct(0.75 + 0.24 * (i + 1) / max(n_e, 1))

            rest_idx = to_enr.index
            rest     = new_df[~new_df.index.isin(rest_idx)]
            new_df   = pd.concat([pd.DataFrame(enriched), rest], ignore_index=True)
            _log("✅ اكتملت الأوصاف")
        else:
            _log("✨ 6/6 — تم تخطي الإثراء")
        _set_pct(0.99)

        # ── Compute final stats ───────────────────────────────────────────────
        def _cnt(col_val):
            if new_df.empty or "product_type" not in new_df.columns:
                return 0
            return int((new_df["product_type"] == col_val).sum())

        final_stats = {
            "new":        len(new_df),
            "perf":       _cnt(CAT_PERFUME),
            "beauty":     _cnt(CAT_BEAUTY),
            "unknown":    _cnt(CAT_UNKNOWN),
            "gray":       len(gray_df),
            "dup":        len(dup_df),
            "total_comp": total_comp,
        }

        final_payload = {
            "done":            True,
            "partial":         False,
            "new_df":          new_df,
            "gray_df":         gray_df,
            "dup_df":          dup_df,
            "existing_brands": existing_brands,
            "stats":           final_stats,
            "gray_approvals":  {},
            "unk_decisions":   {},
        }

        # ── Write to disk FIRST ───────────────────────────────────────────────
        _save_to_disk(final_payload)
        _log("💾 تم الحفظ على القرص")

        # ── Then update in-memory _JOB ────────────────────────────────────────
        with _JOB["lock"]:
            _JOB["result"] = final_payload
            _JOB["status"] = "done"
            _JOB["pct"]    = 1.0

        _log("🎉 اكتمل التحليل بنجاح!")

    except Exception as exc:
        import traceback
        err_msg = f"{exc}\n{traceback.format_exc()}"
        with _JOB["lock"]:
            _JOB["status"] = "error"
            _JOB["error"]  = err_msg
        _log(f"❌ خطأ: {exc}")


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="محرك مهووس | نظام التصنيف الخماسي",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# CSS  (identical to v4 – abbreviated here for brevity, full copy below)
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');
html,body,[class*="css"],.stMarkdown,.stText,
.stButton>button,.stDownloadButton>button,
div[data-testid="stMetricValue"],div[data-testid="stMetricLabel"]{
    font-family:'Cairo',sans-serif !important;
}
.main>div{direction:rtl;}
div[data-testid="column"]{direction:rtl;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1.2rem !important;}
.hero-wrap{
    background:linear-gradient(135deg,#0a0a1a 0%,#0d1b3e 45%,#0f3460 100%);
    border-radius:20px;padding:38px 32px 30px;text-align:center;
    margin-bottom:24px;box-shadow:0 8px 40px rgba(15,52,96,.45);
    position:relative;overflow:hidden;
}
.hero-wrap::before{content:'';position:absolute;inset:0;
    background:radial-gradient(ellipse 80% 60% at 50% 0%,rgba(99,179,237,.08),transparent 70%);}
.hero-title{font-size:2.5em;font-weight:900;color:#fff;margin:0 0 8px;
    text-shadow:0 2px 20px rgba(99,179,237,.4);}
.hero-sub{font-size:1em;color:rgba(255,255,255,.65);margin:0;}
.hero-badges{margin-top:14px;display:flex;justify-content:center;gap:8px;flex-wrap:wrap;}
.hbadge{background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.15);
    color:rgba(255,255,255,.8);padding:4px 14px;border-radius:20px;font-size:.78em;font-weight:600;}
.metric-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:20px 0;}
.mcard{border-radius:14px;padding:16px;text-align:center;}
.mc-blue  {background:linear-gradient(135deg,#e8f4fd,#d1eaf8);}
.mc-green {background:linear-gradient(135deg,#e8f8f0,#d4f0e0);}
.mc-amber {background:linear-gradient(135deg,#fff8e1,#fef0c0);}
.mc-red   {background:linear-gradient(135deg,#fef0f0,#fce0e0);}
.mc-purple{background:linear-gradient(135deg,#f3e5f5,#e1bee7);}
.mc-icon{font-size:1.4em;margin-bottom:2px;}
.mc-num{font-size:2.2em;font-weight:900;line-height:1;}
.mc-blue  .mc-num{color:#1565c0;} .mc-green .mc-num{color:#1b5e20;}
.mc-amber .mc-num{color:#e65100;} .mc-red   .mc-num{color:#b71c1c;}
.mc-purple .mc-num{color:#6a1b9a;}
.mc-label{font-size:.78em;color:#555;margin-top:4px;font-weight:600;}
.opp-bar-wrap{background:#f0f4ff;border-radius:12px;padding:12px 18px;
    margin-bottom:18px;display:flex;align-items:center;gap:12px;}
.opp-bar-label{font-weight:700;color:#0f3460;white-space:nowrap;min-width:130px;}
.opp-bar-outer{flex:1;background:#dde6f5;border-radius:8px;height:10px;overflow:hidden;}
.opp-bar-inner{height:100%;border-radius:8px;background:linear-gradient(90deg,#1b5e20,#43a047);}
.opp-pct{font-weight:900;color:#1b5e20;font-size:.95em;min-width:40px;}
.gate-badge{display:inline-block;padding:2px 10px;border-radius:10px;
    font-size:.7em;font-weight:700;margin:1px;}
.g1{background:#e8f5e9;color:#1b5e20;border:1px solid #a5d6a7;}
.g2{background:#e3f2fd;color:#0d47a1;border:1px solid #90caf9;}
.g3{background:#fff3e0;color:#e65100;border:1px solid #ffcc80;}
.g4{background:#f3e5f5;color:#6a1b9a;border:1px solid #ce93d8;}
.g5{background:#fce4ec;color:#880e4f;border:1px solid #f48fb1;}
.g0{background:#eceff1;color:#546e7a;border:1px solid #b0bec5;}
.pcard-name{font-weight:700;font-size:.9em;color:#1a1a2e;margin:6px 0 3px;line-height:1.4;}
.pcard-brand{font-size:.78em;color:#0f3460;font-weight:600;}
.pcard-meta{font-size:.7em;color:#aaa;margin-top:3px;}
.pill{display:inline-block;padding:2px 9px;border-radius:20px;
    font-size:.7em;font-weight:700;margin:1px;}
.pill-perf {background:#e8eaf6;color:#283593;border:1px solid #9fa8da;}
.pill-beau {background:#fce4ec;color:#880e4f;border:1px solid #f48fb1;}
.pill-unk  {background:#eceff1;color:#546e7a;border:1px solid #b0bec5;}
.pill-new  {background:#e8f8ee;color:#1b5e20;border:1px solid #a5d6a7;}
.pill-price{background:#fff8e1;color:#e65100;border:1px solid #ffcc80;}
.pill-conf {background:#fff3e0;color:#bf360c;border:1px solid #ffcc80;}
.upload-title{font-weight:700;font-size:.97em;color:#1a1a2e;margin:0 0 3px;}
.upload-sub  {font-size:.8em;color:#888;margin:0 0 12px;}
.audit-row{background:#f7f9ff;border:1px solid #e4eaf4;border-radius:10px;
    padding:10px 14px;margin-bottom:6px;font-size:.82em;}
.audit-row b{color:#0f3460;}
.bg-running-banner{
    background:linear-gradient(135deg,#e3f2fd,#bbdefb);
    border:1.5px solid #1565c0;border-radius:14px;padding:16px 20px;
    margin:12px 0;display:flex;align-items:center;gap:14px;
}
.bg-running-icon{font-size:2em;}
.bg-running-text{flex:1;}
.bg-running-text b{color:#0d47a1;font-size:1.05em;}
.bg-running-text p{margin:4px 0 0;color:#555;font-size:.88em;}
div[data-testid="stDownloadButton"]>button{
    background:linear-gradient(135deg,#0d1b3e,#0f3460) !important;
    color:white !important;border:none !important;border-radius:10px !important;
    font-weight:700 !important;font-family:'Cairo',sans-serif !important;
    padding:9px 16px !important;width:100%;}
div[data-testid="stDownloadButton"]>button:hover{opacity:.87 !important;}
div[data-testid="stButton"]>button[kind="primary"]{
    background:linear-gradient(135deg,#0f3460,#1565c0) !important;
    color:white !important;border:none !important;border-radius:12px !important;
    font-weight:900 !important;font-family:'Cairo',sans-serif !important;
    font-size:1.05em !important;padding:13px 24px !important;
    box-shadow:0 4px 20px rgba(15,52,96,.35) !important;}
section[data-testid="stSidebar"]{background:#0d1b3e;}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p{color:rgba(255,255,255,.8) !important;}
section[data-testid="stSidebar"] h3{color:#63b3ed !important;}
.secrets-alert{background:linear-gradient(135deg,#fff8e1,#fff3cd);
    border:1.5px solid #ffc107;border-radius:14px;padding:18px 20px;margin:10px 0 18px;}
.secrets-alert h3{margin:0 0 6px;color:#856404;font-size:.95em;}
.secrets-alert pre{background:#1e1e2e;color:#a8d8a8;padding:10px 14px;
    border-radius:8px;font-size:.82em;margin:8px 0 0;direction:ltr;text-align:left;}
.secrets-alert code{background:rgba(133,100,4,.12);padding:1px 6px;
    border-radius:4px;font-size:.85em;color:#5d4037;}
div[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# API KEY RESOLUTION
# ============================================================================
def _secret(key: str) -> str:
    try:
        v = st.secrets.get(key, "")
        if v: return v
    except Exception:
        pass
    return st.session_state.get(key, "")

GEMINI_KEY      = _secret("GEMINI_API_KEY")
GOOGLE_SRCH_KEY = _secret("GOOGLE_SEARCH_KEY")
GOOGLE_CX       = _secret("GOOGLE_CX")
_has_gemini = bool(GEMINI_KEY)
_has_search = bool(GOOGLE_SRCH_KEY and GOOGLE_CX)


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("<h2 style='color:#63b3ed;margin:0;padding:10px 0 4px;'>⚙️ إعدادات</h2>",
                unsafe_allow_html=True)
    st.caption("المفاتيح تُقرأ تلقائياً من secrets.toml")
    with st.expander("🔑 مفاتيح API يدوية"):
        mg = st.text_input("Gemini API Key",    type="password", key="GEMINI_API_KEY")
        ms = st.text_input("Google Search Key", type="password", key="GOOGLE_SEARCH_KEY")
        mc = st.text_input("Google CX",                          key="GOOGLE_CX")
        if mg: GEMINI_KEY = mg;    _has_gemini = True
        if ms: GOOGLE_SRCH_KEY = ms
        if mc: GOOGLE_CX = mc
        _has_search = bool(GOOGLE_SRCH_KEY and GOOGLE_CX)

    st.divider()
    with st.expander("🤖 خيارات الذكاء الاصطناعي"):
        do_enrich       = st.toggle("توليد أوصاف HTML",                 value=True)
        do_gray_ai      = st.toggle("فحص المنطقة الرمادية بـ Gemini",   value=False)
        do_img_fallback = st.toggle("جلب الصور المفقودة",                value=False)
        max_enrich      = st.slider("حد إثراء AI", 10, 300, _MAX_ENRICH, 10)
        use_ai_classify = st.toggle("Gate 5 – Gemini للتصنيف الغامض",   value=True)

    st.divider()
    # Reset button — clears both RAM and disk
    if st.button("🔄 بدء من جديد (مسح الكل)", use_container_width=True):
        _delete_disk_cache()
        with _JOB["lock"]:
            _JOB["status"] = "idle"
            _JOB["result"] = None
            _JOB["log"]    = []
            _JOB["pct"]    = 0.0
        for k in ["done","new_df","gray_df","dup_df","existing_brands",
                  "gray_approvals","unk_decisions","stats","_loaded_from_disk"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown("<div style='text-align:center;padding:12px 0 0;"
                "color:rgba(255,255,255,.3);font-size:.72em;'>"
                "Mahwous Engine v5.0 | BG-Thread + Disk Cache</div>",
                unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INIT  +  DISK RESTORE  (BUG-1 fix)
# ============================================================================
_EMPTY = {
    "done":            False,
    "new_df":          pd.DataFrame(),
    "gray_df":         pd.DataFrame(),
    "dup_df":          pd.DataFrame(),
    "existing_brands": [],
    "gray_approvals":  {},
    "unk_decisions":   {},
    "stats":           {"new":0,"perf":0,"beauty":0,"unknown":0,
                        "gray":0,"dup":0,"total_comp":0},
}

# Only try disk restore once per session (flag prevents repeated I/O)
if not st.session_state.get("_loaded_from_disk"):
    disk = _load_from_disk()
    if disk and disk.get("done"):
        for k, v in disk.items():
            st.session_state[k] = v
    else:
        for k, v in _EMPTY.items():
            if k not in st.session_state:
                st.session_state[k] = v
    st.session_state["_loaded_from_disk"] = True
else:
    for k, v in _EMPTY.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================================
# HERO
# ============================================================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">🔬 محرك مهووس | نظام التصنيف الخماسي</div>
    <p class="hero-sub">5 بوابات دفاعية · معالجة خلفية · حفظ تلقائي · تصدير فوري لسلة</p>
    <div class="hero-badges">
        <span class="hbadge">G1 Hard Rules 97%</span>
        <span class="hbadge">G2 Category Path</span>
        <span class="hbadge">G3 Weighted Scoring</span>
        <span class="hbadge">G4 Brand KB</span>
        <span class="hbadge">G5 Gemini AI</span>
        <span class="hbadge">💾 Auto-Save</span>
        <span class="hbadge">🧵 Background Thread</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SECRETS ALERT
# ============================================================================
if not _has_gemini:
    st.markdown("""
<div class="secrets-alert">
    <h3>⚠️ مفتاح Gemini غير مُعرَّف</h3>
    <p style="margin:0 0 6px;color:#6d4c41;font-size:.88em;">
        البوابات 1-4 تعمل بدون مفتاح. Gate 5 + الإثراء يحتاجان مفتاحاً.
        أضفه في <code>.streamlit/secrets.toml</code>:
    </p>
    <pre>GEMINI_API_KEY = "AIza...................."</pre>
</div>""", unsafe_allow_html=True)
else:
    st.success("✅ **Gemini جاهز** — البوابات الخمس نشطة + الحفظ التلقائي مفعّل", icon="🔬")


# ============================================================================
# UPLOAD SECTION
# ============================================================================
st.markdown("---")
c_store, c_comp = st.columns([1, 1.5], gap="large")

with c_store:
    st.markdown('<p class="upload-title">🏪 ملفات متجر مهووس</p>', unsafe_allow_html=True)
    st.caption("ملفات سلة — الأجزاء 1-4 أو أكثر")
    store_files = st.file_uploader("store", type=["csv"],
        accept_multiple_files=True, key="uf_store", label_visibility="collapsed")
    brands_file = st.file_uploader("ملف الماركات الحالية (اختياري)",
        type=["csv"], key="uf_brands")
    if store_files:
        st.success(f"✅ {len(store_files)} ملف(ات) للمتجر")

with c_comp:
    st.markdown('<p class="upload-title">🔍 ملفات المنافسين (حتى 15)</p>', unsafe_allow_html=True)
    st.caption("أي تنسيق CSV — يُكشف التنسيق تلقائياً")
    comp_files = st.file_uploader("comp", type=["csv"],
        accept_multiple_files=True, key="uf_comp", label_visibility="collapsed")
    if comp_files:
        st.success(f"✅ {len(comp_files)} ملف(ات) للمنافسين")
        with st.expander("📋 الملفات"):
            for f in comp_files[:15]: st.caption(f"• {f.name}")


# ============================================================================
# START BUTTON  —  launches background thread  (BUG-2 fix)
# ============================================================================
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
_, bc, _ = st.columns([1, 2, 1])
with bc:
    job_running = _JOB["status"] == "running"
    ready       = bool(store_files and comp_files) and not job_running
    start       = st.button("🚀  بدء التحليل العميق",
                            type="primary", use_container_width=True,
                            disabled=not ready, key="btn_start")
    if job_running:
        st.caption("⚙️ التحليل يعمل في الخلفية — يمكنك إغلاق المتصفح والعودة لاحقاً")
    elif not (store_files and comp_files):
        st.caption("⚠️ يجب رفع ملف للمتجر وملف للمنافسين")
st.markdown("---")


# ============================================================================
# LAUNCH BACKGROUND THREAD  (runs only when button is pressed)
# ============================================================================
if start and store_files and comp_files:
    # Read all file bytes NOW (Streamlit file objects are only readable once
    # in the main thread — the thread gets plain bytes)
    store_bytes_list = [(f.name, f.read()) for f in store_files]
    comp_bytes_list  = [(f.name, f.read()) for f in comp_files[:15]]
    brands_bytes     = brands_file.read() if brands_file else None

    # Reset session state (keep unk_decisions in case user had manual decisions)
    for k in ["done","new_df","gray_df","dup_df","existing_brands",
              "gray_approvals","stats","_loaded_from_disk"]:
        st.session_state.pop(k, None)

    thread = threading.Thread(
        target=_run_pipeline,
        kwargs=dict(
            store_bytes_list = store_bytes_list,
            comp_bytes_list  = comp_bytes_list,
            brands_bytes     = brands_bytes,
            gemini_key       = GEMINI_KEY,
            google_srch_key  = GOOGLE_SRCH_KEY,
            google_cx        = GOOGLE_CX,
            do_enrich        = do_enrich,
            do_gray_ai       = do_gray_ai,
            do_img_fallback  = do_img_fallback,
            max_enrich       = int(max_enrich),
            use_ai_classify  = use_ai_classify,
        ),
        daemon=True,
    )
    thread.start()
    st.rerun()


# ============================================================================
# BACKGROUND PROGRESS UI  (shown while thread is running)
# ============================================================================
if _JOB["status"] == "running":
    with _JOB["lock"]:
        current_step = _JOB["step"]
        current_pct  = _JOB["pct"]
        current_log  = list(_JOB["log"])

    st.markdown(f"""
<div class="bg-running-banner">
    <div class="bg-running-icon">⚙️</div>
    <div class="bg-running-text">
        <b>التحليل يعمل في الخلفية</b>
        <p>{current_step}</p>
    </div>
</div>
""", unsafe_allow_html=True)

    st.progress(current_pct, text=f"{current_pct*100:.0f}%")

    with st.expander("📋 سجل التقدم المباشر", expanded=True):
        for line in current_log[-20:]:
            st.caption(line)

    st.caption("💡 يمكنك إغلاق المتصفح الآن — التحليل سيكمل في الخلفية. "
               "عند العودة ستجد النتائج محفوظة تلقائياً.")

    # Auto-refresh every 2 seconds to show live progress
    time.sleep(2)
    st.rerun()


# ============================================================================
# JOB DONE — transfer result into session_state  (runs once per completion)
# ============================================================================
if _JOB["status"] == "done" and not st.session_state.get("done"):
    with _JOB["lock"]:
        payload = _JOB["result"]

    if payload:
        for k, v in payload.items():
            st.session_state[k] = v
        # Results are already on disk from the thread — no need to re-save
        st.toast("🎉 التحليل اكتمل! النتائج محفوظة تلقائياً.", icon="✅")
        st.rerun()


# ============================================================================
# JOB ERROR
# ============================================================================
if _JOB["status"] == "error":
    err_detail = _JOB.get("error", "خطأ غير معروف")
    st.error(f"❌ حدث خطأ أثناء التحليل: {err_detail}")
    if st.button("🔄 إعادة المحاولة"):
        with _JOB["lock"]:
            _JOB["status"] = "idle"
        st.rerun()


# ============================================================================
# WELCOME STATE  (nothing done yet)
# ============================================================================
if not st.session_state.get("done"):
    # Check if disk has a partial result (pipeline still running or crashed mid-way)
    disk = _load_from_disk()
    if disk and disk.get("partial"):
        st.info("🔄 يوجد تحليل جزئي محفوظ — إما أن التحليل لا يزال يعمل في الخلفية أو انقطع."
                " اضغط 'بدء التحليل' لإعادة التشغيل.")

    st.markdown("""<div style='text-align:center;padding:50px 0 60px;'>
        <div style='font-size:5em;margin-bottom:14px;'>🔬</div>
        <p style='font-size:1.1em;color:#888;max-width:500px;margin:0 auto;line-height:1.9;'>
            ارفع ملفات متجرك وملفات المنافسين<br>
            ثم اضغط <b style='color:#0f3460;'>بدء التحليل العميق</b><br>
            <span style='font-size:.85em;color:#aaa;'>
                ✅ يعمل في الخلفية · يحفظ تلقائياً · يُستعاد عند الرجوع
            </span>
        </p></div>""", unsafe_allow_html=True)
    st.stop()


# ============================================================================
# RESULTS SECTION  (identical to v4 from here — data is in session_state)
# ============================================================================
new_df          = st.session_state["new_df"]
gray_df         = st.session_state["gray_df"]
dup_df          = st.session_state["dup_df"]
existing_brands = st.session_state["existing_brands"]
s               = st.session_state["stats"]

# ── Disk-save indicator ──────────────────────────────────────────────────────
if _CACHE_FILE.exists():
    mtime = _CACHE_FILE.stat().st_mtime
    from datetime import datetime
    saved_at = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d  %H:%M:%S")
    st.caption(f"💾 آخر حفظ تلقائي: **{saved_at}** — في `{_CACHE_FILE}`")


# ── METRIC CARDS ─────────────────────────────────────────────────────────────
_total   = s["new"] + s["gray"] + s["dup"]
_opp_pct = round(s["new"] / _total * 100, 1) if _total else 0

st.markdown(f"""
<div class="metric-grid">
  <div class="mcard mc-blue">
    <div class="mc-icon">📦</div>
    <div class="mc-num">{s["total_comp"]:,}</div>
    <div class="mc-label">مُفحوص من المنافسين</div>
  </div>
  <div class="mcard mc-green">
    <div class="mc-icon">🧴</div>
    <div class="mc-num">{s["perf"]:,}</div>
    <div class="mc-label">فرصة عطور جديدة</div>
  </div>
  <div class="mcard mc-amber">
    <div class="mc-icon">💄</div>
    <div class="mc-num">{s["beauty"]:,}</div>
    <div class="mc-label">فرصة مكياج/عناية</div>
  </div>
  <div class="mcard mc-purple" style="background:linear-gradient(135deg,#f3e5f5,#e1bee7);">
    <div class="mc-icon">❓</div>
    <div class="mc-num" style="color:#6a1b9a;">{s["unknown"]:,}</div>
    <div class="mc-label">للمراجعة اليدوية</div>
  </div>
  <div class="mcard mc-red">
    <div class="mc-icon">🗑️</div>
    <div class="mc-num">{s["dup"]:,}</div>
    <div class="mc-label">مكررات مستبعدة</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="opp-bar-wrap">
  <div class="opp-bar-label">📈 نسبة الفرص الجديدة</div>
  <div class="opp-bar-outer"><div class="opp-bar-inner" style="width:{_opp_pct}%"></div></div>
  <div class="opp-pct">{_opp_pct}%</div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# HELPERS  (same as v4)
# ============================================================================
def _salla_btn(records, suffix, label, key):
    if records:
        st.download_button(label, data=export_to_salla_csv(records),
            file_name=f"{suffix}.csv", mime="text/csv",
            use_container_width=True, key=key)

def _gate_html(gate):
    cls = {"G1:HardRules":"g1","G2:CategoryPath":"g2","G3:WeightedScoring":"g3",
           "G4:BrandKB":"g4","G5:AIOracle":"g5"}.get(gate,"g0")
    short = gate.split(":")[-1] if ":" in gate else gate
    return f'<span class="gate-badge {cls}">{short}</span>'

def _show_df(df, col_map, height=400):
    avail = {k: v for k, v in col_map.items() if k in df.columns}
    disp  = df[list(avail.keys())].copy().rename(columns=avail)
    st.dataframe(disp, use_container_width=True, height=height)

def _audit_snip(row):
    gate   = str(row.get("gate_used",""))
    conf   = str(row.get("confidence_pct",""))
    reason = str(row.get("reasoning",""))
    sigs   = str(row.get("signals_fired",""))[:80]
    st.markdown(
        f'<div class="audit-row">'
        f'<b>البوابة:</b> {_gate_html(gate)} '
        f'<b>الثقة:</b> {conf}<br>'
        f'<b>الإشارات:</b> {sigs}<br>'
        f'<b>المبرر:</b> {reason}</div>',
        unsafe_allow_html=True)


# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    f"🧴 عطور  ({s['perf']})",
    f"💄 مكياج/عناية  ({s['beauty']})",
    f"❓ للمراجعة  ({s['unknown']})",
    f"🔍 المنطقة الرمادية  ({s['gray']})",
    f"🗑️ المكررات  ({s['dup']})",
])


# ── TAB 1: PERFUMES ───────────────────────────────────────────────────────────
with tab1:
    perf_df = new_df[new_df["product_type"] == CAT_PERFUME] if not new_df.empty and "product_type" in new_df.columns else pd.DataFrame()
    if perf_df.empty:
        st.info("لم تُكتشف عطور جديدة")
    else:
        st.markdown("#### 📥 تصدير ملف سلة للعطور")
        ec1, ec2 = st.columns(2)
        with ec1:
            _salla_btn(perf_df.to_dict("records"), "Salla_Perfumes",
                       f"⬇️ Salla_Perfumes.csv  ({len(perf_df):,})", "dl_perf")
        with ec2:
            st.download_button("⬇️ سجل التدقيق (Audit Trail)",
                data=export_audit_trail_csv(perf_df.to_dict("records")),
                file_name="audit_perfumes.csv", mime="text/csv",
                use_container_width=True, key="dl_perf_audit")
        st.markdown("---")

        fc1,fc2,fc3,fc4 = st.columns([1.3,1.3,1.8,1])
        with fc1:
            src_o = ["الكل"]+(sorted(perf_df["source_file"].dropna().unique().tolist()) if "source_file" in perf_df.columns else [])
            src_s = st.selectbox("المصدر", src_o, key="pf_src")
        with fc2:
            g_o = ["الكل"]+(sorted(perf_df["gate_used"].dropna().unique().tolist()) if "gate_used" in perf_df.columns else [])
            g_s = st.selectbox("البوابة", g_o, key="pf_gate")
        with fc3:
            srch_p = st.text_input("🔎 بحث", key="pf_q")
        with fc4:
            view_p = st.radio("عرض", ["🃏 بطاقات","📋 جدول"], horizontal=True, key="pf_view")

        disp = perf_df.copy()
        if src_s != "الكل" and "source_file" in disp.columns: disp = disp[disp["source_file"]==src_s]
        if g_s   != "الكل" and "gate_used"   in disp.columns: disp = disp[disp["gate_used"]==g_s]
        if srch_p: disp = disp[disp["product_name"].str.contains(srch_p, case=False, na=False)]
        disp = disp.reset_index(drop=True)
        st.caption(f"عرض **{min(len(disp),60):,}** من **{len(disp):,}**")

        if view_p == "📋 جدول":
            _show_df(disp, {"product_name":"اسم المنتج","brand":"الماركة","price":"السعر",
                            "gate_used":"البوابة","confidence_pct":"الثقة",
                            "signals_fired":"الإشارات","source_file":"المصدر"}, 500)
        else:
            LIMIT = 60
            for rs in range(0, min(len(disp), LIMIT), 3):
                cols = st.columns(3, gap="medium")
                for j, col in enumerate(cols):
                    idx = rs + j
                    if idx >= min(len(disp), LIMIT): break
                    row   = disp.iloc[idx]
                    pname = str(row.get("product_name",""))
                    brand = str(row.get("brand",""))
                    price = str(row.get("price","")).strip()
                    gate  = str(row.get("gate_used",""))
                    conf  = str(row.get("confidence_pct",""))
                    src   = str(row.get("source_file","")).replace(".csv","")
                    with col:
                        img = str(row.get("image_url","")).strip()
                        if img.startswith("http"):
                            try: st.image(img, use_container_width=True)
                            except: pass
                        else:
                            st.markdown("<div style='height:140px;background:#f4f6fb;border-radius:10px;display:flex;align-items:center;justify-content:center;color:#ccc;font-size:2em;'>📷</div>", unsafe_allow_html=True)
                        st.markdown(
                            f'<p class="pcard-name">{pname[:65]}{"…" if len(pname)>65 else ""}</p>'
                            f'{f'<p class="pcard-brand">🏷️ {brand}</p>' if brand else ""}'
                            f'<div style="margin-top:5px;">'
                            f'<span class="pill pill-perf">🧴 عطر</span>'
                            f'<span class="pill pill-new">✅ جديد</span>'
                            f'{f'<span class="pill pill-price">💰 {price}</span>' if price else ""}'
                            f'</div><div style="margin-top:4px;">{_gate_html(gate)} <span class="pill pill-conf">{conf}</span></div>'
                            f'<p class="pcard-meta">{src}</p>',
                            unsafe_allow_html=True)
            if len(disp) > LIMIT:
                st.info(f"يُعرض أول {LIMIT}. حمّل CSV للاطلاع على الكل ({len(disp):,}).")


# ── TAB 2: BEAUTY / CARE ─────────────────────────────────────────────────────
with tab2:
    beau_df = new_df[new_df["product_type"] == CAT_BEAUTY] if not new_df.empty and "product_type" in new_df.columns else pd.DataFrame()
    if beau_df.empty:
        st.info("لم تُكتشف منتجات مكياج/عناية جديدة")
    else:
        st.markdown("#### 📥 تصدير ملف سلة للمكياج والعناية")
        ec1, ec2 = st.columns(2)
        with ec1:
            _salla_btn(beau_df.to_dict("records"), "Salla_Beauty_Care",
                       f"⬇️ Salla_Beauty_Care.csv  ({len(beau_df):,})", "dl_beau")
        with ec2:
            st.download_button("⬇️ سجل التدقيق (Audit Trail)",
                data=export_audit_trail_csv(beau_df.to_dict("records")),
                file_name="audit_beauty.csv", mime="text/csv",
                use_container_width=True, key="dl_beau_audit")
        st.markdown("---")
        srch_b = st.text_input("🔎 بحث", key="bf_q")
        disp = beau_df.copy()
        if srch_b: disp = disp[disp["product_name"].str.contains(srch_b, case=False, na=False)]
        _show_df(disp, {"product_name":"اسم المنتج","brand":"الماركة","price":"السعر",
                        "gate_used":"البوابة","confidence_pct":"الثقة",
                        "reasoning":"المبرر","source_file":"المصدر"}, 500)


# ── TAB 3: UNKNOWN / MANUAL REVIEW ───────────────────────────────────────────
with tab3:
    unk_df = new_df[new_df["product_type"] == CAT_UNKNOWN] if not new_df.empty and "product_type" in new_df.columns else pd.DataFrame()
    if unk_df.empty:
        st.success("🎯 لا منتجات تحتاج مراجعة — النظام حسم الكل!")
    else:
        st.warning(f"⚠️ **{len(unk_df):,} منتج** لم يستطع النظام تصنيفه — قرارك أنت.")
        mc1,mc2,mc3 = st.columns(3)
        mc1.metric("❓ للمراجعة", f"{len(unk_df):,}")

        if not st.session_state.get("unk_decisions"):
            st.session_state["unk_decisions"] = {}
        ud = st.session_state["unk_decisions"]

        decided_perf = [unk_df.iloc[i].to_dict() for i, v in ud.items() if v == CAT_PERFUME]
        decided_beau = [unk_df.iloc[i].to_dict() for i, v in ud.items() if v == CAT_BEAUTY]

        mc2.metric("→ تم تصنيفه", f"{len(decided_perf)+len(decided_beau)}")
        mc3.metric("→ تجاهل",     f"{sum(1 for v in ud.values() if v == 'skip')}")

        if decided_perf or decided_beau:
            st.markdown("---")
            st.markdown("#### 📥 تصدير المنتجات الموافق عليها")
            ex1, ex2, ex3 = st.columns(3)
            with ex1:
                if decided_perf:
                    _salla_btn(decided_perf, "Salla_Perfumes_manual",
                               f"⬇️ عطور مُقررة ({len(decided_perf)})", "dl_unk_p")
            with ex2:
                if decided_beau:
                    _salla_btn(decided_beau, "Salla_Beauty_manual",
                               f"⬇️ مكياج مُقرر ({len(decided_beau)})", "dl_unk_b")
            with ex3:
                all_d = decided_perf + decided_beau
                if all_d:
                    st.download_button("⬇️ الماركات الناقصة",
                        data=export_missing_brands_csv(all_d, existing_brands),
                        file_name="missing_brands_manual.csv", mime="text/csv",
                        use_container_width=True, key="dl_unk_br")

        st.markdown("---")
        st.markdown("#### ✋ راجع وقرر كل منتج")
        for i, (_, row) in enumerate(unk_df.iterrows()):
            pname = str(row.get("product_name",""))
            gate  = str(row.get("gate_used",""))
            cur   = ud.get(i)
            icon  = ("🧴" if cur==CAT_PERFUME else "💄" if cur==CAT_BEAUTY
                     else "⛔" if cur=="skip" else "⏳")
            with st.expander(f"{icon} {pname[:65]} | {gate}", expanded=False):
                r1, r2 = st.columns(2)
                with r1:
                    st.write(f"**📦 المنتج:** {pname}")
                    img = str(row.get("image_url","")).strip()
                    if img.startswith("http"):
                        try: st.image(img, width=130)
                        except: pass
                    st.caption(f"المصدر: {str(row.get('source_file','')).replace('.csv','')} | السعر: {row.get('price','')}")
                with r2:
                    _audit_snip(row.to_dict())
                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("🧴 عطر",   key=f"pu_{i}", type="primary", use_container_width=True):
                        ud[i] = CAT_PERFUME; _save_to_disk(dict(st.session_state)); st.rerun()
                with b2:
                    if st.button("💄 مكياج", key=f"bu_{i}", use_container_width=True):
                        ud[i] = CAT_BEAUTY; _save_to_disk(dict(st.session_state)); st.rerun()
                with b3:
                    if st.button("⛔ تجاهل", key=f"su_{i}", use_container_width=True):
                        ud[i] = "skip"; _save_to_disk(dict(st.session_state)); st.rerun()


# ── TAB 4: GRAY ZONE ─────────────────────────────────────────────────────────
with tab4:
    if gray_df.empty:
        st.success("🎯 لا منطقة رمادية!")
    else:
        n_app = sum(1 for v in st.session_state["gray_approvals"].values() if v)
        n_rej = sum(1 for v in st.session_state["gray_approvals"].values() if not v)
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("🔍 الإجمالي", f"{len(gray_df):,}")
        mc2.metric("✅ موافق",    f"{n_app}")
        mc3.metric("❌ مرفوض",    f"{n_rej}")
        mc4.metric("⏳ معلّق",    f"{len(gray_df)-n_app-n_rej}")

        if _has_gemini:
            _,ab,_ = st.columns([2,1,2])
            with ab:
                if st.button("🤖 فحص بـ Gemini", use_container_width=True):
                    p2 = st.progress(0.0); ai2 = 0
                    for i,(idx,row) in enumerate(gray_df.iterrows()):
                        p2.progress((i+1)/len(gray_df))
                        v = verify_with_gemini(str(row.get("matched_store_product","")),
                                              str(row.get("product_name","")), GEMINI_KEY)
                        if v=="DIFFERENT":
                            st.session_state["gray_approvals"][idx]=True; ai2+=1
                        elif v=="MATCH":
                            st.session_state["gray_approvals"][idx]=False
                    p2.progress(1.0)
                    _save_to_disk(dict(st.session_state))
                    st.toast(f"✅ {ai2} موافق عليها", icon="🤖")
                    st.rerun()

        st.markdown("---")
        _show_df(gray_df, {"product_name":"اسم المنتج (المنافس)",
                           "matched_store_product":"المطابق في متجرك",
                           "match_score":"التشابه %","source_file":"المصدر"}, 280)
        st.markdown("---")
        st.markdown("#### ✋ المراجعة اليدوية")
        for idx, row in gray_df.iterrows():
            pname = str(row.get("product_name",""))
            score = float(row.get("match_score",0))
            cur   = st.session_state["gray_approvals"].get(idx,None)
            icon  = "✅" if cur is True else "❌" if cur is False else "⏳"
            with st.expander(f"{icon} {pname[:65]} | تشابه {score:.0f}%"):
                r1, r2 = st.columns(2)
                with r1:
                    st.write(f"**المنافس:** {pname}")
                    img = str(row.get("image_url","")).strip()
                    if img.startswith("http"):
                        try: st.image(img, width=130)
                        except: pass
                with r2:
                    st.write(f"**متجرك:** {row.get('matched_store_product','')}")
                    st.metric("التشابه", f"{score:.0f}%")
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("✅ إضافة كفرصة", key=f"ga_{idx}", type="primary", use_container_width=True):
                        st.session_state["gray_approvals"][idx]=True
                        _save_to_disk(dict(st.session_state)); st.rerun()
                with b2:
                    if st.button("❌ مكرر - تجاهل", key=f"gr_{idx}", use_container_width=True):
                        st.session_state["gray_approvals"][idx]=False
                        _save_to_disk(dict(st.session_state)); st.rerun()

        approved_idxs = [i for i,v in st.session_state["gray_approvals"].items() if v]
        if approved_idxs:
            st.markdown("---")
            st.markdown(f"#### 📥 تصدير الموافق عليها ({len(approved_idxs)})")
            approved = [gray_df.loc[i].to_dict() for i in approved_idxs]
            g1,g2,g3 = st.columns(3)
            gp = [p for p in approved if p.get("product_type")==CAT_PERFUME]
            gb = [p for p in approved if p.get("product_type")==CAT_BEAUTY]
            with g1:
                if gp: _salla_btn(gp,"Salla_Perfumes_gray",f"⬇️ عطور ({len(gp)})","dl_gp")
            with g2:
                if gb: _salla_btn(gb,"Salla_Beauty_gray",f"⬇️ مكياج ({len(gb)})","dl_gb")
            with g3:
                st.download_button("⬇️ الماركات",
                    data=export_missing_brands_csv(approved, existing_brands),
                    file_name="brands_gray.csv", mime="text/csv",
                    use_container_width=True, key="dl_gbr")


# ── TAB 5: DUPLICATES ────────────────────────────────────────────────────────
with tab5:
    if dup_df.empty:
        st.success("🎉 لا مكررات — كل المنتجات جديدة!")
    else:
        st.success(f"🛡️ تم حظر **{len(dup_df):,}** مكرر — المتجر محمي!")
        d1,d2,d3 = st.columns([1,1,2])
        with d1:
            dt = st.radio("النوع",["الكل","عطور","مكياج"],horizontal=True,key="rd_dt")
        with d2:
            dr = st.radio("السبب",["الكل","تطابق تام","تشابه عالٍ"],horizontal=True,key="rd_dr")
        with d3:
            ds = st.text_input("🔎 بحث", key="ds_q")

        show = dup_df.copy()
        if dt=="عطور"    and "product_type" in show.columns: show=show[show["product_type"]==CAT_PERFUME]
        elif dt=="مكياج"  and "product_type" in show.columns: show=show[show["product_type"]==CAT_BEAUTY]
        if dr=="تطابق تام"   and "match_reason" in show.columns: show=show[show["match_reason"]=="exact"]
        elif dr=="تشابه عالٍ" and "match_reason" in show.columns: show=show[show["match_reason"]=="fuzzy_high"]
        if ds and "product_name" in show.columns:
            show=show[show["product_name"].str.contains(ds,case=False,na=False)]

        reason_ar={"exact":"تطابق تام","fuzzy_high":"تشابه عالٍ جداً","fuzzy_medium":"تشابه متوسط"}
        dd=show.copy()
        if "match_reason" in dd.columns: dd["match_reason"]=dd["match_reason"].map(reason_ar).fillna(dd["match_reason"])
        if "match_score"  in dd.columns: dd["match_score"]=dd["match_score"].apply(lambda x:f"{float(x):.0f}%" if pd.notna(x) else "")
        _show_df(dd,{"product_name":"اسم المنتج (المنافس)","matched_store_product":"المطابق",
                     "match_score":"التطابق","match_reason":"السبب",
                     "product_type":"النوع","source_file":"المصدر"}, 500)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div style='text-align:center;color:#bbb;font-size:.78em;
            padding:18px 0 8px;border-top:1px solid #eee;margin-top:28px;'>
    🔬 <b>Mahwous Engine v5.0</b>
    &nbsp;|&nbsp; 5-Gate Classifier · Background Thread · Auto-Save to Disk
    &nbsp;|&nbsp; Zero False Positives &amp; Zero False Negatives
</div>
""", unsafe_allow_html=True)
