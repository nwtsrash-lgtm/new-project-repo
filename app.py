"""
app.py  -  Mahwous Opportunity Engine  v6.0
===========================================
Automated Resolution Engine (ARE)
Zero Gray Zone | Two-Section UI | Full Audit Trail
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
    ClassificationResult,
    REASON_EXACT_NAME, REASON_FUZZY_HIGH, REASON_SKU_MATCH,
    REASON_VISION_AI, REASON_SAFE_DEFAULT,
    classify_product_5gate,
    deduplicate_and_resolve,
    enrich_product_with_gemini,
    export_audit_trail_csv,
    export_missing_brands_csv,
    export_to_salla_csv,
    fetch_fallback_image,
    load_brands_list,
    load_competitor_products,
    load_store_products,
)

# ── Disk cache ────────────────────────────────────────────────────────────────
_CACHE_DIR  = Path.home() / ".mahwous_cache"
_CACHE_FILE = _CACHE_DIR / "results_v6.pkl"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _save(payload: dict) -> None:
    try:
        with open(_CACHE_FILE, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass

def _load() -> dict | None:
    try:
        if _CACHE_FILE.exists():
            with open(_CACHE_FILE, "rb") as fh:
                return pickle.load(fh)
    except Exception:
        pass
    return None

def _clear_cache() -> None:
    try:
        _CACHE_FILE.unlink(missing_ok=True)
    except Exception:
        pass

# ── Module-level background job ───────────────────────────────────────────────
_JOB: dict = {
    "status": "idle",
    "step":   "",
    "pct":    0.0,
    "log":    [],
    "result": None,
    "error":  None,
    "lock":   threading.Lock(),
}

# ── Defaults ──────────────────────────────────────────────────────────────────
_HIGH_THR   = 90
_LOW_THR    = 50
_MAX_ENRICH = 80


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="محرك مهووس | محكمة الحسم الآلي",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================================
# CSS
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

/* ── HERO ─────────────────────────────────────────────────────────────── */
.hero-wrap{
    background:linear-gradient(135deg,#0a0a1a 0%,#0d1b3e 45%,#0f3460 100%);
    border-radius:20px;padding:36px 32px 28px;text-align:center;
    margin-bottom:22px;box-shadow:0 8px 40px rgba(15,52,96,.45);
    position:relative;overflow:hidden;
}
.hero-wrap::before{content:'';position:absolute;inset:0;
    background:radial-gradient(ellipse 80% 60% at 50% 0%,rgba(99,179,237,.08),transparent 70%);}
.hero-title{font-size:2.4em;font-weight:900;color:#fff;margin:0 0 8px;
    text-shadow:0 2px 20px rgba(99,179,237,.4);}
.hero-sub{font-size:.95em;color:rgba(255,255,255,.65);margin:0;}
.hero-badges{margin-top:14px;display:flex;justify-content:center;gap:8px;flex-wrap:wrap;}
.hbadge{background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.15);
    color:rgba(255,255,255,.8);padding:4px 14px;border-radius:20px;
    font-size:.76em;font-weight:600;}

/* ── METRICS ──────────────────────────────────────────────────────────── */
.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0;}
.mcard{border-radius:14px;padding:18px;text-align:center;}
.mc-blue  {background:linear-gradient(135deg,#e8f4fd,#d1eaf8);}
.mc-green {background:linear-gradient(135deg,#e8f8f0,#d4f0e0);}
.mc-red   {background:linear-gradient(135deg,#fef0f0,#fce0e0);}
.mc-purple{background:linear-gradient(135deg,#f3e5f5,#e1bee7);}
.mc-icon{font-size:1.5em;margin-bottom:2px;}
.mc-num{font-size:2.2em;font-weight:900;line-height:1;}
.mc-blue  .mc-num{color:#1565c0;} .mc-green .mc-num{color:#1b5e20;}
.mc-red   .mc-num{color:#b71c1c;} .mc-purple .mc-num{color:#6a1b9a;}
.mc-label{font-size:.78em;color:#555;margin-top:4px;font-weight:600;}

/* ── OPPORTUNITY RATE BAR ─────────────────────────────────────────────── */
.opp-bar-wrap{background:#f0f4ff;border-radius:12px;padding:12px 18px;
    margin-bottom:18px;display:flex;align-items:center;gap:12px;}
.opp-bar-label{font-weight:700;color:#0f3460;white-space:nowrap;min-width:130px;}
.opp-bar-outer{flex:1;background:#dde6f5;border-radius:8px;height:10px;overflow:hidden;}
.opp-bar-inner{height:100%;border-radius:8px;
    background:linear-gradient(90deg,#1b5e20,#43a047);}
.opp-pct{font-weight:900;color:#1b5e20;font-size:.95em;min-width:40px;}

/* ── SECTION HEADERS ──────────────────────────────────────────────────── */
.section-header-new{
    background:linear-gradient(135deg,#e8f8f0,#d4f0e0);
    border:1.5px solid #a5d6a7;border-radius:14px;
    padding:14px 20px;margin:18px 0 14px;
    display:flex;align-items:center;gap:12px;
}
.section-header-dup{
    background:linear-gradient(135deg,#fef0f0,#fce0e0);
    border:1.5px solid #ef9a9a;border-radius:14px;
    padding:14px 20px;margin:18px 0 14px;
    display:flex;align-items:center;gap:12px;
}
.sh-icon{font-size:2em;}
.sh-text h3{margin:0;font-size:1.1em;font-weight:900;}
.sh-text p{margin:2px 0 0;font-size:.84em;color:#666;}

/* ── PRODUCT CARDS ────────────────────────────────────────────────────── */
.pcard-name{font-weight:700;font-size:.88em;color:#1a1a2e;margin:6px 0 3px;line-height:1.4;}
.pcard-brand{font-size:.78em;color:#0f3460;font-weight:600;}
.pcard-meta{font-size:.68em;color:#aaa;margin-top:3px;}
.pill{display:inline-block;padding:2px 9px;border-radius:20px;
    font-size:.68em;font-weight:700;margin:1px;}
.pill-perf {background:#e8eaf6;color:#283593;border:1px solid #9fa8da;}
.pill-beau {background:#fce4ec;color:#880e4f;border:1px solid #f48fb1;}
.pill-new  {background:#e8f8ee;color:#1b5e20;border:1px solid #a5d6a7;}
.pill-price{background:#fff8e1;color:#e65100;border:1px solid #ffcc80;}
.pill-are  {background:#fff3e0;color:#bf360c;border:1px solid #ffcc80;}
.pill-feat {background:#e3f2fd;color:#0d47a1;border:1px solid #90caf9;}

/* ── RESOLUTION BADGES ────────────────────────────────────────────────── */
.res-badge{display:inline-block;padding:3px 12px;border-radius:10px;
    font-size:.72em;font-weight:700;margin:2px 0;}
.rb-feature{background:#e3f2fd;color:#0d47a1;border:1px solid #90caf9;}
.rb-vision {background:#f3e5f5;color:#6a1b9a;border:1px solid #ce93d8;}
.rb-fuzzy  {background:#fff3e0;color:#e65100;border:1px solid #ffcc80;}
.rb-exact  {background:#ffebee;color:#c62828;border:1px solid #ef9a9a;}
.rb-safe   {background:#f1f8e9;color:#33691e;border:1px solid #aed581;}

/* ── DUPLICATE COMPARISON TABLE ───────────────────────────────────────── */
.dup-header{
    display:grid;grid-template-columns:2fr 2fr 1.5fr 1fr;
    gap:8px;padding:8px 12px;border-radius:8px;
    background:#f0f4ff;font-weight:700;font-size:.82em;color:#0f3460;
    margin-bottom:4px;
}
.dup-row{
    display:grid;grid-template-columns:2fr 2fr 1.5fr 1fr;
    gap:8px;padding:10px 12px;border-radius:10px;
    background:white;border:1px solid #e8ecf4;
    margin-bottom:6px;font-size:.82em;align-items:center;
    transition:box-shadow .15s;
}
.dup-row:hover{box-shadow:0 3px 12px rgba(0,0,0,.08);}
.dup-comp-img{width:40px;height:40px;object-fit:contain;
    border-radius:6px;vertical-align:middle;margin-left:6px;}
.dup-no-img{display:inline-block;width:40px;height:40px;background:#f4f6fb;
    border-radius:6px;vertical-align:middle;margin-left:6px;
    text-align:center;line-height:40px;font-size:1.2em;}

/* ── DOWNLOAD BUTTONS ─────────────────────────────────────────────────── */
div[data-testid="stDownloadButton"]>button{
    background:linear-gradient(135deg,#0d1b3e,#0f3460) !important;
    color:white !important;border:none !important;border-radius:10px !important;
    font-weight:700 !important;font-family:'Cairo',sans-serif !important;
    padding:9px 16px !important;width:100%;}
div[data-testid="stDownloadButton"]>button:hover{opacity:.87 !important;}

/* ── PRIMARY BUTTON ───────────────────────────────────────────────────── */
div[data-testid="stButton"]>button[kind="primary"]{
    background:linear-gradient(135deg,#0f3460,#1565c0) !important;
    color:white !important;border:none !important;border-radius:12px !important;
    font-weight:900 !important;font-family:'Cairo',sans-serif !important;
    font-size:1.05em !important;padding:13px 24px !important;
    box-shadow:0 4px 20px rgba(15,52,96,.35) !important;}

/* ── SIDEBAR ──────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"]{background:#0d1b3e;}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p{color:rgba(255,255,255,.8) !important;}
section[data-testid="stSidebar"] h3{color:#63b3ed !important;}

/* ── BG RUNNING BANNER ────────────────────────────────────────────────── */
.bg-banner{background:linear-gradient(135deg,#e3f2fd,#bbdefb);
    border:1.5px solid #1565c0;border-radius:14px;padding:16px 20px;
    margin:12px 0;display:flex;align-items:center;gap:14px;}
.bg-banner-icon{font-size:2em;}
.bg-banner-text b{color:#0d47a1;font-size:1.05em;}
.bg-banner-text p{margin:4px 0 0;color:#555;font-size:.87em;}

/* ── SECRETS ALERT ────────────────────────────────────────────────────── */
.sec-alert{background:linear-gradient(135deg,#fff8e1,#fff3cd);
    border:1.5px solid #ffc107;border-radius:14px;padding:18px 20px;margin:10px 0 18px;}
.sec-alert h3{margin:0 0 6px;color:#856404;font-size:.95em;}
.sec-alert pre{background:#1e1e2e;color:#a8d8a8;padding:10px 14px;
    border-radius:8px;font-size:.82em;margin:8px 0 0;direction:ltr;text-align:left;}
.sec-alert code{background:rgba(133,100,4,.12);padding:1px 6px;
    border-radius:4px;font-size:.85em;color:#5d4037;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# API KEY RESOLUTION
# ============================================================================
def _secret(k: str) -> str:
    try:
        v = st.secrets.get(k, ""); return v if v else ""
    except Exception:
        pass
    return st.session_state.get(k, "")

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
        do_enrich       = st.toggle("توليد أوصاف HTML لسلة",            value=True)
        do_img_fallback = st.toggle("جلب الصور المفقودة (Google Search)", value=False)
        max_enrich      = st.slider("حد إثراء AI", 10, 300, _MAX_ENRICH, 10)
        use_ai_classify = st.toggle("Gate 5 – Gemini للتصنيف الغامض",   value=True)

    st.divider()
    if st.button("🔄 بدء من جديد (مسح الكل)", use_container_width=True):
        _clear_cache()
        with _JOB["lock"]:
            _JOB.update({"status":"idle","result":None,"log":[],"pct":0.0,"error":None})
        for k in ["done","new_df","dup_df","existing_brands","stats","_loaded"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown("<div style='text-align:center;padding:12px 0 0;"
                "color:rgba(255,255,255,.3);font-size:.72em;'>"
                "Mahwous Engine v6.0 | ARE + Vision AI</div>",
                unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INIT  +  DISK RESTORE
# ============================================================================
_EMPTY_STATE = {
    "done":            False,
    "new_df":          pd.DataFrame(),
    "dup_df":          pd.DataFrame(),
    "existing_brands": [],
    "stats":           {"new":0,"perf":0,"beauty":0,"dup":0,"total_comp":0,
                        "are_resolved":0,"vision_ai_calls":0},
}
if not st.session_state.get("_loaded"):
    disk = _load()
    if disk and disk.get("done"):
        for k, v in disk.items():
            st.session_state[k] = v
    else:
        for k, v in _EMPTY_STATE.items():
            if k not in st.session_state:
                st.session_state[k] = v
    st.session_state["_loaded"] = True
else:
    for k, v in _EMPTY_STATE.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================================
# BACKGROUND PIPELINE
# ============================================================================

def _log(msg: str) -> None:
    with _JOB["lock"]:
        _JOB["log"].append(msg)
        _JOB["step"] = msg

def _pct(p: float) -> None:
    with _JOB["lock"]:
        _JOB["pct"] = min(p, 1.0)


def _run_pipeline(
    store_bytes:    list[tuple[str, bytes]],
    comp_bytes:     list[tuple[str, bytes]],
    brands_bytes:   bytes | None,
    gemini_key:     str,
    g_srch:         str,
    g_cx:           str,
    do_enrich:      bool,
    do_img_fallback:bool,
    max_enrich:     int,
    use_ai_cls:     bool,
) -> None:
    """Complete pipeline in background thread. Saves to disk on completion."""
    try:
        with _JOB["lock"]:
            _JOB.update({"status":"running","log":[],"pct":0.0,"result":None,"error":None})

        class _Buf:
            def __init__(self, d, n): self._b = io.BytesIO(d); self.name = n
            def read(self):           self._b.seek(0); return self._b.read()

        store_files = [_Buf(b, n) for n, b in store_bytes]
        comp_files  = [_Buf(b, n) for n, b in comp_bytes]
        brands_file = _Buf(brands_bytes, "brands.csv") if brands_bytes else None

        # ── Step 1: Load store ────────────────────────────────────────────────
        _log("📂 1/5 — تحميل جدار الحماية…")
        store_df        = load_store_products(store_files)
        existing_brands = load_brands_list(brands_file)
        _log(f"✅ {len(store_df):,} منتج في المتجر | {len(existing_brands):,} ماركة")
        _pct(0.10)

        # ── Step 2: Load competitors ──────────────────────────────────────────
        _log("📦 2/5 — طحن ملفات المنافسين…")
        all_comp: list[dict] = []
        for cf in comp_files:
            dfc = load_competitor_products(cf)
            all_comp.extend(dfc.to_dict("records"))
            _log(f"  • {cf.name} ← {len(dfc):,} منتج")
        total_comp = len(all_comp)
        _log(f"📊 إجمالي: {total_comp:,} منتج للمقارنة")
        _pct(0.20)

        # ── Step 3: Deduplication + ARE ───────────────────────────────────────
        _log("⚖️ 3/5 — محكمة الحسم الآلي تعمل (ARE)…")
        _stages: dict = {"comparing": 0, "resolving": 0}

        def _upd(i, t, name, stage):
            _stages[stage] = _stages.get(stage, 0) + 1
            base = 0.20 + 0.40 * i / max(t, 1)
            _pct(base)

        new_df, dup_df = deduplicate_and_resolve(
            store_df       = store_df,
            comp_products  = all_comp,
            high_threshold = _HIGH_THR,
            low_threshold  = _LOW_THR,
            api_key        = gemini_key,
            progress_callback = _upd,
        )
        are_resolved = len([
            r for r in new_df.to_dict("records") + dup_df.to_dict("records")
            if r.get("resolution_path") in ("feature_mismatch","vision_ai","safe_fallback")
        ]) if not new_df.empty or not dup_df.empty else 0
        vision_calls = len([
            r for r in new_df.to_dict("records") + dup_df.to_dict("records")
            if r.get("resolution_path") == "vision_ai"
        ]) if not new_df.empty or not dup_df.empty else 0

        _log(f"✅ {len(new_df):,} فرصة جديدة | {len(dup_df):,} مكرر | {are_resolved} حُسم بـ ARE")
        _pct(0.60)

        # ── Step 4: 5-Gate Classification ────────────────────────────────────
        _log(f"🔬 4/5 — تصنيف {len(new_df):,} فرصة بالبوابات الخمس…")
        api_cls = gemini_key if use_ai_cls else ""
        if not new_df.empty:
            classified = []
            rows = new_df.to_dict("records")
            for i, row in enumerate(rows):
                res = classify_product_5gate(
                    str(row.get("product_name", "")),
                    str(row.get("category", "")),
                    str(row.get("brand", "")),
                    api_cls,
                )
                d = dict(row); d.update(res.to_dict()); d["product_type"] = res.category
                classified.append(d)
                if (i+1) % 30 == 0:
                    _pct(0.60 + 0.20 * (i+1) / max(len(rows), 1))
            new_df = pd.DataFrame(classified)
        _pct(0.80)

        # ── Step 5: Enrichment ────────────────────────────────────────────────
        if do_enrich and gemini_key and not new_df.empty:
            perf_mask = new_df["product_type"] == CAT_PERFUME if "product_type" in new_df.columns else pd.Series([False]*len(new_df))
            to_enr    = new_df[perf_mask].head(int(max_enrich))
            n_e       = len(to_enr)
            _log(f"✨ 5/5 — تحضير أوصاف HTML لـ {n_e} عطر…")
            enriched = []
            for i, (_, row) in enumerate(to_enr.iterrows()):
                enr = enrich_product_with_gemini(
                    str(row.get("product_name","")), str(row.get("image_url","")),
                    str(row.get("product_type",CAT_PERFUME)), gemini_key,
                )
                d = row.to_dict(); d.update(enr)
                if do_img_fallback and g_srch and not d.get("image_url"):
                    d["image_url"] = fetch_fallback_image(d.get("product_name",""), g_srch, g_cx)
                enriched.append(d)
                _pct(0.80 + 0.19 * (i+1) / max(n_e, 1))
            rest = new_df[~new_df.index.isin(to_enr.index)]
            new_df = pd.concat([pd.DataFrame(enriched), rest], ignore_index=True)
            _log("✅ اكتملت الأوصاف")
        else:
            _log("✨ 5/5 — تم تخطي الإثراء")

        _pct(0.99)

        # ── Compute stats ─────────────────────────────────────────────────────
        def _cnt(col, val):
            if new_df.empty or col not in new_df.columns: return 0
            return int((new_df[col] == val).sum())

        stats = {
            "new":           len(new_df),
            "perf":          _cnt("product_type", CAT_PERFUME),
            "beauty":        _cnt("product_type", CAT_BEAUTY),
            "dup":           len(dup_df),
            "total_comp":    total_comp,
            "are_resolved":  are_resolved,
            "vision_ai_calls": vision_calls,
        }

        payload = {
            "done": True, "new_df": new_df, "dup_df": dup_df,
            "existing_brands": existing_brands, "stats": stats,
        }
        _save(payload)
        _log("💾 تم الحفظ التلقائي على القرص")

        with _JOB["lock"]:
            _JOB["result"] = payload
            _JOB["status"] = "done"
            _JOB["pct"]    = 1.0
        _log("🎉 اكتمل التحليل!")

    except Exception as exc:
        import traceback
        err = f"{exc}\n{traceback.format_exc()}"
        with _JOB["lock"]:
            _JOB["status"] = "error"
            _JOB["error"]  = err
        _log(f"❌ خطأ: {exc}")


# ============================================================================
# HERO
# ============================================================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">⚖️ محرك مهووس | محكمة الحسم الآلي</div>
    <p class="hero-sub">
        محرك الحسم الآلي (ARE) · استخراج عميق للخصائص · رؤية اصطناعية ·
        قرار قاطع لكل منتج · صفر منطقة رمادية
    </p>
    <div class="hero-badges">
        <span class="hbadge">⚙️ استخراج الحجم / التركيز / الجنس</span>
        <span class="hbadge">🔬 مقارنة SKU تلقائية</span>
        <span class="hbadge">👁️ Vision AI Judge</span>
        <span class="hbadge">💾 حفظ تلقائي</span>
        <span class="hbadge">🧵 معالجة خلفية</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SECRETS ALERT
# ============================================================================
if not _has_gemini:
    st.markdown("""<div class="sec-alert">
    <h3>⚠️ مفتاح Gemini غير مُعرَّف</h3>
    <p style="margin:0 0 6px;color:#6d4c41;font-size:.88em;">
        بدون مفتاح، محكمة الرؤية الاصطناعية ستعتمد على الخصائص النصية فقط.
        <code>.streamlit/secrets.toml</code>:
    </p>
    <pre>GEMINI_API_KEY = "AIza...................."</pre>
</div>""", unsafe_allow_html=True)
else:
    st.success(
        "✅ **Gemini جاهز** — استخراج الخصائص + محكمة الرؤية الاصطناعية نشطة",
        icon="⚖️",
    )


# ============================================================================
# UPLOAD
# ============================================================================
st.markdown("---")
c1, c2 = st.columns([1, 1.5], gap="large")
with c1:
    st.markdown('<p style="font-weight:700;font-size:.97em;color:#1a1a2e;margin:0 0 3px;">🏪 ملفات متجر مهووس</p>', unsafe_allow_html=True)
    st.caption("ملفات سلة — الجدار الواقي")
    store_files = st.file_uploader("store", type=["csv"],
        accept_multiple_files=True, key="uf_store", label_visibility="collapsed")
    brands_file = st.file_uploader("ملف الماركات (اختياري)", type=["csv"], key="uf_brands")
    if store_files:
        st.success(f"✅ {len(store_files)} ملف(ات)")

with c2:
    st.markdown('<p style="font-weight:700;font-size:.97em;color:#1a1a2e;margin:0 0 3px;">🔍 ملفات المنافسين (حتى 15)</p>', unsafe_allow_html=True)
    st.caption("أي تنسيق CSV — كشف تلقائي للأعمدة")
    comp_files = st.file_uploader("comp", type=["csv"],
        accept_multiple_files=True, key="uf_comp", label_visibility="collapsed")
    if comp_files:
        st.success(f"✅ {len(comp_files)} ملف(ات)")
        with st.expander("📋 الملفات"):
            for f in comp_files[:15]: st.caption(f"• {f.name}")


# ============================================================================
# START BUTTON
# ============================================================================
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
_, bc, _ = st.columns([1, 2, 1])
with bc:
    running = _JOB["status"] == "running"
    ready   = bool(store_files and comp_files) and not running
    start   = st.button("🚀  بدء التحليل والحسم الآلي",
                        type="primary", use_container_width=True,
                        disabled=not ready, key="btn_start")
    if running:
        st.caption("⚙️ محكمة ARE تعمل في الخلفية — يمكنك إغلاق المتصفح")
    elif not ready and not running:
        st.caption("⚠️ يجب رفع ملفات المتجر والمنافسين")
st.markdown("---")


# ============================================================================
# LAUNCH BACKGROUND THREAD
# ============================================================================
if start and store_files and comp_files:
    store_bytes = [(f.name, f.read()) for f in store_files]
    comp_bytes  = [(f.name, f.read()) for f in comp_files[:15]]
    brands_bytes = brands_file.read() if brands_file else None
    for k in ["done","new_df","dup_df","existing_brands","stats","_loaded"]:
        st.session_state.pop(k, None)
    threading.Thread(
        target=_run_pipeline,
        kwargs=dict(
            store_bytes=store_bytes, comp_bytes=comp_bytes,
            brands_bytes=brands_bytes,
            gemini_key=GEMINI_KEY, g_srch=GOOGLE_SRCH_KEY, g_cx=GOOGLE_CX,
            do_enrich=do_enrich, do_img_fallback=do_img_fallback,
            max_enrich=int(max_enrich), use_ai_cls=use_ai_classify,
        ),
        daemon=True,
    ).start()
    st.rerun()


# ============================================================================
# BACKGROUND PROGRESS UI
# ============================================================================
if _JOB["status"] == "running":
    with _JOB["lock"]:
        step = _JOB["step"]
        pct  = _JOB["pct"]
        logs = list(_JOB["log"])

    st.markdown(f"""
<div class="bg-banner">
    <div class="bg-banner-icon">⚖️</div>
    <div class="bg-banner-text">
        <b>محكمة الحسم الآلي تعمل في الخلفية</b>
        <p>{step}</p>
    </div>
</div>""", unsafe_allow_html=True)
    st.progress(pct, text=f"{pct*100:.0f}%")
    with st.expander("📋 سجل ARE المباشر", expanded=True):
        for line in logs[-20:]: st.caption(line)
    st.caption("💡 يمكنك إغلاق المتصفح — النتائج تُحفظ تلقائياً عند الانتهاء.")
    time.sleep(2); st.rerun()


# ============================================================================
# JOB DONE → transfer to session_state
# ============================================================================
if _JOB["status"] == "done" and not st.session_state.get("done"):
    with _JOB["lock"]:
        payload = _JOB["result"]
    if payload:
        for k, v in payload.items():
            st.session_state[k] = v
        st.toast("🎉 اكتمل الحسم الآلي! النتائج محفوظة.", icon="⚖️")
        time.sleep(0.3); st.rerun()


# ============================================================================
# JOB ERROR
# ============================================================================
if _JOB["status"] == "error":
    err = _JOB.get("error", "خطأ غير معروف")
    st.error(f"❌ حدث خطأ: {err[:300]}")
    if st.button("🔄 إعادة المحاولة"):
        with _JOB["lock"]: _JOB["status"] = "idle"
        st.rerun()


# ============================================================================
# WELCOME STATE
# ============================================================================
if not st.session_state.get("done"):
    st.markdown("""<div style='text-align:center;padding:50px 0 60px;'>
        <div style='font-size:5em;margin-bottom:14px;'>⚖️</div>
        <p style='font-size:1.1em;color:#888;max-width:520px;margin:0 auto;line-height:1.9;'>
            ارفع ملفات متجرك وملفات المنافسين<br>
            ثم اضغط <b style='color:#0f3460;'>بدء التحليل والحسم الآلي</b><br>
            <span style='font-size:.85em;color:#aaa;'>
                ✅ صفر منطقة رمادية ·
                ⚖️ حسم بالخصائص أو الرؤية الاصطناعية ·
                💾 حفظ تلقائي
            </span>
        </p></div>""", unsafe_allow_html=True)
    st.stop()


# ============================================================================
# RESULTS
# ============================================================================
new_df          = st.session_state["new_df"]
dup_df          = st.session_state["dup_df"]
existing_brands = st.session_state["existing_brands"]
s               = st.session_state["stats"]

# ── Disk timestamp ─────────────────────────────────────────────────────────
if _CACHE_FILE.exists():
    from datetime import datetime
    ts = datetime.fromtimestamp(_CACHE_FILE.stat().st_mtime).strftime("%Y-%m-%d  %H:%M")
    st.caption(f"💾 آخر حفظ: **{ts}**")

# ── METRIC CARDS ─────────────────────────────────────────────────────────────
_total   = s["new"] + s["dup"]
_opp_pct = round(s["new"] / _total * 100, 1) if _total else 0

st.markdown(f"""
<div class="metric-grid">
  <div class="mcard mc-blue">
    <div class="mc-icon">📦</div>
    <div class="mc-num">{s["total_comp"]:,}</div>
    <div class="mc-label">منتجات مُحللة</div>
  </div>
  <div class="mcard mc-green">
    <div class="mc-icon">🌟</div>
    <div class="mc-num">{s["new"]:,}</div>
    <div class="mc-label">فرص جديدة</div>
  </div>
  <div class="mcard mc-red">
    <div class="mc-icon">🛡️</div>
    <div class="mc-num">{s["dup"]:,}</div>
    <div class="mc-label">مكررات محظورة</div>
  </div>
  <div class="mcard mc-purple" style="background:linear-gradient(135deg,#f3e5f5,#e1bee7);">
    <div class="mc-icon">⚖️</div>
    <div class="mc-num">{s.get("are_resolved",0):,}</div>
    <div class="mc-label">حُسم بمحكمة ARE</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="opp-bar-wrap">
  <div class="opp-bar-label">📈 نسبة الفرص الجديدة</div>
  <div class="opp-bar-outer">
    <div class="opp-bar-inner" style="width:{_opp_pct}%"></div>
  </div>
  <div class="opp-pct">{_opp_pct}%</div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# HELPERS
# ============================================================================
def _res_badge(path: str, reason: str) -> str:
    cls = {"feature_mismatch": "rb-feature", "vision_ai": "rb-vision",
           "fuzzy_high": "rb-fuzzy", "exact_name": "rb-exact",
           "safe_fallback": "rb-safe"}.get(path, "rb-safe")
    label = {"feature_mismatch": "⚙️ خصائص SKU",
              "vision_ai":        "👁️ رؤية AI",
              "fuzzy_high":       "📊 تشابه نصي",
              "exact_name":       "🎯 تطابق تام",
              "safe_fallback":    "🔒 قرار آمن"}.get(path, path)
    return f'<span class="res-badge {cls}">{label}</span>'

def _salla_btn(records, suffix, label, key):
    if records:
        st.download_button(label, data=export_to_salla_csv(records),
            file_name=f"{suffix}.csv", mime="text/csv",
            use_container_width=True, key=key)

def _show_df(df, col_map, h=400):
    avail = {k:v for k,v in col_map.items() if k in df.columns}
    st.dataframe(df[list(avail)].rename(columns=avail), use_container_width=True, height=h)


# ============================================================================
# SECTION 1 — OPPORTUNITIES  (القسم الأول: غير متوفر — الفرص الجديدة)
# ============================================================================
st.markdown("""
<div class="section-header-new">
  <div class="sh-icon">🌟</div>
  <div class="sh-text">
    <h3>القسم الأول: غير متوفر لدينا — الفرص الجديدة</h3>
    <p>هذه المنتجات غير موجودة في متجرك ولم تُحسم كمكررة. جاهزة للرفع المباشر على سلة.</p>
  </div>
</div>
""", unsafe_allow_html=True)

if new_df.empty:
    st.info("💡 لا فرص جديدة — كل منتجات المنافسين موجودة لديك!")
else:
    perf_recs  = new_df[new_df.get("product_type", pd.Series(dtype=str)) == CAT_PERFUME].to_dict("records") if "product_type" in new_df.columns else []
    beau_recs  = new_df[new_df.get("product_type", pd.Series(dtype=str)).isin([CAT_BEAUTY,CAT_UNKNOWN])].to_dict("records") if "product_type" in new_df.columns else []
    all_recs   = new_df.to_dict("records")

    # ── Export buttons ────────────────────────────────────────────────────────
    st.markdown("#### 📥 تصدير ملفات سلة")
    ex1, ex2, ex3, ex4 = st.columns(4)
    with ex1:
        _salla_btn(perf_recs, "Salla_Perfumes",
                   f"⬇️ العطور ({len(perf_recs):,})", "dl_perf")
    with ex2:
        _salla_btn(beau_recs, "Salla_Beauty_Care",
                   f"⬇️ المكياج والعناية ({len(beau_recs):,})", "dl_beau")
    with ex3:
        st.download_button(
            "⬇️ الماركات الناقصة",
            data      = export_missing_brands_csv(all_recs, existing_brands),
            file_name = "missing_brands.csv", mime = "text/csv",
            use_container_width=True, key="dl_brands",
        )
    with ex4:
        st.download_button(
            "⬇️ سجل التدقيق (ARE)",
            data      = export_audit_trail_csv(all_recs),
            file_name = "audit_are.csv", mime = "text/csv",
            use_container_width=True, key="dl_audit",
        )

    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.columns([1.3, 1.3, 1.8, 1])
    with fc1:
        t_sel = st.selectbox("🗂️ النوع", ["الكل","عطور","مكياج وعناية"], key="nt")
    with fc2:
        res_opts = ["الكل"] + sorted(new_df["resolution_path"].dropna().unique().tolist()) if "resolution_path" in new_df.columns else ["الكل"]
        res_sel  = st.selectbox("⚖️ مسار الحسم", res_opts, key="nr")
    with fc3:
        srch = st.text_input("🔎 بحث في اسم المنتج", key="nq")
    with fc4:
        view = st.radio("عرض", ["🃏 بطاقات","📋 جدول"], horizontal=True, key="nv")

    disp = new_df.copy()
    if t_sel == "عطور"       and "product_type" in disp.columns: disp = disp[disp["product_type"]==CAT_PERFUME]
    elif t_sel == "مكياج وعناية" and "product_type" in disp.columns: disp = disp[disp["product_type"].isin([CAT_BEAUTY,CAT_UNKNOWN])]
    if res_sel != "الكل"    and "resolution_path" in disp.columns: disp = disp[disp["resolution_path"]==res_sel]
    if srch: disp = disp[disp["product_name"].str.contains(srch,case=False,na=False)]
    disp = disp.reset_index(drop=True)

    st.caption(f"عرض **{min(len(disp),60):,}** من **{len(disp):,}** فرصة")

    if view == "📋 جدول":
        _show_df(disp, {
            "product_name":"اسم المنتج","brand":"الماركة","product_type":"النوع",
            "price":"السعر","resolution_path":"مسار الحسم",
            "feature_details":"تفاصيل ARE","source_file":"المصدر",
        }, 520)
    else:
        LIMIT = 60
        for rs in range(0, min(len(disp), LIMIT), 3):
            cols = st.columns(3, gap="medium")
            for j, col in enumerate(cols):
                i = rs + j
                if i >= min(len(disp), LIMIT): break
                row  = disp.iloc[i]
                pname= str(row.get("product_name",""))
                brand= str(row.get("brand",""))
                price= str(row.get("price","")).strip()
                ptype= str(row.get("product_type","other"))
                path = str(row.get("resolution_path",""))
                feat = str(row.get("feature_details",""))[:55]
                src  = str(row.get("source_file","")).replace(".csv","")
                t_pill = ('<span class="pill pill-perf">🧴 عطر</span>' if ptype==CAT_PERFUME
                          else '<span class="pill pill-beau">💄 مكياج</span>')
                with col:
                    img = str(row.get("image_url","")).strip()
                    if img.startswith("http"):
                        try: st.image(img, use_container_width=True)
                        except: st.markdown("<div style='height:130px;background:#f4f6fb;border-radius:10px;display:flex;align-items:center;justify-content:center;color:#ccc;font-size:2em;'>📷</div>",unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='height:130px;background:#f4f6fb;border-radius:10px;display:flex;align-items:center;justify-content:center;color:#ccc;font-size:2em;'>📷</div>",unsafe_allow_html=True)
                    brand_html = f'<p class="pcard-brand">🏷️ {brand}</p>' if brand else ""
                    price_html = f'<span class="pill pill-price">💰 {price}</span>' if price else ""
                    st.markdown(
                        f'<p class="pcard-name">{pname[:65]}{"…" if len(pname)>65 else ""}</p>'
                        f'{brand_html}'
                        f'<div style="margin-top:5px;">{t_pill}'
                        f'<span class="pill pill-new">✅ جديد</span>'
                        f'{price_html}'
                        f'</div>'
                        f'<div style="margin-top:4px;">{_res_badge(path, "")}</div>'
                        f'<p class="pcard-meta">{feat}<br>{src}</p>',
                        unsafe_allow_html=True,
                    )
        if len(disp) > LIMIT:
            st.info(f"يُعرض أول {LIMIT}. حمّل CSV للاطلاع على الكل ({len(disp):,}).")


# ============================================================================
# SECTION 2 — DUPLICATES  (القسم الثاني: متوفر لدينا — المكررات)
# ============================================================================
st.markdown("""
<div class="section-header-dup">
  <div class="sh-icon">🛡️</div>
  <div class="sh-text">
    <h3>القسم الثاني: متوفر لدينا — المكررات المحظورة</h3>
    <p>هذه المنتجات موجودة أو مطابقة لما لديك. تم حسمها بالخصائص أو الرؤية الاصطناعية.</p>
  </div>
</div>
""", unsafe_allow_html=True)

if dup_df.empty:
    st.success("🎉 لا مكررات — كل منتجات المنافسين فرص جديدة!")
else:
    st.success(f"🛡️ تم حظر **{len(dup_df):,}** مكرر — متجرك محمي بالكامل!")

    # ── Stats row ─────────────────────────────────────────────────────────────
    if "resolution_path" in dup_df.columns:
        path_counts = dup_df["resolution_path"].value_counts().to_dict()
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("🎯 تطابق تام",      path_counts.get("exact_name",0))
        mc2.metric("📊 تشابه نصي ≥90%", path_counts.get("fuzzy_high",0))
        mc3.metric("⚙️ حسم بالخصائص",  path_counts.get("feature_mismatch",0))  # This shouldn't happen for dups but just in case
        mc4.metric("👁️ رؤية AI",        path_counts.get("vision_ai",0))

    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────────────
    df1, df2, df3 = st.columns([1, 1, 2])
    with df1:
        reason_opts = ["الكل"] + (sorted(dup_df["resolution_path"].dropna().unique().tolist()) if "resolution_path" in dup_df.columns else [])
        r_sel = st.selectbox("⚖️ سبب الحظر", reason_opts, key="dr")
    with df2:
        src_opts = ["الكل"] + (sorted(dup_df["source_file"].dropna().unique().tolist()) if "source_file" in dup_df.columns else [])
        s_sel = st.selectbox("🏪 المصدر", src_opts, key="ds")
    with df3:
        d_srch = st.text_input("🔎 بحث", placeholder="ابحث في المكررات…", key="dq")

    show = dup_df.copy()
    if r_sel != "الكل" and "resolution_path" in show.columns: show=show[show["resolution_path"]==r_sel]
    if s_sel != "الكل" and "source_file"     in show.columns: show=show[show["source_file"]==s_sel]
    if d_srch and "product_name" in show.columns: show=show[show["product_name"].str.contains(d_srch,case=False,na=False)]

    st.caption(f"عرض **{len(show):,}** مكرر محظور")

    # ── SIDE-BY-SIDE COMPARISON TABLE ─────────────────────────────────────────
    st.markdown("""
<div class="dup-header">
    <div>🔍 منتج المنافس</div>
    <div>🏪 منتجنا المطابق له</div>
    <div>⚖️ سبب التطابق</div>
    <div>📊 التشابه</div>
</div>
""", unsafe_allow_html=True)

    DISPLAY_LIMIT = 200
    for _, row in show.head(DISPLAY_LIMIT).iterrows():
        comp_name  = str(row.get("product_name",""))
        store_name = str(row.get("matched_store_product",""))
        path       = str(row.get("resolution_path",""))
        feat_det   = str(row.get("feature_details",""))[:80]
        score      = row.get("match_score", 0)
        try: score_f = float(score)
        except: score_f = 0.0
        source     = str(row.get("source_file","")).replace(".csv","")
        comp_img   = str(row.get("image_url","")).strip()

        img_html = (f'<img src="{comp_img}" class="dup-comp-img" onerror="this.style.display=\'none\'">'
                    if comp_img.startswith("http") else
                    '<span class="dup-no-img">📷</span>')

        badge      = _res_badge(path, "")
        score_pill = (f'<span class="pill" style="background:#e8f4fd;color:#1565c0;'
                      f'border:1px solid #90caf9;">{score_f:.0f}%</span>')

        # Feature details as sub-note
        feat_note = (f'<div style="font-size:.68em;color:#888;margin-top:3px;">{feat_det}</div>'
                     if feat_det and feat_det not in ("nan","") else "")

        st.markdown(
            f'<div class="dup-row">'
            f'<div>{img_html}<b>{comp_name[:55]}{"…" if len(comp_name)>55 else ""}</b>'
            f'<div style="font-size:.7em;color:#888;">{source}</div></div>'
            f'<div style="color:#0f3460;font-weight:600;">{store_name[:60]}{"…" if len(store_name)>60 else ""}</div>'
            f'<div>{badge}{feat_note}</div>'
            f'<div>{score_pill}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if len(show) > DISPLAY_LIMIT:
        st.info(f"يُعرض أول {DISPLAY_LIMIT}. حمّل CSV للاطلاع على الكل ({len(show):,}).")
        st.download_button(
            f"⬇️ تصدير جدول المكررات الكامل ({len(dup_df):,})",
            data      = export_audit_trail_csv(dup_df.to_dict("records")),
            file_name = "duplicates_full.csv",
            mime      = "text/csv",
            use_container_width=True,
            key       = "dl_dup_full",
        )


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div style='text-align:center;color:#bbb;font-size:.78em;
            padding:18px 0 8px;border-top:1px solid #eee;margin-top:28px;'>
    ⚖️ <b>Mahwous Engine v6.0</b>
    &nbsp;|&nbsp; Automated Resolution Engine (ARE)
    &nbsp;|&nbsp; Deep Feature Extraction · SKU Comparison · Vision AI Court
    &nbsp;|&nbsp; Zero False Positives & Zero False Negatives
</div>
""", unsafe_allow_html=True)
