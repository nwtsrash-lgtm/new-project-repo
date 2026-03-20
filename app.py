"""
app.py  -  Mahwous Opportunity Engine  v3.0
Zero-Configuration Interactive Dashboard
"""

import time
import pandas as pd
import streamlit as st

from logic import (
    deduplicate_products,
    enrich_product_with_gemini,
    export_missing_brands_csv,
    export_to_salla_csv,
    fetch_fallback_image,
    load_brands_list,
    load_competitor_products,
    load_store_products,
    verify_with_gemini,
)

# ============================================================
# 0.  OPTIMAL DEFAULTS  (Zero-Config -- user never sees these)
# ============================================================
_HIGH_THRESHOLD = 90   # score >= 90  --> duplicate
_LOW_THRESHOLD  = 50   # score <  50  --> confirmed new
_MAX_ENRICH     = 80   # max products to enrich with AI

# ============================================================
# 1.  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="محرك مهووس لاستكشاف الفرص",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 2.  GLOBAL CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"],
.stMarkdown, .stText,
.stButton > button,
.stDownloadButton > button,
div[data-testid="stMetricValue"],
div[data-testid="stMetricLabel"] {
    font-family: 'Cairo', sans-serif !important;
}
.main > div { direction: rtl; }
div[data-testid="column"] { direction: rtl; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

/* HERO */
.hero-wrap {
    background: linear-gradient(135deg,#0a0a1a 0%,#0d1b3e 45%,#0f3460 100%);
    border-radius: 20px;
    padding: 42px 36px 36px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 8px 40px rgba(15,52,96,.45);
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content:'';
    position:absolute;inset:0;
    background:radial-gradient(ellipse 80% 60% at 50% 0%,
        rgba(99,179,237,.08) 0%,transparent 70%);
}
.hero-title {
    font-size:2.7em;font-weight:900;color:#fff;
    margin:0 0 10px;text-shadow:0 2px 20px rgba(99,179,237,.4);
}
.hero-sub   { font-size:1.1em;color:rgba(255,255,255,.65);margin:0;font-weight:400; }
.hero-badges{
    margin-top:18px;display:flex;justify-content:center;
    gap:10px;flex-wrap:wrap;
}
.hbadge {
    background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.15);
    color:rgba(255,255,255,.8);padding:5px 16px;border-radius:20px;
    font-size:.8em;font-weight:600;backdrop-filter:blur(4px);
}

/* METRIC CARDS */
.metric-grid {
    display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0;
}
.mcard {
    border-radius:16px;padding:20px 16px;text-align:center;
}
.mc-blue  { background:linear-gradient(135deg,#e8f4fd,#d1eaf8); }
.mc-green { background:linear-gradient(135deg,#e8f8f0,#d4f0e0); }
.mc-amber { background:linear-gradient(135deg,#fff8e1,#fef0c0); }
.mc-red   { background:linear-gradient(135deg,#fef0f0,#fce0e0); }
.mc-icon  { font-size:1.6em;margin-bottom:4px; }
.mc-num   { font-size:2.4em;font-weight:900;line-height:1; }
.mc-blue  .mc-num{color:#1565c0;} .mc-green .mc-num{color:#1b5e20;}
.mc-amber .mc-num{color:#e65100;} .mc-red   .mc-num{color:#b71c1c;}
.mc-label { font-size:.82em;color:#555;margin-top:6px;font-weight:600; }

/* OPP BAR */
.opp-bar-wrap {
    background:#f0f4ff;border-radius:12px;padding:14px 20px;
    margin-bottom:20px;display:flex;align-items:center;gap:14px;
}
.opp-bar-label{font-weight:700;color:#0f3460;white-space:nowrap;min-width:140px;}
.opp-bar-outer{flex:1;background:#dde6f5;border-radius:8px;height:12px;overflow:hidden;}
.opp-bar-inner{height:100%;border-radius:8px;
    background:linear-gradient(90deg,#1b5e20,#43a047);transition:width .6s;}
.opp-pct{font-weight:900;color:#1b5e20;font-size:1em;min-width:44px;}

/* UPLOAD ZONES */
.upload-title{font-weight:700;font-size:1em;color:#1a1a2e;margin:0 0 4px;}
.upload-sub  {font-size:.82em;color:#888;margin:0 0 14px;}

/* PRODUCT CARDS */
.pcard-name {
    font-weight:700;font-size:.92em;color:#1a1a2e;
    margin:8px 0 4px;line-height:1.4;
}
.pcard-brand{font-size:.8em;color:#0f3460;font-weight:600;}
.pcard-meta {font-size:.72em;color:#aaa;margin-top:4px;}
.pill {
    display:inline-block;padding:2px 10px;border-radius:20px;
    font-size:.72em;font-weight:700;margin:2px 1px;
}
.pill-new  {background:#e8f8ee;color:#1b5e20;border:1px solid #a5d6a7;}
.pill-perf {background:#e8eaf6;color:#283593;border:1px solid #9fa8da;}
.pill-beau {background:#fce4ec;color:#880e4f;border:1px solid #f48fb1;}
.pill-price{background:#fff8e1;color:#e65100;border:1px solid #ffcc80;}

/* DOWNLOAD BUTTON */
div[data-testid="stDownloadButton"] > button {
    background:linear-gradient(135deg,#0d1b3e,#0f3460) !important;
    color:white !important;border:none !important;border-radius:10px !important;
    font-weight:700 !important;font-family:'Cairo',sans-serif !important;
    padding:10px 18px !important;width:100%;
}
div[data-testid="stDownloadButton"] > button:hover{opacity:.87 !important;}

/* PRIMARY BUTTON */
div[data-testid="stButton"] > button[kind="primary"] {
    background:linear-gradient(135deg,#0f3460,#1565c0) !important;
    color:white !important;border:none !important;border-radius:12px !important;
    font-weight:900 !important;font-family:'Cairo',sans-serif !important;
    font-size:1.05em !important;padding:14px 28px !important;
    box-shadow:0 4px 20px rgba(15,52,96,.35) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover{
    transform:translateY(-2px) !important;
    box-shadow:0 8px 28px rgba(15,52,96,.45) !important;
}

/* SECRETS ALERT */
.secrets-alert {
    background:linear-gradient(135deg,#fff8e1,#fff3cd);
    border:1.5px solid #ffc107;border-radius:14px;padding:20px 22px;margin:12px 0 20px;
}
.secrets-alert h3{margin:0 0 8px;color:#856404;font-size:1em;}
.secrets-alert code{
    background:rgba(133,100,4,.12);padding:2px 8px;
    border-radius:6px;font-size:.88em;color:#5d4037;
}
.secrets-alert pre{
    background:#1e1e2e;color:#a8d8a8;padding:12px 16px;
    border-radius:10px;font-size:.85em;margin:10px 0 0;
    direction:ltr;text-align:left;
}

/* SIDEBAR */
section[data-testid="stSidebar"]{background:#0d1b3e;}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p{color:rgba(255,255,255,.8) !important;}
section[data-testid="stSidebar"] h3{color:#63b3ed !important;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3.  API KEY RESOLUTION  (Zero-Config from st.secrets)
# ============================================================

def _secret(key: str) -> str:
    try:
        v = st.secrets.get(key, "")
        if v:
            return v
    except Exception:
        pass
    return st.session_state.get(key, "")

GEMINI_KEY      = _secret("GEMINI_API_KEY")
GOOGLE_SRCH_KEY = _secret("GOOGLE_SEARCH_KEY")
GOOGLE_CX       = _secret("GOOGLE_CX")
_has_gemini = bool(GEMINI_KEY)
_has_search = bool(GOOGLE_SRCH_KEY and GOOGLE_CX)


# ============================================================
# 4.  SIDEBAR  (advanced / collapsed by default)
# ============================================================
with st.sidebar:
    st.markdown(
        "<h2 style='color:#63b3ed;margin:0;padding:10px 0 4px;'>⚙️ إعدادات متقدمة</h2>",
        unsafe_allow_html=True,
    )
    st.caption("المفاتيح تُقرأ تلقائياً من secrets.toml")

    with st.expander("🔑 مفاتيح API يدوية"):
        m_gem = st.text_input("Gemini API Key",     type="password", key="GEMINI_API_KEY")
        m_src = st.text_input("Google Search Key",  type="password", key="GOOGLE_SEARCH_KEY")
        m_cx  = st.text_input("Google CX",                           key="GOOGLE_CX")
        if m_gem: GEMINI_KEY = m_gem; _has_gemini = True
        if m_src: GOOGLE_SRCH_KEY = m_src
        if m_cx:  GOOGLE_CX = m_cx
        _has_search = bool(GOOGLE_SRCH_KEY and GOOGLE_CX)

    st.divider()
    with st.expander("🤖 خيارات الذكاء الاصطناعي"):
        do_enrich       = st.toggle("توليد أوصاف HTML",                  value=True)
        do_gray_ai      = st.toggle("فحص المنطقة الرمادية بـ Gemini",    value=False)
        do_img_fallback = st.toggle("جلب الصور المفقودة (Google Search)", value=False)
        max_enrich      = st.slider("حد الإثراء بـ AI", 10, 300, _MAX_ENRICH, 10)
    
    st.divider()
    if st.button("🔄 بدء من جديد", use_container_width=True):
        for k in ["analysis_done","new_df","gray_df","dup_df","gray_approvals","stats"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown(
        "<div style='text-align:center;padding:14px 0 4px;color:rgba(255,255,255,.35);font-size:.75em;'>"
        "Mahwous Engine v3.0</div>",
        unsafe_allow_html=True,
    )

# sidebar defaults for AI options when sidebar not rendered
if "do_enrich"       not in dir(): do_enrich       = True
if "do_gray_ai"      not in dir(): do_gray_ai      = False
if "do_img_fallback" not in dir(): do_img_fallback = False
if "max_enrich"      not in dir(): max_enrich      = _MAX_ENRICH


# ============================================================
# 5.  SESSION STATE INIT
# ============================================================
_DEFAULTS = {
    "analysis_done":   False,
    "new_df":          pd.DataFrame(),
    "gray_df":         pd.DataFrame(),
    "dup_df":          pd.DataFrame(),
    "existing_brands": [],
    "gray_approvals":  {},
    "stats":           {"new":0,"gray":0,"dup":0,"total_comp":0},
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ============================================================
# 6.  HERO HEADER
# ============================================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">🚀 محرك مهووس لاستكشاف الفرص</div>
    <p class="hero-sub">
        ارفع ملفات المتجر والمنافسين — نكتشف الفرص الجديدة ونجهّز ملفات سلة في ثوانٍ
    </p>
    <div class="hero-badges">
        <span class="hbadge">🛡️ منع التكرار الذكي</span>
        <span class="hbadge">🤖 Gemini AI</span>
        <span class="hbadge">⚡ Fuzzy Matching</span>
        <span class="hbadge">📦 تصدير لسلة مباشرةً</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 7.  SECRETS ALERT
# ============================================================
if not _has_gemini:
    st.markdown("""
<div class="secrets-alert">
    <h3>⚠️ مفتاح Gemini API غير مُعرَّف</h3>
    <p style="margin:0 0 8px;color:#6d4c41;font-size:.9em;">
        التطبيق يعمل بدونه لكن ميزات الإثراء والفحص الذكي ستكون معطّلة.<br>
        أضف المفتاح في ملف <code>.streamlit/secrets.toml</code>:
    </p>
    <pre>GEMINI_API_KEY    = "AIza...................."
GOOGLE_SEARCH_KEY = "..."   # اختياري
GOOGLE_CX         = "..."   # اختياري</pre>
</div>""", unsafe_allow_html=True)
else:
    st.success("✅ **Gemini AI جاهز** — الإثراء والفحص الذكي مُفعَّلان", icon="🤖")


# ============================================================
# 8.  UPLOAD SECTION  (side-by-side)
# ============================================================
st.markdown("---")
col_store, col_comp = st.columns([1, 1.5], gap="large")

with col_store:
    st.markdown('<p class="upload-title">🏪 ملفات متجر مهووس</p>'
                '<p class="upload-sub">الجدار الواقي — ملفات سلة (الأجزاء 1–4 أو أكثر)</p>',
                unsafe_allow_html=True)
    store_files = st.file_uploader(
        "store", type=["csv"], accept_multiple_files=True,
        key="uf_store", label_visibility="collapsed",
    )
    brands_file = st.file_uploader(
        "ملف الماركات الحالية (اختياري)", type=["csv"], key="uf_brands",
    )
    if store_files:
        st.markdown(
            f"<span style='color:#1b5e20;font-weight:700;font-size:.9em;'>"
            f"✅ {len(store_files)} ملف(ات) محملة</span>",
            unsafe_allow_html=True,
        )

with col_comp:
    st.markdown('<p class="upload-title">🔍 ملفات المنافسين</p>'
                '<p class="upload-sub">ارفع حتى 15 ملف CSV من مواقع المنافسين دفعةً واحدة</p>',
                unsafe_allow_html=True)
    comp_files = st.file_uploader(
        "comp", type=["csv"], accept_multiple_files=True,
        key="uf_comp", label_visibility="collapsed",
    )
    if comp_files:
        st.markdown(
            f"<span style='color:#1565c0;font-weight:700;font-size:.9em;'>"
            f"✅ {len(comp_files)} ملف(ات) محملة</span>",
            unsafe_allow_html=True,
        )
        with st.expander("📋 الملفات المرفوعة"):
            for f in comp_files[:15]:
                st.caption(f"• {f.name}")


# ============================================================
# 9.  START BUTTON
# ============================================================
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    ready = bool(store_files and comp_files)
    start = st.button(
        "🚀  بدء التحليل العميق",
        type="primary", use_container_width=True,
        disabled=not ready, key="btn_start",
    )
    if not ready:
        st.caption("⚠️ يجب رفع ملف للمتجر وملف واحد للمنافسين على الأقل")

st.markdown("---")


# ============================================================
# 10.  ANALYSIS PIPELINE
# ============================================================
if start:
    st.session_state["analysis_done"]  = False
    st.session_state["gray_approvals"] = {}

    with st.status("⏳ جاري معالجة الملفات…", expanded=True) as pipeline:

        # STEP 1: Store
        st.write("📂 **الخطوة 1/5** — تحميل ملفات متجر مهووس وبناء جدار الحماية…")
        store_df = load_store_products(store_files)
        st.write(f"✅ جدار الحماية جاهز: **{len(store_df):,}** منتج محمي")

        existing_brands = load_brands_list(brands_file)
        if existing_brands:
            st.write(f"🏷️ ماركات حالية محمّلة: **{len(existing_brands):,}**")

        # STEP 2: Competitors
        st.write("📦 **الخطوة 2/5** — طحن بيانات المنافسين…")
        all_comp: list[dict] = []
        for cf in comp_files[:15]:
            df_c = load_competitor_products(cf)
            all_comp.extend(df_c.to_dict("records"))
            st.write(f"  &nbsp;&nbsp;• **{cf.name}** ← {len(df_c):,} منتج")
        st.write(f"📊 إجمالي المنتجات للتحليل: **{len(all_comp):,}**")

        # STEP 3: Deduplication
        st.write("🔍 **الخطوة 3/5** — تشغيل مصفاة منع التكرار الثلاثية…")
        prog_bar = st.progress(0.0, text="جاري المقارنة…")

        def _upd(i, total, name):
            if total > 0:
                prog_bar.progress(
                    min(i / total, 1.0),
                    text=f"⚙️ {i}/{total} — {name[:55]}",
                )

        new_df, gray_df, dup_df = deduplicate_products(
            store_df=store_df,
            comp_products=all_comp,
            high_threshold=_HIGH_THRESHOLD,
            low_threshold=_LOW_THRESHOLD,
            progress_callback=_upd,
        )
        prog_bar.progress(1.0, text="✅ اكتملت المقارنة")
        st.write(
            f"🌟 **{len(new_df):,}** فرصة جديدة  |  "
            f"🔍 **{len(gray_df):,}** رمادية  |  "
            f"🗑️ **{len(dup_df):,}** مكرر"
        )

        # STEP 4: Gray AI
        if do_gray_ai and _has_gemini and not gray_df.empty:
            st.write(
                f"🤖 **الخطوة 4/5** — فحص المنطقة الرمادية "
                f"({len(gray_df)} منتج) بالذكاء الاصطناعي…"
            )
            ai_p    = st.progress(0.0)
            ai_added = 0
            extra   = []
            for i, (idx, row) in enumerate(gray_df.iterrows()):
                ai_p.progress((i+1)/len(gray_df), text=f"🧠 {i+1}/{len(gray_df)}")
                verdict = verify_with_gemini(
                    store_name=str(row.get("matched_store_product","")),
                    comp_name =str(row.get("product_name","")),
                    api_key   =GEMINI_KEY,
                )
                if verdict == "DIFFERENT":
                    extra.append(row.to_dict())
                    st.session_state["gray_approvals"][idx] = True
                    ai_added += 1
                elif verdict == "MATCH":
                    st.session_state["gray_approvals"][idx] = False
            if extra:
                new_df = pd.concat([new_df, pd.DataFrame(extra)], ignore_index=True)
            ai_p.progress(1.0, text=f"✅ Gemini أضاف {ai_added} للفرص")
        else:
            st.write("🤖 **الخطوة 4/5** — تم تخطي فحص الرمادية (معطّل أو لا مفتاح)")

        # STEP 5: Enrichment
        if do_enrich and _has_gemini and not new_df.empty:
            n_e = min(int(max_enrich), len(new_df))
            st.write(f"✨ **الخطوة 5/5** — تحضير أوصاف HTML لـ {n_e} منتج بـ Gemini…")
            enr_p    = st.progress(0.0)
            to_enr   = new_df.head(n_e)
            rest_df  = new_df.iloc[n_e:]
            enriched = []
            for i, (_, row) in enumerate(to_enr.iterrows()):
                enr_p.progress(
                    (i+1)/n_e,
                    text=f"✍️ {i+1}/{n_e} — {str(row.get('product_name',''))[:45]}",
                )
                enrich = enrich_product_with_gemini(
                    product_name=str(row.get("product_name","")),
                    image_url   =str(row.get("image_url","")),
                    product_type=str(row.get("product_type","perfume")),
                    api_key     =GEMINI_KEY,
                )
                d = row.to_dict()
                d.update(enrich)
                if do_img_fallback and _has_search and not d.get("image_url"):
                    d["image_url"] = fetch_fallback_image(
                        d.get("product_name",""), GOOGLE_SRCH_KEY, GOOGLE_CX
                    )
                enriched.append(d)
            enr_p.progress(1.0, text="✅ اكتملت جميع الأوصاف")
            new_df = pd.concat([pd.DataFrame(enriched), rest_df], ignore_index=True)
        else:
            st.write("✨ **الخطوة 5/5** — تم تخطي الإثراء (معطّل أو لا مفتاح)")

        # Save
        st.session_state.update({
            "analysis_done":   True,
            "new_df":          new_df,
            "gray_df":         gray_df,
            "dup_df":          dup_df,
            "existing_brands": existing_brands,
            "stats": {
                "new":       len(new_df),
                "gray":      len(gray_df),
                "dup":       len(dup_df),
                "total_comp":len(all_comp),
            },
        })
        pipeline.update(label="🎉 اكتمل التحليل بنجاح!", state="complete", expanded=False)

    st.toast("🎉 التحليل اكتمل — راجع نتائجك أدناه!", icon="✅")
    time.sleep(0.3)
    st.rerun()


# ============================================================
# 11.  WELCOME STATE  (before first run)
# ============================================================
if not st.session_state["analysis_done"]:
    st.markdown("""
<div style='text-align:center;padding:50px 0 60px;'>
    <div style='font-size:5em;margin-bottom:16px;'>🚀</div>
    <p style='font-size:1.15em;color:#888;max-width:480px;margin:0 auto;line-height:1.8;'>
        ارفع ملفات متجرك وملفات المنافسين<br>
        ثم اضغط <b style='color:#0f3460;'>بدء التحليل العميق</b><br>
        لاكتشاف الفرص الجديدة فوراً
    </p>
</div>""", unsafe_allow_html=True)
    st.stop()


# ============================================================
# 12.  RESULTS
# ============================================================
new_df          = st.session_state["new_df"]
gray_df         = st.session_state["gray_df"]
dup_df          = st.session_state["dup_df"]
existing_brands = st.session_state["existing_brands"]
s               = st.session_state["stats"]


# --- 12-A METRIC CARDS -------------------------------------------------------
_total   = s["new"] + s["gray"] + s["dup"]
_opp_pct = round(s["new"] / _total * 100, 1) if _total else 0.0

st.markdown(f"""
<div class="metric-grid">
  <div class="mcard mc-blue">
    <div class="mc-icon">📦</div>
    <div class="mc-num">{s["total_comp"]:,}</div>
    <div class="mc-label">إجمالي المنتجات المفحوصة</div>
  </div>
  <div class="mcard mc-green">
    <div class="mc-icon">🌟</div>
    <div class="mc-num">{s["new"]:,}</div>
    <div class="mc-label">الفرص الجديدة المكتشفة</div>
  </div>
  <div class="mcard mc-amber">
    <div class="mc-icon">🔍</div>
    <div class="mc-num">{s["gray"]:,}</div>
    <div class="mc-label">تحتاج مراجعة (رمادية)</div>
  </div>
  <div class="mcard mc-red">
    <div class="mc-icon">🗑️</div>
    <div class="mc-num">{s["dup"]:,}</div>
    <div class="mc-label">مكررات مستبعدة</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Opportunity rate bar
st.markdown(f"""
<div class="opp-bar-wrap">
  <div class="opp-bar-label">📈 نسبة الفرص الجديدة</div>
  <div class="opp-bar-outer">
    <div class="opp-bar-inner" style="width:{_opp_pct}%"></div>
  </div>
  <div class="opp-pct">{_opp_pct}%</div>
</div>
""", unsafe_allow_html=True)


# --- 12-B HELPERS ------------------------------------------------------------

def _split_types(records: list[dict]):
    perf, beau = [], []
    for r in records:
        if r.get("product_type") == "perfume":
            perf.append(r)
        else:
            beau.append(r)
    return perf, beau


def _download_trio(records: list[dict], brands: list[str], suffix: str):
    perf, beau = _split_types(records)
    c1, c2, c3 = st.columns(3)
    with c1:
        if perf:
            st.download_button(
                f"⬇️ ملف العطور لسلة  ({len(perf):,})",
                data=export_to_salla_csv(perf),
                file_name=f"mahwous_perfumes_{suffix}.csv",
                mime="text/csv", use_container_width=True,
                key=f"dl_p_{suffix}",
            )
        else:
            st.info("لا عطور جديدة")
    with c2:
        if beau:
            st.download_button(
                f"⬇️ ملف مكياج/عناية لسلة  ({len(beau):,})",
                data=export_to_salla_csv(beau),
                file_name=f"mahwous_beauty_{suffix}.csv",
                mime="text/csv", use_container_width=True,
                key=f"dl_b_{suffix}",
            )
        else:
            st.info("لا منتجات تجميل")
    with c3:
        st.download_button(
            "⬇️ الماركات الناقصة",
            data=export_missing_brands_csv(records, brands),
            file_name=f"missing_brands_{suffix}.csv",
            mime="text/csv", use_container_width=True,
            key=f"dl_br_{suffix}",
        )


def _show_df(df: pd.DataFrame, col_map: dict, height: int = 420):
    cols   = {k: v for k, v in col_map.items() if k in df.columns}
    disp   = df[list(cols.keys())].copy().rename(columns=cols)
    st.dataframe(disp, use_container_width=True, height=height)


# --- 12-C TABS ---------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    f"🌟 الفرص الجديدة  ({len(new_df):,})",
    f"🔍 المراجعة اليدوية  ({len(gray_df):,})",
    f"🗑️ المكررات المستبعدة  ({len(dup_df):,})",
])


# ── TAB 1: NEW OPPORTUNITIES ─────────────────────────────────────────────────
with tab1:
    if new_df.empty:
        st.info("💡 لم يُكتشف أي منتج جديد — كل منتجات المنافسين موجودة لديك بالفعل!")
    else:
        st.markdown("#### 📥 ملفات التصدير الجاهزة")
        _download_trio(new_df.to_dict("records"), existing_brands, "new")
        st.markdown("---")

        # Filters
        fc1, fc2, fc3, fc4 = st.columns([1.2, 1.2, 1.6, 1])
        with fc1:
            t_sel = st.selectbox("🗂️ النوع", ["الكل","عطور","مكياج وعناية"], key="f_t")
        with fc2:
            src_opts = ["الكل"] + (
                sorted(new_df["source_file"].dropna().unique().tolist())
                if "source_file" in new_df.columns else []
            )
            s_sel = st.selectbox("🏪 المصدر", src_opts, key="f_s")
        with fc3:
            srch  = st.text_input("🔎 بحث في الاسم", placeholder="اكتب للبحث…", key="f_q")
        with fc4:
            view  = st.radio("عرض", ["🃏 بطاقات","📋 جدول"], horizontal=True, key="f_v")

        disp = new_df.copy()
        if t_sel == "عطور"          and "product_type" in disp.columns:
            disp = disp[disp["product_type"] == "perfume"]
        elif t_sel == "مكياج وعناية" and "product_type" in disp.columns:
            disp = disp[disp["product_type"].isin(["beauty","other"])]
        if s_sel != "الكل" and "source_file" in disp.columns:
            disp = disp[disp["source_file"] == s_sel]
        if srch:
            disp = disp[disp["product_name"].str.contains(srch, case=False, na=False)]
        disp = disp.reset_index(drop=True)

        st.caption(f"عرض **{min(len(disp),60):,}** من أصل **{len(disp):,}** منتج")

        if view == "📋 جدول":
            _show_df(disp, {
                "product_name":   "اسم المنتج",
                "brand":          "الماركة",
                "product_type":   "النوع",
                "price":          "السعر",
                "match_score":    "نسبة الاختلاف %",
                "source_file":    "المصدر",
                "salla_category": "تصنيف سلة",
            }, 520)
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
                    ptype = str(row.get("product_type","other"))
                    score = float(row.get("match_score",0))
                    price = str(row.get("price","")).strip()
                    src   = str(row.get("source_file","")).replace(".csv","")
                    t_pill = ('<span class="pill pill-perf">🧴 عطر</span>'
                              if ptype=="perfume"
                              else '<span class="pill pill-beau">💄 مكياج</span>')
                    p_pill = (f'<span class="pill pill-price">💰 {price} ر.س</span>'
                              if price else "")
                    with col:
                        img = str(row.get("image_url","")).strip()
                        if img.startswith("http"):
                            try: st.image(img, use_container_width=True)
                            except Exception: st.markdown(
                                "<div style='height:150px;background:#f4f6fb;"
                                "border-radius:10px;display:flex;align-items:center;"
                                "justify-content:center;color:#ccc;font-size:2em;'>📷</div>",
                                unsafe_allow_html=True)
                        else:
                            st.markdown(
                                "<div style='height:150px;background:#f4f6fb;"
                                "border-radius:10px;display:flex;align-items:center;"
                                "justify-content:center;color:#ccc;font-size:2em;'>📷</div>",
                                unsafe_allow_html=True)
                        st.markdown(
                            f'<p class="pcard-name">{pname[:70]}{"…" if len(pname)>70 else ""}</p>'
                            f'{f'<p class="pcard-brand">🏷️ {brand}</p>' if brand else ""}'
                            f'<div style="margin-top:6px;">{t_pill}'
                            f'<span class="pill pill-new">✅ جديد</span>{p_pill}</div>'
                            f'<p class="pcard-meta">تشابه: {score:.0f}% • {src}</p>',
                            unsafe_allow_html=True)
            if len(disp) > LIMIT:
                st.info(f"يُعرض أول {LIMIT}. حمّل الـ CSV أعلاه للاطلاع على الكل ({len(disp):,}).")


# ── TAB 2: GRAY ZONE ─────────────────────────────────────────────────────────
with tab2:
    if gray_df.empty:
        st.success("🎯 لا منطقة رمادية — الفلترة دقيقة 100%!")
    else:
        # Metrics
        n_app = sum(1 for v in st.session_state["gray_approvals"].values() if v)
        n_rej = sum(1 for v in st.session_state["gray_approvals"].values() if not v)
        n_pnd = len(gray_df) - len(st.session_state["gray_approvals"])
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("🔍 الإجمالي",    f"{len(gray_df):,}")
        mc2.metric("✅ موافق",        f"{n_app:,}")
        mc3.metric("❌ مرفوض",        f"{n_rej:,}")
        mc4.metric("⏳ معلّق",        f"{n_pnd:,}")

        # AI bulk-verify button
        if _has_gemini:
            _, ab, _ = st.columns([2,1,2])
            with ab:
                if st.button("🤖 فحص كل الرمادية بـ Gemini", use_container_width=True):
                    ai_p2 = st.progress(0.0)
                    ai_a  = 0
                    for i,(idx,row) in enumerate(gray_df.iterrows()):
                        ai_p2.progress((i+1)/len(gray_df))
                        v = verify_with_gemini(
                            store_name=str(row.get("matched_store_product","")),
                            comp_name =str(row.get("product_name","")),
                            api_key   =GEMINI_KEY,
                        )
                        if v=="DIFFERENT":
                            st.session_state["gray_approvals"][idx]=True; ai_a+=1
                        elif v=="MATCH":
                            st.session_state["gray_approvals"][idx]=False
                    ai_p2.progress(1.0)
                    st.toast(f"✅ Gemini أضاف {ai_a} منتج", icon="🤖")
                    st.rerun()

        st.markdown("---")

        # Interactive dataframe
        st.markdown("##### 📋 عرض المنطقة الرمادية")
        _show_df(gray_df, {
            "product_name":          "اسم المنتج (المنافس)",
            "matched_store_product": "المنتج المشابه في متجرك",
            "match_score":           "درجة التشابه %",
            "product_type":          "النوع",
            "price":                 "السعر",
            "source_file":           "المصدر",
        }, 320)

        st.markdown("---")
        st.markdown("#### ✋ المراجعة اليدوية المنتج بالمنتج")

        for idx, row in gray_df.iterrows():
            pname   = str(row.get("product_name",""))
            score   = float(row.get("match_score",0))
            matched = str(row.get("matched_store_product","غير محدد"))
            cur     = st.session_state["gray_approvals"].get(idx, None)
            icon    = "✅" if cur is True else "❌" if cur is False else "⏳"

            with st.expander(f"{icon}  {pname[:65]}  │  تشابه {score:.0f}%"):
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown("**📦 منتج المنافس**")
                    st.write(pname)
                    img = str(row.get("image_url","")).strip()
                    if img.startswith("http"):
                        try: st.image(img, width=140)
                        except: pass
                    st.caption(str(row.get("source_file","")).replace(".csv",""))
                with r2:
                    st.markdown("**🏪 المطابق في متجرك**")
                    st.write(matched)
                    st.metric("التشابه", f"{score:.0f}%")
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("✅ إضافة كفرصة جديدة",
                                 key=f"add_{idx}", type="primary",
                                 use_container_width=True):
                        st.session_state["gray_approvals"][idx] = True
                        st.rerun()
                with b2:
                    if st.button("❌ تجاهل — هو مكرر",
                                 key=f"rej_{idx}", use_container_width=True):
                        st.session_state["gray_approvals"][idx] = False
                        st.rerun()

        approved_idxs = [i for i,v in st.session_state["gray_approvals"].items() if v]
        if approved_idxs:
            st.markdown("---")
            st.markdown(f"#### 📥 تحميل الموافق عليها ({len(approved_idxs)})")
            _download_trio(
                gray_df.loc[approved_idxs].to_dict("records"),
                existing_brands, "gray",
            )


# ── TAB 3: DUPLICATES ────────────────────────────────────────────────────────
with tab3:
    if dup_df.empty:
        st.success("🎉 لا مكررات — كل منتجات المنافسين جديدة بالنسبة لمتجرك!")
    else:
        st.success(
            f"🛡️ تم حظر **{len(dup_df):,} منتج مكرر** — متجرك محمي من الازدواجية!",
        )

        df1,df2,df3 = st.columns([1,1,2])
        with df1:
            dt = st.radio("النوع",["الكل","عطور","مكياج"],horizontal=True,key="rd_dt")
        with df2:
            dr = st.radio("السبب",["الكل","تطابق تام","تشابه عالٍ"],
                          horizontal=True,key="rd_dr")
        with df3:
            ds = st.text_input("🔎 بحث",placeholder="ابحث في المكررات…",key="ds_q")

        show = dup_df.copy()
        if dt == "عطور"   and "product_type" in show.columns:
            show = show[show["product_type"]=="perfume"]
        elif dt == "مكياج" and "product_type" in show.columns:
            show = show[show["product_type"].isin(["beauty","other"])]
        if dr == "تطابق تام"  and "match_reason" in show.columns:
            show = show[show["match_reason"]=="exact"]
        elif dr == "تشابه عالٍ" and "match_reason" in show.columns:
            show = show[show["match_reason"]=="fuzzy_high"]
        if ds and "product_name" in show.columns:
            show = show[show["product_name"].str.contains(ds,case=False,na=False)]

        reason_ar = {
            "exact":        "تطابق تام",
            "fuzzy_high":   "تشابه عالٍ جداً",
            "fuzzy_medium": "تشابه متوسط",
        }
        disp_dup = show.copy()
        if "match_reason" in disp_dup.columns:
            disp_dup["match_reason"] = disp_dup["match_reason"].map(reason_ar).fillna(disp_dup["match_reason"])
        if "match_score" in disp_dup.columns:
            disp_dup["match_score"] = disp_dup["match_score"].apply(
                lambda x: f"{float(x):.0f}%" if pd.notna(x) else ""
            )

        _show_df(disp_dup, {
            "product_name":          "اسم المنتج (المنافس)",
            "matched_store_product": "المنتج المطابق في متجرك",
            "match_score":           "درجة التطابق",
            "match_reason":          "سبب الاستبعاد",
            "product_type":          "النوع",
            "source_file":           "المصدر",
        }, 500)


# ============================================================
# 13.  FOOTER
# ============================================================
st.markdown("""
<div style='text-align:center;color:#bbb;font-size:.8em;
            padding:20px 0 8px;border-top:1px solid #eee;margin-top:32px;'>
    🚀 <b>Mahwous Opportunity Engine v3.0</b>
    &nbsp;|&nbsp; Gemini AI + Fuzzy Matching
    &nbsp;|&nbsp; صُمم حصرياً لمتجر <b>مهووس</b> 🧴
</div>
""", unsafe_allow_html=True)
