"""
app.py  ─  Mahwous Opportunity Engine  |  Streamlit UI
=======================================================
Enterprise-grade interface for the متجر مهووس sourcing &
deduplication engine.

Run with:  streamlit run app.py
"""

import time
from typing import Optional

import pandas as pd
import streamlit as st

from logic import (
    classify_product,
    deduplicate_products,
    enrich_product_with_gemini,
    export_missing_brands_csv,
    export_to_salla_csv,
    fetch_fallback_image,
    load_brands_list,
    load_competitor_products,
    load_store_products,
    normalize_name,
    verify_with_gemini,
)

# ══════════════════════════════════════════════════════════════════════════════
# 0.  PAGE CONFIG  &  CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="مهووس | محرك الفرص الذكي",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ── Global RTL & font ─────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap');
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif !important;
        direction: rtl;
    }
    /* ── Hero header ───────────────────────────────────── */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 36px 28px 28px;
        text-align: center;
        margin-bottom: 24px;
        color: white;
    }
    .hero h1 { font-size: 2.4em; font-weight: 900; margin: 0; }
    .hero p  { font-size: 1.1em; color: #ccc; margin-top: 8px; }

    /* ── Stat chips ─────────────────────────────────────── */
    .stat-chip {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 24px;
        font-weight: 700;
        font-size: 0.9em;
        margin: 4px;
    }
    .chip-green  { background:#d4edda; color:#155724; }
    .chip-yellow { background:#fff3cd; color:#856404; }
    .chip-red    { background:#f8d7da; color:#721c24; }

    /* ── Product cards ──────────────────────────────────── */
    .pcard {
        border: 1px solid #e8ecf0;
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 12px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,.07);
        transition: transform .15s, box-shadow .15s;
    }
    .pcard:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0,0,0,.12);
    }
    .pcard-title { font-weight: 700; font-size: 1em; color: #1a1a2e; margin: 6px 0 4px; }
    .pcard-brand { color: #0f3460; font-size: .85em; margin: 0; }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 10px;
        font-size: .75em;
        font-weight: 600;
        margin-top: 4px;
    }
    .badge-new  { background:#28a745; color:white; }
    .badge-gray { background:#ffc107; color:#333;  }
    .badge-dup  { background:#dc3545; color:white; }
    .badge-perf { background:#0f3460; color:white; }
    .badge-beau { background:#e91e8c; color:white; }

    /* ── Upload zones ───────────────────────────────────── */
    .upload-box {
        background: #f8f9ff;
        border: 2px dashed #bcc5e0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    /* ── Buttons ────────────────────────────────────────── */
    .stButton > button { font-family: 'Cairo', sans-serif !important; }
    div[data-testid="stDownloadButton"] > button {
        width: 100%;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

def _init_state() -> None:
    defaults = {
        "analysis_done":  False,
        "new_df":         pd.DataFrame(),
        "gray_df":        pd.DataFrame(),
        "dup_df":         pd.DataFrame(),
        "existing_brands": [],
        "gray_approvals": {},   # {row_index: True/False}
        "stats":          {"new": 0, "gray": 0, "dup": 0, "total_comp": 0},
        "expert_prompt":  "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding:12px 0;'>"
        "<h2 style='color:#0f3460; margin:0;'>🧴 مهووس</h2>"
        "<p style='color:#888; font-size:.85em; margin:2px 0;'>محرك الفرص الذكي</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### 🔑 مفاتيح API")
    gemini_key       = st.text_input("Gemini API Key",        type="password", key="sk_gemini")
    google_search_key = st.text_input("Google Search API Key", type="password", key="sk_gsearch")
    google_cx        = st.text_input("Google CX (Search Engine ID)", key="sk_cx")

    st.divider()
    st.markdown("### 🎛️ إعدادات الفلترة")

    high_thr = st.slider(
        "حد التطابق العالي (مكرر ✗)", 70, 99, 90,
        help="Score أعلى من هذا = مكرر يُستبعد",
    )
    low_thr = st.slider(
        "حد التطابق المنخفض (جديد ✓)", 10, 70, 50,
        help="Score أقل من هذا = فرصة جديدة مؤكدة",
    )

    st.divider()
    st.markdown("### ✨ خيارات الذكاء الاصطناعي")
    do_enrich        = st.checkbox("توليد أوصاف HTML تلقائياً",          value=True)
    do_gray_ai       = st.checkbox("فحص المنطقة الرمادية بـ Gemini",      value=False)
    do_img_fallback  = st.checkbox("جلب الصور عند غيابها (Google Search)", value=False)
    max_enrich       = st.number_input("أقصى عدد منتجات للإثراء", 1, 500, 50, step=10)

    st.divider()
    # Live stats
    s = st.session_state["stats"]
    if st.session_state["analysis_done"]:
        st.markdown("### 📊 نتائج آخر تحليل")
        st.markdown(
            f'<span class="stat-chip chip-green">🌟 جديد: {s["new"]}</span>'
            f'<span class="stat-chip chip-yellow">🔍 رمادي: {s["gray"]}</span>'
            f'<span class="stat-chip chip-red">🗑️ مكرر: {s["dup"]}</span>',
            unsafe_allow_html=True,
        )
        total = s["new"] + s["gray"] + s["dup"]
        if total:
            pct = round(s["new"] / total * 100, 1)
            st.progress(s["new"] / total, text=f"نسبة الفرص: {pct}%")

    st.divider()
    if st.button("🔄 إعادة ضبط كل شيء", use_container_width=True):
        for k in ["analysis_done", "new_df", "gray_df", "dup_df",
                  "gray_approvals", "stats"]:
            st.session_state.pop(k, None)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="hero">
        <h1>🧴 مهووس | محرك اكتشاف الفرص الذكي</h1>
        <p>اكشف الفرص الجديدة • امنع التكرار • صدّر ملفاتك جاهزةً لرفعها مباشرةً على سلة</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FILE UPLOAD SECTION
# ══════════════════════════════════════════════════════════════════════════════

left_col, right_col = st.columns([1, 1.4], gap="large")

with left_col:
    st.markdown("#### 🏪 ملفات متجرك (جدار الحماية)")
    st.caption(
        "ارفع ملفات متجر مهووس  (الأجزاء 1-4 أو أكثر) — صيغة Salla"
    )
    store_files = st.file_uploader(
        "ملفات متجر مهووس",
        type=["csv"],
        accept_multiple_files=True,
        key="uf_store",
        label_visibility="collapsed",
    )

    brands_file = st.file_uploader(
        "ملف الماركات الحالية  (اختياري — لاكتشاف الماركات الناقصة بدقة أكبر)",
        type=["csv"],
        key="uf_brands",
    )

    if store_files:
        st.success(f"✅ {len(store_files)} ملف(ات) للمتجر محملة")

with right_col:
    st.markdown("#### 🔍 ملفات المنافسين (حتى 15 ملف)")
    st.caption("ارفع ملفات CSV المستخرجة من مواقع المنافسين — يدعم أي تنسيق")
    comp_files = st.file_uploader(
        "ملفات المنافسين",
        type=["csv"],
        accept_multiple_files=True,
        key="uf_comp",
        label_visibility="collapsed",
    )
    if comp_files:
        st.success(f"✅ {len(comp_files)} ملف(ات) للمنافسين محملة")
        with st.expander("📋 عرض الملفات المرفوعة"):
            for cf in comp_files[:15]:
                st.caption(f"• {cf.name}")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ANALYSIS TRIGGER
# ══════════════════════════════════════════════════════════════════════════════

_, btn_col, _ = st.columns([1, 2.5, 1])
with btn_col:
    ready = bool(store_files and comp_files)
    start = st.button(
        "🚀 بدء التحليل العميق",
        type="primary",
        use_container_width=True,
        disabled=not ready,
        key="btn_start",
    )
    if not ready:
        st.caption("⚠️ يجب رفع ملفات المتجر وملف منافس واحد على الأقل")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

if start:
    st.session_state["analysis_done"] = False
    st.session_state["gray_approvals"] = {}

    # ── Step 1: Load store ───────────────────────────────────────────────────
    with st.status("⏳ جاري تحميل بيانات المتجر…", expanded=True) as status:
        store_df = load_store_products(store_files)
        st.write(f"✅ تم تحميل **{len(store_df):,}** منتج من المتجر")

        existing_brands = load_brands_list(brands_file)
        st.write(f"✅ تم تحميل **{len(existing_brands):,}** ماركة حالية")

        # ── Step 2: Load competitors ─────────────────────────────────────────
        all_comp: list[dict] = []
        for cf in comp_files[:15]:
            df_c = load_competitor_products(cf)
            all_comp.extend(df_c.to_dict("records"))
            st.write(f"  • {cf.name}: {len(df_c):,} منتج")

        st.write(f"📦 إجمالي منتجات المنافسين: **{len(all_comp):,}**")
        status.update(label="✅ اكتمل التحميل", state="complete", expanded=False)

    # ── Step 3: Deduplication ────────────────────────────────────────────────
    prog = st.progress(0.0)
    status_txt = st.empty()

    def _upd(i: int, total: int, name: str) -> None:
        if total > 0:
            prog.progress(min(i / total, 1.0))
        status_txt.caption(f"⚙️ فحص {i}/{total}:  {name[:60]}")

    with st.spinner("🔍 جاري الفلترة ومنع التكرار…"):
        new_df, gray_df, dup_df = deduplicate_products(
            store_df         = store_df,
            comp_products    = all_comp,
            high_threshold   = high_thr,
            low_threshold    = low_thr,
            progress_callback= _upd,
        )

    prog.progress(1.0)
    status_txt.empty()

    # ── Step 4: Gray-zone AI verification (optional) ─────────────────────────
    if do_gray_ai and gemini_key and not gray_df.empty:
        with st.status(f"🤖 فحص المنطقة الرمادية ({len(gray_df)} منتج)…", expanded=True) as gs:
            gray_approved   = []
            gray_rejected   = []
            gray_unresolved = []
            ai_prog = st.progress(0.0)

            for i, (idx, row) in enumerate(gray_df.iterrows()):
                ai_prog.progress((i + 1) / len(gray_df))
                verdict = verify_with_gemini(
                    store_name = row.get("matched_store_product", ""),
                    comp_name  = row.get("product_name", ""),
                    api_key    = gemini_key,
                )
                if verdict == "DIFFERENT":
                    gray_approved.append(row.to_dict())
                    st.session_state["gray_approvals"][idx] = True
                elif verdict == "MATCH":
                    gray_rejected.append(row.to_dict())
                    st.session_state["gray_approvals"][idx] = False
                else:
                    gray_unresolved.append(row.to_dict())

            # Merge AI-approved gray into new_df
            if gray_approved:
                approved_df = pd.DataFrame(gray_approved)
                new_df      = pd.concat([new_df, approved_df], ignore_index=True)

            gs.update(
                label=(
                    f"✅ AI: أضاف {len(gray_approved)} | "
                    f"رفض {len(gray_rejected)} | "
                    f"يحتاج مراجعة {len(gray_unresolved)}"
                ),
                state="complete",
                expanded=False,
            )

    # ── Step 5: AI Enrichment ────────────────────────────────────────────────
    expert_prompt = st.session_state.get("expert_prompt", "")

    if do_enrich and gemini_key and not new_df.empty:
        to_enrich   = new_df.head(int(max_enrich))
        rest        = new_df.iloc[int(max_enrich):]
        enriched    = []
        enrich_prog = st.progress(0.0)
        enr_status  = st.empty()
        n = len(to_enrich)

        for i, (_, row) in enumerate(to_enrich.iterrows()):
            enr_status.caption(
                f"✨ تحضير المنتج {i+1}/{n}:  {row.get('product_name','')[:50]}"
            )
            enrich_prog.progress((i + 1) / n)

            enrichment = enrich_product_with_gemini(
                product_name  = row.get("product_name", ""),
                image_url     = row.get("image_url", ""),
                product_type  = row.get("product_type", "perfume"),
                api_key       = gemini_key,
                expert_prompt = expert_prompt,
            )
            d = row.to_dict()
            d.update(enrichment)

            # Image fallback
            if (do_img_fallback and google_search_key
                    and not d.get("image_url")):
                d["image_url"] = fetch_fallback_image(
                    d.get("product_name", ""),
                    google_search_key,
                    google_cx or "",
                )

            enriched.append(d)

        enrich_prog.progress(1.0)
        enr_status.empty()

        new_df = pd.concat(
            [pd.DataFrame(enriched), rest], ignore_index=True
        )

    # ── Persist results ──────────────────────────────────────────────────────
    st.session_state.update(
        {
            "analysis_done":   True,
            "new_df":          new_df,
            "gray_df":         gray_df,
            "dup_df":          dup_df,
            "existing_brands": existing_brands,
            "stats": {
                "new":        len(new_df),
                "gray":       len(gray_df),
                "dup":        len(dup_df),
                "total_comp": len(all_comp),
            },
        }
    )

    st.toast("🎉 اكتمل التحليل!", icon="✅")
    time.sleep(0.4)
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 7.  RESULTS  (rendered every run if analysis_done)
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state["analysis_done"]:
    # Welcome state
    st.markdown(
        """
        <div style='text-align:center; color:#aaa; padding:60px 0;'>
            <div style='font-size:4em;'>🚀</div>
            <p style='font-size:1.2em;'>ارفع الملفات واضغط <b>بدء التحليل العميق</b> لتبدأ</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

new_df          = st.session_state["new_df"]
gray_df         = st.session_state["gray_df"]
dup_df          = st.session_state["dup_df"]
existing_brands = st.session_state["existing_brands"]

# ── Summary banner ────────────────────────────────────────────────────────────
s = st.session_state["stats"]
st.markdown(
    f"""
    <div style='display:flex; gap:16px; justify-content:center;
                flex-wrap:wrap; margin:12px 0 24px;'>
        <div style='background:#e8f5e9; border-radius:12px; padding:14px 28px; text-align:center;'>
            <div style='font-size:2em; font-weight:900; color:#2e7d32;'>{s["new"]:,}</div>
            <div style='color:#555;'>🌟 فرصة جديدة</div>
        </div>
        <div style='background:#fff8e1; border-radius:12px; padding:14px 28px; text-align:center;'>
            <div style='font-size:2em; font-weight:900; color:#f57f17;'>{s["gray"]:,}</div>
            <div style='color:#555;'>🔍 منطقة رمادية</div>
        </div>
        <div style='background:#fce4ec; border-radius:12px; padding:14px 28px; text-align:center;'>
            <div style='font-size:2em; font-weight:900; color:#b71c1c;'>{s["dup"]:,}</div>
            <div style='color:#555;'>🗑️ منتج مكرر</div>
        </div>
        <div style='background:#e3f2fd; border-radius:12px; padding:14px 28px; text-align:center;'>
            <div style='font-size:2em; font-weight:900; color:#1565c0;'>{s["total_comp"]:,}</div>
            <div style='color:#555;'>📦 منتج فحصناه</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    [
        f"🌟 الفرص الجديدة  ({len(new_df)})",
        f"🔍 المراجعة اليدوية  ({len(gray_df)})",
        f"🗑️ المنتجات المكررة  ({len(dup_df)})",
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1  –  NEW OPPORTUNITIES
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if new_df.empty:
        st.info("لم يتم العثور على فرص جديدة — جميع المنتجات موجودة بالفعل في متجرك 🎉")
    else:
        # ── Export row ────────────────────────────────────────────────────────
        st.markdown("#### 📥 تحميل ملفات التصدير")
        ex1, ex2, ex3 = st.columns(3)

        def _products_of_type(df: pd.DataFrame, ptype: str) -> list[dict]:
            if df.empty or "product_type" not in df.columns:
                return []
            if ptype == "perfume":
                return df[df["product_type"] == "perfume"].to_dict("records")
            else:
                return df[df["product_type"].isin(["beauty", "other"])].to_dict("records")

        perf_list   = _products_of_type(new_df, "perfume")
        beauty_list = _products_of_type(new_df, "beauty")

        with ex1:
            if perf_list:
                st.download_button(
                    f"⬇️ ملف العطور لسلة ({len(perf_list)})",
                    data      = export_to_salla_csv(perf_list),
                    file_name = "mahwous_perfumes_salla.csv",
                    mime      = "text/csv",
                    use_container_width=True,
                    key="dl_perfumes",
                )
            else:
                st.info("لا عطور جديدة")

        with ex2:
            if beauty_list:
                st.download_button(
                    f"⬇️ ملف المكياج والعناية لسلة ({len(beauty_list)})",
                    data      = export_to_salla_csv(beauty_list),
                    file_name = "mahwous_beauty_salla.csv",
                    mime      = "text/csv",
                    use_container_width=True,
                    key="dl_beauty",
                )
            else:
                st.info("لا منتجات تجميل جديدة")

        with ex3:
            brands_csv = export_missing_brands_csv(
                new_df.to_dict("records"), existing_brands
            )
            st.download_button(
                "⬇️ الماركات الناقصة",
                data      = brands_csv,
                file_name = "missing_brands.csv",
                mime      = "text/csv",
                use_container_width=True,
                key="dl_brands",
            )

        st.divider()

        # ── Filters ───────────────────────────────────────────────────────────
        fc1, fc2, fc3 = st.columns([1, 1, 2])
        with fc1:
            type_sel = st.selectbox(
                "نوع المنتج", ["الكل", "عطور", "مكياج وعناية"], key="sel_type"
            )
        with fc2:
            src_opts = ["الكل"]
            if "source_file" in new_df.columns:
                src_opts += sorted(new_df["source_file"].dropna().unique().tolist())
            src_sel = st.selectbox("المصدر (متجر المنافس)", src_opts, key="sel_src")
        with fc3:
            search_q = st.text_input("🔎 بحث في اسم المنتج", key="srch_new")

        disp = new_df.copy()
        if type_sel == "عطور" and "product_type" in disp.columns:
            disp = disp[disp["product_type"] == "perfume"]
        elif type_sel == "مكياج وعناية" and "product_type" in disp.columns:
            disp = disp[disp["product_type"].isin(["beauty", "other"])]
        if src_sel != "الكل" and "source_file" in disp.columns:
            disp = disp[disp["source_file"] == src_sel]
        if search_q:
            disp = disp[
                disp["product_name"].str.contains(search_q, case=False, na=False)
            ]

        disp = disp.reset_index(drop=True)
        st.caption(f"عرض **{min(len(disp), 60)}** منتج من أصل **{len(disp)}**")

        # ── Product cards  (3-column grid, max 60) ────────────────────────────
        DISPLAY_LIMIT = 60
        for row_start in range(0, min(len(disp), DISPLAY_LIMIT), 3):
            cols = st.columns(3, gap="medium")
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx >= len(disp):
                    break
                row = disp.iloc[idx]
                with col:
                    img_url = str(row.get("image_url", "")).strip()
                    if img_url.startswith("http"):
                        try:
                            st.image(img_url, use_container_width=True)
                        except Exception:
                            st.markdown(
                                "<div style='height:160px; background:#f0f0f0; "
                                "border-radius:8px; display:flex; align-items:center; "
                                "justify-content:center; color:#bbb;'>📷</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown(
                            "<div style='height:160px; background:#f0f0f0; "
                            "border-radius:8px; display:flex; align-items:center; "
                            "justify-content:center; color:#bbb;'>📷 لا صورة</div>",
                            unsafe_allow_html=True,
                        )

                    pname  = str(row.get("product_name", ""))
                    brand  = str(row.get("brand", ""))
                    ptype  = str(row.get("product_type", "other"))
                    score  = row.get("match_score", 0)
                    price  = str(row.get("price", "")).strip()
                    source = str(row.get("source_file", "")).replace(".csv", "")

                    type_badge = (
                        '<span class="badge badge-perf">عطر</span>'
                        if ptype == "perfume"
                        else '<span class="badge badge-beau">مكياج</span>'
                    )
                    price_txt = (
                        f'<span style="color:#e91e8c; font-weight:700;">{price} ر.س</span>'
                        if price else ""
                    )

                    st.markdown(
                        f"""
                        <p class="pcard-title">{pname[:65]}{"…" if len(pname)>65 else ""}</p>
                        {f'<p class="pcard-brand">🏷️ {brand}</p>' if brand else ""}
                        <div style="margin-top:6px;">
                            {type_badge}
                            <span class="badge badge-new">🆕 جديد</span>
                            {price_txt}
                        </div>
                        <div style="margin-top:4px; font-size:.75em; color:#aaa;">
                            تشابه: {score:.0f}% • {source}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        if len(disp) > DISPLAY_LIMIT:
            st.info(
                f"يُعرض أول {DISPLAY_LIMIT} منتج. "
                f"حمّل ملف CSV لرؤية جميع الـ {len(disp)} منتج."
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2  –  GRAY ZONE (manual + AI review)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if gray_df.empty:
        st.success("✅ لا توجد منتجات في المنطقة الرمادية — الفلترة دقيقة جداً!")
    else:
        st.warning(
            f"⚠️ **{len(gray_df)}** منتج في المنطقة الرمادية (تشابه {low_thr}–{high_thr}%) "
            "— راجعها يدوياً أو بالذكاء الاصطناعي"
        )

        # ── AI bulk-verify button ─────────────────────────────────────────────
        if gemini_key:
            ai_col, _ = st.columns([1, 3])
            with ai_col:
                if st.button("🤖 فحص الكل بـ Gemini الآن", type="secondary"):
                    ai_prog2 = st.progress(0.0)
                    added = 0
                    for i, (idx, row) in enumerate(gray_df.iterrows()):
                        ai_prog2.progress((i + 1) / len(gray_df))
                        verdict = verify_with_gemini(
                            store_name = row.get("matched_store_product", ""),
                            comp_name  = row.get("product_name", ""),
                            api_key    = gemini_key,
                        )
                        if verdict == "DIFFERENT":
                            st.session_state["gray_approvals"][idx] = True
                            added += 1
                        elif verdict == "MATCH":
                            st.session_state["gray_approvals"][idx] = False
                    ai_prog2.progress(1.0)
                    st.toast(f"✅ Gemini أضاف {added} منتج جديد", icon="🤖")
                    st.rerun()

        # Stats
        n_approved = sum(1 for v in st.session_state["gray_approvals"].values() if v)
        n_rejected = sum(1 for v in st.session_state["gray_approvals"].values() if not v)
        n_pending  = len(gray_df) - len(st.session_state["gray_approvals"])

        st.markdown(
            f'<span class="stat-chip chip-green">✅ موافق: {n_approved}</span>'
            f'<span class="stat-chip chip-red">❌ مرفوض: {n_rejected}</span>'
            f'<span class="stat-chip chip-yellow">⏳ معلّق: {n_pending}</span>',
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Per-product expanders ─────────────────────────────────────────────
        for idx, row in gray_df.iterrows():
            pname    = row.get("product_name", "")
            score    = row.get("match_score", 0)
            matched  = row.get("matched_store_product", "غير محدد")
            current  = st.session_state["gray_approvals"].get(idx, None)

            status_icon = (
                "✅" if current is True
                else "❌" if current is False
                else "⏳"
            )

            with st.expander(
                f"{status_icon}  {pname[:70]}  │  تشابه {score:.0f}%", expanded=False
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**📦 منتج المنافس:**")
                    st.write(pname)
                    img = str(row.get("image_url", "")).strip()
                    if img.startswith("http"):
                        try:
                            st.image(img, width=150)
                        except Exception:
                            pass
                    src = str(row.get("source_file", "")).replace(".csv", "")
                    st.caption(f"المصدر: {src}")

                with c2:
                    st.markdown("**🏪 المنتج المشابه في متجرك:**")
                    st.write(matched)
                    st.metric("درجة التشابه", f"{score:.0f}%")
                    ptype = row.get("product_type", "other")
                    st.caption(
                        f"النوع: {'عطر' if ptype=='perfume' else 'مكياج/عناية'}"
                    )

                b1, b2 = st.columns(2)
                with b1:
                    btn_type_add = "primary" if current is True else "secondary"
                    if st.button(
                        "✅ إضافة كفرصة جديدة",
                        key=f"add_{idx}",
                        type=btn_type_add,
                        use_container_width=True,
                    ):
                        st.session_state["gray_approvals"][idx] = True
                        st.rerun()
                with b2:
                    btn_type_rej = "primary" if current is False else "secondary"
                    if st.button(
                        "❌ تجاهل (هو مكرر)",
                        key=f"rej_{idx}",
                        type=btn_type_rej,
                        use_container_width=True,
                    ):
                        st.session_state["gray_approvals"][idx] = False
                        st.rerun()

        # ── Export approved gray products ─────────────────────────────────────
        approved_idxs = [i for i, v in st.session_state["gray_approvals"].items() if v]
        if approved_idxs:
            st.divider()
            approved_records = gray_df.loc[approved_idxs].to_dict("records")

            ap1, ap2, ap3 = st.columns(3)
            perf_ap   = [p for p in approved_records if p.get("product_type") == "perfume"]
            beauty_ap = [
                p for p in approved_records
                if p.get("product_type") in ("beauty", "other")
            ]

            with ap1:
                if perf_ap:
                    st.download_button(
                        f"⬇️ عطور موافق عليها ({len(perf_ap)})",
                        data      = export_to_salla_csv(perf_ap),
                        file_name = "gray_approved_perfumes_salla.csv",
                        mime      = "text/csv",
                        use_container_width=True,
                        key="dl_gray_perf",
                    )
            with ap2:
                if beauty_ap:
                    st.download_button(
                        f"⬇️ مكياج/عناية موافق عليه ({len(beauty_ap)})",
                        data      = export_to_salla_csv(beauty_ap),
                        file_name = "gray_approved_beauty_salla.csv",
                        mime      = "text/csv",
                        use_container_width=True,
                        key="dl_gray_beau",
                    )
            with ap3:
                brands_gray = export_missing_brands_csv(
                    approved_records, existing_brands
                )
                st.download_button(
                    "⬇️ ماركات موافق عليها",
                    data      = brands_gray,
                    file_name = "gray_approved_brands.csv",
                    mime      = "text/csv",
                    use_container_width=True,
                    key="dl_gray_brands",
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3  –  DUPLICATES
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    if dup_df.empty:
        st.info("لا توجد مكررات — جميع المنتجات جديدة!")
    else:
        st.success(
            f"✅ تم استبعاد **{len(dup_df):,}** منتج مكرر بنجاح — "
            "المتجر محمي من التكرار!"
        )

        dup_search = st.text_input("🔎 بحث في المكررات", key="srch_dup")
        show_dup   = dup_df.copy()
        if dup_search:
            show_dup = show_dup[
                show_dup["product_name"].str.contains(
                    dup_search, case=False, na=False
                )
            ]

        # Type filter
        type_dup_sel = st.radio(
            "تصفية حسب النوع",
            ["الكل", "عطور", "مكياج وعناية"],
            horizontal=True,
            key="rad_dup",
        )
        if type_dup_sel == "عطور" and "product_type" in show_dup.columns:
            show_dup = show_dup[show_dup["product_type"] == "perfume"]
        elif type_dup_sel == "مكياج وعناية" and "product_type" in show_dup.columns:
            show_dup = show_dup[show_dup["product_type"].isin(["beauty", "other"])]

        reason_map = {
            "exact":       "تطابق تام",
            "fuzzy_high":  "تشابه عالٍ جداً",
            "fuzzy_medium":"تشابه متوسط",
        }
        if not show_dup.empty:
            disp_cols = {
                "product_name":           "اسم المنتج (المنافس)",
                "matched_store_product":  "المنتج المطابق في متجرك",
                "match_score":            "درجة التطابق %",
                "match_reason":           "سبب الاستبعاد",
                "source_file":            "مصدر الملف",
            }
            display_dup = show_dup[
                [c for c in disp_cols if c in show_dup.columns]
            ].copy()
            if "match_reason" in display_dup.columns:
                display_dup["match_reason"] = display_dup["match_reason"].map(
                    reason_map
                ).fillna(display_dup["match_reason"])
            if "match_score" in display_dup.columns:
                display_dup["match_score"] = display_dup["match_score"].apply(
                    lambda x: f"{float(x):.0f}%" if pd.notna(x) else ""
                )
            display_dup.rename(columns=disp_cols, inplace=True)
            st.dataframe(display_dup, use_container_width=True, height=420)
        else:
            st.info("لا نتائج تطابق الفلتر المحدد")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown(
    """
    <div style='text-align:center; color:#aaa; font-size:.82em; padding:8px 0;'>
        🧴 <b>مهووس</b> – محرك اكتشاف الفرص الذكي  |
        مدعوم بـ Gemini AI & Fuzzy Matching  |
        صُمم لمتجر مهووس السعودي للعطور
    </div>
    """,
    unsafe_allow_html=True,
)
