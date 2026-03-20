"""
logic.py  -  Mahwous Opportunity Engine  v6.0
=============================================
Automated Resolution Engine (ARE)
Zero Gray Zone | Zero Manual Review | 100% Definitive Decisions

Resolution Pipeline for ambiguous products (50-90% fuzzy match):
  Step 1: Deep Feature Extraction  → volume / concentration / gender
  Step 2: SKU Feature Comparison   → any mismatch = DIFFERENT (new opp)
  Step 3: Vision AI Court          → Gemini image+text judge
  Step 4: Safe Fallback            → DIFFERENT (business-safe default)

Classification Pipeline (5-Gate):
  G1 Hard Rules → G2 Category Path → G3 Weighted Scoring
  → G4 Brand KB → G5 AI Oracle
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd
import requests
from thefuzz import fuzz, process

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

CAT_PERFUME = "perfume"
CAT_BEAUTY  = "beauty"
CAT_UNKNOWN = "unknown"

GATE_HARD_RULES = "G1:HardRules"
GATE_CATEGORY   = "G2:CategoryPath"
GATE_SCORING    = "G3:WeightedScoring"
GATE_BRAND_KB   = "G4:BrandKB"
GATE_AI_ORACLE  = "G5:AIOracle"
GATE_FALLBACK   = "G0:Fallback"

CONFIDENCE_FINALIZE = {
    GATE_HARD_RULES: 0.85,
    GATE_CATEGORY:   0.85,
    GATE_SCORING:    0.75,
    GATE_BRAND_KB:   0.70,
    GATE_AI_ORACLE:  0.50,
}

# Resolution verdicts
RES_DUPLICATE  = "duplicate"
RES_DIFFERENT  = "different"

# Match reasons (for duplicates table)
REASON_EXACT_NAME    = "تطابق نصي تام 100%"
REASON_FUZZY_HIGH    = "تشابه نصي عالٍ ≥90%"
REASON_SKU_MATCH     = "حسم بالخصائص — نفس الـ SKU"
REASON_VISION_AI     = "حسم بالذكاء الاصطناعي البصري"
REASON_SAFE_DEFAULT  = "قرار آمن — إعادة السعي"

# ── Rate limiter ─────────────────────────────────────────────────────────────
_LAST_CALL: float = 0.0
_MIN_GAP:   float = 1.8

def _rate_sleep() -> None:
    global _LAST_CALL
    gap = time.time() - _LAST_CALL
    if gap < _MIN_GAP:
        time.sleep(_MIN_GAP - gap)
    _LAST_CALL = time.time()

def _gemini_retry(model, prompt, parts=None, retries=4) -> Optional[str]:
    for attempt in range(retries):
        try:
            _rate_sleep()
            content = parts if parts else prompt
            resp = model.generate_content(content)
            return resp.text.strip()
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ("quota","rate","429","exhausted")):
                wait = 5 * (2 ** attempt)
                logger.warning(f"Rate-limit attempt {attempt+1}, waiting {wait}s")
                time.sleep(wait)
            else:
                logger.error(f"Gemini error attempt {attempt+1}: {e}")
                break
    return None


# ============================================================================
# SECTION A  —  DEEP FEATURE EXTRACTION
# ============================================================================

# Volume: captures "100ml", "50 مل", "200g", "1.7oz" etc.
_VOL_RE = re.compile(
    r"(\d+\.?\d*)\s*(ml|مل|g|gr|غ|oz|fl\.?\s*oz)",
    re.IGNORECASE | re.UNICODE,
)

# Concentration keyword map  (longest match first, applied on lowercased text)
_CONC_MAP: list[tuple[str, str]] = [
    # Extrait first (most specific)
    ("اكستريت دو بارفان",      "Extrait"),
    ("اكستريت دو بارفيوم",     "Extrait"),
    ("extrait de parfum",      "Extrait"),
    ("extrait de perfum",      "Extrait"),
    ("extrait",                "Extrait"),
    ("اليكسير دي بارفيوم",     "Extrait"),
    ("اليكسير دي بارفان",      "Extrait"),
    # EDP
    ("eau de parfum",          "EDP"),
    ("او دو برفيوم",            "EDP"),
    ("او دو بارفيوم",           "EDP"),
    ("او دي بارفيوم",           "EDP"),
    ("او دو بيرفيوم",           "EDP"),
    ("او دو بارفان",            "EDP"),
    ("اودي برفيوم",             "EDP"),
    ("اوه دو بارفيوم",          "EDP"),
    ("لو دي بارفان",            "EDP"),
    ("لو بارفان",               "EDP"),
    ("اي دي بيرفيوم",           "EDP"),
    ("\bedp\b",               "EDP"),
    ("بارفيوم",                 "EDP"),
    ("برفيوم",                  "EDP"),
    ("بيرفيوم",                 "EDP"),
    ("بارفان",                  "EDP"),
    # EDT
    ("eau de toilette",        "EDT"),
    ("او دو تواليت",            "EDT"),
    ("او دي تواليت",            "EDT"),
    ("اي دي تواليت",            "EDT"),
    ("\bedt\b",               "EDT"),
    # EDC
    ("eau de cologne",         "EDC"),
    ("او دو كولون",             "EDC"),
    ("او دي كولون",             "EDC"),
    ("\bedc\b",               "EDC"),
    ("كولونيا",                 "EDC"),
    ("cologne",                "EDC"),
]

# Gender keywords
_MASC_KW  = ["رجالي","للرجال","رجال","men","homme","man",
              "pour homme","for men","for him","هوم"]
_FEM_KW   = ["نسائي","للنساء","نساء","women","femme","woman",
              "pour femme","for women","for her","فيم","وومن"]
_UNI_KW   = ["للجنسين","unisex","mixte","both","للجنسين"]


@dataclass
class ProductFeatures:
    """Extracted SKU-level features from product name."""
    volume_ml:     str = ""   # normalised e.g. "100.0"
    concentration: str = ""   # EDP / EDT / EDC / Extrait / …
    gender:        str = ""   # male / female / unisex / ""


def extract_features(name: str) -> ProductFeatures:
    """
    Extract volume, concentration and gender from a product name.
    Returns empty strings for fields that cannot be determined.
    """
    if not isinstance(name, str):
        return ProductFeatures()

    text = name.lower().strip()
    text_ar = re.sub(r"[أإآ]", "ا", text)
    text_ar = re.sub(r"ى", "ي", text_ar)

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_ml = ""
    m = _VOL_RE.search(name)
    if m:
        num  = float(m.group(1))
        unit = m.group(2).lower().strip()
        oz_units = ("oz", "fl.oz", "fl. oz", "floz")
        factor = 29.5735 if any(o in unit for o in oz_units) else 1.0
        vol_ml = f"{num * factor:.1f}"

    # ── Concentration ─────────────────────────────────────────────────────────
    # Fast path: check plain abbreviations first (avoids \b regex issues)
    conc = ""
    _text_words = set(re.split(r"[^a-zA-Z0-9]+", text.upper()))
    if   "EXTRAIT" in _text_words:                        conc = "Extrait"
    elif "EDP"     in _text_words:                        conc = "EDP"
    elif "EDT"     in _text_words:                        conc = "EDT"
    elif "EDC"     in _text_words:                        conc = "EDC"

    if not conc:
        for kw, label in _CONC_MAP:
            pattern = kw if kw.startswith("(") else re.escape(kw)
            if re.search(pattern, text_ar, re.IGNORECASE | re.UNICODE):
                conc = label
                break

    # ── Gender ────────────────────────────────────────────────────────────────
    gend = ""
    if any(k in text_ar for k in _UNI_KW):
        gend = "unisex"
    elif any(k in text_ar for k in _MASC_KW):
        gend = "male"
    elif any(k in text_ar for k in _FEM_KW):
        gend = "female"

    return ProductFeatures(volume_ml=vol_ml, concentration=conc, gender=gend)


def features_conflict(comp: ProductFeatures, store: ProductFeatures) -> tuple[bool, str]:
    """
    Compare two products' SKU features.
    Returns (has_conflict: bool, reason: str).

    Rule: conflict ONLY when BOTH sides have a value AND they differ.
    If one side is empty → cannot determine → no conflict (proceed to Vision AI).
    """
    reasons = []

    if comp.volume_ml and store.volume_ml:
        # Allow 10% tolerance for rounding (e.g. 29.6 vs 30.0 oz→ml)
        try:
            c_vol = float(comp.volume_ml)
            s_vol = float(store.volume_ml)
            if abs(c_vol - s_vol) > max(c_vol, s_vol) * 0.12:
                reasons.append(f"حجم مختلف: {comp.volume_ml}ml ≠ {store.volume_ml}ml")
        except ValueError:
            if comp.volume_ml != store.volume_ml:
                reasons.append(f"حجم مختلف: {comp.volume_ml} ≠ {store.volume_ml}")

    # Normalise concentration for comparison
    conc_groups = {
        "EDP": {"EDP", "Parfum"},
        "EDT": {"EDT"},
        "EDC": {"EDC"},
        "Extrait": {"Extrait", "Elixir"},
    }
    def _conc_group(c: str) -> str:
        for g, members in conc_groups.items():
            if c in members:
                return g
        return c

    if comp.concentration and store.concentration:
        cg = _conc_group(comp.concentration)
        sg = _conc_group(store.concentration)
        if cg != sg:
            reasons.append(
                f"تركيز مختلف: {comp.concentration} ≠ {store.concentration}"
            )

    if comp.gender and store.gender:
        if comp.gender != store.gender:
            reasons.append(f"جنس مختلف: {comp.gender} ≠ {store.gender}")

    return bool(reasons), " | ".join(reasons)


# ============================================================================
# SECTION B  —  VISION AI COURT  (Gemini image+text judge)
# ============================================================================

def _fetch_image_bytes(url: str, timeout: int = 8) -> Optional[bytes]:
    """Download image bytes from URL. Returns None on any failure."""
    if not url or not url.startswith("http"):
        return None
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        if resp.status_code == 200:
            content_type = resp.headers.get("content-type", "image/jpeg").lower()
            if "image" in content_type:
                return resp.content
    except Exception as exc:
        logger.warning(f"Image fetch failed for {url[:60]}: {exc}")
    return None


def _mime_from_url(url: str) -> str:
    """Guess MIME type from URL extension."""
    url_low = url.lower()
    if ".png"  in url_low: return "image/png"
    if ".webp" in url_low: return "image/webp"
    if ".gif"  in url_low: return "image/gif"
    return "image/jpeg"


def vision_verify(
    store_name: str,
    store_img:  str,
    comp_name:  str,
    comp_img:   str,
    api_key:    str,
) -> tuple[str, str]:
    """
    Ask Gemini Vision to judge whether two products are the same SKU.

    Returns
    -------
    (verdict: "MATCH"|"DIFFERENT", method: str)
        method describes what was sent to AI (images / text-only)
    """
    if not (api_key and GENAI_AVAILABLE):
        return RES_DIFFERENT, "no_api_key"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Build text-based prompt (works even without images)
        text_prompt = (
            "أنت قاضي مطابقة منتجات دقيق ومحايد.\n\n"
            f"🏪 منتجنا     : {store_name}\n"
            f"🔍 منتج المنافس: {comp_name}\n\n"
            "مهمتك: تحديد ما إذا كان هذان المنتجان نفس الـ SKU تماماً.\n"
            "دقق في: الماركة، اسم العطر، الحجم، نوع التركيز (EDP/EDT/Extrait/etc).\n"
            "لا تسمح للاختلافات في اللغة (عربي/إنجليزي) أن تؤثر على قرارك.\n\n"
            "أجب بكلمة واحدة فقط:\n"
            "MATCH      (إذا كانا نفس المنتج بنفس الـ SKU)\n"
            "DIFFERENT  (إذا كان هناك أي اختلاف حقيقي)"
        )

        # Try to include images
        store_bytes = _fetch_image_bytes(store_img)
        comp_bytes  = _fetch_image_bytes(comp_img)

        if store_bytes and comp_bytes:
            # Full vision prompt with both images
            vision_instruction = (
                "أنت قاضي مطابقة منتجات دقيق ومحايد.\n\n"
                f"🏪 منتجنا     : {store_name}\n"
                f"🔍 منتج المنافس: {comp_name}\n\n"
                "الصورة الأولى لمنتجنا. الصورة الثانية لمنتج المنافس.\n"
                "دقق في شكل الزجاجة، الماركة، الحجم المكتوب، "
                "ونوع التركيز على العلبة.\n\n"
                "أجب بكلمة واحدة فقط:\n"
                "MATCH      (إذا كانا نفس المنتج بنفس الـ SKU)\n"
                "DIFFERENT  (إذا كان هناك أي اختلاف حقيقي)"
            )
            parts = [
                vision_instruction,
                {"mime_type": _mime_from_url(store_img), "data": store_bytes},
                {"mime_type": _mime_from_url(comp_img),  "data": comp_bytes},
            ]
            raw = _gemini_retry(model, text_prompt, parts=parts)
            method = "vision+text"
        else:
            # Text-only fallback
            raw = _gemini_retry(model, text_prompt)
            method = "text_only"

        if raw:
            upper = raw.strip().upper()
            if "MATCH"     in upper: return RES_DUPLICATE, method
            if "DIFFERENT" in upper: return RES_DIFFERENT, method

    except Exception as exc:
        logger.error(f"vision_verify error: {exc}")

    # Safe default: treat as DIFFERENT (new opportunity is business-safer than missed dup)
    return RES_DIFFERENT, "error_fallback"


# ============================================================================
# SECTION C  —  AUTOMATED RESOLUTION ENGINE (replaces gray zone)
# ============================================================================

@dataclass
class ResolutionRecord:
    """Full record for a resolved ambiguous product."""
    verdict:         str = RES_DIFFERENT   # RES_DUPLICATE or RES_DIFFERENT
    match_reason:    str = ""              # human-readable reason
    resolution_path: str = ""             # feature_mismatch / vision_ai / fallback
    feature_details: str = ""             # e.g. "حجم مختلف: 50ml ≠ 100ml"
    comp_features:   ProductFeatures = field(default_factory=ProductFeatures)
    store_features:  ProductFeatures = field(default_factory=ProductFeatures)
    vision_method:   str = ""             # vision+text / text_only / skipped


def resolve_gray_product(
    comp_product:        dict,
    matched_store_name:  str,
    matched_store_img:   str,
    api_key:             str,
) -> ResolutionRecord:
    """
    Resolve a single ambiguous product through the 3-step ARE pipeline.

    Step 1: Feature Extraction + SKU Comparison
      → Any hard mismatch (volume/concentration/gender) = DIFFERENT immediately.
    Step 2: Vision AI Court
      → Send both product images + names to Gemini for binary verdict.
    Step 3: Safe Fallback
      → If API unavailable/failed = DIFFERENT (never blocks a new opportunity).
    """
    comp_name = str(comp_product.get("product_name", ""))
    comp_img  = str(comp_product.get("image_url", ""))

    # ── Step 1: Feature Extraction & Comparison ───────────────────────────────
    comp_feat  = extract_features(comp_name)
    store_feat = extract_features(matched_store_name)

    has_conflict, conflict_detail = features_conflict(comp_feat, store_feat)

    if has_conflict:
        return ResolutionRecord(
            verdict          = RES_DIFFERENT,
            match_reason     = REASON_SKU_MATCH,
            resolution_path  = "feature_mismatch",
            feature_details  = conflict_detail,
            comp_features    = comp_feat,
            store_features   = store_feat,
        )

    # ── Step 2: Vision AI Court ───────────────────────────────────────────────
    if api_key and GENAI_AVAILABLE:
        verdict, vision_method = vision_verify(
            store_name = matched_store_name,
            store_img  = matched_store_img,
            comp_name  = comp_name,
            comp_img   = comp_img,
            api_key    = api_key,
        )
        match_reason = (
            REASON_VISION_AI if verdict == RES_DUPLICATE
            else f"حسم بصري → مختلف ({vision_method})"
        )
        return ResolutionRecord(
            verdict          = verdict,
            match_reason     = match_reason,
            resolution_path  = "vision_ai",
            feature_details  = conflict_detail or "الخصائص النصية متطابقة أو غير محددة",
            comp_features    = comp_feat,
            store_features   = store_feat,
            vision_method    = vision_method,
        )

    # ── Step 3: Safe Fallback ─────────────────────────────────────────────────
    return ResolutionRecord(
        verdict          = RES_DIFFERENT,
        match_reason     = REASON_SAFE_DEFAULT,
        resolution_path  = "safe_fallback",
        feature_details  = "لا يوجد API — القرار الآمن: فرصة جديدة",
        comp_features    = comp_feat,
        store_features   = store_feat,
    )


# ============================================================================
# SECTION D  —  MASTER DEDUPLICATION + RESOLUTION ENGINE
# ============================================================================

def deduplicate_and_resolve(
    store_df:          pd.DataFrame,
    comp_products:     list[dict],
    high_threshold:    int = 90,
    low_threshold:     int = 50,
    api_key:           str = "",
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete deduplication + automated resolution pipeline.
    Returns ONLY (new_opportunities_df, duplicates_df) — ZERO gray zone.

    For each competitor product:
      Score ≥ high_threshold  →  duplicate (fuzzy)
      Score < low_threshold   →  new opportunity (clear)
      50 ≤ Score < 90         →  Automated Resolution Engine (ARE):
                                  features mismatch → new opp
                                  vision AI MATCH   → duplicate
                                  vision AI DIFFERENT / fallback → new opp

    Parameters
    ----------
    store_df           : Reference store products DataFrame.
    comp_products      : List of competitor product dicts.
    high_threshold     : Fuzzy score ≥ this → definite duplicate.
    low_threshold      : Fuzzy score < this → definite new opportunity.
    api_key            : Gemini API key for ARE Vision AI step.
    progress_callback  : fn(current, total, product_name, stage)
    """
    # Build lookup structures
    store_norms    = store_df["normalized_name"].tolist() if not store_df.empty else []
    store_names    = store_df["product_name"].tolist()    if not store_df.empty else []
    store_images   = store_df["image_url"].tolist()       if not store_df.empty and "image_url"  in store_df.columns else [""] * len(store_names)
    store_norm_set = set(store_norms)

    new_opps:   list[dict] = []
    duplicates: list[dict] = []
    total = len(comp_products)

    for i, product in enumerate(comp_products):
        comp_name = product.get("product_name", "")
        comp_norm = product.get("normalized_name", "")

        if progress_callback:
            progress_callback(i, total, comp_name, "comparing")

        if not comp_norm or len(comp_norm) < 3:
            continue

        # ── Layer 1: Exact normalized match ───────────────────────────────────
        if comp_norm in store_norm_set:
            sidx = store_norms.index(comp_norm)
            duplicates.append({
                **product,
                "match_score":           100,
                "matched_store_product": store_names[sidx],
                "matched_store_image":   store_images[sidx] if sidx < len(store_images) else "",
                "match_reason":          REASON_EXACT_NAME,
                "resolution_path":       "exact_name",
                "feature_details":       "تطابق نصي حرفي بعد التطبيع",
            })
            continue

        if not store_norms:
            new_opps.append({**product, "match_score": 0,
                             "matched_store_product": "", "match_reason": "new_clear",
                             "resolution_path": "no_store_data",
                             "feature_details": "لا توجد بيانات مرجعية للمتجر"})
            continue

        # ── Layer 2: Fuzzy match ───────────────────────────────────────────────
        best = process.extractOne(comp_norm, store_norms, scorer=fuzz.token_sort_ratio)
        if not best:
            new_opps.append({**product, "match_score": 0,
                             "matched_store_product": "", "match_reason": "new_clear",
                             "resolution_path": "no_fuzzy_match",
                             "feature_details": "لم يُعثر على أي تشابه نصي"})
            continue

        score         = best[1]
        best_norm     = best[0]
        sidx          = store_norms.index(best_norm)
        matched_name  = store_names[sidx]
        matched_img   = store_images[sidx] if sidx < len(store_images) else ""

        if score >= high_threshold:
            # Definite duplicate
            duplicates.append({
                **product,
                "match_score":           score,
                "matched_store_product": matched_name,
                "matched_store_image":   matched_img,
                "match_reason":          REASON_FUZZY_HIGH,
                "resolution_path":       "fuzzy_high",
                "feature_details":       f"تشابه نصي {score}%",
            })

        elif score < low_threshold:
            # Definite new opportunity — score too low to warrant ARE
            new_opps.append({
                **product,
                "match_score":           score,
                "matched_store_product": "",
                "match_reason":          "new_clear",
                "resolution_path":       "low_score_new",
                "feature_details":       f"تشابه منخفض جداً {score}% — فرصة جديدة مؤكدة",
            })

        else:
            # ── ARE: Automated Resolution Engine ─────────────────────────────
            if progress_callback:
                progress_callback(i, total, comp_name, "resolving")

            rec = resolve_gray_product(
                comp_product       = product,
                matched_store_name = matched_name,
                matched_store_img  = matched_img,
                api_key            = api_key,
            )

            base = {
                **product,
                "match_score":           score,
                "matched_store_product": matched_name,
                "matched_store_image":   matched_img,
                "match_reason":          rec.match_reason,
                "resolution_path":       rec.resolution_path,
                "feature_details":       rec.feature_details,
                "comp_volume":           rec.comp_features.volume_ml,
                "comp_concentration":    rec.comp_features.concentration,
                "comp_gender":           rec.comp_features.gender,
                "store_volume":          rec.store_features.volume_ml,
                "store_concentration":   rec.store_features.concentration,
                "store_gender":          rec.store_features.gender,
                "vision_method":         rec.vision_method,
            }

            if rec.verdict == RES_DUPLICATE:
                duplicates.append(base)
            else:
                new_opps.append(base)

    def _df(lst: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(lst) if lst else pd.DataFrame()

    return _df(new_opps), _df(duplicates)


# ============================================================================
# SECTION E  —  5-GATE PRODUCT CLASSIFIER  (unchanged from v4)
# ============================================================================

# ── Salla 40-column schema ────────────────────────────────────────────────────
SALLA_COLS: list[str] = [
    "النوع ", "أسم المنتج", "تصنيف المنتج", "صورة المنتج",
    "وصف صورة المنتج", "نوع المنتج", "سعر المنتج", "الوصف",
    "هل يتطلب شحن؟", "رمز المنتج sku", "سعر التكلفة",
    "السعر المخفض", "تاريخ بداية التخفيض", "تاريخ نهاية التخفيض",
    "اقصي كمية لكل عميل", "إخفاء خيار تحديد الكمية",
    "اضافة صورة عند الطلب", "الوزن", "وحدة الوزن", "الماركة",
    "العنوان الترويجي", "تثبيت المنتج", "الباركود",
    "السعرات الحرارية", "MPN", "GTIN", "خاضع للضريبة ؟",
    "سبب عدم الخضوع للضريبة",
    "[1] الاسم", "[1] النوع", "[1] القيمة", "[1] الصورة / اللون",
    "[2] الاسم", "[2] النوع", "[2] القيمة", "[2] الصورة / اللون",
    "[3] الاسم", "[3] النوع", "[3] القيمة", "[3] الصورة / اللون",
]
assert len(SALLA_COLS) == 40


# ── Brand Knowledge Base (abbreviated — full version in v4) ──────────────────
BRAND_KB: dict[str, tuple[str, float, str]] = {
    "ديور":(CAT_PERFUME,0.88,""),  "dior":(CAT_PERFUME,0.88,""),
    "شانيل":(CAT_PERFUME,0.85,""), "chanel":(CAT_PERFUME,0.85,""),
    "لانكوم":(CAT_PERFUME,0.80,""),"lancome":(CAT_PERFUME,0.80,""),
    "جيفنشي":(CAT_PERFUME,0.85,""),"givenchy":(CAT_PERFUME,0.85,""),
    "توم فورد":(CAT_PERFUME,0.90,""),"tom ford":(CAT_PERFUME,0.90,""),
    "لطافة":(CAT_PERFUME,0.95,""),  "lattafa":(CAT_PERFUME,0.95,""),
    "ارماني":(CAT_PERFUME,0.85,""), "armani":(CAT_PERFUME,0.85,""),
    "كريد":(CAT_PERFUME,0.97,""),   "creed":(CAT_PERFUME,0.97,""),
    "امواج":(CAT_PERFUME,0.97,""),  "amouage":(CAT_PERFUME,0.97,""),
    "باكو رابان":(CAT_PERFUME,0.90,""),
    "كارولينا هيريرا":(CAT_PERFUME,0.88,""),
    "فالنتينو":(CAT_PERFUME,0.88,""),
    "بربري":(CAT_PERFUME,0.87,""),  "burberry":(CAT_PERFUME,0.87,""),
    "ايف سان لوران":(CAT_PERFUME,0.85,""),
    "هوغو بوس":(CAT_PERFUME,0.90,""),"hugo boss":(CAT_PERFUME,0.90,""),
    "جان بول غوتييه":(CAT_PERFUME,0.90,""),
    "ديبتيك":(CAT_PERFUME,0.95,""), "diptyque":(CAT_PERFUME,0.95,""),
    "كيليان":(CAT_PERFUME,0.96,""), "kilian":(CAT_PERFUME,0.96,""),
    "فريدريك مال":(CAT_PERFUME,0.97,""),
    "جيرلان":(CAT_PERFUME,0.90,""), "guerlain":(CAT_PERFUME,0.90,""),
    "مون بلانك":(CAT_PERFUME,0.90,""),"montblanc":(CAT_PERFUME,0.90,""),
    "بينهاليغونز":(CAT_PERFUME,0.95,""),
    "ميزون كريفلي":(CAT_PERFUME,0.97,""),
    "كارتير":(CAT_PERFUME,0.88,""), "cartier":(CAT_PERFUME,0.88,""),
    "اكوا دي بارما":(CAT_PERFUME,0.92,""),
    "زيرجوف":(CAT_PERFUME,0.95,""),
    "افنان":(CAT_PERFUME,0.94,""),  "afnan":(CAT_PERFUME,0.94,""),
    "مارلي":(CAT_PERFUME,0.95,""),
    "ثامين":(CAT_PERFUME,0.95,""),  "ثمين":(CAT_PERFUME,0.95,""),
    "مانسيرا":(CAT_PERFUME,0.96,""),
    "عساف":(CAT_PERFUME,0.90,""),
    "كارتييه":(CAT_PERFUME,0.88,""),
    "فيرزاتشي":(CAT_PERFUME,0.87,""),"versace":(CAT_PERFUME,0.87,""),
    "دولتشي":(CAT_PERFUME,0.85,""), "dolce":(CAT_PERFUME,0.85,""),
    "لوريال":(CAT_BEAUTY,0.85,""),  "loreal":(CAT_BEAUTY,0.85,""),
    "بورجوا":(CAT_BEAUTY,0.90,""),  "bourjois":(CAT_BEAUTY,0.90,""),
    "نيوتري":(CAT_UNKNOWN,0.90,""), "nutri":(CAT_UNKNOWN,0.90,""),
    "جاميز":(CAT_UNKNOWN,0.90,""),
    "كيرستاسي":(CAT_BEAUTY,0.90,""),"kerastase":(CAT_BEAUTY,0.90,""),
}

# ── Hard patterns ─────────────────────────────────────────────────────────────
_PERF_HARD = [
    (r"\bedp\b","EDP"),(r"\bedt\b","EDT"),(r"\bedc\b","EDC"),
    (r"\bextrait\b","Extrait"),(r"eau\s+de\s+(parfum|toilette|cologne)","EauDe"),
    (r"او\s+دو\s+(برفيوم|بارفيوم|تواليت|كولون|بيرفيوم|بارفان)","ar:EauDe"),
    (r"اودي\s+(بارفيوم|برفيوم|تواليت)","ar:Ody"),
    (r"لو\s+دي\s+بارفان","ar:LParfum"),(r"\b(parfum|perfume|cologne|fragrance)\b","en:Perf"),
    (r"معطر\s+(شعر|الشعر)","HairMist→PERF"),
    (r"مجموعة\s+(هدايا\s+)?عطور","GiftSet→PERF"),
    (r"مجموعة\s+هدايا\s+عطرية","GiftSet2→PERF"),
    (r"طقم\s+(هدايا\s+)?عطر","GiftKit→PERF"),
    (r"عينة\s+(عطر|ماء\s+عطر)","Sample→PERF"),
    (r"تستر\s+(عطر|لعطر)","Tester→PERF"),
    (r"ماء\s+عطر","EauPerf"),(r"بخور","Bakhoor→PERF"),
    (r"دهن\s+عطر","OilPerf"),
]
_BEAU_HARD = [
    (r"(استشوار|مجفف\s+شعر|مكواة\s+شعر)","Device→BEAU"),
    (r"(مسكارا|ريميل|كحل|مكياج)","Makeup→BEAU"),
    (r"(كونسيلر|كريم\s+اساس|فاونديشن)","Foundation→BEAU"),
    (r"(ظل\s+عيون|بلاشر|هايلايتر)","Eyeshadow→BEAU"),
    (r"(شامبو|بلسم\s+شعر|غسول\s+(شعر|وجه))","Haircare→BEAU"),
    (r"مزيل\s+عرق","Deodorant→BEAU"),
    (r"(ديودورانت|antiperspirant)","Deo_en→BEAU"),
    (r"(سيروم|تونر|ماسك\s+وجه)","Skincare→BEAU"),
    (r"شمعة","Candle→BEAU"),
    (r"(مرطب\s+(جسم|يدين)|لوشن|lotion)","Lotion→BEAU"),
    (r"زيت\s+(جسم|الجسم)","BodyOil→BEAU"),
    (r"جل\s+(استحمام|الاستحمام)","ShowerGel→BEAU"),
    (r"(بودرة\s+جسم|سكراب)","BodyCare→BEAU"),
]
_UNK_HARD = [
    (r"مكمل\s+غذائي","FoodSup→UNK"),
    (r"(فيتامين|vitamin)","Vitamin→UNK"),
    (r"جاميز\s+مكمل","Gummies→UNK"),
    (r"حلوى\s+(جاميز|صحية)","Candy→UNK"),
]
_PERF_SCORED = [
    (r"عطر\b",2.5),(r"\bتستر\b",2.0),(r"\bعينة\b",1.5),
    (r"\b(عود|oud)\b",1.8),(r"\b(مسك|musk)\b",1.5),
    (r"\b\d+\s*مل\b",1.0),(r"\bml\b",1.0),(r"\bسبراي\b",1.0),
    (r"\b(نيش|niche)\b",1.5),(r"بارفيوم\b",2.0),(r"برفيوم\b",2.0),
]
_BEAU_SCORED = [
    (r"(كريم|cream)\b",2.0),(r"(لوشن|lotion)\b",2.0),
    (r"(سيروم|serum)\b",2.5),(r"مزيل",2.5),
    (r"(مكياج|makeup)",3.0),(r"(بشرة|skin)\b",2.0),
    (r"(مقشر|scrub)\b",2.5),(r"شمعة",3.0),(r"(مرطب|moisturizer)\b",2.0),
]
_UNK_SCORED = [
    (r"(مكمل|supplement)\b",4.0),(r"(فيتامين|vitamin)\b",4.0),
    (r"(قطعة|capsule|حبة)",2.0),(r"(حلوى|candy|gummy)",3.0),
]
_CAT_RULES = [
    (r"عطور",CAT_PERFUME,0.95),(r"perfume",CAT_PERFUME,0.95),
    (r"(نسائي|رجالي|نيش)",CAT_PERFUME,0.80),
    (r"(مكياج|تجميل)",CAT_BEAUTY,0.90),(r"(عناية|skincare)",CAT_BEAUTY,0.90),
    (r"(مكملات|supplement)",CAT_UNKNOWN,0.90),
]


@dataclass
class ClassificationResult:
    category:      str   = CAT_UNKNOWN
    confidence:    float = 0.0
    gate_used:     str   = GATE_FALLBACK
    signals_fired: list  = field(default_factory=list)
    reasoning:     str   = ""
    ai_used:       bool  = False

    def to_dict(self) -> dict:
        return {
            "classified_as":  self.category,
            "confidence_pct": f"{self.confidence*100:.1f}%",
            "gate_used":      self.gate_used,
            "signals_fired":  " | ".join(self.signals_fired),
            "reasoning":      self.reasoning,
            "ai_used":        self.ai_used,
        }


def _norm_ar(t: str) -> str:
    if not isinstance(t, str): return ""
    t = t.lower().strip()
    t = re.sub(r"[أإآ]","ا",t); t = re.sub(r"ى","ي",t)
    return re.sub(r"\s+"," ",t)

def _gate1(txt: str) -> Optional[ClassificationResult]:
    for pat, lbl in _PERF_HARD:
        if re.search(pat, txt, re.IGNORECASE|re.UNICODE):
            return ClassificationResult(CAT_PERFUME,0.97,GATE_HARD_RULES,[lbl],lbl)
    sigs=[]
    for pat, lbl in _BEAU_HARD:
        if re.search(pat, txt, re.IGNORECASE|re.UNICODE):
            sigs.append(lbl)
    if sigs:
        return ClassificationResult(CAT_BEAUTY,0.97,GATE_HARD_RULES,sigs,sigs[0])
    for pat, lbl in _UNK_HARD:
        if re.search(pat, txt, re.IGNORECASE|re.UNICODE):
            return ClassificationResult(CAT_UNKNOWN,0.95,GATE_HARD_RULES,[lbl],lbl)
    return None

def _gate2(cat_path: str) -> Optional[ClassificationResult]:
    if not cat_path: return None
    txt = _norm_ar(cat_path)
    for pat, cat, conf in _CAT_RULES:
        if re.search(pat, txt, re.IGNORECASE|re.UNICODE):
            if conf >= 0.85:
                return ClassificationResult(cat,conf,GATE_CATEGORY,[pat],f"cat:{pat}")
    return None

def _gate3(txt: str) -> Optional[ClassificationResult]:
    ps=bs=us=0.0; fired=[]
    for pat,w in _PERF_SCORED:
        if re.search(pat,txt,re.IGNORECASE|re.UNICODE): ps+=w; fired.append(f"p+{w}")
    for pat,w in _BEAU_SCORED:
        if re.search(pat,txt,re.IGNORECASE|re.UNICODE): bs+=w; fired.append(f"b+{w}")
    for pat,w in _UNK_SCORED:
        if re.search(pat,txt,re.IGNORECASE|re.UNICODE): us+=w; fired.append(f"u+{w}")
    tot=ps+bs+us
    if tot<1.0: return None
    scores={CAT_PERFUME:ps,CAT_BEAUTY:bs,CAT_UNKNOWN:us}
    win=max(scores,key=scores.get); conf=scores[win]/tot
    if conf>=0.75:
        return ClassificationResult(win,min(conf,0.95),GATE_SCORING,fired[:6],
                                    f"P:{ps:.1f} B:{bs:.1f} U:{us:.1f}")
    return None

def _gate4(txt: str) -> Optional[ClassificationResult]:
    best=None
    for slug,(cat,conf,_) in BRAND_KB.items():
        if slug in txt and (best is None or conf>best[1]):
            best=(slug,conf,cat)
    if best and best[1]>=0.70:
        return ClassificationResult(best[2],best[1],GATE_BRAND_KB,
                                    [f"brand:{best[0]}"],f"brand:{best[0]}")
    return None

def _gate5_ai(name,cat,brand,api_key) -> ClassificationResult:
    fb=ClassificationResult(CAT_UNKNOWN,0.0,GATE_FALLBACK,["NoSignals"],"fallback→unknown")
    if not (api_key and GENAI_AVAILABLE): return fb
    try:
        genai.configure(api_key=api_key)
        model=genai.GenerativeModel("gemini-1.5-flash")
        prompt=(f"صنّف المنتج التالي: {name}\n"
                f"التصنيف: {cat or 'غير محدد'}\nالماركة: {brand or 'غير محددة'}\n"
                "الفئات: PERFUME / BEAUTY / UNKNOWN\n"
                "أجب بـ JSON فقط: {{\"category\":\"PERFUME|BEAUTY|UNKNOWN\","
                "\"confidence\":0.0-1.0,\"reasoning\":\"سبب\"}} ")
        raw=_gemini_retry(model,prompt)
        if not raw: return fb
        raw=re.sub(r"^```(?:json)?\s*","",raw,flags=re.MULTILINE)
        raw=re.sub(r"```\s*$","",raw,flags=re.MULTILINE).strip()
        d=json.loads(raw)
        mp={"PERFUME":CAT_PERFUME,"BEAUTY":CAT_BEAUTY,"UNKNOWN":CAT_UNKNOWN}
        c=mp.get(d.get("category","").upper(),CAT_UNKNOWN)
        return ClassificationResult(c,float(d.get("confidence",0.5)),GATE_AI_ORACLE,
                                    ["AI:Gemini"],d.get("reasoning","AI"),True)
    except Exception as exc:
        logger.error(f"gate5: {exc}")
    return fb


def classify_product_5gate(name:str, category:str="", brand:str="", api_key:str="") -> ClassificationResult:
    """5-Gate product type classifier."""
    if not isinstance(name,str) or not name.strip():
        return ClassificationResult(CAT_UNKNOWN,0.0,GATE_FALLBACK,[],"empty name")
    full=_norm_ar(f"{name} {category} {brand}")
    for fn,thr in [(_gate1,0.85),(_gate2,0.85),(_gate3,0.75),(_gate4,0.70)]:
        try:
            arg=full if fn not in (_gate2,) else category
            r=fn(arg)
            if r and r.confidence>=thr: return r
        except Exception as e: logger.error(f"{fn.__name__}: {e}")
    if api_key and GENAI_AVAILABLE:
        try: return _gate5_ai(name,category,brand,api_key)
        except Exception as e: logger.error(f"gate5: {e}")
    return ClassificationResult(CAT_UNKNOWN,0.0,GATE_FALLBACK,[],"all gates below threshold")


def classify_product(name:str, category:str="") -> str:
    """Backward-compatible wrapper."""
    return classify_product_5gate(name,category).category


# ============================================================================
# SECTION F  —  TEXT NORMALISATION  &  CSV LOADER
# ============================================================================

_AR_SW=["عطر","او دي","او","دي","بارفيوم","برفيوم","تواليت","تستر","بديل","كولونيا"]
_EN_SW=["eau","de","parfum","toilette","cologne","edp","edt","edc","ml","tester","spray","the"]

def normalize_name(name: str) -> str:
    if not isinstance(name,str) or not name.strip(): return ""
    t=name.lower().strip()
    t=re.sub(r"[أإآ]","ا",t); t=re.sub(r"ى","ي",t); t=re.sub(r"[ةه]","h",t)
    for w in _AR_SW: t=re.sub(r"\b"+re.escape(w)+r"\b"," ",t)
    for w in _EN_SW: t=re.sub(r"\b"+re.escape(w)+r"\b"," ",t)
    t=re.sub(r"\d+\.?\d*\s*(?:ml|مل|g|gr|oz)","",t,flags=re.IGNORECASE)
    t=re.sub(r"[^\w\s]"," ",t); t=re.sub(r"\s+"," ",t).strip()
    return t


def _read_csv_safe(f, **kw) -> pd.DataFrame:
    raw=(f.read() if hasattr(f,"read") else open(f,"rb").read())
    if isinstance(raw,str): raw=raw.encode()
    for enc in ("utf-8-sig","utf-8","cp1256","latin-1"):
        try: return pd.read_csv(io.BytesIO(raw),encoding=enc,**kw)
        except: pass
    raise ValueError("Cannot parse CSV")


def load_store_products(files: list) -> pd.DataFrame:
    frames=[]
    for f in files:
        try:
            raw=_read_csv_safe(f,header=None,low_memory=False,dtype=str)
            hrow=next((i for i,r in raw.iterrows() if any("أسم المنتج" in str(v) for v in r)),None)
            if hrow is None: continue
            raw.columns=[str(c).strip() for c in raw.iloc[hrow]]
            data=raw.iloc[hrow+1:].reset_index(drop=True)
            nc=next((c for c in data.columns if "أسم المنتج" in c),None)
            if not nc: continue
            fr=pd.DataFrame()
            fr["product_name"]=data[nc].fillna("").astype(str)
            bc=next((c for c in data.columns if "الماركة" in c),None)
            cc=next((c for c in data.columns if "تصنيف" in c),None)
            ic=next((c for c in data.columns if "صورة المنتج" in c),None)
            fr["brand"]   =data[bc].fillna("").astype(str) if bc else ""
            fr["category"]=data[cc].fillna("").astype(str) if cc else ""
            fr["image_url"]=(data[ic].apply(lambda x: str(x).split(",")[0].strip() if pd.notna(x) else "")
                             if ic else "")
            fr["normalized_name"]=fr["product_name"].apply(normalize_name)
            frames.append(fr[fr["product_name"].str.strip()!=""])
        except Exception as e: logger.error(f"load_store: {e}")
    if not frames:
        return pd.DataFrame(columns=["product_name","brand","category","image_url","normalized_name"])
    return pd.concat(frames,ignore_index=True).drop_duplicates(subset=["product_name"]).reset_index(drop=True)


def load_competitor_products(f) -> pd.DataFrame:
    try:
        df=_read_csv_safe(f,low_memory=False,dtype=str)
        df.columns=[str(c).strip() for c in df.columns]
        name_h=["name","اسم","productcard","styles_product","title","عنوان"]
        img_h =["src","image","img","صورة","photo"]
        price_h=["price","سعر","text-sm","cost"]
        nc=next((c for c in df.columns if any(h in c.lower() for h in name_h)),df.columns[2] if len(df.columns)>=3 else None)
        ic=next((c for c in df.columns if any(h in c.lower() for h in img_h)), df.columns[1] if len(df.columns)>=2 else None)
        pc=next((c for c in df.columns if any(h in c.lower() for h in price_h)),df.columns[3] if len(df.columns)>=4 else None)
        r=pd.DataFrame()
        r["product_name"]=df[nc].fillna("").astype(str) if nc else ""
        r["image_url"]   =df[ic].fillna("").astype(str).str.split(",").str[0].str.strip() if ic else ""
        r["price"]       =df[pc].fillna("").astype(str) if pc else ""
        r["normalized_name"]=r["product_name"].apply(normalize_name)
        r["product_type"]   =r["product_name"].apply(classify_product)
        r["source_file"]    =getattr(f,"name","unknown")
        return r[r["product_name"].str.strip().str.len()>1].reset_index(drop=True)
    except Exception as e:
        logger.error(f"load_comp: {e}")
        return pd.DataFrame(columns=["product_name","image_url","price","normalized_name","product_type","source_file"])


def load_brands_list(f) -> list[str]:
    if f is None: return []
    try:
        df=_read_csv_safe(f,dtype=str)
        df.columns=[str(c).strip() for c in df.columns]
        col=next((c for c in df.columns if "اسم" in c),None)
        if col: return df[col].dropna().astype(str).tolist()
    except: pass
    return []


# ============================================================================
# SECTION G  —  ENRICHMENT
# ============================================================================

def enrich_product_with_gemini(name:str, img:str, ptype:str, api_key:str, expert:str="") -> dict:
    default={"brand":"","gender":"للجنسين",
             "salla_category":"العطور > عطور للجنسين" if ptype==CAT_PERFUME else "مكياج وعناية",
             "description":f"<p>{name}</p>","sku_suggestion":"","enriched":False}
    if not (api_key and GENAI_AVAILABLE and name.strip()): return default
    try:
        genai.configure(api_key=api_key)
        model=genai.GenerativeModel("gemini-1.5-flash")
        lbl="عطر" if ptype==CAT_PERFUME else "منتج تجميل"
        prompt=(f"خبير {lbl} لمتجر مهووس.\nالمنتج: {name}\n"
                f"{f'رابط صورة: {img}' if img else ''}\n"
                "أعد JSON فقط:\n"
                "{{\"brand_ar\":\"\",\"brand_en\":\"\",\"gender\":\"رجالي|نسائي|للجنسين|نيش\","
                "\"salla_category\":\"\",\"sku_suggestion\":\"\","
                "\"description_html\":\"وصف HTML 200-300 كلمة\"}}")
        raw=_gemini_retry(model,prompt)
        if not raw: return default
        raw=re.sub(r"^```(?:json)?\s*","",raw,flags=re.MULTILINE)
        raw=re.sub(r"```\s*$","",raw,flags=re.MULTILINE).strip()
        d=json.loads(raw)
        brand=f"{d.get('brand_ar','').strip()} | {d.get('brand_en','').strip()}".strip(" |")
        return {"brand":brand,"gender":d.get("gender","للجنسين"),
                "salla_category":d.get("salla_category",default["salla_category"]),
                "description":d.get("description_html",default["description"]),
                "sku_suggestion":d.get("sku_suggestion",""),"enriched":True}
    except Exception as e: logger.error(f"enrich: {e}")
    return default


def fetch_fallback_image(name:str, key:str="", cx:str="") -> str:
    if not (key and cx and name.strip()): return ""
    try:
        r=requests.get("https://www.googleapis.com/customsearch/v1",
            params={"key":key,"cx":cx,"q":f"{name} official","searchType":"image","num":1},timeout=10)
        if r.status_code==200:
            items=r.json().get("items",[])
            if items: return items[0].get("link","")
    except Exception as e: logger.warning(f"img_fetch: {e}")
    return ""


# ============================================================================
# SECTION H  —  EXPORT ENGINE
# ============================================================================

def _salla_row(p: dict) -> dict:
    return {
        "النوع ":"منتج","أسم المنتج":p.get("product_name",""),
        "تصنيف المنتج":p.get("salla_category",""),"صورة المنتج":p.get("image_url",""),
        "وصف صورة المنتج":p.get("product_name",""),"نوع المنتج":"منتج جاهز",
        "سعر المنتج":p.get("price",""),"الوصف":p.get("description",""),
        "هل يتطلب شحن؟":"نعم","رمز المنتج sku":p.get("sku_suggestion",""),
        "سعر التكلفة":"","السعر المخفض":"","تاريخ بداية التخفيض":"",
        "تاريخ نهاية التخفيض":"","اقصي كمية لكل عميل":"0",
        "إخفاء خيار تحديد الكمية":"لا","اضافة صورة عند الطلب":"",
        "الوزن":"0.5","وحدة الوزن":"kg","الماركة":p.get("brand",""),
        "العنوان الترويجي":"","تثبيت المنتج":"لا","الباركود":"",
        "السعرات الحرارية":"","MPN":"","GTIN":"","خاضع للضريبة ؟":"نعم",
        "سبب عدم الخضوع للضريبة":"",
        "[1] الاسم":"","[1] النوع":"","[1] القيمة":"","[1] الصورة / اللون":"",
        "[2] الاسم":"","[2] النوع":"","[2] القيمة":"","[2] الصورة / اللون":"",
        "[3] الاسم":"","[3] النوع":"","[3] القيمة":"","[3] الصورة / اللون":"",
    }


def export_to_salla_csv(products: list[dict]) -> bytes:
    """Salla-compatible CSV: Row1=magic header, Row2=cols, Row3+=data. UTF-8-BOM."""
    df=pd.DataFrame([_salla_row(p) for p in (products or [])],columns=SALLA_COLS)
    buf=io.StringIO()
    buf.write("بيانات المنتج"+(","*(len(SALLA_COLS)-1))+"\n")
    df.to_csv(buf,index=False,encoding="utf-8")
    return ("\ufeff"+buf.getvalue()).encode("utf-8")


def export_missing_brands_csv(products: list[dict], existing: list[str]) -> bytes:
    COLS=["اسم الماركة","وصف مختصر عن الماركة","صورة شعار الماركة",
          "(إختياري) صورة البانر","(Page Title) عنوان صفحة العلامة التجارية",
          "(SEO Page URL) رابط صفحة العلامة التجارية",
          "(Page Description) وصف صفحة العلامة التجارية"]
    el={b.lower().strip() for b in existing if b}
    nb={}
    for p in products:
        br=str(p.get("brand","")).strip()
        if not br: continue
        if not any(fuzz.partial_ratio(br.lower(),e)>85 for e in el) and br not in nb:
            slug=re.sub(r"[^\w\s-]","",br.lower()).strip()
            slug=re.sub(r"[\s|]+" ,"-",slug).strip("-")
            nb[br]={"اسم الماركة":br,"وصف مختصر عن الماركة":"","صورة شعار الماركة":"",
                    "(إختياري) صورة البانر":"",
                    "(Page Title) عنوان صفحة العلامة التجارية":f"{br} | مهووس للعطور",
                    "(SEO Page URL) رابط صفحة العلامة التجارية":slug,
                    "(Page Description) وصف صفحة العلامة التجارية":f"تسوق {br} الأصلية لدى مهووس."}
    df=pd.DataFrame(list(nb.values()),columns=COLS) if nb else pd.DataFrame(columns=COLS)
    buf=io.BytesIO(); df.to_csv(buf,index=False,encoding="utf-8-sig")
    return buf.getvalue()


def export_audit_trail_csv(products: list[dict]) -> bytes:
    COLS=["product_name","classified_as","confidence_pct","gate_used",
          "signals_fired","reasoning","ai_used","source_file","price","image_url",
          "match_reason","resolution_path","feature_details"]
    rows=[{c:p.get(c,"") for c in COLS} for p in products]
    df=pd.DataFrame(rows,columns=COLS) if rows else pd.DataFrame(columns=COLS)
    buf=io.BytesIO(); df.to_csv(buf,index=False,encoding="utf-8-sig")
    return buf.getvalue()
