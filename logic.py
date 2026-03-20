"""
logic.py  ─  Mahwous Opportunity Engine
========================================
Core data-processing pipeline for the Mahwous perfume-store
sourcing and deduplication tool.

Author  : Senior AI & Data Engineer (for متجر مهووس)
Version : 2.0.0  –  Production-Ready
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from typing import Callable, Optional

import pandas as pd
import requests
from thefuzz import fuzz, process

# ── Optional Gemini import (graceful fallback if key absent) ──────────────────
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Exact 40 Salla column names (matches منتج_جديد.csv template)
SALLA_COLS: list[str] = [
    "النوع ",
    "أسم المنتج",
    "تصنيف المنتج",
    "صورة المنتج",
    "وصف صورة المنتج",
    "نوع المنتج",
    "سعر المنتج",
    "الوصف",
    "هل يتطلب شحن؟",
    "رمز المنتج sku",
    "سعر التكلفة",
    "السعر المخفض",
    "تاريخ بداية التخفيض",
    "تاريخ نهاية التخفيض",
    "اقصي كمية لكل عميل",
    "إخفاء خيار تحديد الكمية",
    "اضافة صورة عند الطلب",
    "الوزن",
    "وحدة الوزن",
    "الماركة",
    "العنوان الترويجي",
    "تثبيت المنتج",
    "الباركود",
    "السعرات الحرارية",
    "MPN",
    "GTIN",
    "خاضع للضريبة ؟",
    "سبب عدم الخضوع للضريبة",
    "[1] الاسم",
    "[1] النوع",
    "[1] القيمة",
    "[1] الصورة / اللون",
    "[2] الاسم",
    "[2] النوع",
    "[2] القيمة",
    "[2] الصورة / اللون",
    "[3] الاسم",
    "[3] النوع",
    "[3] القيمة",
    "[3] الصورة / اللون",
]
assert len(SALLA_COLS) == 40, "SALLA_COLS must have exactly 40 entries"

# Keywords for auto-classification
_PERFUME_KW = [
    "عطر", "بخاخ", "برفيوم", "بارفيوم", "كولونيا", "تواليت", "ماء العطر",
    "eau de parfum", "eau de toilette", "edp", "edt", "edc", "parfum",
    "fragrance", "cologne", "scent", "perfume", "oud", "عود", "بخور",
]
_BEAUTY_KW = [
    "كريم", "مكياج", "ماكياج", "روج", "ظل عيون", "بودرة", "مسكارا",
    "اساس", "كونسيلر", "تونر", "سيروم", "شامبو", "بلسم", "مرطب",
    "واقي شمس", "حمام زيت", "استشوار", "مجفف", "مكواة", "فرشاة",
    "مقشر", "غسول", "لوشن", "زيت شعر", "ماسك", "بي بي كريم",
    "cream", "makeup", "mascara", "foundation", "concealer", "toner",
    "serum", "shampoo", "conditioner", "moisturizer", "sunscreen",
    "hair dryer", "straightener", "brush", "cleanser", "lotion",
]

# Arabic stopwords to remove before comparison
_AR_STOPWORDS = [
    "عطر", "او دي", "او", "دي", "بارفيوم", "برفيوم", "تواليت",
    "تستر", "بديل", "كولونيا", "رائحة", "بخاخ", "اوه", "ذا",
]
_EN_STOPWORDS = [
    "eau", "de", "parfum", "toilette", "cologne", "edp", "edt", "edc",
    "ml", "tester", "dupe", "perfume", "fragrance", "spray", "the",
]

# Gemini rate-limit guard
_LAST_API_CALL: float = 0.0
_MIN_INTERVAL: float = 1.8   # seconds between consecutive Gemini calls


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TEXT NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_name(name: str) -> str:
    """
    Return a clean, comparable 'root' string for a product name.

    Steps
    -----
    1. Lower-case + strip.
    2. Normalize Arabic variant letters (أ إ آ → ا, ى → ي, ة/ه mutual).
    3. Remove Arabic & English stopwords.
    4. Strip volume tokens (100ml, 50 مل, …).
    5. Remove non-word characters and collapse whitespace.
    """
    if not isinstance(name, str) or not name.strip():
        return ""

    text = name.lower().strip()

    # Arabic letter normalization
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    # Normalize ة and ه to the same token for comparison
    text = re.sub(r"[ةه]", "h", text)

    # Remove Arabic stopwords (whole-word)
    for w in _AR_STOPWORDS:
        text = re.sub(r"\b" + re.escape(w) + r"\b", " ", text)

    # Remove English stopwords (whole-word)
    for w in _EN_STOPWORDS:
        text = re.sub(r"\b" + re.escape(w) + r"\b", " ", text)

    # Remove volume: 100ml / 50 مل / 1.7oz / etc.
    text = re.sub(r"\d+\.?\d*\s*(?:ml|مل|g|gr|غ|oz|fl\.?\s*oz)", " ", text)

    # Remove special characters (keep letters, digits, spaces)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PRODUCT-TYPE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def classify_product(name: str, category: str = "") -> str:
    """
    Classify a product as 'perfume', 'beauty', or 'other'.

    Parameters
    ----------
    name     : Product display name.
    category : Optional category string for additional context.

    Returns
    -------
    'perfume' | 'beauty' | 'other'
    """
    combined = f"{name} {category}".lower()

    for kw in _PERFUME_KW:
        if kw in combined:
            return "perfume"

    for kw in _BEAUTY_KW:
        if kw in combined:
            return "beauty"

    # EDP / EDT pattern without full keyword
    if re.search(r"\b(edp|edt|edc)\b", combined):
        return "perfume"

    return "other"   # treated as beauty/general


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SAFE CSV READER  (handles UTF-8-sig, UTF-8, CP1256, …)
# ══════════════════════════════════════════════════════════════════════════════

def _read_csv_safe(file_obj, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file-like object trying multiple encodings in priority order.
    Raises ValueError only if all attempts fail.
    """
    if hasattr(file_obj, "read"):
        raw_bytes = file_obj.read()
        if isinstance(raw_bytes, str):
            raw_bytes = raw_bytes.encode("utf-8")
    else:
        with open(file_obj, "rb") as fh:
            raw_bytes = fh.read()

    for enc in ("utf-8-sig", "utf-8", "cp1256", "iso-8859-6", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, **kwargs)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
        except Exception as exc:
            logger.warning(f"CSV parse error with {enc}: {exc}")

    raise ValueError("Could not parse CSV with any supported encoding.")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FILE LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_store_products(store_files: list) -> pd.DataFrame:
    """
    Load one or more Salla-format CSV files (متجر مهووس parts 1-4).

    Auto-detects the header row containing 'أسم المنتج' and skips the
    magic 'بيانات المنتج' first row.

    Returns
    -------
    DataFrame with columns:
        product_name, brand, category, image_url, normalized_name
    """
    frames: list[pd.DataFrame] = []

    for f in store_files:
        try:
            raw = _read_csv_safe(f, header=None, low_memory=False, dtype=str)

            # Find the header row
            header_row_idx = None
            for idx, row in raw.iterrows():
                if any("أسم المنتج" in str(v) for v in row.values):
                    header_row_idx = idx
                    break

            if header_row_idx is None:
                logger.warning(f"Could not find header in store file: {getattr(f, 'name', f)}")
                continue

            raw.columns = [str(c).strip() for c in raw.iloc[header_row_idx].values]
            data = raw.iloc[header_row_idx + 1:].reset_index(drop=True)

            # Map columns flexibly
            def _col(keyword: str) -> Optional[str]:
                return next((c for c in data.columns if keyword in c), None)

            name_col  = _col("أسم المنتج")
            brand_col = _col("الماركة")
            cat_col   = _col("تصنيف المنتج")
            img_col   = _col("صورة المنتج")

            if not name_col:
                continue

            frame = pd.DataFrame()
            frame["product_name"]   = data[name_col].fillna("").astype(str)
            frame["brand"]          = data[brand_col].fillna("").astype(str) if brand_col else ""
            frame["category"]       = data[cat_col].fillna("").astype(str)   if cat_col  else ""

            # Some rows have multiple images comma-separated → keep first
            if img_col:
                frame["image_url"] = data[img_col].apply(
                    lambda x: str(x).split(",")[0].strip() if pd.notna(x) else ""
                )
            else:
                frame["image_url"] = ""

            frame["normalized_name"] = frame["product_name"].apply(normalize_name)
            frames.append(frame[frame["product_name"].str.strip() != ""])

        except Exception as exc:
            logger.error(f"Error loading store file '{getattr(f, 'name', f)}': {exc}")

    if not frames:
        return pd.DataFrame(
            columns=["product_name", "brand", "category", "image_url", "normalized_name"]
        )

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["product_name"])
        .reset_index(drop=True)
    )


def load_competitor_products(comp_file) -> pd.DataFrame:
    """
    Load a competitor CSV file (scraped from mahally.com or any other source).

    Auto-detects name / image / price columns via column-name heuristics,
    then falls back to positional detection.

    Returns
    -------
    DataFrame with columns:
        product_name, image_url, price, normalized_name, product_type, source_file
    """
    try:
        df = _read_csv_safe(comp_file, low_memory=False, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]

        # ── Column detection ─────────────────────────────────────────────────
        name_col  = None
        img_col   = None
        price_col = None

        name_hints  = ["name", "اسم", "productcard", "styles_product", "title", "عنوان"]
        img_hints   = ["src", "image", "img", "صورة", "photo"]
        price_hints = ["price", "سعر", "text-sm", "cost"]

        for col in df.columns:
            cl = col.lower()
            if name_col is None  and any(h in cl for h in name_hints):
                name_col  = col
            if img_col is None   and any(h in cl for h in img_hints):
                img_col   = col
            if price_col is None and any(h in cl for h in price_hints):
                price_col = col

        # Positional fallback
        if name_col  is None and len(df.columns) >= 3:
            name_col  = df.columns[2]
        if img_col   is None and len(df.columns) >= 2:
            img_col   = df.columns[1]
        if price_col is None and len(df.columns) >= 4:
            price_col = df.columns[3]

        result = pd.DataFrame()
        result["product_name"] = df[name_col].fillna("").astype(str) if name_col else ""
        result["image_url"]    = df[img_col].fillna("").astype(str)  if img_col  else ""
        result["price"]        = df[price_col].fillna("").astype(str) if price_col else ""

        # Clean up Salla CDN image URLs (strip extra transforms if present)
        result["image_url"] = result["image_url"].str.split(",").str[0].str.strip()

        result["normalized_name"] = result["product_name"].apply(normalize_name)
        result["product_type"]    = result["product_name"].apply(classify_product)
        result["source_file"]     = getattr(comp_file, "name", "unknown")

        return (
            result[result["product_name"].str.strip().str.len() > 1]
            .reset_index(drop=True)
        )

    except Exception as exc:
        logger.error(
            f"Error loading competitor file '{getattr(comp_file, 'name', '')}': {exc}"
        )
        return pd.DataFrame(
            columns=["product_name", "image_url", "price",
                     "normalized_name", "product_type", "source_file"]
        )


def load_brands_list(brands_file) -> list[str]:
    """
    Load existing brand names from a Salla brands CSV.

    Returns
    -------
    List of brand name strings.
    """
    if brands_file is None:
        return []
    try:
        df = _read_csv_safe(brands_file, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]
        col = next((c for c in df.columns if "اسم" in c), None)
        if col:
            return df[col].dropna().astype(str).tolist()
    except Exception as exc:
        logger.error(f"Could not load brands file: {exc}")
    return []


# ══════════════════════════════════════════════════════════════════════════════
# 5.  DEDUPLICATION ENGINE  (3-layer fuzzy filter)
# ══════════════════════════════════════════════════════════════════════════════

def deduplicate_products(
    store_df: pd.DataFrame,
    comp_products: list[dict],
    high_threshold: int = 90,
    low_threshold:  int = 50,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Three-layer deduplication filter.

    Layer 1  – Exact normalized match      → duplicate
    Layer 2  – Fuzzy token_sort_ratio
                 ≥ high_threshold           → duplicate
                 between thresholds         → gray zone  (human / AI review)
                 < low_threshold            → new opportunity ✅
    Layer 3  – (Vision AI) handled in app.py for gray zone on-demand

    Parameters
    ----------
    store_df          : Reference store products with 'normalized_name'.
    comp_products     : List of competitor product dicts.
    high_threshold    : Fuzzy score above which product is considered duplicate.
    low_threshold     : Fuzzy score below which product is a confirmed new opp.
    progress_callback : Optional fn(current_idx, total, product_name).

    Returns
    -------
    (new_opportunities_df, gray_zone_df, duplicates_df)
    """
    store_norms     = store_df["normalized_name"].tolist()
    store_names_raw = store_df["product_name"].tolist()
    store_norm_set  = set(store_norms)   # O(1) exact-match lookup

    new_opps:   list[dict] = []
    gray_zone:  list[dict] = []
    duplicates: list[dict] = []

    total = len(comp_products)

    for i, product in enumerate(comp_products):
        if progress_callback:
            progress_callback(i, total, product.get("product_name", ""))

        comp_norm = product.get("normalized_name", "")
        comp_name = product.get("product_name", "")

        if not comp_norm or len(comp_norm) < 3:
            continue   # skip garbage rows

        # ── Layer 1: exact match ──────────────────────────────────────────
        if comp_norm in store_norm_set:
            duplicates.append({
                **product,
                "match_score": 100,
                "matched_store_product": "",
                "match_reason": "exact",
            })
            continue

        # ── Layer 2: fuzzy match ──────────────────────────────────────────
        if store_norms:
            best = process.extractOne(
                comp_norm, store_norms, scorer=fuzz.token_sort_ratio
            )
            if best:
                score        = best[1]
                matched_idx  = store_norms.index(best[0])
                matched_name = store_names_raw[matched_idx]

                if score >= high_threshold:
                    duplicates.append({
                        **product,
                        "match_score": score,
                        "matched_store_product": matched_name,
                        "match_reason": "fuzzy_high",
                    })
                elif score >= low_threshold:
                    gray_zone.append({
                        **product,
                        "match_score": score,
                        "matched_store_product": matched_name,
                        "match_reason": "fuzzy_medium",
                    })
                else:
                    new_opps.append({
                        **product,
                        "match_score": score,
                        "matched_store_product": "",
                        "match_reason": "new",
                    })
            else:
                new_opps.append({**product, "match_score": 0,
                                 "matched_store_product": "", "match_reason": "new"})
        else:
            # No store products loaded → everything is new
            new_opps.append({**product, "match_score": 0,
                             "matched_store_product": "", "match_reason": "new"})

    def _to_df(lst: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(lst) if lst else pd.DataFrame()

    return _to_df(new_opps), _to_df(gray_zone), _to_df(duplicates)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  GEMINI AI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _rate_limited_sleep() -> None:
    """Enforce a minimum gap between consecutive Gemini API calls."""
    global _LAST_API_CALL
    elapsed = time.time() - _LAST_API_CALL
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _LAST_API_CALL = time.time()


def _gemini_call_with_retry(
    model,
    prompt: str,
    max_retries: int = 4,
) -> Optional[str]:
    """
    Call a Gemini model with exponential-backoff retry on rate-limit errors.

    Returns the stripped response text, or None on complete failure.
    """
    for attempt in range(max_retries):
        try:
            _rate_limited_sleep()
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as exc:
            err = str(exc).lower()
            if any(k in err for k in ("quota", "rate", "429", "resource exhausted")):
                wait = 5 * (2 ** attempt)   # 5 → 10 → 20 → 40 s
                logger.warning(
                    f"Gemini rate-limit (attempt {attempt+1}/{max_retries}). "
                    f"Sleeping {wait}s…"
                )
                time.sleep(wait)
            else:
                logger.error(f"Gemini error (attempt {attempt+1}): {exc}")
                break
    return None


def verify_with_gemini(
    store_name: str,
    comp_name:  str,
    api_key:    str,
) -> Optional[str]:
    """
    Ask Gemini whether two product names refer to the same item.

    Returns
    -------
    'MATCH' | 'DIFFERENT' | None (on API failure)
    """
    if not (api_key and GENAI_AVAILABLE):
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            "أنت خبير عطور ومنتجات تجميل دقيق جداً.\n"
            f"المنتج الأول : {store_name}\n"
            f"المنتج الثاني: {comp_name}\n\n"
            "هل هذان المنتجان نفس الشيء تماماً (نفس الماركة، الاسم، التركيز)؟\n"
            "أجب بكلمة واحدة فقط:  MATCH  أو  DIFFERENT"
        )

        result = _gemini_call_with_retry(model, prompt)
        if result:
            upper = result.upper()
            if "MATCH" in upper:
                return "MATCH"
            if "DIFFERENT" in upper:
                return "DIFFERENT"
    except Exception as exc:
        logger.error(f"verify_with_gemini: {exc}")
    return None


def enrich_product_with_gemini(
    product_name: str,
    image_url:    str,
    product_type: str,
    api_key:      str,
    expert_prompt: str = "",
) -> dict:
    """
    Use Gemini to generate rich marketing data for a single product.

    Returns a dict with keys:
        brand, gender, salla_category, description, sku_suggestion, enriched
    """
    default = {
        "brand": "",
        "gender": "للجنسين",
        "salla_category": (
            "العطور > عطور للجنسين" if product_type == "perfume"
            else "مكياج وعناية"
        ),
        "description": f"<p>{product_name}</p>",
        "sku_suggestion": "",
        "enriched": False,
    }

    if not (api_key and GENAI_AVAILABLE and product_name.strip()):
        return default

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        type_label = "عطر" if product_type == "perfume" else "منتج تجميل أو عناية"
        img_note   = f"\nرابط صورة المنتج: {image_url}" if image_url else ""
        extra_inst = (
            f"\nتعليمات إضافية للكتابة:\n{expert_prompt[:600]}"
            if expert_prompt else ""
        )

        prompt = f"""أنت خبير {type_label} وكاتب محتوى تسويقي لمتجر "مهووس" السعودي.
المنتج: {product_name}
النوع : {type_label}{img_note}{extra_inst}

أعد بيانات JSON فقط (لا نص خارج JSON، لا ```):
{{
  "brand_ar": "الماركة بالعربية",
  "brand_en": "Brand in English",
  "gender": "رجالي | نسائي | للجنسين | نيش",
  "salla_category": "مثال: العطور > عطور رجالية > عطور رجالية فاخرة",
  "sku_suggestion": "رمز SKU قصير بالإنجليزية",
  "description_html": "وصف HTML تسويقي عربي 200-300 كلمة يشمل: مقدمة جذابة، هرم عطري أو مكونات، لماذا تشتريه، خاتمة مقنعة. استخدم فقط: <h2> <h3> <ul> <li> <strong> <p>"
}}"""

        raw = _gemini_call_with_retry(model, prompt)
        if not raw:
            return default

        # Clean potential markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$",          "", raw, flags=re.MULTILINE)
        raw = raw.strip()

        data = json.loads(raw)

        brand_ar = data.get("brand_ar", "").strip()
        brand_en = data.get("brand_en", "").strip()
        brand    = f"{brand_ar} | {brand_en}".strip(" |")

        return {
            "brand":         brand,
            "gender":        data.get("gender", "للجنسين"),
            "salla_category": data.get("salla_category", default["salla_category"]),
            "description":   data.get("description_html", default["description"]),
            "sku_suggestion": data.get("sku_suggestion", ""),
            "enriched":      True,
        }

    except json.JSONDecodeError:
        logger.warning(f"Gemini returned non-JSON for '{product_name}'")
    except Exception as exc:
        logger.error(f"enrich_product_with_gemini('{product_name}'): {exc}")

    return default


# ══════════════════════════════════════════════════════════════════════════════
# 7.  IMAGE FETCHING  (graceful fallback)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_fallback_image(
    product_name:   str,
    search_api_key: str = "",
    cx:             str = "",
) -> str:
    """
    Try to fetch a product image via Google Custom Search.
    Returns URL string, or empty string if anything fails.
    Never raises an exception.
    """
    if not search_api_key or not cx or not product_name.strip():
        return ""
    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key":        search_api_key,
                "cx":         cx,
                "q":          f"{product_name} official",
                "searchType": "image",
                "num":        1,
                "imgSize":    "large",
                "imgType":    "photo",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            items = resp.json().get("items", [])
            if items:
                return items[0].get("link", "")
    except Exception as exc:
        logger.warning(f"fetch_fallback_image('{product_name}'): {exc}")
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# 8.  EXPORT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _build_salla_row(product: dict) -> dict:
    """Map an enriched product dict onto the 40-column Salla schema."""
    return {
        "النوع ":                         "منتج",
        "أسم المنتج":                     product.get("product_name", ""),
        "تصنيف المنتج":                   product.get("salla_category", ""),
        "صورة المنتج":                    product.get("image_url", ""),
        "وصف صورة المنتج":               product.get("product_name", ""),
        "نوع المنتج":                     "منتج جاهز",
        "سعر المنتج":                     product.get("price", ""),
        "الوصف":                          product.get("description", ""),
        "هل يتطلب شحن؟":                  "نعم",
        "رمز المنتج sku":                 product.get("sku_suggestion", ""),
        "سعر التكلفة":                    "",
        "السعر المخفض":                   "",
        "تاريخ بداية التخفيض":            "",
        "تاريخ نهاية التخفيض":            "",
        "اقصي كمية لكل عميل":             "0",
        "إخفاء خيار تحديد الكمية":        "لا",
        "اضافة صورة عند الطلب":           "",
        "الوزن":                          "0.5",
        "وحدة الوزن":                    "kg",
        "الماركة":                        product.get("brand", ""),
        "العنوان الترويجي":               "",
        "تثبيت المنتج":                   "لا",
        "الباركود":                       "",
        "السعرات الحرارية":               "",
        "MPN":                            "",
        "GTIN":                           "",
        "خاضع للضريبة ؟":                 "نعم",
        "سبب عدم الخضوع للضريبة":        "",
        "[1] الاسم":                      "",
        "[1] النوع":                      "",
        "[1] القيمة":                     "",
        "[1] الصورة / اللون":             "",
        "[2] الاسم":                      "",
        "[2] النوع":                      "",
        "[2] القيمة":                     "",
        "[2] الصورة / اللون":             "",
        "[3] الاسم":                      "",
        "[3] النوع":                      "",
        "[3] القيمة":                     "",
        "[3] الصورة / اللون":             "",
    }


def export_to_salla_csv(products: list[dict]) -> bytes:
    """
    Build a Salla-compatible CSV blob.

    Structure
    ---------
    • Row 1 : 'بيانات المنتج' + 39 commas  (magic Salla header – 40 fields)
    • Row 2 : Column headers
    • Row 3+: Product data

    Encoding: UTF-8-BOM (utf-8-sig) so Excel / Salla read Arabic correctly.

    Note
    ----
    Row 1 is written as a raw string BEFORE Pandas, so Pandas never
    touches or corrupts it.
    """
    if not products:
        products = []

    rows = [_build_salla_row(p) for p in products]
    df   = pd.DataFrame(rows, columns=SALLA_COLS)

    # ── Write manually to control Row 1 exactly ──────────────────────────────
    buf = io.StringIO()

    # Row 1: magic Salla header (40 fields = header + 39 empty)
    buf.write("بيانات المنتج" + "," * (len(SALLA_COLS) - 1) + "\n")

    # Row 2+: headers then data via pandas (no index, no BOM here)
    df.to_csv(buf, index=False, encoding="utf-8")

    # Prepend UTF-8 BOM and return as bytes
    final_str = buf.getvalue()
    return ("\ufeff" + final_str).encode("utf-8")


def export_missing_brands_csv(
    products:        list[dict],
    existing_brands: list[str],
) -> bytes:
    """
    Extract brands from the new-opportunity products that do NOT already
    exist in the store's brand list, and export them in Salla brand format.

    Returns UTF-8-BOM encoded bytes.
    """
    BRAND_COLS = [
        "اسم الماركة",
        "وصف مختصر عن الماركة",
        "صورة شعار الماركة",
        "(إختياري) صورة البانر",
        "(Page Title) عنوان صفحة العلامة التجارية",
        "(SEO Page URL) رابط صفحة العلامة التجارية",
        "(Page Description) وصف صفحة العلامة التجارية",
    ]

    existing_lower = {b.lower().strip() for b in existing_brands if b}

    new_brands: dict[str, dict] = {}

    for p in products:
        brand = str(p.get("brand", "")).strip()
        if not brand:
            continue

        brand_key = brand.lower()
        is_new = not any(
            fuzz.partial_ratio(brand_key, eb) > 85
            for eb in existing_lower
        )

        if is_new and brand not in new_brands:
            slug = re.sub(r"[^\w\s-]", "", brand.lower()).strip()
            slug = re.sub(r"[\s|]+", "-", slug).strip("-")
            new_brands[brand] = {
                "اسم الماركة":                             brand,
                "وصف مختصر عن الماركة":                   "",
                "صورة شعار الماركة":                      "",
                "(إختياري) صورة البانر":                   "",
                "(Page Title) عنوان صفحة العلامة التجارية":
                    f"{brand} | مهووس للعطور",
                "(SEO Page URL) رابط صفحة العلامة التجارية": slug,
                "(Page Description) وصف صفحة العلامة التجارية":
                    f"تسوق منتجات {brand} الأصلية الفاخرة لدى مهووس.",
            }

    df  = pd.DataFrame(list(new_brands.values()), columns=BRAND_COLS) \
          if new_brands else pd.DataFrame(columns=BRAND_COLS)

    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()
