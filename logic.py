"""
logic.py — Mahwous Hybrid Semantic Engine v7.0
================================================
5-Layer Pipeline:
  L1  Deterministic Blocking & Feature Parsing
  L2  Semantic Vector Search (multilingual FAISS)
  L3  Lexical Weighted Verification (rapidfuzz)
  L4  LLM Oracle (Gemini 1.5 Flash) — gray-zone only
  L5  Image Fetching Agent

Architecture: OOP, lazy loading, thread-safe, production-ready.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz as rfuzz
from rapidfuzz import process as rprocess

try:
    import google.generativeai as genai
    _GENAI_OK = True
except ImportError:
    _GENAI_OK = False

try:
    import faiss
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False

log = logging.getLogger("mahwous")
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")

# ── Rate-limiter state ──────────────────────────────────────────────────────
_LAST_GEMINI: float = 0.0
_GEMINI_GAP: float  = 1.5          # seconds between calls


# ═══════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ProductFeatures:
    """Parsed product attributes extracted in Layer 1."""
    volume_ml:     str = ""
    concentration: str = ""
    brand_ar:      str = ""
    brand_en:      str = ""
    category:      str = ""   # perfume | beauty | unknown
    gtin:          str = ""
    sku:           str = ""


@dataclass
class MatchResult:
    """Full match record for one competitor product."""
    verdict:          str   = "review"   # new | duplicate | review
    confidence:       float = 0.0
    layer_used:       str   = ""         # L1-GTIN | L2-FAISS | L3-LEX | L4-LLM | safe
    store_name:       str   = ""
    store_image:      str   = ""
    comp_name:        str   = ""
    comp_image:       str   = ""
    comp_price:       str   = ""
    comp_source:      str   = ""
    feature_details:  str   = ""
    faiss_score:      float = 0.0
    lex_score:        float = 0.0
    llm_reasoning:    str   = ""
    product_type:     str   = "perfume"  # perfume | beauty | unknown
    brand:            str   = ""
    salla_category:   str   = ""


# ═══════════════════════════════════════════════════════════════════════════
#  LAYER 0 — Salla export schema
# ═══════════════════════════════════════════════════════════════════════════

SALLA_COLS = [
    "النوع ","أسم المنتج","تصنيف المنتج","صورة المنتج","وصف صورة المنتج",
    "نوع المنتج","سعر المنتج","الوصف","هل يتطلب شحن؟","رمز المنتج sku",
    "سعر التكلفة","السعر المخفض","تاريخ بداية التخفيض","تاريخ نهاية التخفيض",
    "اقصي كمية لكل عميل","إخفاء خيار تحديد الكمية","اضافة صورة عند الطلب",
    "الوزن","وحدة الوزن","الماركة","العنوان الترويجي","تثبيت المنتج",
    "الباركود","السعرات الحرارية","MPN","GTIN","خاضع للضريبة ؟",
    "سبب عدم الخضوع للضريبة",
    "[1] الاسم","[1] النوع","[1] القيمة","[1] الصورة / اللون",
    "[2] الاسم","[2] النوع","[2] القيمة","[2] الصورة / اللون",
    "[3] الاسم","[3] النوع","[3] القيمة","[3] الصورة / اللون",
]
assert len(SALLA_COLS) == 40


# ═══════════════════════════════════════════════════════════════════════════
#  LAYER 1 — Deterministic Blocking & Feature Parsing
# ═══════════════════════════════════════════════════════════════════════════

class FeatureParser:
    """Extracts structured features from raw product name strings."""

    _VOL = re.compile(
        r"(\d+\.?\d*)\s*(ml|مل|g|gr|غ|oz|fl\.?\s*oz|مل|cc)",
        re.IGNORECASE | re.UNICODE,
    )
    _CONC_AR = {
        "EDP":      ["او دو برفيوم","او دي بارفيوم","اودو بارفيوم","او دو بارفيوم",
                     "او دو بيرفيوم","او دو برفوم","اودي برفيوم","اود برفيوم",
                     "بارفيوم","برفيوم","بيرفيوم","بارفان","لو بارفان","لو دي بارفان"],
        "EDT":      ["او دو تواليت","او دي تواليت","اودي تواليت","تواليت"],
        "EDC":      ["او دو كولون","كولون","كولونيا"],
        "Extrait":  ["اكستريت","إكستريت","اليكسير دي بارفيوم","اليكسير دو بارفيوم",
                     "اليكسير دي بارفان","انتنس اكستريت"],
        "Parfum":   ["بارفيوم ناتورال","ماء العطر"],
        "HairMist": ["رذاذ الشعر","بخاخ الشعر","معطر الشعر"],
        "BodyMist": ["بخاخ الجسم","بخاخ للجسم","بخاخ معطر"],
    }
    _CONC_EN = {
        "EDP":      [r"\bedp\b", r"eau\s+de\s+parfum", r"eau\s+du?\s+parfu"],
        "EDT":      [r"\bedt\b", r"eau\s+de\s+toilette"],
        "EDC":      [r"\bedc\b", r"eau\s+de\s+cologne"],
        "Extrait":  [r"\bextrait\b", r"elixir\s+de\s+parfum", r"\belixir\b",
                     r"\bintense\b", r"\bintens\b"],
        "Parfum":   [r"\bparfum\b", r"\bperfume\b"],
        "HairMist": [r"hair\s+mist", r"hair\s+perfume"],
        "BodyMist": [r"body\s+mist", r"body\s+spray"],
    }
    _PERF_KW = [
        "عطر","بارفيوم","برفيوم","تواليت","تستر","بخور","دهن","عود","مسك","معطر",
        "عطرية","هدايا عطر","مجموعة عطر","مجموعة هدايا عطرية","عطر الشعر",
        "perfume","cologne","fragrance","parfum","edp","edt","edc","oud","musk","mist",
    ]
    _BEAUTY_KW = [
        "مكياج","ريمل","مسكارا","روج","بودرة","سيروم","كريم وجه",
        "شامبو","بلسم","لوشن","واقي شمس","ماسك","تونر","فاونديشن",
        "كونسيلر","بلاشر","هايلايتر","كاجال","احمر شفاه","استشوار","مجفف",
        "mascara","lipstick","foundation","concealer","blush","serum","moisturizer",
        "shampoo","conditioner","hairdryer","straightener","toner","sunscreen",
        "غسول","كريم جسم","لوشن جسم","غسول وجه","مرطب جسم","واقي",
    ]

    @classmethod
    def parse(cls, name: str, sku: str = "", gtin: str = "") -> ProductFeatures:
        """Parse a product name into structured features."""
        name_lower = name.lower().strip()
        name_norm  = cls._normalize_ar(name_lower)

        # Volume
        vol = cls._extract_volume(name)

        # Concentration
        conc = cls._extract_concentration(name_norm)

        # Category
        cat = cls._classify_product(name_norm)

        # Brand extraction (heuristic: first 1-2 words before Arabic perfume keyword)
        brand_ar, brand_en = cls._extract_brand(name)

        # Salla category suggestion
        salla_cat = cls._suggest_salla_category(cat, name_norm)

        return ProductFeatures(
            volume_ml     = vol,
            concentration = conc,
            brand_ar      = brand_ar,
            brand_en      = brand_en,
            category      = cat,
            gtin          = str(gtin).strip() if gtin else "",
            sku           = str(sku).strip() if sku else "",
        )

    @staticmethod
    def _normalize_ar(text: str) -> str:
        """Normalize Arabic text for matching."""
        t = re.sub(r"[أإآ]", "ا", text)
        t = re.sub(r"ى", "ي", t)
        t = re.sub(r"ة", "ه", t)
        return t

    @classmethod
    def _extract_volume(cls, name: str) -> str:
        m = cls._VOL.search(name)
        if not m:
            return ""
        num  = float(m.group(1))
        unit = m.group(2).lower().strip()
        # Normalise to ml
        if "oz" in unit:
            num = round(num * 29.5735, 1)
        formatted = f"{num:.1f}".rstrip("0").rstrip(".")
        return formatted + "ml" if num else ""

    @classmethod
    def _extract_concentration(cls, name_norm: str) -> str:
        # English patterns first (more precise)
        for conc, patterns in cls._CONC_EN.items():
            for pat in patterns:
                if re.search(pat, name_norm, re.IGNORECASE):
                    return conc
        # Arabic keywords
        for conc, keywords in cls._CONC_AR.items():
            for kw in keywords:
                if kw in name_norm:
                    return conc
        return ""

    @classmethod
    def _classify_product(cls, name_norm: str) -> str:
        """Classify as perfume / beauty / unknown with word-boundary checks."""
        def _has(kw: str, text: str) -> bool:
            """Check keyword with boundary check for short words."""
            if len(kw) <= 4:
                # Prevent substring matches: "مسك" should NOT match inside "مسكارا"
                return bool(re.search(
                    r"(?:^|[\s،,.+])" + re.escape(kw) + r"(?=[\s،,.+]|$)",
                    " " + text + " ", re.UNICODE | re.IGNORECASE
                ))
            return kw in text

        beauty_score = sum(1 for kw in cls._BEAUTY_KW if _has(kw, name_norm))
        perf_score   = sum(1 for kw in cls._PERF_KW   if _has(kw, name_norm))
        if perf_score > 0 and perf_score >= beauty_score:
            return "perfume"
        if beauty_score > 0:
            return "beauty"
        return "unknown"

    @classmethod
    def _extract_brand(cls, name: str) -> tuple[str, str]:
        """Heuristic brand extraction."""
        # Try to find known brand pattern: Word | Word
        m = re.search(r"([A-Za-z\u0600-\u06FF ]+?)\s*\|\s*([A-Za-z ]+)", name)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        # First 1-2 Arabic words before "عطر" or common marker
        ar_m = re.match(r"^([^\s]+(?:\s+[^\s]+)?)\s+(?:عطر|تستر|عطور)", name)
        if ar_m:
            return ar_m.group(1).strip(), ""
        # First English word(s)
        en_m = re.match(r"^([A-Za-z]+(?:\s+[A-Za-z]+)?)", name)
        if en_m:
            return "", en_m.group(1).strip()
        return "", ""

    @staticmethod
    def _suggest_salla_category(cat: str, name: str) -> str:
        """Suggest a Salla-style category path."""
        if cat == "perfume":
            if any(k in name for k in ["نسائي","للنساء","women","femme","pour elle"]):
                return "العطور > عطور نسائية > عطور نسائية فاخرة"
            if any(k in name for k in ["رجالي","للرجال","men","homme","pour lui"]):
                return "العطور > عطور رجالية > عطور رجالية فاخرة"
            if any(k in name for k in ["نيش","niche"]):
                return "العطور > عطور نيش > عطور نيش فاخرة"
            return "العطور > عطور للجنسين"
        if cat == "beauty":
            if any(k in name for k in ["مكياج","makeup","foundation","mascara"]):
                return "المكياج والعناية > مستحضرات المكياج"
            if any(k in name for k in ["شعر","hair","شامبو"]):
                return "المكياج والعناية > العناية بالشعر"
            if any(k in name for k in ["كريم","سيروم","serum","cream","بشرة","skin"]):
                return "المكياج والعناية > العناية بالبشرة"
            return "المكياج والعناية"
        return "العطور"


def features_mismatch(f1: ProductFeatures, f2: ProductFeatures) -> tuple[bool, str]:
    """
    Return (True, reason) if two products are definitely different SKUs.
    Only triggers when BOTH sides have a value that differs.
    """
    reasons = []
    if f1.volume_ml and f2.volume_ml:
        v1, v2 = float(f1.volume_ml.replace("ml","")), float(f2.volume_ml.replace("ml",""))
        if abs(v1 - v2) > max(v1, v2) * 0.12:
            reasons.append(f"حجم مختلف ({f1.volume_ml} ≠ {f2.volume_ml})")
    _CONC_GRP = {
        "EDP":"EDP","Parfum":"EDP","Intense":"EDP",
        "EDT":"EDT","EDC":"EDC",
        "Extrait":"Extrait","HairMist":"HairMist","BodyMist":"BodyMist",
    }
    c1 = _CONC_GRP.get(f1.concentration, f1.concentration)
    c2 = _CONC_GRP.get(f2.concentration, f2.concentration)
    if c1 and c2 and c1 != c2:
        reasons.append(f"تركيز مختلف ({f1.concentration} ≠ {f2.concentration})")
    return bool(reasons), " | ".join(reasons)


# ═══════════════════════════════════════════════════════════════════════════
#  SAFE CSV READER
# ═══════════════════════════════════════════════════════════════════════════

def _read_csv(file_obj, **kwargs) -> pd.DataFrame:
    """Read CSV with multiple encoding fallbacks."""
    if hasattr(file_obj, "read"):
        raw = file_obj.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
    else:
        with open(file_obj, "rb") as fh:
            raw = fh.read()
    for enc in ("utf-8-sig", "utf-8", "cp1256", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, **kwargs)
        except Exception:
            continue
    raise ValueError("Cannot decode CSV")


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════

def load_store_products(files: list) -> pd.DataFrame:
    """Load Salla-format store CSV files (parts 1-N)."""
    frames = []
    for f in files:
        try:
            raw_df = _read_csv(f, header=None, low_memory=False, dtype=str)
            # Find header row (row containing "أسم المنتج")
            hrow = None
            for i, row in raw_df.iterrows():
                if any("أسم المنتج" in str(v) or "اسم المنتج" in str(v) for v in row.values):
                    hrow = i; break
            if hrow is None:
                continue
            raw_df.columns = [str(c).strip() for c in raw_df.iloc[hrow].values]
            data = raw_df.iloc[hrow + 1:].reset_index(drop=True)

            def _c(kw: str) -> Optional[str]:
                return next((c for c in data.columns if kw in c), None)

            frame = pd.DataFrame()
            frame["product_name"] = data[_c("أسم المنتج")].fillna("").astype(str)
            frame["brand"]        = data[_c("الماركة")].fillna("").astype(str) if _c("الماركة") else ""
            frame["category"]     = data[_c("تصنيف المنتج")].fillna("").astype(str) if _c("تصنيف المنتج") else ""
            frame["sku"]          = data[_c("رمز المنتج")].fillna("").astype(str) if _c("رمز المنتج") else ""
            frame["gtin"]         = data[_c("GTIN")].fillna("").astype(str)      if _c("GTIN") else ""
            frame["price"]        = data[_c("سعر المنتج")].fillna("").astype(str) if _c("سعر المنتج") else ""

            if _c("صورة المنتج"):
                frame["image_url"] = data[_c("صورة المنتج")].apply(
                    lambda x: str(x).split(",")[0].strip() if pd.notna(x) else ""
                )
            else:
                frame["image_url"] = ""

            frames.append(frame[frame["product_name"].str.strip() != ""])
        except Exception as e:
            log.error(f"load_store: {e}")

    if not frames:
        return pd.DataFrame(columns=["product_name","brand","category","sku","gtin","price","image_url"])
    return (pd.concat(frames, ignore_index=True)
              .drop_duplicates(subset=["product_name"])
              .reset_index(drop=True))


def load_competitor_products(files: list) -> pd.DataFrame:
    """Load scraped competitor CSV files."""
    frames = []
    for f in files:
        try:
            df = _read_csv(f, low_memory=False, dtype=str)
            df.columns = [str(c).strip() for c in df.columns]

            name_h  = ["name","اسم","productcard","styles_product","title","عنوان"]
            img_h   = ["src","image","img","صورة","photo","w-full src"]
            price_h = ["price","سعر","text-sm","cost"]

            nc = next((c for c in df.columns if any(h in c.lower() for h in name_h)),
                      df.columns[2] if len(df.columns)>2 else None)
            ic = next((c for c in df.columns if any(h in c.lower() for h in img_h)),
                      df.columns[1] if len(df.columns)>1 else None)
            pc = next((c for c in df.columns if any(h in c.lower() for h in price_h)),
                      df.columns[3] if len(df.columns)>3 else None)

            frame = pd.DataFrame()
            frame["product_name"] = df[nc].fillna("").astype(str) if nc else ""
            frame["image_url"]    = df[ic].fillna("").astype(str).str.split(",").str[0].str.strip() if ic else ""
            frame["price"]        = df[pc].fillna("").astype(str) if pc else ""
            frame["source_file"]  = getattr(f, "name", str(f))
            frames.append(frame[frame["product_name"].str.strip().str.len() > 1])
        except Exception as e:
            log.error(f"load_comp: {e}")

    if not frames:
        return pd.DataFrame(columns=["product_name","image_url","price","source_file"])
    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def load_brands(file) -> list[str]:
    """Load existing brand names from Salla brands CSV."""
    try:
        df = _read_csv(file, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]
        col = next((c for c in df.columns if "اسم" in c), None)
        return df[col].dropna().astype(str).tolist() if col else []
    except Exception as e:
        log.error(f"load_brands: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════
#  LAYER 2 — Semantic Vector Search (FAISS)
# ═══════════════════════════════════════════════════════════════════════════

class SemanticIndex:
    """
    Multilingual FAISS index for semantic product matching.
    Lifecycle: build once per store DataFrame (hashed), cached in session_state.
    """

    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model):
        self._model = model
        self._index: Optional["faiss.IndexFlatIP"] = None
        self._store_names: list[str] = []
        self._store_hash: str = ""
        self._dim: int = 384

    @property
    def is_built(self) -> bool:
        return self._index is not None and len(self._store_names) > 0

    def build(
        self,
        store_df: pd.DataFrame,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Build FAISS index from store products. Returns store hash."""
        names = store_df["product_name"].tolist()
        h = hashlib.md5("|".join(names[:50]).encode()).hexdigest()[:12]
        if h == self._store_hash:
            return h  # Already built for this data

        if progress_cb:
            progress_cb("🧠 تحويل أسماء المتجر لمتجهات دلالية…")

        try:
            embeddings = self._model.encode(
                names, batch_size=64, show_progress_bar=False,
                normalize_embeddings=True,
            )
            dim = embeddings.shape[1]
            idx = faiss.IndexFlatIP(dim)
            idx.add(embeddings.astype("float32"))
            self._index       = idx
            self._store_names = names
            self._store_hash  = h
            self._dim         = dim
            log.info(f"FAISS built: {len(names)} vectors, dim={dim}")
        except Exception as e:
            log.error(f"FAISS build failed: {e}")
        return h

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        """Search for k nearest store products. Returns [(name, score)]."""
        if not self.is_built:
            return []
        try:
            qvec = self._model.encode(
                [query], normalize_embeddings=True
            ).astype("float32")
            scores, idxs = self._index.search(qvec, k)
            results = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx >= 0 and score > 0.1:
                    results.append((self._store_names[idx], float(score)))
            return results
        except Exception as e:
            log.error(f"FAISS search failed: {e}")
            return []


# ═══════════════════════════════════════════════════════════════════════════
#  LAYER 3 — Lexical Weighted Verification
# ═══════════════════════════════════════════════════════════════════════════

class LexicalVerifier:
    """
    Weighted lexical scoring using rapidfuzz.
    Weights: brand 40% | name 30% | volume 20% | concentration 10%
    """

    WEIGHTS = {"brand": 0.40, "name": 0.30, "volume": 0.20, "conc": 0.10}

    @classmethod
    def score(
        cls,
        comp_name: str,
        store_name: str,
        comp_feat: ProductFeatures,
        store_feat: ProductFeatures,
    ) -> float:
        """Return weighted [0-1] lexical similarity."""
        scores = {}

        # Brand similarity
        b1 = (comp_feat.brand_ar or comp_feat.brand_en or "").lower()
        b2 = (store_feat.brand_ar or store_feat.brand_en or "").lower()
        scores["brand"] = rfuzz.token_sort_ratio(b1, b2) / 100 if (b1 and b2) else 0.5

        # Name similarity
        scores["name"] = rfuzz.token_sort_ratio(
            comp_name.lower(), store_name.lower()
        ) / 100

        # Volume exact match
        if comp_feat.volume_ml and store_feat.volume_ml:
            scores["volume"] = 1.0 if comp_feat.volume_ml == store_feat.volume_ml else 0.0
        else:
            scores["volume"] = 0.5  # unknown → neutral

        # Concentration match
        if comp_feat.concentration and store_feat.concentration:
            scores["conc"] = 1.0 if comp_feat.concentration == store_feat.concentration else 0.0
        else:
            scores["conc"] = 0.5

        return sum(cls.WEIGHTS[k] * v for k, v in scores.items())


# ═══════════════════════════════════════════════════════════════════════════
#  LAYER 4 — LLM Oracle (Gemini)
# ═══════════════════════════════════════════════════════════════════════════

class GeminiOracle:
    """Gemini 1.5 Flash judge — called only in gray zone (55%-84%)."""

    _SYSTEM = (
        "أنت قاضٍ متخصص في مطابقة منتجات العطور والتجميل. "
        "مهمتك: تحديد ما إذا كان منتجان يمثلان نفس الـ SKU تماماً "
        "(نفس الماركة، الاسم، الحجم، التركيز). "
        "انظر للصورتين والاسمين بعناية. "
        "أجب بكلمة واحدة فقط: MATCH أو DIFFERENT"
    )

    def __init__(self, api_key: str):
        self._key = api_key
        self._ok  = bool(api_key and _GENAI_OK)
        if self._ok:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel("gemini-1.5-flash")

    def judge(
        self,
        store_name: str,
        comp_name: str,
        store_img: str = "",
        comp_img:  str = "",
    ) -> tuple[str, str]:
        """
        Returns ("MATCH"|"DIFFERENT"|"UNKNOWN", reasoning).
        Uses images when available; falls back to text-only.
        """
        if not self._ok:
            return "UNKNOWN", "no API key"

        global _LAST_GEMINI
        elapsed = time.time() - _LAST_GEMINI
        if elapsed < _GEMINI_GAP:
            time.sleep(_GEMINI_GAP - elapsed)
        _LAST_GEMINI = time.time()

        prompt_text = (
            f"🏪 منتجنا: {store_name}\n"
            f"🔍 منتج المنافس: {comp_name}\n\n"
            f"{self._SYSTEM}\n"
            "تذكر: الفرق في اللغة (عربي/إنجليزي) ليس سبباً للاختلاف. "
            "الاختلاف الحقيقي: حجم مختلف، تركيز مختلف، ماركة مختلفة، اسم مختلف تماماً."
        )

        for attempt in range(4):
            try:
                parts: list = [prompt_text]

                # Add images if available
                for url in [store_img, comp_img]:
                    if url and url.startswith("http"):
                        img_bytes = _fetch_image_bytes(url)
                        if img_bytes:
                            parts.append({
                                "mime_type": "image/jpeg",
                                "data": img_bytes,
                            })

                resp = self._model.generate_content(parts if len(parts) > 1 else prompt_text)
                text = resp.text.strip().upper()

                if "MATCH"     in text: return "MATCH",     text
                if "DIFFERENT" in text: return "DIFFERENT", text
                return "UNKNOWN", text

            except Exception as e:
                err = str(e).lower()
                if "429" in err or "quota" in err or "rate" in err:
                    time.sleep(5 * 2**attempt)
                else:
                    log.error(f"Gemini error attempt {attempt+1}: {e}")
                    break
        return "UNKNOWN", "failed"


# ═══════════════════════════════════════════════════════════════════════════
#  LAYER 5 — Image Fetching Agent
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_image_bytes(url: str, timeout: int = 8) -> Optional[bytes]:
    """Download image bytes from URL, None on failure."""
    if not url or not url.startswith("http"):
        return None
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        if resp.status_code == 200 and "image" in resp.headers.get("content-type",""):
            return resp.content
    except Exception:
        pass
    return None


def fetch_product_image(
    product_name: str,
    search_api_key: str = "",
    cx: str = "",
) -> str:
    """Attempt to find a product image via Google Custom Search."""
    if not (search_api_key and cx and product_name.strip()):
        return ""
    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": search_api_key,
                "cx": cx,
                "q": f"{product_name} official product",
                "searchType": "image",
                "num": 1,
                "imgSize": "large",
                "imgType": "photo",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            items = resp.json().get("items", [])
            if items:
                return items[0].get("link", "")
    except Exception as e:
        log.warning(f"Image fetch failed: {e}")
    return ""


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════

# Decision thresholds
_HIGH  = 0.85   # ≥ 85% → duplicate  (no LLM needed)
_LOW   = 0.40   # < 40% → new opp    (no LLM needed)
_GRAY_LOW  = 0.40
_GRAY_HIGH = 0.85
# Within gray zone: LLM is called
# Above _HIGH: definite duplicate
# Below _LOW:  definite new opportunity


class MahwousEngine:
    """
    Orchestrates the 5-layer hybrid pipeline.
    Thread-safe progress callbacks for Streamlit background processing.
    """

    def __init__(
        self,
        semantic_index: SemanticIndex,
        gemini_oracle: Optional[GeminiOracle] = None,
        search_api_key: str = "",
        search_cx: str = "",
        fetch_images: bool = False,
    ):
        self.idx          = semantic_index
        self.oracle       = gemini_oracle
        self.search_key   = search_api_key
        self.search_cx    = search_cx
        self.fetch_images = fetch_images
        self._lock        = threading.Lock()

    def run(
        self,
        store_df: pd.DataFrame,
        comp_df: pd.DataFrame,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        log_cb:      Optional[Callable[[str], None]] = None,
    ) -> tuple[list[MatchResult], list[MatchResult], list[MatchResult]]:
        """
        Process all competitor products through the 5-layer pipeline.
        Returns (new_opps, duplicates, reviews).
        """
        # Pre-build store lookup structures
        store_names = store_df["product_name"].tolist()
        store_imgs  = store_df.get("image_url", pd.Series([""] * len(store_df))).tolist()
        store_gtins = store_df.get("gtin", pd.Series([""] * len(store_df))).tolist()
        store_skus  = store_df.get("sku",  pd.Series([""] * len(store_df))).tolist()

        # Parse store features once
        store_feats = {}
        for i, row in store_df.iterrows():
            store_feats[store_names[i]] = FeatureParser.parse(
                store_names[i], store_skus[i], store_gtins[i]
            )

        # GTIN / SKU lookup sets
        gtin_set = {g.strip() for g in store_gtins if g and g.strip() not in ("nan","None","")}
        sku_set  = {s.strip() for s in store_skus  if s and s.strip() not in ("nan","None","")}

        new_opps:   list[MatchResult] = []
        duplicates: list[MatchResult] = []
        reviews:    list[MatchResult] = []
        total = len(comp_df)

        def _log(msg: str) -> None:
            if log_cb:
                log_cb(msg)

        for i, (_, row) in enumerate(comp_df.iterrows()):
            if progress_cb:
                progress_cb(i, total, str(row.get("product_name",""))[:50])

            comp_name  = str(row.get("product_name","")).strip()
            comp_img   = str(row.get("image_url","")).strip()
            comp_price = str(row.get("price","")).strip()
            comp_src   = str(row.get("source_file","")).strip()

            if not comp_name or len(comp_name) < 3:
                continue

            # Parse competitor features
            comp_feat = FeatureParser.parse(comp_name)

            result = MatchResult(
                comp_name   = comp_name,
                comp_image  = comp_img,
                comp_price  = comp_price,
                comp_source = comp_src,
                product_type= comp_feat.category,
            )

            # ──────────────────────────────────────────────────────────────
            # L1: GTIN / SKU exact match → definite duplicate
            # ──────────────────────────────────────────────────────────────
            if comp_feat.gtin and comp_feat.gtin in gtin_set:
                idx_g = store_gtins.index(comp_feat.gtin)
                result.verdict     = "duplicate"
                result.confidence  = 1.0
                result.layer_used  = "L1-GTIN"
                result.store_name  = store_names[idx_g]
                result.store_image = store_imgs[idx_g]
                result.feature_details = f"GTIN: {comp_feat.gtin}"
                _log(f"  🎯 L1-GTIN │ {comp_name[:40]}")
                duplicates.append(result)
                continue

            if comp_feat.sku and comp_feat.sku in sku_set:
                idx_s = store_skus.index(comp_feat.sku)
                result.verdict     = "duplicate"
                result.confidence  = 1.0
                result.layer_used  = "L1-SKU"
                result.store_name  = store_names[idx_s]
                result.store_image = store_imgs[idx_s]
                result.feature_details = f"SKU: {comp_feat.sku}"
                _log(f"  🎯 L1-SKU  │ {comp_name[:40]}")
                duplicates.append(result)
                continue

            # ──────────────────────────────────────────────────────────────
            # L2: Semantic vector search
            # ──────────────────────────────────────────────────────────────
            faiss_hits = self.idx.search(comp_name, k=3)

            if not faiss_hits:
                # No semantic match → new opportunity
                result.verdict    = "new"
                result.confidence = 0.95
                result.layer_used = "L2-FAISS-miss"
                _log(f"  ✨ L2 no-hit │ {comp_name[:40]}")
                # Optionally fetch missing image
                if self.fetch_images and not comp_img:
                    result.comp_image = fetch_product_image(comp_name, self.search_key, self.search_cx)
                new_opps.append(result)
                continue

            # ──────────────────────────────────────────────────────────────
            # L3: Lexical weighted scoring on top FAISS candidates
            # ──────────────────────────────────────────────────────────────
            best_lex     = 0.0
            best_store   = ""
            best_store_i = -1
            best_faiss_s = 0.0

            for (sname, faiss_score) in faiss_hits:
                if sname not in store_feats:
                    continue
                lex_s = LexicalVerifier.score(
                    comp_name, sname, comp_feat, store_feats[sname]
                )
                # Fusion: 60% semantic + 40% lexical
                fused = 0.60 * faiss_score + 0.40 * lex_s
                if fused > best_lex:
                    best_lex    = fused
                    best_store  = sname
                    best_store_i = store_names.index(sname)
                    best_faiss_s = faiss_score

            result.store_name  = best_store
            result.store_image = store_imgs[best_store_i] if best_store_i >= 0 else ""
            result.faiss_score = best_faiss_s
            result.lex_score   = best_lex

            # ──────────────────────────────────────────────────────────────
            # Feature mismatch check (Deterministic override)
            # ──────────────────────────────────────────────────────────────
            if best_store and best_store in store_feats:
                has_mismatch, mismatch_reason = features_mismatch(
                    comp_feat, store_feats[best_store]
                )
                if has_mismatch:
                    result.verdict         = "new"
                    result.confidence      = 0.92
                    result.layer_used      = "L3-FEAT-MISMATCH"
                    result.feature_details = mismatch_reason
                    _log(f"  ⚙️ L3 mismatch │ {comp_name[:35]} | {mismatch_reason}")
                    if self.fetch_images and not comp_img:
                        result.comp_image = fetch_product_image(comp_name, self.search_key, self.search_cx)
                    new_opps.append(result)
                    continue

            # ──────────────────────────────────────────────────────────────
            # Decision by fused score
            # ──────────────────────────────────────────────────────────────
            if best_lex >= _HIGH:
                result.verdict    = "duplicate"
                result.confidence = best_lex
                result.layer_used = "L3-LEX-HIGH"
                _log(f"  🚫 L3-dup   │ {comp_name[:35]} ≈ {best_store[:30]} [{best_lex:.0%}]")
                duplicates.append(result)

            elif best_lex < _LOW:
                result.verdict    = "new"
                result.confidence = 1.0 - best_lex
                result.layer_used = "L3-LEX-LOW"
                _log(f"  ✨ L3-new   │ {comp_name[:40]} [{best_lex:.0%}]")
                if self.fetch_images and not comp_img:
                    result.comp_image = fetch_product_image(comp_name, self.search_key, self.search_cx)
                new_opps.append(result)

            else:
                # ──────────────────────────────────────────────────────────
                # GRAY ZONE → L4: Gemini LLM Oracle
                # ──────────────────────────────────────────────────────────
                if self.oracle:
                    _log(f"  🤖 L4-LLM   │ {comp_name[:35]} [{best_lex:.0%}]")
                    verdict_llm, reasoning = self.oracle.judge(
                        store_name = best_store,
                        comp_name  = comp_name,
                        store_img  = store_imgs[best_store_i] if best_store_i >= 0 else "",
                        comp_img   = comp_img,
                    )
                    result.llm_reasoning = reasoning
                    if verdict_llm == "MATCH":
                        result.verdict    = "duplicate"
                        result.confidence = 0.96
                        result.layer_used = "L4-LLM-MATCH"
                        duplicates.append(result)
                    elif verdict_llm == "DIFFERENT":
                        result.verdict    = "new"
                        result.confidence = 0.96
                        result.layer_used = "L4-LLM-DIFF"
                        if self.fetch_images and not comp_img:
                            result.comp_image = fetch_product_image(comp_name, self.search_key, self.search_cx)
                        new_opps.append(result)
                    else:
                        # LLM inconclusive → safe default: keep as review
                        result.verdict    = "review"
                        result.confidence = best_lex
                        result.layer_used = "L4-LLM-UNSURE"
                        reviews.append(result)
                else:
                    # No LLM → send to review queue
                    result.verdict    = "review"
                    result.confidence = best_lex
                    result.layer_used = "L3-GRAY"
                    reviews.append(result)

        return new_opps, duplicates, reviews


# ═══════════════════════════════════════════════════════════════════════════
#  EXPORT ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def _salla_row(r: MatchResult) -> dict:
    return {
        "النوع ":                  "منتج",
        "أسم المنتج":              r.comp_name,
        "تصنيف المنتج":            r.salla_category or FeatureParser._suggest_salla_category(
                                       r.product_type, r.comp_name.lower()),
        "صورة المنتج":             r.comp_image,
        "وصف صورة المنتج":        r.comp_name,
        "نوع المنتج":              "منتج جاهز",
        "سعر المنتج":              r.comp_price,
        "الوصف":                   f"<p>{r.comp_name}</p>",
        "هل يتطلب شحن؟":           "نعم",
        "رمز المنتج sku":          "",
        "سعر التكلفة":             "",
        "السعر المخفض":            "",
        "تاريخ بداية التخفيض":     "",
        "تاريخ نهاية التخفيض":     "",
        "اقصي كمية لكل عميل":      "0",
        "إخفاء خيار تحديد الكمية": "لا",
        "اضافة صورة عند الطلب":    "",
        "الوزن":                   "0.5",
        "وحدة الوزن":              "kg",
        "الماركة":                  r.brand,
        "العنوان الترويجي":         "",
        "تثبيت المنتج":             "لا",
        "الباركود":                 "",
        "السعرات الحرارية":         "",
        "MPN":                      "",
        "GTIN":                     "",
        "خاضع للضريبة ؟":           "نعم",
        "سبب عدم الخضوع للضريبة":  "",
        "[1] الاسم":  "", "[1] النوع":  "", "[1] القيمة":  "", "[1] الصورة / اللون":  "",
        "[2] الاسم":  "", "[2] النوع":  "", "[2] القيمة":  "", "[2] الصورة / اللون":  "",
        "[3] الاسم":  "", "[3] النوع":  "", "[3] القيمة":  "", "[3] الصورة / اللون":  "",
    }


def export_salla_csv(results: list[MatchResult]) -> bytes:
    """Export to Salla-compatible CSV (Row1=magic header, Row2=columns)."""
    rows = [_salla_row(r) for r in results]
    df   = pd.DataFrame(rows, columns=SALLA_COLS)
    buf  = io.StringIO()
    buf.write("بيانات المنتج" + "," * (len(SALLA_COLS) - 1) + "\n")
    df.to_csv(buf, index=False, encoding="utf-8")
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


def export_brands_csv(results: list[MatchResult], existing_brands: list[str]) -> bytes:
    """Export missing brands that are not already in the store."""
    COLS = ["اسم الماركة","وصف مختصر عن الماركة","صورة شعار الماركة",
            "(إختياري) صورة البانر",
            "(Page Title) عنوان صفحة العلامة التجارية",
            "(SEO Page URL) رابط صفحة العلامة التجارية",
            "(Page Description) وصف صفحة العلامة التجارية"]

    existing_lower = {b.lower().strip() for b in existing_brands if b}
    seen: set[str] = set()
    rows = []
    for r in results:
        brand = (r.brand or "").strip()
        if not brand:
            continue
        if brand.lower() in existing_lower or brand in seen:
            continue
        seen.add(brand)
        slug = re.sub(r"[^\w\s-]","",brand.lower()).strip()
        slug = re.sub(r"[\s|]+","-",slug).strip("-")
        rows.append({
            "اسم الماركة": brand,
            "وصف مختصر عن الماركة": "",
            "صورة شعار الماركة": "",
            "(إختياري) صورة البانر": "",
            "(Page Title) عنوان صفحة العلامة التجارية": f"{brand} | مهووس",
            "(SEO Page URL) رابط صفحة العلامة التجارية": slug,
            "(Page Description) وصف صفحة العلامة التجارية": f"تسوق منتجات {brand} الأصلية لدى مهووس.",
        })

    df  = pd.DataFrame(rows, columns=COLS) if rows else pd.DataFrame(columns=COLS)
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()
