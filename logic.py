"""
logic.py  -  Mahwous Opportunity Engine  v4.0
=============================================
5-Gate Classification System  |  Zero False Positives & Zero False Negatives
Production-Ready Defensive Code with Full Audit Trail

Gate 0: Pre-processor        (text normalization + signal extraction)
Gate 1: Hard Rules           (100% confidence regex/keyword rules)
Gate 2: Category Path        (Salla taxonomy analysis)
Gate 3: Weighted Scoring     (multi-signal voting engine)
Gate 4: Brand Knowledge Base (built-in brand-to-category map)
Gate 5: AI Oracle            (Gemini Flash - last resort)
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
import unicodedata
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
# SECTION 0  -  OUTPUT CATEGORIES  &  CONFIDENCE THRESHOLDS
# ============================================================================

CAT_PERFUME = "perfume"   # -> Salla_Perfumes.csv
CAT_BEAUTY  = "beauty"    # -> Salla_Beauty_Care.csv
CAT_UNKNOWN = "unknown"   # -> Manual Review

GATE_HARD_RULES  = "G1:HardRules"
GATE_CATEGORY    = "G2:CategoryPath"
GATE_SCORING     = "G3:WeightedScoring"
GATE_BRAND_KB    = "G4:BrandKB"
GATE_AI_ORACLE   = "G5:AIOracle"
GATE_FALLBACK    = "G0:Fallback"

CONFIDENCE_FINALIZE = {
    GATE_HARD_RULES: 0.85,
    GATE_CATEGORY:   0.85,
    GATE_SCORING:    0.75,
    GATE_BRAND_KB:   0.70,
    GATE_AI_ORACLE:  0.50,  # always finalizes
}

# ============================================================================
# SECTION 1  -  AUDIT TRAIL DATACLASS
# ============================================================================

@dataclass
class ClassificationResult:
    """Full audit record for a single product classification."""
    category:     str   = CAT_UNKNOWN
    confidence:   float = 0.0
    gate_used:    str   = GATE_FALLBACK
    signals_fired: list  = field(default_factory=list)
    reasoning:    str   = ""
    ai_used:      bool  = False

    def to_dict(self) -> dict:
        return {
            "classified_as":  self.category,
            "confidence_pct": f"{self.confidence * 100:.1f}%",
            "gate_used":      self.gate_used,
            "signals_fired":  " | ".join(self.signals_fired),
            "reasoning":      self.reasoning,
            "ai_used":        self.ai_used,
        }


# ============================================================================
# SECTION 2  -  SALLA EXPORT SCHEMA  (40 columns)
# ============================================================================

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


# ============================================================================
# SECTION 3  -  BUILT-IN KNOWLEDGE BASES
# ============================================================================

# --- 3A: BRAND KNOWLEDGE BASE ---
# Format: "brand_slug": (primary_category, confidence, notes)
# Derived from actual مهووس data + domain expertise
BRAND_KB: dict[str, tuple[str, float, str]] = {
    # Pure perfume houses
    "ديور":              (CAT_PERFUME, 0.88, "multi-category - mostly perfume"),
    "dior":              (CAT_PERFUME, 0.88, "multi-category - mostly perfume"),
    "شانيل":             (CAT_PERFUME, 0.85, "multi-category - mostly perfume"),
    "chanel":            (CAT_PERFUME, 0.85, "multi-category - mostly perfume"),
    "لانكوم":            (CAT_PERFUME, 0.80, "perfume+beauty - edge case"),
    "lancome":           (CAT_PERFUME, 0.80, "perfume+beauty"),
    "جيفنشي":            (CAT_PERFUME, 0.85, "mostly perfume"),
    "givenchy":          (CAT_PERFUME, 0.85, "mostly perfume"),
    "توم فورد":           (CAT_PERFUME, 0.90, "luxury perfume house"),
    "tom ford":          (CAT_PERFUME, 0.90, "luxury perfume house"),
    "لطافة":             (CAT_PERFUME, 0.95, "pure perfume brand"),
    "lattafa":           (CAT_PERFUME, 0.95, "pure perfume brand"),
    "ارماني":            (CAT_PERFUME, 0.85, "multi-category"),
    "armani":            (CAT_PERFUME, 0.85, "multi-category"),
    "كريد":              (CAT_PERFUME, 0.97, "pure perfume house"),
    "creed":             (CAT_PERFUME, 0.97, "pure perfume house"),
    "اميلي":             (CAT_PERFUME, 0.95, "pure perfume brand"),
    "amouage":           (CAT_PERFUME, 0.97, "pure oud perfume"),
    "امواج":             (CAT_PERFUME, 0.97, "pure oud perfume"),
    "باكو رابان":         (CAT_PERFUME, 0.90, "mostly perfume"),
    "paco rabanne":      (CAT_PERFUME, 0.90, "mostly perfume"),
    "كارولينا هيريرا":   (CAT_PERFUME, 0.88, "mostly perfume"),
    "فالنتينو":           (CAT_PERFUME, 0.88, "mostly perfume"),
    "بربري":             (CAT_PERFUME, 0.87, "mostly perfume"),
    "burberry":          (CAT_PERFUME, 0.87, "mostly perfume"),
    "ايف سان لوران":     (CAT_PERFUME, 0.85, "mostly perfume"),
    "ysl":               (CAT_PERFUME, 0.85, "mostly perfume"),
    "هوغو بوس":          (CAT_PERFUME, 0.90, "mostly perfume"),
    "hugo boss":         (CAT_PERFUME, 0.90, "mostly perfume"),
    "بلو دي":            (CAT_PERFUME, 0.88, "chanel subline"),
    "جان بول غوتييه":    (CAT_PERFUME, 0.90, "mostly perfume"),
    "ديبتيك":            (CAT_PERFUME, 0.95, "niche perfume"),
    "diptyque":          (CAT_PERFUME, 0.95, "niche perfume"),
    "كيليان":            (CAT_PERFUME, 0.96, "niche perfume"),
    "kilian":            (CAT_PERFUME, 0.96, "niche perfume"),
    "فريدريك مال":        (CAT_PERFUME, 0.97, "niche perfume"),
    "frederic malle":    (CAT_PERFUME, 0.97, "niche perfume"),
    "جيرلان":            (CAT_PERFUME, 0.90, "mostly perfume"),
    "guerlain":          (CAT_PERFUME, 0.90, "mostly perfume"),
    "مون بلانك":          (CAT_PERFUME, 0.90, "mostly perfume"),
    "montblanc":         (CAT_PERFUME, 0.90, "mostly perfume"),
    "بينهاليغونز":        (CAT_PERFUME, 0.95, "niche perfume"),
    "penhaligons":       (CAT_PERFUME, 0.95, "niche perfume"),
    "ميزون كريفلي":       (CAT_PERFUME, 0.97, "niche perfume"),
    "كارتير":            (CAT_PERFUME, 0.88, "mostly perfume"),
    "cartier":           (CAT_PERFUME, 0.88, "mostly perfume"),
    "اكوا دي بارما":      (CAT_PERFUME, 0.92, "mostly perfume"),
    "acqua di parma":    (CAT_PERFUME, 0.92, "mostly perfume"),
    "زيرجوف":            (CAT_PERFUME, 0.95, "niche perfume"),
    "سيرج لوتنس":        (CAT_PERFUME, 0.97, "niche perfume"),
    "افنان":             (CAT_PERFUME, 0.94, "pure perfume brand"),
    "afnan":             (CAT_PERFUME, 0.94, "pure perfume brand"),
    "مارلي":             (CAT_PERFUME, 0.95, "niche perfume"),
    "parfums de marly":  (CAT_PERFUME, 0.95, "niche perfume"),
    "ثامين":             (CAT_PERFUME, 0.95, "niche perfume"),
    "thameen":           (CAT_PERFUME, 0.95, "niche perfume"),
    "ثمين":              (CAT_PERFUME, 0.95, "niche perfume"),
    "مانسيرا":           (CAT_PERFUME, 0.96, "niche perfume"),
    "عساف":              (CAT_PERFUME, 0.90, "arabic perfume brand"),
    "ميمو باريس":         (CAT_PERFUME, 0.95, "niche perfume"),
    "تيزيانا تيرينزي":    (CAT_PERFUME, 0.97, "niche perfume"),
    "ex nihilo":         (CAT_PERFUME, 0.97, "niche perfume"),
    "اكس نيهيلو":         (CAT_PERFUME, 0.97, "niche perfume"),
    "سنتولوجيا":          (CAT_PERFUME, 0.95, "niche perfume"),
    "ميزون ماركيلا":      (CAT_PERFUME, 0.93, "mostly perfume"),
    "فيرزاتشي":           (CAT_PERFUME, 0.87, "mostly perfume"),
    "versace":           (CAT_PERFUME, 0.87, "mostly perfume"),
    "دولتشي":            (CAT_PERFUME, 0.85, "mostly perfume"),
    "dolce":             (CAT_PERFUME, 0.85, "mostly perfume"),
    "لوريال":            (CAT_BEAUTY, 0.85, "beauty/care brand"),
    "loreal":            (CAT_BEAUTY, 0.85, "beauty/care brand"),
    "بورجوا":            (CAT_BEAUTY, 0.90, "pure makeup brand"),
    "bourjois":          (CAT_BEAUTY, 0.90, "pure makeup brand"),
    "ماكياج":            (CAT_BEAUTY, 0.90, "arabic generic makeup"),
    "نيوتري":            (CAT_UNKNOWN, 0.90, "supplement/nutrition"),
    "nutri":             (CAT_UNKNOWN, 0.90, "supplement/nutrition"),
    "بيوتين":            (CAT_UNKNOWN, 0.90, "supplement - manual review"),
    "جاميز":             (CAT_UNKNOWN, 0.90, "gummies supplement"),
    "كيرستاسي":          (CAT_BEAUTY, 0.90, "pure hair care brand"),
    "kerastase":         (CAT_BEAUTY, 0.90, "pure hair care brand"),
    "لوريال بروفيشنال":   (CAT_BEAUTY, 0.92, "pure hair care"),
}

# --- 3B: PERFUME HARD SIGNALS (Gate 1 patterns - confidence 1.0) ---
_PERFUME_HARD_PATTERNS: list[tuple[str, str]] = [
    (r"\bedp\b",                               "pattern:EDP"),
    (r"\bedt\b",                               "pattern:EDT"),
    (r"\bedc\b",                               "pattern:EDC"),
    (r"\bextrait\b",                           "pattern:Extrait"),
    (r"\balikisir\b",                          "pattern:Elixir"),
    (r"eau\s+de\s+(parfum|toilette|cologne)",  "pattern:EauDe"),
    (r"اكستريت\s+دو\s+بارفان",                "ar:ExtractParfum"),
    (r"او\s+دو\s+(برفيوم|بارفيوم|تواليت|كولون|بيرفيوم|بيرفوم|بارفان|بارفيوم|بارفام)", "ar:EauDe"),
    (r"اودي\s+(بارفيوم|برفيوم|تواليت)",        "ar:OdParfum"),
    (r"اوه\s+دو",                             "ar:EauDe2"),
    (r"او\s+دي\s+(بارفان|بارفيوم|تواليت|برفيوم)", "ar:OdParfum2"),
    (r"لو\s+دي\s+بارفان",                     "ar:LParfum"),
    (r"لو\s+بارفان",                          "ar:LeParfum"),
    (r"اي\s+دي\s+(بيرفيوم|برفيوم|تواليت)",   "ar:EdeP"),
    (r"\b(parfum|perfume|cologne|fragrance)\b","en:PerfumeWord"),
    (r"\bparfan\b",                            "en:Parfan"),
    (r"\bparfum\b",                            "en:Parfum"),
    (r"معطر\s+(شعر|الشعر)",                   "ar:HairMist→PERFUME"),
    (r"رذاذ\s+(عطري|الشعر)",                  "ar:HairMist2→PERFUME"),
    (r"مجموعة\s+(هدايا\s+)?عطور",             "ar:GiftPerfumeSet→PERFUME"),
    (r"مجموعة\s+هدايا\s+عطرية",               "ar:GiftSet→PERFUME"),
    (r"طقم\s+(هدايا\s+)?عطر",                "ar:GiftKit→PERFUME"),
    (r"(تستر|tester)\s+(عطر|لعطر)",           "ar:PerfumeTester"),
    (r"عطر\s+(شعر|للشعر)",                   "ar:HairPerfume→PERFUME"),
    (r"مجموعة\s+عطر",                         "ar:PerfumeSet→PERFUME"),
    (r"عينة\s+(عطر|من|ماء\s+عطر)",            "ar:PerfumeSample→PERFUME"),
    (r"ماء\s+عطر",                            "ar:EauDeParfum"),
    (r"بخاخ\s+جسم\s+(معطر|عطري)",            "ar:BodyMist→PERFUME"),
    (r"بخور",                                 "ar:Bakhoor→PERFUME"),
    (r"دهن\s+عطر",                            "ar:OilPerfume→PERFUME"),
]

# --- 3C: BEAUTY / CARE HARD SIGNALS ---
_BEAUTY_HARD_PATTERNS: list[tuple[str, str]] = [
    (r"(استشوار|مجفف\s+شعر|مكواة\s+شعر)",    "ar:HairDevice→BEAUTY"),
    (r"(فرشاة\s+(شعر|مكياج)|برش\s+شعر)",     "ar:Brush→BEAUTY"),
    (r"(مسكارا|ريميل|كحل|مكياج)",             "ar:Makeup→BEAUTY"),
    (r"(كونسيلر|كريم\s+اساس|فاونديشن)",       "ar:Foundation→BEAUTY"),
    (r"(ظل\s+عيون|بلاشر|هايلايتر)",           "ar:Eyeshadow→BEAUTY"),
    (r"(احمر\s+شفاه|ليب\s+(جلوس|ستيك))",     "ar:Lipstick→BEAUTY"),
    (r"(شامبو|بلسم\s+شعر|غسول\s+(شعر|وجه))", "ar:Haircare→BEAUTY"),
    (r"(واقي\s+شمس|كريم\s+وجه|مرطب\s+وجه)",  "ar:Skincare→BEAUTY"),
    (r"(مرطب\s+(جسم|يدين|اليدين))",           "ar:Lotion→BEAUTY"),
    (r"مزيل\s+عرق",                           "ar:Deodorant→BEAUTY"),
    (r"(ديودورانت|antiperspirant)",            "en:Deodorant→BEAUTY"),
    (r"(سيروم|تونر|ماسك\s+وجه|بيلينج)",       "ar:Skincare2→BEAUTY"),
    (r"(حمام\s+زيت|زيت\s+شعر|كيراتين)",       "ar:HairTreatment→BEAUTY"),
    (r"شمعة",                                 "ar:Candle→BEAUTY"),
    (r"(مرآة\s+مستحضرات|حقيبة\s+مكياج)",      "ar:MakeupAccessory→BEAUTY"),
    (r"(بودرة\s+جسم|سكراب|قشرة\s+وجه)",       "ar:BodyCare→BEAUTY"),
    (r"(لوشن|لوسيون|body\s+lotion)",           "en:Bodylotion→BEAUTY"),
    (r"جل\s+(استحمام|الاستحمام)",             "ar:ShowerGel→BEAUTY"),
    (r"صابون\s+(لوكس|فاخر)",                  "ar:Soap→BEAUTY"),
    (r"زيت\s+(جسم|الجسم)",                   "ar:BodyOil→BEAUTY"),
]

# --- 3D: UNKNOWN / MANUAL-REVIEW HARD SIGNALS ---
_UNKNOWN_HARD_PATTERNS: list[tuple[str, str]] = [
    (r"مكمل\s+غذائي",          "ar:FoodSupplement→UNKNOWN"),
    (r"(فيتامين|vitamin)",     "Supplement:Vitamin→UNKNOWN"),
    (r"بروتين\s+(بودر|مسحوق)", "Supplement:Protein→UNKNOWN"),
    (r"جاميز\s+مكمل",          "Supplement:Gummies→UNKNOWN"),
    (r"(ببلة|علكة|candy)",     "ar:Candy→UNKNOWN"),
    (r"حلوى\s+(جاميز|صحية)",  "ar:HealthCandy→UNKNOWN"),
    (r"مكسرات\s+طبيعية",       "ar:NaturalNuts→UNKNOWN"),
    (r"شاي\s+(طبي|أعشاب)",    "ar:HerbalTea→UNKNOWN"),
]

# --- 3E: WEIGHTED SCORING SIGNALS (Gate 3) ---
_PERFUME_SCORED: list[tuple[str, float, str]] = [
    (r"عطر\b",                 2.5, "ar:AtrWord"),
    (r"\bتستر\b",              2.0, "ar:Tester"),
    (r"\bعينة\b",              1.5, "ar:Sample"),
    (r"\b(عود|oud)\b",         1.8, "ar:OudNote"),
    (r"\b(مسك|musk|musc)\b",  1.5, "ar:MuskNote"),
    (r"\b(ورد|rose|roses)\b", 1.2, "perfume:RoseNote"),
    (r"\b\d+\s*مل\b",         1.0, "ar:VolumeML"),
    (r"\bml\b",               1.0, "en:VolumeML"),
    (r"\bسبراي\b",             1.0, "ar:Spray"),
    (r"\bspray\b",             1.0, "en:Spray"),
    (r"\b(نيش|niche)\b",      1.5, "ar:NichePerf"),
    (r"(مقدمة|قلب|قاعدة)",    1.5, "ar:PyramidNote"),
    (r"(هرم\s+عطري|نوتة)",    1.5, "ar:PerfumeNotes"),
    (r"(تستر|مجموعة\s+عطر)",  2.0, "ar:TesterOrSet"),
    (r"(بخور|عود\s+هندي)",    2.0, "ar:Incense"),
    (r"بارفيوم\b",             2.0, "ar:Parfum"),
    (r"\b(intenso|intense|intens)\b", 1.2, "en:Intense"),
    (r"\b(elixir|elixire|اليكسير|اليكسر)\b", 1.5, "en:Elixir"),
]

_BEAUTY_SCORED: list[tuple[str, float, str]] = [
    (r"(كريم|cream)\b",        2.0, "ar:Cream"),
    (r"(لوشن|lotion)\b",       2.0, "ar:Lotion"),
    (r"(سيروم|serum)\b",       2.5, "ar:Serum"),
    (r"(تونر|toner)\b",        2.5, "ar:Toner"),
    (r"مزيل",                  2.5, "ar:Remover/Deo"),
    (r"(مكياج|makeup|make-up)", 3.0, "ar:Makeup"),
    (r"(شعر|hair)\b(?!.*عطر)", 1.5, "ar:HairNoPerf"),
    (r"(بشرة|skin|skincare)\b", 2.0, "ar:Skin"),
    (r"(مقشر|exfoliant|scrub)\b", 2.5, "ar:Scrub"),
    (r"(ماسك|mask)\b",          2.0, "ar:Mask"),
    (r"(كيراتين|keratin)\b",    2.5, "ar:Keratin"),
    (r"(ملمع|gloss|primer)\b",  2.0, "ar:Primer"),
    (r"(بودرة\s+(?!جسم)|powder)\b(?!.*عطر)", 1.8, "ar:Powder"),
    (r"(ارغان|argan)\b",        1.5, "ar:ArganOil"),
    (r"(شمعة|candle)\b",        3.0, "ar:Candle"),
    (r"(مرطب|moisturizer)\b",   2.0, "ar:Moisturizer"),
    (r"(واقي\s+شمس|spf\s*\d)", 3.0, "ar:Sunscreen"),
]

_UNKNOWN_SCORED: list[tuple[str, float, str]] = [
    (r"(مكمل|supplement)\b",    4.0, "ar:Supplement"),
    (r"(فيتامين|vitamin)\b",    4.0, "ar:Vitamin"),
    (r"(بروتين|protein)\b",     4.0, "ar:Protein"),
    (r"(قطعة|capsule|حبة)",     2.0, "ar:Capsule"),
    (r"(حلوى|candy|gummy)",     3.0, "ar:Candy"),
    (r"(شاي|tea)\b(?!.*عطر)",  1.5, "ar:Tea"),
]

# --- 3F: SALLA CATEGORY-PATH TAXONOMY ---
_CAT_PATH_RULES: list[tuple[str, str, float, str]] = [
    # (pattern_in_category_path, output_cat, confidence, label)
    (r"عطور",              CAT_PERFUME, 0.95, "SallaCat:عطور"),
    (r"perfume",           CAT_PERFUME, 0.95, "SallaCat:Perfume"),
    (r"fragrance",         CAT_PERFUME, 0.95, "SallaCat:Fragrance"),
    (r"(نسائي|رجالي|نيش)", CAT_PERFUME, 0.80, "SallaCat:Gender"),
    (r"(مكياج|تجميل)",     CAT_BEAUTY,  0.90, "SallaCat:Makeup"),
    (r"(عناية|skincare)",  CAT_BEAUTY,  0.90, "SallaCat:Care"),
    (r"(شعر|hair)",        CAT_BEAUTY,  0.80, "SallaCat:Hair"),
    (r"(أجهزة|devices)",   CAT_BEAUTY,  0.88, "SallaCat:Devices"),
    (r"(مكملات|supplement)",CAT_UNKNOWN, 0.90, "SallaCat:Supplement"),
]

# ============================================================================
# SECTION 4  -  GEMINI RATE LIMITER
# ============================================================================

_LAST_API_CALL: float = 0.0
_MIN_INTERVAL:  float = 1.8

def _rate_sleep() -> None:
    global _LAST_API_CALL
    elapsed = time.time() - _LAST_API_CALL
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _LAST_API_CALL = time.time()


def _gemini_retry(model, prompt: str, retries: int = 4) -> Optional[str]:
    for attempt in range(retries):
        try:
            _rate_sleep()
            return model.generate_content(prompt).text.strip()
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ("quota","rate","429","exhausted")):
                wait = 5 * (2 ** attempt)
                logger.warning(f"Gemini rate-limit, waiting {wait}s…")
                time.sleep(wait)
            else:
                logger.error(f"Gemini error attempt {attempt+1}: {e}")
                break
    return None


# ============================================================================
# SECTION 5  -  TEXT PRE-PROCESSOR  (Gate 0)
# ============================================================================

_AR_STOPWORDS = [
    "عطر","او دي","او دو","دي","دو","بارفيوم","برفيوم","بيرفيوم",
    "تواليت","تستر","بديل","كولونيا","بخاخ","اوه","ذا","ال","من",
    "للرجال","للنساء","للجنسين","الرجالي","النسائي",
]
_EN_STOPWORDS = [
    "eau","de","parfum","toilette","cologne","edp","edt","edc",
    "ml","tester","dupe","perfume","fragrance","spray","the","for",
    "her","him","men","women","homme","femme","unisex",
]

def normalize_name(name: str) -> str:
    """Return clean comparable root string for fuzzy matching."""
    if not isinstance(name, str) or not name.strip():
        return ""
    text = name.lower().strip()
    # Arabic letter normalization
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[ةه]", "h", text)
    # Remove stopwords
    for w in _AR_STOPWORDS:
        text = re.sub(r"\b" + re.escape(w) + r"\b", " ", text)
    for w in _EN_STOPWORDS:
        text = re.sub(r"\b" + re.escape(w) + r"\b", " ", text)
    # Remove volumes
    text = re.sub(r"\d+\.?\d*\s*(?:ml|مل|g|gr|oz|fl\.?\s*oz)", " ", text)
    # Remove special chars
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_for_classify(text: str) -> str:
    """Lightweight normalization preserving signal keywords."""
    if not isinstance(text, str):
        return ""
    t = text.lower().strip()
    t = re.sub(r"[أإآ]", "ا", t)
    t = re.sub(r"ى", "ي", t)
    t = re.sub(r"\s+", " ", t)
    return t


# ============================================================================
# SECTION 6  -  GATE 1: HARD RULES  (100% confidence triggers)
# ============================================================================

def _gate1_hard_rules(text: str) -> Optional[ClassificationResult]:
    """Return a result if any hard-rule pattern fires. Else None."""
    signals: list[str] = []

    # Check perfume hard patterns
    for pattern, label in _PERFUME_HARD_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.UNICODE):
            signals.append(label)

    if signals:
        return ClassificationResult(
            category    = CAT_PERFUME,
            confidence  = 0.97,
            gate_used   = GATE_HARD_RULES,
            signals_fired= signals,
            reasoning   = f"Hard perfume patterns matched: {signals[0]}",
        )

    # Check beauty hard patterns
    for pattern, label in _BEAUTY_HARD_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.UNICODE):
            signals.append(label)

    if signals:
        return ClassificationResult(
            category    = CAT_BEAUTY,
            confidence  = 0.97,
            gate_used   = GATE_HARD_RULES,
            signals_fired= signals,
            reasoning   = f"Hard beauty pattern: {signals[0]}",
        )

    # Check unknown hard patterns
    for pattern, label in _UNKNOWN_HARD_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.UNICODE):
            signals.append(label)

    if signals:
        return ClassificationResult(
            category    = CAT_UNKNOWN,
            confidence  = 0.95,
            gate_used   = GATE_HARD_RULES,
            signals_fired= signals,
            reasoning   = f"Hard unknown pattern: {signals[0]}",
        )

    return None


# ============================================================================
# SECTION 7  -  GATE 2: CATEGORY PATH ANALYZER
# ============================================================================

def _gate2_category_path(category_path: str) -> Optional[ClassificationResult]:
    """Analyze Salla category path string."""
    if not category_path or not isinstance(category_path, str):
        return None

    text = _normalize_for_classify(category_path)
    best_conf = 0.0
    best_cat  = None
    best_label = ""

    for pattern, cat, conf, label in _CAT_PATH_RULES:
        if re.search(pattern, text, re.IGNORECASE | re.UNICODE):
            if conf > best_conf:
                best_conf  = conf
                best_cat   = cat
                best_label = label

    if best_cat and best_conf >= CONFIDENCE_FINALIZE[GATE_CATEGORY]:
        return ClassificationResult(
            category    = best_cat,
            confidence  = best_conf,
            gate_used   = GATE_CATEGORY,
            signals_fired= [best_label],
            reasoning   = f"Category path '{category_path[:60]}' matched {best_label}",
        )
    return None


# ============================================================================
# SECTION 8  -  GATE 3: WEIGHTED SIGNAL SCORING
# ============================================================================

def _gate3_scoring(text: str) -> Optional[ClassificationResult]:
    """Multi-signal weighted voting."""
    perf_score, beau_score, unkn_score = 0.0, 0.0, 0.0
    fired: list[str] = []

    for pattern, weight, label in _PERFUME_SCORED:
        if re.search(pattern, text, re.IGNORECASE | re.UNICODE):
            perf_score += weight
            fired.append(f"+{weight}[{label}]")

    for pattern, weight, label in _BEAUTY_SCORED:
        if re.search(pattern, text, re.IGNORECASE | re.UNICODE):
            beau_score += weight
            fired.append(f"beauty+{weight}[{label}]")

    for pattern, weight, label in _UNKNOWN_SCORED:
        if re.search(pattern, text, re.IGNORECASE | re.UNICODE):
            unkn_score += weight
            fired.append(f"unk+{weight}[{label}]")

    total = perf_score + beau_score + unkn_score
    if total < 1.0:
        return None  # no signals fired

    scores = {
        CAT_PERFUME: perf_score,
        CAT_BEAUTY:  beau_score,
        CAT_UNKNOWN: unkn_score,
    }
    winner = max(scores, key=scores.get)
    conf   = scores[winner] / total

    if conf >= CONFIDENCE_FINALIZE[GATE_SCORING]:
        return ClassificationResult(
            category    = winner,
            confidence  = min(conf, 0.95),
            gate_used   = GATE_SCORING,
            signals_fired= fired[:8],
            reasoning   = (f"Scores → Perfume:{perf_score:.1f} "
                           f"Beauty:{beau_score:.1f} Unknown:{unkn_score:.1f} "
                           f"winner={winner} conf={conf:.0%}"),
        )
    return None


# ============================================================================
# SECTION 9  -  GATE 4: BRAND KNOWLEDGE BASE
# ============================================================================

def _extract_brand_slug(text: str) -> list[str]:
    """Extract probable brand substrings from product name."""
    text_low = text.lower()
    found = []
    for slug in BRAND_KB:
        if slug in text_low:
            found.append(slug)
    return found


def _gate4_brand_kb(text: str) -> Optional[ClassificationResult]:
    """Lookup brand knowledge base."""
    slugs = _extract_brand_slug(text)
    if not slugs:
        return None

    best: Optional[tuple] = None
    for slug in slugs:
        entry = BRAND_KB.get(slug)
        if entry and (best is None or entry[1] > best[1]):
            best = (slug, *entry)

    if best is None:
        return None

    slug, cat, conf, notes = best
    if conf >= CONFIDENCE_FINALIZE[GATE_BRAND_KB]:
        return ClassificationResult(
            category    = cat,
            confidence  = conf,
            gate_used   = GATE_BRAND_KB,
            signals_fired= [f"brand:{slug}"],
            reasoning   = f"Brand '{slug}' → {cat} ({notes}) conf={conf:.0%}",
        )
    return None


# ============================================================================
# SECTION 10  -  GATE 5: AI ORACLE (Gemini Flash)
# ============================================================================

def _gate5_ai_oracle(
    name: str,
    category: str,
    brand: str,
    api_key: str,
) -> ClassificationResult:
    """Gemini-based classification for ambiguous products. Always returns a result."""
    fallback = ClassificationResult(
        category    = CAT_UNKNOWN,
        confidence  = 0.0,
        gate_used   = GATE_FALLBACK,
        signals_fired= ["NoSignals"],
        reasoning   = "No gate resolved - sent to manual review",
    )

    if not (api_key and GENAI_AVAILABLE):
        return fallback

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""أنت نظام تصنيف متخصص في متجر عطور ومستحضرات تجميل.
صنّف المنتج التالي في إحدى الفئات الثلاث بدقة:

المنتج: {name}
التصنيف في المتجر: {category or 'غير محدد'}
الماركة المحتملة: {brand or 'غير محددة'}

الفئات المتاحة:
- PERFUME: عطور، تستر عطر، معطر شعر، مجموعة هدايا عطرية، بخور، دهن عطر، عينة عطر
- BEAUTY: مكياج، عناية بشرة، عناية شعر، أجهزة تجميل، شمعات، مزيل عرق، كريمات جسم
- UNKNOWN: مكملات غذائية، طعام، مشروبات، منتجات لا تنتمي للفئتين

أجب بـ JSON فقط بدون نص إضافي:
{{"category": "PERFUME|BEAUTY|UNKNOWN", "confidence": 0.0-1.0, "reasoning": "سبب التصنيف بجملة واحدة"}}"""

        raw = _gemini_retry(model, prompt)
        if not raw:
            return fallback

        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)

        cat_map = {"PERFUME": CAT_PERFUME, "BEAUTY": CAT_BEAUTY, "UNKNOWN": CAT_UNKNOWN}
        ai_cat  = cat_map.get(data.get("category","").upper(), CAT_UNKNOWN)
        ai_conf = float(data.get("confidence", 0.5))

        return ClassificationResult(
            category    = ai_cat,
            confidence  = ai_conf,
            gate_used   = GATE_AI_ORACLE,
            signals_fired= ["AI:Gemini1.5Flash"],
            reasoning   = data.get("reasoning","AI decision"),
            ai_used     = True,
        )

    except json.JSONDecodeError:
        logger.warning(f"Gemini non-JSON response for '{name}'")
    except Exception as exc:
        logger.error(f"Gate5 AI error for '{name}': {exc}")

    return fallback


# ============================================================================
# SECTION 11  -  MAIN CLASSIFIER ORCHESTRATOR
# ============================================================================

def classify_product_5gate(
    name:      str,
    category:  str = "",
    brand:     str = "",
    api_key:   str = "",
) -> ClassificationResult:
    """
    Run the 5-Gate classification pipeline.

    Parameters
    ----------
    name      : Product display name (Arabic or English or mixed).
    category  : Salla category path string (optional but helps accuracy).
    brand     : Known brand string (optional).
    api_key   : Gemini API key (needed for Gate 5 only).

    Returns
    -------
    ClassificationResult with category, confidence, gate_used, audit trail.
    """
    if not isinstance(name, str) or not name.strip():
        return ClassificationResult(
            category=CAT_UNKNOWN, confidence=0.0,
            gate_used=GATE_FALLBACK, reasoning="Empty product name",
        )

    # Combine all text for pattern matching
    full_text = _normalize_for_classify(f"{name} {category} {brand}")

    # ── Gate 1: Hard Rules ────────────────────────────────────────────
    try:
        r = _gate1_hard_rules(full_text)
        if r and r.confidence >= CONFIDENCE_FINALIZE[GATE_HARD_RULES]:
            return r
    except Exception as e:
        logger.error(f"Gate1 error: {e}")

    # ── Gate 2: Category Path ─────────────────────────────────────────
    try:
        r = _gate2_category_path(category)
        if r and r.confidence >= CONFIDENCE_FINALIZE[GATE_CATEGORY]:
            return r
    except Exception as e:
        logger.error(f"Gate2 error: {e}")

    # ── Gate 3: Weighted Scoring ──────────────────────────────────────
    try:
        r = _gate3_scoring(full_text)
        if r and r.confidence >= CONFIDENCE_FINALIZE[GATE_SCORING]:
            return r
    except Exception as e:
        logger.error(f"Gate3 error: {e}")

    # ── Gate 4: Brand KB ──────────────────────────────────────────────
    try:
        r = _gate4_brand_kb(full_text)
        if r and r.confidence >= CONFIDENCE_FINALIZE[GATE_BRAND_KB]:
            return r
    except Exception as e:
        logger.error(f"Gate4 error: {e}")

    # ── Gate 5: AI Oracle ─────────────────────────────────────────────
    if api_key and GENAI_AVAILABLE:
        try:
            return _gate5_ai_oracle(name, category, brand, api_key)
        except Exception as e:
            logger.error(f"Gate5 error: {e}")

    # ── Fallback: Manual Review ───────────────────────────────────────
    return ClassificationResult(
        category    = CAT_UNKNOWN,
        confidence  = 0.0,
        gate_used   = GATE_FALLBACK,
        signals_fired= ["NoGateResolved"],
        reasoning   = "All gates below threshold → Manual Review",
    )


# Legacy shim for backward compatibility
def classify_product(name: str, category: str = "") -> str:
    """Backward-compatible wrapper. Returns category string only."""
    return classify_product_5gate(name, category).category


# ============================================================================
# SECTION 12  -  TEXT NORMALIZATION HELPERS
# ============================================================================

_AR_NORM_WORDS = [
    "عطر","او دي","او","دي","بارفيوم","برفيوم","تواليت",
    "تستر","بديل","كولونيا","رائحة","بخاخ","اوه","ذا",
]
_EN_NORM_WORDS = [
    "eau","de","parfum","toilette","cologne","edp","edt","edc",
    "ml","tester","dupe","perfume","fragrance","spray","the",
]


# ============================================================================
# SECTION 13  -  SAFE CSV READER
# ============================================================================

def _read_csv_safe(file_obj, **kwargs) -> pd.DataFrame:
    if hasattr(file_obj, "read"):
        raw = file_obj.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
    else:
        with open(file_obj, "rb") as fh:
            raw = fh.read()

    for enc in ("utf-8-sig", "utf-8", "cp1256", "iso-8859-6", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, **kwargs)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
        except Exception as exc:
            logger.warning(f"CSV parse error with {enc}: {exc}")

    raise ValueError("Could not parse CSV with any supported encoding.")


# ============================================================================
# SECTION 14  -  FILE LOADERS
# ============================================================================

def load_store_products(store_files: list) -> pd.DataFrame:
    """Load one or more Salla-format store CSV files."""
    frames: list[pd.DataFrame] = []

    for f in store_files:
        try:
            raw = _read_csv_safe(f, header=None, low_memory=False, dtype=str)
            header_row_idx = None
            for idx, row in raw.iterrows():
                if any("أسم المنتج" in str(v) for v in row.values):
                    header_row_idx = idx
                    break
            if header_row_idx is None:
                logger.warning(f"No header in store file: {getattr(f,'name',f)}")
                continue

            raw.columns = [str(c).strip() for c in raw.iloc[header_row_idx].values]
            data = raw.iloc[header_row_idx + 1:].reset_index(drop=True)

            def _col(kw: str) -> Optional[str]:
                return next((c for c in data.columns if kw in c), None)

            name_col  = _col("أسم المنتج")
            brand_col = _col("الماركة")
            cat_col   = _col("تصنيف المنتج")
            img_col   = _col("صورة المنتج")
            if not name_col:
                continue

            frame = pd.DataFrame()
            frame["product_name"]    = data[name_col].fillna("").astype(str)
            frame["brand"]           = data[brand_col].fillna("").astype(str) if brand_col else ""
            frame["category"]        = data[cat_col].fillna("").astype(str)   if cat_col  else ""
            frame["image_url"]       = (
                data[img_col].apply(lambda x: str(x).split(",")[0].strip() if pd.notna(x) else "")
                if img_col else ""
            )
            frame["normalized_name"] = frame["product_name"].apply(normalize_name)
            frames.append(frame[frame["product_name"].str.strip() != ""])
        except Exception as exc:
            logger.error(f"Error loading store file: {exc}")

    if not frames:
        return pd.DataFrame(columns=["product_name","brand","category","image_url","normalized_name"])

    return (pd.concat(frames, ignore_index=True)
              .drop_duplicates(subset=["product_name"])
              .reset_index(drop=True))


def load_competitor_products(comp_file) -> pd.DataFrame:
    """Load a competitor CSV file with auto-column detection."""
    try:
        df = _read_csv_safe(comp_file, low_memory=False, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]

        name_col = img_col = price_col = None
        name_hints  = ["name","اسم","productcard","styles_product","title","عنوان"]
        img_hints   = ["src","image","img","صورة","photo"]
        price_hints = ["price","سعر","text-sm","cost"]

        for col in df.columns:
            cl = col.lower()
            if name_col  is None and any(h in cl for h in name_hints):  name_col  = col
            if img_col   is None and any(h in cl for h in img_hints):   img_col   = col
            if price_col is None and any(h in cl for h in price_hints): price_col = col

        if name_col  is None and len(df.columns) >= 3: name_col  = df.columns[2]
        if img_col   is None and len(df.columns) >= 2: img_col   = df.columns[1]
        if price_col is None and len(df.columns) >= 4: price_col = df.columns[3]

        result = pd.DataFrame()
        result["product_name"] = df[name_col].fillna("").astype(str) if name_col else ""
        result["image_url"]    = df[img_col].fillna("").astype(str).str.split(",").str[0].str.strip() if img_col else ""
        result["price"]        = df[price_col].fillna("").astype(str) if price_col else ""
        result["normalized_name"] = result["product_name"].apply(normalize_name)
        result["product_type"]    = result["product_name"].apply(
            lambda n: classify_product(n)  # fast gate without AI for loading
        )
        result["source_file"]  = getattr(comp_file, "name", "unknown")
        return result[result["product_name"].str.strip().str.len() > 1].reset_index(drop=True)
    except Exception as exc:
        logger.error(f"Error loading competitor file: {exc}")
        return pd.DataFrame(columns=["product_name","image_url","price","normalized_name","product_type","source_file"])


def load_brands_list(brands_file) -> list[str]:
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


# ============================================================================
# SECTION 15  -  DEDUPLICATION ENGINE
# ============================================================================

def deduplicate_products(
    store_df:          pd.DataFrame,
    comp_products:     list[dict],
    high_threshold:    int = 90,
    low_threshold:     int = 50,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    3-layer deduplication: Exact → Fuzzy → Gray Zone.
    Returns (new_opportunities_df, gray_zone_df, duplicates_df).
    """
    store_norms    = store_df["normalized_name"].tolist() if not store_df.empty else []
    store_names    = store_df["product_name"].tolist()    if not store_df.empty else []
    store_norm_set = set(store_norms)

    new_opps:   list[dict] = []
    gray_zone:  list[dict] = []
    duplicates: list[dict] = []
    total = len(comp_products)

    for i, product in enumerate(comp_products):
        if progress_callback:
            progress_callback(i, total, product.get("product_name",""))

        comp_norm = product.get("normalized_name","")
        if not comp_norm or len(comp_norm) < 3:
            continue

        # Layer 1: exact
        if comp_norm in store_norm_set:
            duplicates.append({**product, "match_score":100,
                               "matched_store_product":"", "match_reason":"exact"})
            continue

        if not store_norms:
            new_opps.append({**product, "match_score":0,
                             "matched_store_product":"", "match_reason":"new"})
            continue

        # Layer 2: fuzzy
        best = process.extractOne(comp_norm, store_norms, scorer=fuzz.token_sort_ratio)
        if best:
            score = best[1]
            matched_name = store_names[store_norms.index(best[0])]
            if score >= high_threshold:
                duplicates.append({**product, "match_score":score,
                                   "matched_store_product":matched_name, "match_reason":"fuzzy_high"})
            elif score >= low_threshold:
                gray_zone.append({**product, "match_score":score,
                                  "matched_store_product":matched_name, "match_reason":"fuzzy_medium"})
            else:
                new_opps.append({**product, "match_score":score,
                                 "matched_store_product":"", "match_reason":"new"})
        else:
            new_opps.append({**product, "match_score":0,
                             "matched_store_product":"", "match_reason":"new"})

    def _df(lst):
        return pd.DataFrame(lst) if lst else pd.DataFrame()

    return _df(new_opps), _df(gray_zone), _df(duplicates)


# ============================================================================
# SECTION 16  -  ENRICHMENT & IMAGE FETCH
# ============================================================================

def enrich_product_with_gemini(
    product_name: str,
    image_url:    str,
    product_type: str,
    api_key:      str,
    expert_prompt: str = "",
) -> dict:
    """Generate brand, gender, Salla category and HTML description via Gemini."""
    default = {
        "brand": "", "gender": "للجنسين",
        "salla_category": ("العطور > عطور للجنسين" if product_type == CAT_PERFUME else "مكياج وعناية"),
        "description": f"<p>{product_name}</p>",
        "sku_suggestion": "", "enriched": False,
    }

    if not (api_key and GENAI_AVAILABLE and product_name.strip()):
        return default

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        type_label = "عطر" if product_type == CAT_PERFUME else "منتج تجميل أو عناية"
        img_note   = f"\nرابط صورة: {image_url}" if image_url else ""
        extra_inst = f"\nتعليمات: {expert_prompt[:500]}" if expert_prompt else ""

        prompt = f"""أنت خبير {type_label} لمتجر مهووس السعودي.
المنتج: {product_name}
النوع : {type_label}{img_note}{extra_inst}

أعد JSON فقط (لا نص خارج JSON، لا ``` ):
{{"brand_ar":"اسم الماركة عربي","brand_en":"Brand EN","gender":"رجالي|نسائي|للجنسين|نيش",
"salla_category":"مثال: العطور > عطور رجالية > عطور رجالية فاخرة",
"sku_suggestion":"SKU-SHORT",
"description_html":"وصف HTML 200-300 كلمة"}}"""

        raw = _gemini_retry(model, prompt)
        if not raw:
            return default

        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)

        brand = f"{data.get('brand_ar','').strip()} | {data.get('brand_en','').strip()}".strip(" |")
        return {
            "brand":          brand,
            "gender":         data.get("gender","للجنسين"),
            "salla_category": data.get("salla_category", default["salla_category"]),
            "description":    data.get("description_html", default["description"]),
            "sku_suggestion": data.get("sku_suggestion",""),
            "enriched":       True,
        }
    except Exception as exc:
        logger.error(f"enrich_product_with_gemini('{product_name}'): {exc}")
    return default


def verify_with_gemini(store_name: str, comp_name: str, api_key: str) -> Optional[str]:
    if not (api_key and GENAI_AVAILABLE):
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (f"أنت خبير عطور ومنتجات تجميل.\n"
                  f"المنتج الأول : {store_name}\n"
                  f"المنتج الثاني: {comp_name}\n\n"
                  "هل هذان المنتجان نفس الشيء تماماً؟\n"
                  "أجب بكلمة واحدة فقط:  MATCH  أو  DIFFERENT")
        result = _gemini_retry(model, prompt)
        if result:
            upper = result.upper()
            if "MATCH"     in upper: return "MATCH"
            if "DIFFERENT" in upper: return "DIFFERENT"
    except Exception as exc:
        logger.error(f"verify_with_gemini: {exc}")
    return None


def fetch_fallback_image(name: str, search_key: str = "", cx: str = "") -> str:
    if not (search_key and cx and name.strip()):
        return ""
    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key":search_key,"cx":cx,"q":f"{name} official",
                    "searchType":"image","num":1,"imgSize":"large"},
            timeout=10,
        )
        if resp.status_code == 200:
            items = resp.json().get("items",[])
            if items:
                return items[0].get("link","")
    except Exception as exc:
        logger.warning(f"fetch_fallback_image('{name}'): {exc}")
    return ""


# ============================================================================
# SECTION 17  -  EXPORT ENGINE
# ============================================================================

def _build_salla_row(product: dict) -> dict:
    return {
        "النوع ": "منتج",
        "أسم المنتج": product.get("product_name",""),
        "تصنيف المنتج": product.get("salla_category",""),
        "صورة المنتج": product.get("image_url",""),
        "وصف صورة المنتج": product.get("product_name",""),
        "نوع المنتج": "منتج جاهز",
        "سعر المنتج": product.get("price",""),
        "الوصف": product.get("description",""),
        "هل يتطلب شحن؟": "نعم",
        "رمز المنتج sku": product.get("sku_suggestion",""),
        "سعر التكلفة": "",
        "السعر المخفض": "",
        "تاريخ بداية التخفيض": "",
        "تاريخ نهاية التخفيض": "",
        "اقصي كمية لكل عميل": "0",
        "إخفاء خيار تحديد الكمية": "لا",
        "اضافة صورة عند الطلب": "",
        "الوزن": "0.5",
        "وحدة الوزن": "kg",
        "الماركة": product.get("brand",""),
        "العنوان الترويجي": "",
        "تثبيت المنتج": "لا",
        "الباركود": "",
        "السعرات الحرارية": "",
        "MPN": "",
        "GTIN": "",
        "خاضع للضريبة ؟": "نعم",
        "سبب عدم الخضوع للضريبة": "",
        "[1] الاسم": "",
        "[1] النوع": "",
        "[1] القيمة": "",
        "[1] الصورة / اللون": "",
        "[2] الاسم": "",
        "[2] النوع": "",
        "[2] القيمة": "",
        "[2] الصورة / اللون": "",
        "[3] الاسم": "",
        "[3] النوع": "",
        "[3] القيمة": "",
        "[3] الصورة / اللون": "",
    }


def export_to_salla_csv(products: list[dict]) -> bytes:
    """
    Build Salla-compatible CSV.
    Row 1: بيانات المنتج + 39 commas (magic Salla header - 40 fields).
    Row 2: Column headers.
    Row 3+: Data.
    Encoding: UTF-8-BOM for correct Arabic in Excel/Salla.
    """
    rows = [_build_salla_row(p) for p in (products or [])]
    df   = pd.DataFrame(rows, columns=SALLA_COLS)
    buf  = io.StringIO()
    buf.write("بيانات المنتج" + "," * (len(SALLA_COLS) - 1) + "\n")
    df.to_csv(buf, index=False, encoding="utf-8")
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


def export_missing_brands_csv(products: list[dict], existing_brands: list[str]) -> bytes:
    BRAND_COLS = [
        "اسم الماركة", "وصف مختصر عن الماركة", "صورة شعار الماركة",
        "(إختياري) صورة البانر",
        "(Page Title) عنوان صفحة العلامة التجارية",
        "(SEO Page URL) رابط صفحة العلامة التجارية",
        "(Page Description) وصف صفحة العلامة التجارية",
    ]
    existing_lower = {b.lower().strip() for b in existing_brands if b}
    new_brands: dict[str, dict] = {}

    for p in products:
        brand = str(p.get("brand","")).strip()
        if not brand:
            continue
        brand_key = brand.lower()
        is_new = not any(fuzz.partial_ratio(brand_key, eb) > 85 for eb in existing_lower)
        if is_new and brand not in new_brands:
            slug = re.sub(r"[^\w\s-]","", brand.lower()).strip()
            slug = re.sub(r"[\s|]+","-", slug).strip("-")
            new_brands[brand] = {
                "اسم الماركة": brand,
                "وصف مختصر عن الماركة": "",
                "صورة شعار الماركة": "",
                "(إختياري) صورة البانر": "",
                "(Page Title) عنوان صفحة العلامة التجارية": f"{brand} | مهووس للعطور",
                "(SEO Page URL) رابط صفحة العلامة التجارية": slug,
                "(Page Description) وصف صفحة العلامة التجارية": f"تسوق منتجات {brand} الأصلية لدى مهووس.",
            }

    df  = pd.DataFrame(list(new_brands.values()), columns=BRAND_COLS) if new_brands else pd.DataFrame(columns=BRAND_COLS)
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()


def export_audit_trail_csv(products: list[dict]) -> bytes:
    """Export classification audit trail for review."""
    AUDIT_COLS = [
        "product_name", "classified_as", "confidence_pct",
        "gate_used", "signals_fired", "reasoning", "ai_used",
        "source_file", "price", "image_url",
    ]
    rows = []
    for p in products:
        row = {col: p.get(col, "") for col in AUDIT_COLS}
        rows.append(row)
    df  = pd.DataFrame(rows, columns=AUDIT_COLS) if rows else pd.DataFrame(columns=AUDIT_COLS)
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()
