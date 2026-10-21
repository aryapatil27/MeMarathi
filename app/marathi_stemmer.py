
"""
marathi_stemmer.py
------------------
Rule-based Marathi stemmer and dictionary-based lemmatizer.

Functions:
    stem_word(word)
    lemmatize_word(word)
    stem_text(text)
    lemmatize_text(text)
    analyze_word(word)
"""

import re
from functools import lru_cache
from typing import Dict


# --------------------------------------------------
# 1. Regular expressions and helper functions
# --------------------------------------------------

# Match Devanagari words, including combining marks and virama.
_DEVANAGARI_WORD = re.compile(
    r"[\u0900-\u097F]+"
)

# Split text into Devanagari words and non-word separators.
# This allows punctuation and spaces to be preserved.
_DEVANAGARI_PARTS = re.compile(
    r"([\u0900-\u097F]+)"
)


def normalize(word: str) -> str:
    """Normalize whitespace and Unicode representation."""

    if not isinstance(word, str):
        return word

    return re.sub(r"\s+", " ", word).strip()


def letters(word: str) -> int:
    """Count Devanagari characters in a word."""

    return len(
        re.findall(r"[\u0900-\u097F]", word)
    )


def _mapping(data: dict) -> dict:
    """Normalize dictionary keys and values."""

    return {
        normalize(key): normalize(value)
        for key, value in data.items()
    }


# --------------------------------------------------
# 2. Protected words
# --------------------------------------------------

# These words should remain unchanged by the stemmer.
PROTECTED_WORDS = {
    normalize(word)
    for word in (
        "दिल्ली", "पणजी", "लक्झरी", "लेक्सस", "निस्सान", "यामाहा",
        "भारत", "शहर", "घर", "गाव", "देश", "राज्य", "शाळा",
        "पाणी", "अपघात", "स्रोत", "इकडे", "तिकडे", "जिकडे",
        "कडे", "जवळ", "समोर", "नंतर", "आधी", "पूर्वी", "नाही",
        "काही", "कोणी", "सर्व", "सरकार", "शेतकरी", "विद्यार्थी",
        "भावना", "कारण", "माहिती", "कार्य", "प्रश्न", "उत्तर",
    )
}


# --------------------------------------------------
# 3. Suffixes used for stemming
# --------------------------------------------------

# Longer suffixes are checked first to avoid removing
# only part of a longer grammatical ending.
#
# Short endings such as "ा", "े", and "ी" are excluded
# because they can change a word's meaning.

STEM_SUFFIXES = sorted({
    "ांमधून", "ांमध्ये", "ांमुळे", "ांसाठी",
    "ांकडून", "ांबरोबर",
    "्यापासून", "्यामुळे", "्यावरून", "्यामधून",
    "्यातील", "्यामध्ये", "्याच्या",
    "ापासून", "ीपासून", "ामधून", "ीमधून",
    "ामुळे", "ीमुळे", "ासाठी", "ीसाठी",
    "ाकडून", "ीकडून", "ाबरोबर", "ीबरोबर",
    "ामध्ये", "ीमध्ये", "ावरून", "ीवरून",
    "ासारखे", "ीसारखे",
    "ांना", "ांनी", "ांचा", "ांची", "ांचे",
    "ाच्या", "ीच्या",
    "ाबद्दल", "ाविषयी", "ानंतर", "ामागे", "ासमोर",
    "ापासून", "ातील", "ीतील",
    "ाकडे", "ीकडे", "ातून", "ीतून",
    "ावर", "ीवर", "ाने", "ीने",
    "ाचा", "ाची", "ाचे", "ीचा", "ीची", "ीचे",
    "ाला", "ीला", "ांचा", "ांची", "ांचे",
    "ात", "ीत", "ास", "ीस",
}, key=len, reverse=True)


# --------------------------------------------------
# 4. Exact rules for irregular word forms
# --------------------------------------------------

# These mappings are approximate stems.
# They should not be treated as a complete Marathi grammar.

STEM_MAP = _mapping({
    "पाण्यात": "पाणी",
    "पाण्यावर": "पाणी",
    "पाण्याचा": "पाणी",
    "पाण्याची": "पाणी",
    "पाण्याचे": "पाणी",

    "शहरात": "शहर",
    "शहरातील": "शहर",
    "शहरांमध्ये": "शहर",

    "भारतात": "भारत",
    "भारताच्या": "भारत",
    "भारताविरुद्ध": "भारत",

    "गेला": "जा",
    "गेली": "जा",
    "गेले": "जा",

    "झाला": "हो",
    "झाली": "हो",
    "झाले": "हो",

    "केला": "कर",
    "केली": "कर",
    "केले": "कर",

    "दिला": "दे",
    "दिली": "दे",
    "दिले": "दे",

    "आला": "ये",
    "आली": "ये",
    "आले": "ये",
})


# --------------------------------------------------
# 5. Lemmatization dictionary
# --------------------------------------------------

# A lemma is a dictionary/base form, not simply a word
# with its suffix removed.
#
# Only explicitly listed forms are transformed.
# Unknown or ambiguous words remain unchanged.

LEMMA_MAP = _mapping({
    "पाण्यात": "पाणी",
    "पाण्यावर": "पाणी",
    "पाण्याचा": "पाणी",
    "पाण्याची": "पाणी",
    "पाण्याचे": "पाणी",

    "शहरात": "शहर",
    "शहरातील": "शहर",
    "शहरांमध्ये": "शहर",

    "भारतात": "भारत",
    "भारताच्या": "भारत",
    "भारताविरुद्ध": "भारत",

    "गेला": "जा",
    "गेली": "जा",
    "गेले": "जा",

    "झाला": "हो",
    "झाली": "हो",
    "झाले": "हो",

    "केला": "कर",
    "केली": "कर",
    "केले": "कर",

    "दिला": "दे",
    "दिली": "दे",
    "दिले": "दे",

    "आला": "ये",
    "आली": "ये",
    "आले": "ये",
})


# --------------------------------------------------
# 6. Stemming a single word
# --------------------------------------------------

@lru_cache(maxsize=100_000)
def stem_word(word: str) -> str:
    """Return a conservative approximate stem."""

    if not isinstance(word, str) or not word:
        return word

    w = normalize(word)

    # Leave non-Devanagari input unchanged.
    if not _DEVANAGARI_WORD.fullmatch(w):
        return word

    # Do not modify protected words.
    if w in PROTECTED_WORDS:
        return w

    # Apply exact mappings for known irregular forms.
    if w in STEM_MAP:
        return STEM_MAP[w]

    # Try removing a known suffix.
    for suffix in STEM_SUFFIXES:
        if w.endswith(suffix):
            base = w[:-len(suffix)]

            # Avoid empty or very short stems.
            if (
                base
                and letters(base) >= 2
                and not base.endswith("्")
                and len(base) >= 3
            ):
                # Preserve recognized base forms.
                if (
                    base in PROTECTED_WORDS
                    or base in LEMMA_MAP.values()
                ):
                    return base

                return base

    # If no safe rule applies, return the original word.
    return w


# --------------------------------------------------
# 7. Lemmatizing a single word
# --------------------------------------------------

@lru_cache(maxsize=100_000)
def lemmatize_word(word: str) -> str:
    """Return a known base form, leaving unknown words unchanged."""

    if not isinstance(word, str) or not word:
        return word

    w = normalize(word)

    if not _DEVANAGARI_WORD.fullmatch(w):
        return word

    # Preserve protected words.
    if w in PROTECTED_WORDS:
        return w

    # Convert only known forms.
    return LEMMA_MAP.get(w, w)


# --------------------------------------------------
# 8. Transforming text
# --------------------------------------------------

def _transform_text(text: str, transform) -> str:
    """Transform Devanagari words and preserve separators."""

    if not isinstance(text, str) or not text:
        return text

    parts = _DEVANAGARI_PARTS.split(text)

    return "".join(
        transform(part)
        if _DEVANAGARI_WORD.fullmatch(part)
        else part
        for part in parts
    )


def stem_text(text: str) -> str:
    """Stem Marathi words while preserving punctuation and spaces."""

    return _transform_text(text, stem_word)


def lemmatize_text(text: str) -> str:
    """Lemmatize known Marathi words while preserving punctuation."""

    return _transform_text(text, lemmatize_word)


# --------------------------------------------------
# 9. Analyze a word
# --------------------------------------------------

def analyze_word(word: str) -> Dict[str, str]:
    """Return the original word, stem, and lemma."""

    return {
        "word": word,
        "stem": stem_word(word),
        "lemma": lemmatize_word(word),
    }


def stem_and_lemmatize(word: str) -> str:
    """
    Backwards-compatible function.

    Returns the lemma, not the stem.
    Use stem_word() or lemmatize_word() explicitly when possible.
    """

    return lemmatize_word(word)


# --------------------------------------------------
# 10. Test the implementation
# --------------------------------------------------

if __name__ == "__main__":

    examples = [
        "पाणी",
        "पाण्यावर",
        "पाण्यात",
        "शहरांमध्ये",
        "शहरात",
        "भारताच्या",
        "भारतात",
        "भारताविरुद्ध",
        "सरकारने",
        "शेतकऱ्यांनी",
        "शेतकरी",
        "विद्यार्थ्यांना",
        "विद्यार्थी",
        "गेला",
        "केलेला",
        "करणार",
        "करण्यासाठी",
        "खेळतात",
        "करून",
        "बोलला",
        "भावना",
        "भावनांना",
        "अपघातात",
        "अपघात",
        "गाडीत",
        "गाड्यांमध्ये",
        "मुलाला",
        "शाळेत",
        "शाळा",
        "दिल्ली",
        "पणजी",
        "लक्झरी",
    ]

    print(
        f"{'Word':18} {'Stem':18} {'Lemma':18}"
    )
    print("-" * 54)

    for word in examples:
        result = analyze_word(word)

        print(
            f"{result['word']:18} "
            f"{result['stem']:18} "
            f"{result['lemma']:18}"
        )

    # Example of transforming a complete sentence.
    sentence = "भारतात पाण्याचा प्रश्न आहे."

    print("\nOriginal sentence:")
    print(sentence)

    print("\nStemmed sentence:")
    print(stem_text(sentence))

    print("\nLemmatized sentence:")
    print(lemmatize_text(sentence))
