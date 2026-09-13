"""
Simple, interpretable, rule-based text features for one transcript
(a turn's text, or a whole-recording-channel's text). No embeddings, no
trained NLP models — tokenization is a plain regex, and "POS"/function
word categories are hand-built Spanish word lists, not a statistical
tagger. This is a deliberate simplification for this exploratory stage;
see nlp/README.md.
"""
import re
from collections import Counter

import numpy as np

# letters only (incl. Spanish accents/ñ/ü) - numbers and punctuation are
# dropped from the *word* count (a spoken phone number like "88, 23, 11"
# would otherwise inflate word counts inconsistently across transcripts).
_WORD_RE = re.compile(r"[a-zà-ÿñ]+", re.IGNORECASE)

# single-token filler/hesitation markers (Spanish). Counted as a
# LINGUISTIC MARKER only — not automatically interpreted as a genuine
# hesitation (that would need prosodic/pause evidence this module
# doesn't have).
FILLER_WORDS = {"eh", "ehh", "em", "emm", "este", "pues", "bueno", "mmm", "mm", "o", "sea"}
# multi-word filler phrases, checked separately against the joined lowercase text
FILLER_PHRASES = ["o sea"]

ARTICLES = {"el", "la", "los", "las", "un", "una", "unos", "unas", "lo"}
PRONOUNS = {
    "yo", "tu", "tú", "el", "él", "ella", "ellos", "ellas", "nosotros", "nosotras",
    "vosotros", "vosotras", "usted", "ustedes", "me", "te", "se", "nos", "os",
    "le", "les", "mi", "mis", "tus", "su", "sus", "mio", "mío", "nuestro", "nuestra",
    "este", "esta", "esto", "estos", "estas", "ese", "esa", "eso", "esos", "esas",
}
CONJUNCTIONS = {"y", "o", "pero", "porque", "aunque", "sino", "ni", "que", "si", "pues", "entonces"}
PREPOSITIONS = {
    "a", "ante", "bajo", "con", "contra", "de", "desde", "en", "entre", "hacia",
    "hasta", "para", "por", "segun", "según", "sin", "sobre", "tras",
}


def tokenize(text: str) -> list:
    if not isinstance(text, str) or not text:
        return []
    return _WORD_RE.findall(text.lower())


def ngrams(tokens: list, n: int) -> list:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def lexical_features(text: str) -> dict:
    tokens = tokenize(text)
    n = len(tokens)
    char_count = len(text) if isinstance(text, str) else 0

    if n == 0:
        return {
            "word_count": 0, "char_count": char_count, "unique_word_count": 0,
            "type_token_ratio": np.nan, "repeated_word_count": 0,
            "immediate_repetition_count": 0, "repeated_bigram_count": 0,
            "filler_count": 0, "filler_rate": np.nan,
            "article_count": 0, "pronoun_count": 0, "conjunction_count": 0,
            "preposition_count": 0, "function_word_ratio": np.nan,
        }

    counts = Counter(tokens)
    unique = len(counts)
    repeated_word_count = sum(1 for w, c in counts.items() if c > 1)
    immediate_repetition = sum(1 for i in range(1, n) if tokens[i] == tokens[i - 1])

    bigrams = ngrams(tokens, 2)
    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(1 for bg, c in bigram_counts.items() if c > 1)

    filler_count = sum(counts.get(w, 0) for w in FILLER_WORDS)
    joined = " " + " ".join(tokens) + " "
    for phrase in FILLER_PHRASES:
        filler_count += joined.count(f" {phrase} ")

    n_articles = sum(counts.get(w, 0) for w in ARTICLES)
    n_pronouns = sum(counts.get(w, 0) for w in PRONOUNS)
    n_conjunctions = sum(counts.get(w, 0) for w in CONJUNCTIONS)
    n_prepositions = sum(counts.get(w, 0) for w in PREPOSITIONS)
    n_function = n_articles + n_pronouns + n_conjunctions + n_prepositions

    return {
        "word_count": n,
        "char_count": char_count,
        "unique_word_count": unique,
        "type_token_ratio": unique / n,
        "repeated_word_count": repeated_word_count,
        "immediate_repetition_count": immediate_repetition,
        "repeated_bigram_count": repeated_bigrams,
        "filler_count": filler_count,
        "filler_rate": filler_count / n,
        "article_count": n_articles,
        "pronoun_count": n_pronouns,
        "conjunction_count": n_conjunctions,
        "preposition_count": n_prepositions,
        "function_word_ratio": n_function / n,
    }


def top_ngrams(texts: list, n: int, top_k: int = 20) -> list:
    """Most common n-grams across a list of texts (recording- or
    group-level exploratory look, not a per-turn feature)."""
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        counter.update(ngrams(tokens, n))
    return counter.most_common(top_k)
