import re
from typing import List, Optional, Set

# -------------------------------------------------------------------
# Common English words that should NEVER be counted as hallucinations.
# These are words that any reasonable summary/response can introduce
# without fabricating factual claims.  Kept intentionally minimal so
# genuinely novel entities ARE still caught.
# -------------------------------------------------------------------
_COMMON_WORDS: Set[str] = {
    # Connectors / transitions
    "also", "however", "therefore", "although", "because", "while",
    "since", "after", "before", "during", "about", "between",
    "through", "within", "without", "along", "above", "below",
    "under", "over", "into", "upon", "toward", "towards", "across",
    # Common verbs / verb forms
    "said", "says", "told", "made", "came", "went", "been", "being",
    "have", "having", "does", "doing", "would", "could", "should",
    "might", "must", "will", "shall", "were", "become", "becomes",
    "became", "taken", "given", "known", "shown", "found", "used",
    "called", "based", "noted", "added", "expected", "reported",
    "according", "including", "following", "leading", "reaching",
    "making", "taking", "getting", "going", "coming", "looking",
    "working", "turning", "running", "playing", "starting",
    # Common adjectives / adverbs
    "more", "most", "many", "much", "some", "other", "such",
    "only", "even", "well", "still", "just", "very", "than",
    "when", "then", "here", "there", "where", "both", "each",
    "every", "like", "same", "different", "several", "various",
    "another", "certain", "particular", "especially", "particularly",
    "recently", "currently", "previously", "significantly",
    "approximately", "potentially", "dramatically", "effectively",
    # Common nouns (generic)
    "people", "time", "year", "years", "part", "case", "point",
    "fact", "place", "thing", "things", "world", "life", "work",
    "system", "number", "group", "area", "level", "result",
    "report", "data", "information", "process", "study", "research",
    "team", "market", "production", "technology", "development",
    # Pronouns / determiners (≥4 chars)
    "that", "this", "these", "those", "their", "them", "they",
    "what", "which", "whom", "your", "with", "from",
}


def compute_hallucination_score(
    responses: List[str],
    input_text: str,
    reference: Optional[str] = None,
) -> float:
    """Hallucination Score (HS).

    Detects facts/words in the response that are entirely missing from
    the input text (and optional reference).

    Returns a score from 0.0 (no hallucinations) to 1.0 (fully
    hallucinated).

    Convention
    ----------
    * HS = 0.0  →  every response word is grounded  (best)
    * HS = 1.0  →  every response word is fabricated (worst)

    Improvements over the previous version:
    1. Common English words are excluded from the "hallucination" set.
    2. An optional *reference* text is accepted so that words from the
       gold output are not counted as hallucinated.
    """
    # Build the grounding vocabulary from input + reference
    grounding_facts = set(re.findall(r'\b\w{4,}\b', input_text.lower()))
    if reference:
        grounding_facts |= set(re.findall(r'\b\w{4,}\b', reference.lower()))

    if not grounding_facts:
        return 0.0

    hs_scores = []
    for r in responses:
        resp_facts = set(re.findall(r'\b\w{4,}\b', r.lower()))
        if not resp_facts:
            hs_scores.append(0.0)
            continue

        # Remove common words — they are never hallucinations
        resp_facts -= _COMMON_WORDS

        if not resp_facts:
            hs_scores.append(0.0)
            continue

        hallucinated = len(resp_facts - grounding_facts)
        score = hallucinated / len(resp_facts)
        hs_scores.append(score)

    return sum(hs_scores) / len(hs_scores) if hs_scores else 0.0

