"""
metrics_advanced.py — Research-grade metric upgrades for the Prompt Robustness Evaluation Framework.

This module contains advanced implementations of TRD, SMS, KPIG, and USD metrics,
designed to complement (not replace) the original basic versions.

Mathematical References:
- TRD Semantic: Uses embedding centroid divergence via cosine distance.
- SMS Wasserstein: Approximates Earth-Mover's Distance via mean pairwise transport cost.
- KPIG Advanced: Extracts and validates explicit instruction constraints (length, style).
- USD: Utility-Stability Divergence gap between automated PRI and human evaluation.
"""

import re
import math
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


# =============================================================================
# STEP 1: Semantic TRD — Thematic Robustness Drift (Embedding-Based)
# =============================================================================

def compute_trd_semantic(responses: List[str], embedder=None) -> float:
    """
    Compute Thematic Robustness Drift (TRD) using embedding-based topic coherence.

    Instead of measuring response length variance (which is a surface-level proxy),
    this computes semantic drift as the average cosine distance of response embeddings
    from their centroid. Higher values indicate greater thematic instability.

    Algorithm:
        1. Encode all responses → embedding matrix E ∈ ℝ^{n × d}
        2. Compute centroid μ = mean(E, axis=0) ∈ ℝ^d
        3. For each embedding e_i, compute cosine_distance(e_i, μ)
        4. TRD = mean(cosine_distances) normalized to [0, 1]

    Args:
        responses: List of model response strings.
        embedder: An EmbeddingHelper instance (if None, one will be created).

    Returns:
        float: Semantic TRD score in [0, 1]. 0 = perfectly coherent, 1 = maximum drift.
    """
    if not responses or len(responses) < 2:
        return 0.0

    # Lazy import to avoid circular dependencies
    if embedder is None:
        from .embeddings import EmbeddingHelper
        embedder = EmbeddingHelper()

    embeddings = embedder.encode(responses)

    # Compute centroid of all response embeddings
    centroid = np.mean(embeddings, axis=0, keepdims=True)  # Shape: (1, d)

    # Compute cosine distance from each embedding to the centroid
    # cosine_distance = 1 - cosine_similarity
    distances = cosine_distances(embeddings, centroid).flatten()  # Shape: (n,)

    # Average cosine distance as the drift measure
    trd = float(np.mean(distances))

    # Cosine distance ∈ [0, 2], but in practice for normalized embeddings ∈ [0, 1]
    # Clamp to [0, 1] for safety
    return max(0.0, min(1.0, trd))


# =============================================================================
# STEP 2: SMS Wasserstein — Approximate Earth-Mover's Distance
# =============================================================================

def compute_sms_wasserstein(embeddings: np.ndarray) -> float:
    """
    Compute Semantic Manifold Stability (SMS) using approximate Wasserstein distance.

    This computes the mean pairwise Euclidean transport cost between embedding pairs,
    approximating the 1-Wasserstein (Earth-Mover's) distance for the embedding distribution.
    The stability score is defined as the complement of the average transport distance.

    Algorithm:
        1. Compute pairwise distance matrix D ∈ ℝ^{n × n} (Euclidean)
        2. Extract upper-triangular entries (avoid double-counting & self-distances)
        3. avg_transport = mean(upper_triangle(D))
        4. SMS_W = 1 - avg_transport (normalized to [0, 1])

    A high SMS_W indicates that response embeddings are tightly clustered (stable),
    while a low SMS_W indicates semantic dispersion (unstable).

    Args:
        embeddings: numpy array of shape (n_responses, embedding_dim).

    Returns:
        float: Wasserstein-approximated SMS score in [0, 1].
    """
    n = embeddings.shape[0]
    if n < 2:
        return 1.0

    # Pairwise Euclidean distance matrix
    dist_matrix = cdist(embeddings, embeddings, metric='euclidean')

    # Extract upper triangle (exclude diagonal = 0)
    upper_tri_indices = np.triu_indices(n, k=1)
    pairwise_distances = dist_matrix[upper_tri_indices]

    # Mean transport distance (average cost to "transport" one distribution to another)
    avg_transport = float(np.mean(pairwise_distances))

    # Normalize: embeddings from sentence-transformers are L2-normalized,
    # so max Euclidean distance between two unit vectors = 2.0
    # We normalize by this theoretical maximum.
    max_distance = 2.0  # For L2-normalized embeddings
    normalized_transport = avg_transport / max_distance

    # SMS = 1 - normalized_transport (higher = more stable)
    sms_w = 1.0 - normalized_transport

    return max(0.0, min(1.0, sms_w))


# =============================================================================
# STEP 4: KPIG Advanced — Semantic Information Gain with Redundancy Penalty
#
# Completely redesigned to fix saturation (~1.0 for all samples).
# Old approach: constraint-satisfaction (almost always 1.0 since few prompts
#               have explicit constraints like "2 sentences").
# New approach: semantic key-unit coverage against the reference, with
#               TF-IDF importance weighting, redundancy penalty, and
#               variance sensitivity across prompt variants.
# =============================================================================


def _extract_key_units(text: str, min_word_len: int = 3) -> List[str]:
    """
    Extract semantic key units from text using multi-layer extraction.

    Extracts three types of units, then deduplicates:
        1. Named entities (capitalized multi-word or single proper nouns)
        2. Significant keywords (words ≥ min_word_len, excluding stopwords)
        3. Bigram phrases (adjacent non-stopword pairs)

    Returns lowercased, deduplicated key units for consistent matching.

    Args:
        text: Input text string.
        min_word_len: Minimum character length for keywords.

    Returns:
        List of unique key unit strings (lowercased).
    """
    if not text or not text.strip():
        return []

    # Common English stopwords (lightweight, no NLTK dependency)
    STOPWORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
        'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'don', 'now', 'and', 'but', 'or',
        'if', 'while', 'that', 'this', 'what', 'which', 'who', 'whom',
        'these', 'those', 'its', 'his', 'her', 'their', 'our', 'your',
        'my', 'he', 'she', 'it', 'we', 'they', 'them', 'him', 'me', 'us',
        'said', 'also', 'about', 'up', 'one', 'two', 'like', 'much',
        'many', 'well', 'back', 'even', 'still', 'way', 'take', 'come',
        'make', 'know', 'get', 'got', 'say', 'says', 'see', 'go', 'went',
        'tell', 'told', 'think', 'thought', 'give', 'gave', 'put',
    }

    key_units = set()

    # 1. Named entities — capitalized sequences (e.g., "Daniel Radcliffe")
    named_entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
    for ent in named_entities:
        lowered = ent.lower().strip()
        if len(lowered) >= min_word_len and lowered not in STOPWORDS:
            key_units.add(lowered)

    # 2. Significant keywords — non-stopword tokens
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    content_tokens = [
        t for t in tokens
        if len(t) >= min_word_len and t not in STOPWORDS
    ]
    key_units.update(content_tokens)

    # 3. Content bigrams — consecutive non-stopword pairs capture phrases
    for i in range(len(content_tokens) - 1):
        bigram = f"{content_tokens[i]} {content_tokens[i+1]}"
        key_units.add(bigram)

    return list(key_units)


def _compute_tfidf_importance(
    key_units: List[str],
    corpus_texts: List[str]
) -> Dict[str, float]:
    """
    Compute TF-IDF importance weights for key units against a text corpus.

    Key units that appear in fewer documents (responses) are weighted higher,
    since they represent more distinctive/informative content.

    Falls back to uniform weighting if TF-IDF computation fails.

    Args:
        key_units: List of key unit strings to weight.
        corpus_texts: List of text documents (responses) forming the corpus.

    Returns:
        Dict mapping each key unit to its importance weight (higher = more important).
    """
    if not key_units or not corpus_texts:
        return {u: 1.0 for u in key_units}

    try:
        # Build TF-IDF matrix from the corpus
        vectorizer = TfidfVectorizer(
            vocabulary={u: i for i, u in enumerate(key_units)},
            token_pattern=r'(?u)\b\w+\b',
            lowercase=True,
        )
        tfidf_matrix = vectorizer.fit_transform(corpus_texts)

        # Mean TF-IDF score per key unit across all documents
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

        importance = {}
        for i, unit in enumerate(key_units):
            score = float(mean_scores[i]) if i < len(mean_scores) else 0.0
            # Floor at 0.1 so no unit is completely ignored
            importance[unit] = max(0.1, score)

        # Normalize so max importance = 1.0
        max_imp = max(importance.values()) if importance else 1.0
        if max_imp > 0:
            importance = {k: v / max_imp for k, v in importance.items()}

        return importance

    except Exception as e:
        logger.debug(f"TF-IDF computation failed, using uniform weights: {e}")
        return {u: 1.0 for u in key_units}


def _semantic_match_units(
    ref_units: List[str],
    resp_units: List[str],
    embedder,
    threshold: float = 0.55
) -> List[Tuple[str, str, float]]:
    """
    Match reference key units to response key units using semantic similarity.

    Instead of exact string matching (which misses paraphrases), this encodes
    all units as sentence-transformer embeddings and finds the best match for
    each reference unit via cosine similarity.

    A match is accepted if similarity ≥ threshold.

    Args:
        ref_units: Key units extracted from the reference.
        resp_units: Key units extracted from a response.
        embedder: EmbeddingHelper instance for encoding.
        threshold: Minimum cosine similarity for a valid match.

    Returns:
        List of (ref_unit, matched_resp_unit, similarity) tuples.
    """
    if not ref_units or not resp_units:
        return []

    # Encode all units
    ref_embeddings = embedder.encode(ref_units)
    resp_embeddings = embedder.encode(resp_units)

    # Compute cosine similarity matrix: (n_ref, n_resp)
    sim_matrix = cosine_similarity(ref_embeddings, resp_embeddings)

    matches = []
    for i, ref_unit in enumerate(ref_units):
        best_j = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i, best_j])
        if best_sim >= threshold:
            matches.append((ref_unit, resp_units[best_j], best_sim))

    return matches


def _compute_redundancy_weights(
    ref_units: List[str],
    all_matches_per_response: List[List[Tuple[str, str, float]]]
) -> Dict[str, float]:
    """
    Compute inverse-frequency redundancy weights for matched reference units.

    If a key unit is matched by every single response, it's easy/trivial content
    and should contribute less. Rare matches (only 1-2 responses captured it)
    indicate more discriminative content.

    Formula:
        weight(unit) = 1 / (1 + frequency_across_responses)

    This penalizes units that appear in all responses (they don't help differentiate)
    and boosts units that only some responses capture.

    Args:
        ref_units: All reference key units.
        all_matches_per_response: For each response, the list of matched
                                  (ref_unit, resp_unit, sim) tuples.

    Returns:
        Dict mapping ref_unit → redundancy weight in (0, 1].
    """
    # Count how many responses matched each reference unit
    match_counts = Counter()
    for response_matches in all_matches_per_response:
        matched_ref_units = set(m[0] for m in response_matches)
        for unit in matched_ref_units:
            match_counts[unit] += 1

    weights = {}
    for unit in ref_units:
        freq = match_counts.get(unit, 0)
        # Inverse frequency: rarer matches → higher weight
        weights[unit] = 1.0 / (1.0 + freq)

    return weights


def compute_kpig_advanced(
    prompts: List[str],
    responses: List[str],
    reference: str = "",
    embedder=None,
    match_threshold: float = 0.55,
    verbose: bool = False,
) -> float:
    """
    Compute research-grade Key Point Information Gain (KPIG).

    Measures how much meaningful information from the reference is captured
    across generated responses, with penalties for redundancy and bonuses
    for output diversity.

    Algorithm:
        1. Extract key semantic units from reference (entities, keywords, bigrams)
        2. Compute TF-IDF importance weights for reference units
        3. For each response:
           a. Extract key units from response
           b. Semantically match response units to reference units (cosine sim)
           c. Compute weighted coverage for this response
        4. Compute redundancy penalties (inverse-frequency weighting)
        5. Combine: base_KPIG = Σ(importance × redundancy × matched) / Σ(importance)
        6. Apply variance sensitivity:
           variance_penalty = std(per_response_coverage)
           final_KPIG = base_KPIG × (1 - exp(-3 × variance_penalty))
           (penalizes identical outputs; rewards diverse coverage)

    Args:
        prompts: List of prompt strings.
        responses: List of corresponding response strings.
        reference: Reference/gold-standard text. If empty, uses responses' union.
        embedder: EmbeddingHelper instance. Created if None.
        match_threshold: Min cosine similarity for semantic match (default 0.55).
        verbose: If True, print detailed extraction and matching logs.

    Returns:
        float: KPIG score in [0, 1].
               Higher = better information coverage with good diversity.
               Designed to NOT saturate at 1.0 for typical outputs.
    """
    if not responses:
        return 0.0

    # Lazy embedder creation
    if embedder is None:
        from .embeddings import EmbeddingHelper
        embedder = EmbeddingHelper()

    # --- Step 1: Extract reference key units ---
    if reference and reference.strip():
        ref_units = _extract_key_units(reference)
    else:
        # Fallback: use the union of all response units as pseudo-reference
        all_units = set()
        for resp in responses:
            all_units.update(_extract_key_units(resp))
        ref_units = list(all_units)

    if not ref_units:
        return 0.0

    if verbose:
        logger.info(f"[KPIG] Reference key units ({len(ref_units)}): {ref_units[:20]}...")

    # --- Step 2: TF-IDF importance weights ---
    # Use responses as the corpus to determine which reference units are
    # discriminative vs trivial
    corpus = [reference] + responses if reference.strip() else responses
    importance = _compute_tfidf_importance(ref_units, corpus)

    if verbose:
        top_important = sorted(importance.items(), key=lambda x: -x[1])[:10]
        logger.info(f"[KPIG] Top important units: {top_important}")

    # --- Step 3: Per-response semantic matching ---
    all_matches_per_response = []
    per_response_coverage = []

    for i, resp in enumerate(responses):
        resp_units = _extract_key_units(resp)

        if not resp_units:
            all_matches_per_response.append([])
            per_response_coverage.append(0.0)
            continue

        # Semantic matching via embeddings
        matches = _semantic_match_units(
            ref_units, resp_units, embedder, threshold=match_threshold
        )
        all_matches_per_response.append(matches)

        # Raw coverage for this response (unweighted)
        coverage = len(matches) / len(ref_units) if ref_units else 0.0
        per_response_coverage.append(coverage)

        if verbose:
            matched_refs = [m[0] for m in matches]
            missing = [u for u in ref_units if u not in matched_refs]
            logger.info(
                f"[KPIG] Response {i}: matched={len(matches)}/{len(ref_units)}, "
                f"coverage={coverage:.3f}, missing={missing[:5]}..."
            )

    # --- Step 4: Redundancy penalty ---
    redundancy_weights = _compute_redundancy_weights(ref_units, all_matches_per_response)

    if verbose:
        logger.info(f"[KPIG] Redundancy weights (sample): "
                    f"{dict(list(redundancy_weights.items())[:5])}")

    # --- Step 5: Weighted KPIG computation ---
    # For each reference unit, check if it was matched by ANY response,
    # then weight by importance × redundancy.
    matched_any = set()
    for response_matches in all_matches_per_response:
        for ref_unit, _, _ in response_matches:
            matched_any.add(ref_unit)

    # Weighted numerator: matched units weighted by importance and redundancy
    weighted_matched = 0.0
    weighted_total = 0.0
    smoothing = 1e-8  # Avoid division by zero

    for unit in ref_units:
        imp = importance.get(unit, 0.1)
        red = redundancy_weights.get(unit, 0.5)

        # Combined weight: importance × redundancy_scaling
        # Redundancy scaling: we want units that are NOT matched by every response
        # to count more. But we also don't want to completely zero out common matches.
        combined_weight = imp * (0.3 + 0.7 * red)

        weighted_total += combined_weight
        if unit in matched_any:
            # Scale by average match quality (similarity score)
            avg_sim = 0.0
            sim_count = 0
            for response_matches in all_matches_per_response:
                for ref_u, _, sim in response_matches:
                    if ref_u == unit:
                        avg_sim += sim
                        sim_count += 1
            avg_sim = avg_sim / sim_count if sim_count > 0 else 0.0

            # Soft match: partial credit based on similarity quality
            weighted_matched += combined_weight * avg_sim

    base_kpig = weighted_matched / (weighted_total + smoothing)

    if verbose:
        logger.info(f"[KPIG] Base KPIG (before variance): {base_kpig:.4f}")
        logger.info(f"[KPIG] Per-response coverage: {[f'{c:.3f}' for c in per_response_coverage]}")

    # --- Step 6: Variance sensitivity ---
    # Penalize if all responses have identical coverage (no diversity).
    # A model that always generates the exact same output gets penalized.
    if len(per_response_coverage) >= 2:
        coverage_std = float(np.std(per_response_coverage))
        # Exponential penalty: identical outputs (std ≈ 0) → strong penalty
        # Diverse outputs (std > 0.1) → minimal penalty
        # Using 1 - exp(-k * std) where k controls sensitivity
        variance_factor = 1.0 - math.exp(-3.0 * coverage_std)
        # Floor at 0.3 so even identical-output models get partial credit
        variance_factor = max(0.3, variance_factor)
    else:
        variance_factor = 1.0

    final_kpig = base_kpig * variance_factor

    if verbose:
        logger.info(
            f"[KPIG] Variance penalty: std={coverage_std:.4f}, "
            f"factor={variance_factor:.4f}, final={final_kpig:.4f}"
        )

    return max(0.0, min(1.0, final_kpig))


# =============================================================================
# STEP 5: USD — Utility-Stability Divergence
# =============================================================================

def compute_usd(pri: float, human_score: float) -> float:
    """
    Compute Utility-Stability Divergence (USD).

    USD measures the absolute gap between the automated Prompt Robustness Index (PRI)
    and the human evaluation score. A large USD indicates that the automated metrics
    and human judgment diverge significantly, which may signal:
        - Metric calibration issues
        - Evaluation artifacts
        - Tasks where human intuition differs from statistical measures

    Definition:
        USD = |PRI - Human_Score|

    Args:
        pri: Prompt Robustness Index score in [0, 1].
        human_score: Human evaluation score in [0, 1].

    Returns:
        float: USD score in [0, 1]. 0 = perfect agreement, 1 = maximum divergence.
    """
    return abs(float(pri) - float(human_score))


# =============================================================================
# STEP 7: Baseline Metrics — ROUGE and BERTScore
# =============================================================================

def compute_rouge_scores(responses: List[str], reference: str) -> Dict[str, float]:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) between responses and reference.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures n-gram overlap
    between generated text and reference. Commonly used in summarization evaluation.

    Returns per-metric averages across all responses and the individual response scores.

    Args:
        responses: List of model response strings.
        reference: The reference/gold-standard text.

    Returns:
        Dict with keys 'rouge1', 'rouge2', 'rougeL' (averaged F-scores),
        and 'per_response' containing individual scores.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {
            'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0,
            'per_response': []
        }

    if not reference.strip() or not responses:
        return {
            'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0,
            'per_response': []
        }

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    all_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    per_response = []

    for resp in responses:
        if not resp.strip():
            for k in all_scores:
                all_scores[k].append(0.0)
            per_response.append({'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0})
            continue

        scores = scorer.score(reference, resp)
        resp_scores = {}
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            fmeasure = scores[metric].fmeasure
            all_scores[metric].append(fmeasure)
            resp_scores[metric] = round(fmeasure, 4)
        per_response.append(resp_scores)

    result = {
        'rouge1': float(np.mean(all_scores['rouge1'])) if all_scores['rouge1'] else 0.0,
        'rouge2': float(np.mean(all_scores['rouge2'])) if all_scores['rouge2'] else 0.0,
        'rougeL': float(np.mean(all_scores['rougeL'])) if all_scores['rougeL'] else 0.0,
        'per_response': per_response,
    }

    return result


def compute_bertscore(responses: List[str], reference: str) -> Dict[str, float]:
    """
    Compute BERTScore between responses and reference.

    BERTScore uses contextual embeddings from BERT to compute token-level similarity,
    providing a more semantically-aware evaluation than n-gram overlap.

    Falls back gracefully if the bert_score package is not installed.

    Args:
        responses: List of model response strings.
        reference: The reference/gold-standard text.

    Returns:
        Dict with keys 'precision', 'recall', 'f1' (averaged scores),
        and 'per_response' containing individual scores.
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'per_response': [],
            'available': False
        }

    if not reference.strip() or not responses:
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'per_response': [],
            'available': True
        }

    try:
        refs = [reference] * len(responses)
        P, R, F1 = bert_score_fn(responses, refs, lang="en", verbose=False)

        per_response = []
        for i in range(len(responses)):
            per_response.append({
                'precision': round(P[i].item(), 4),
                'recall': round(R[i].item(), 4),
                'f1': round(F1[i].item(), 4),
            })

        return {
            'precision': float(P.mean().item()),
            'recall': float(R.mean().item()),
            'f1': float(F1.mean().item()),
            'per_response': per_response,
            'available': True
        }
    except Exception as e:
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'per_response': [],
            'available': False,
            'error': str(e)
        }
