"""
adversarial_textfooler.py — TextFooler-style adversarial paraphrase generator.

PUBLICATION_SPEC §7.3 (Tier-2): replace the current `adversarial_paraphrases.py`
(deterministic noise perturbations — typos, double negation, hedge insertion)
with a real adversarial paraphrase pipeline modelled after TextFooler
(Jin et al., AAAI 2020) and used as the standard adversarial baseline in the
prompt-robustness literature (PromptBench, NeurIPS 2023; etc.).

ALGORITHM
---------
For each base prompt:
  1. Tokenise. Tag CONTENT positions (excludes stopwords + punctuation).
  2. For each remaining content position, generate top-K MLM synonym
     candidates via a masked-language-model fill-mask (default
     `bert-base-uncased`).
  3. Filter candidates with the same gates as production paraphrase
     generation:
       (a) original word excluded
       (b) SBERT cos(base, perturbed) ≥ semantic_floor (default 0.85)
       (c) bidirectional NLI passes (single-model or ensemble)
       (d) length ratio within [0.7, 1.5] of base
  4. Score each qualified candidate by ADVERSARIALITY:
         score = 1.0 - SBERT_cos(base, perturbed)
       (we want maximally divergent surface form WHILE STAYING INSIDE the
        paraphrase gate — that is the operational definition of
        "adversarial paraphrase")
  5. Apply the highest-scoring substitution.
  6. Repeat up to `max_substitutions` times (default 3), without re-using
     any already-substituted position. Stop early when no qualified
     candidate remains.

OUTPUT
------
JSONL with one line per base prompt:
  {
    "instance_id":            "...",
    "base":                   "...",
    "adversarial_paraphrase": "...",
    "substitutions": [
      {"position": 5, "original": "increased", "substitute": "boosted",
       "sbert_sim": 0.91, "nli_fwd": 0.78, "nli_bwd": 0.81,
       "adversariality": 0.09}
    ],
    "num_substitutions":      1,
    "filter_metadata": {
      "sbert_sim":     0.89,
      "token_overlap": 0.66,
      "length_ratio":  1.03,
      "nli_fwd":       0.81, "nli_bwd": 0.83
    },
    "perturbation_family": "adversarial_textfooler"
  }

USAGE
-----
  # 1. Build adversarial paraphrases from an existing GenSens dataset
  python gensens/scripts/adversarial_textfooler.py \\
      --input gensens/data/gensens_summarization_200inst_8var.jsonl \\
      --output gensens/data/adversarial_textfooler_summarization.jsonl \\
      --max-substitutions 3 \\
      --top-k-mlm 8

  # 2. Or from a flat list of base prompts (one per line)
  python gensens/scripts/adversarial_textfooler.py \\
      --input-prompts prompts.txt \\
      --output adv.jsonl \\
      --max-substitutions 3

DEPENDENCIES
------------
  transformers + torch    (BERT MLM fill-mask)
  sentence-transformers   (SBERT semantic gate)
  cross-encoder NLI       (lazy-loaded via the production helpers)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("adversarial_textfooler")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Constants — match production filters (PUBLICATION_SPEC §6)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SEMANTIC_FLOOR    = 0.85      # SBERT cos(base, adversarial) ≥ this
DEFAULT_NLI_THRESHOLD     = 0.50      # bidirectional entail prob ≥ this
DEFAULT_LENGTH_RATIO_MIN  = 0.70
DEFAULT_LENGTH_RATIO_MAX  = 1.50
DEFAULT_MAX_SUBSTITUTIONS = 3
DEFAULT_TOP_K_MLM         = 8

DEFAULT_MLM_MODEL   = "bert-base-uncased"
DEFAULT_SBERT_MODEL = "all-mpnet-base-v2"
DEFAULT_NLI_MODEL   = "cross-encoder/nli-deberta-v3-small"
DEFAULT_NLI_ENSEMBLE = [
    "cross-encoder/nli-deberta-v3-small",
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-roberta-base",
]

# Stopword set mirrors `gensens/scripts/paraphrase_generator.py` _STOPWORDS
# minimally — we keep it local to avoid the dependency cycle (this script may
# be invoked when paraphrase_generator's vLLM imports are not available).
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in",
    "on", "at", "by", "for", "with", "as", "is", "are", "was", "were",
    "be", "been", "being", "do", "does", "did", "doing", "have", "has",
    "had", "having", "this", "that", "these", "those", "i", "you", "he",
    "she", "it", "we", "they", "them", "us", "him", "her", "my", "your",
    "his", "their", "our", "its", "from", "into", "out", "up", "down",
    "so", "than", "then", "too", "very", "can", "will", "just", "not",
    "no", "nor", "only", "what", "which", "who", "whom", "when", "where",
    "why", "how",
}
_PUNCT_RE = re.compile(r"^[^\w]+$")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared with paraphrase_generator.py
# ─────────────────────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _token_overlap(a: str, b: str) -> float:
    sa = set(re.findall(r"\w+", a.lower()))
    sb = set(re.findall(r"\w+", b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _length_ratio(cand: str, base: str) -> float:
    b = _word_count(base)
    return _word_count(cand) / b if b else 0.0


def _is_content_word(token: str) -> bool:
    """A token is a substitution candidate iff it has letters AND is not a
    pure-punctuation token AND is not in the minimal stopword list."""
    stripped = token.strip()
    if not stripped:
        return False
    if _PUNCT_RE.match(stripped):
        return False
    lo = stripped.lower()
    if lo in _STOPWORDS:
        return False
    if not re.search(r"[a-zA-Z]", stripped):
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Lazy model loaders (NLI helpers reused from paraphrase_generator when possible)
# ─────────────────────────────────────────────────────────────────────────────

def _wire_nli_helpers() -> Dict[str, Callable]:
    """Import the production NLI helpers; fall back to a local CrossEncoder reimpl."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from gensens.scripts.paraphrase_generator import (
            _get_nli_model, _nli_entail_prob,
            _get_nli_ensemble, _nli_ensemble_entail,
        )
        logger.info("[wire] Using production NLI helpers")
        return {
            "get_single":      _get_nli_model,
            "entail":          _nli_entail_prob,
            "get_ensemble":    _get_nli_ensemble,
            "ensemble_entail": _nli_ensemble_entail,
        }
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[wire] Production NLI helpers unavailable ({e}); using local fallback")
        return _build_local_nli_helpers()


def _build_local_nli_helpers() -> Dict[str, Callable]:
    from sentence_transformers import CrossEncoder
    import numpy as np

    def get_single(name):
        try:
            return CrossEncoder(name)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Local NLI load failed for {name}: {e}")
            return None

    def entail(model, premise, hypothesis):
        if model is None:
            return None
        try:
            s = np.asarray(model.predict([(premise, hypothesis)])).reshape(-1)
            if s.size == 3:
                ex = np.exp(s - np.max(s))
                return float((ex / ex.sum())[1])
            return float(s[0])
        except Exception:
            return None

    def get_ensemble(names):
        return [(n, get_single(n)) for n in (names or DEFAULT_NLI_ENSEMBLE)]

    def ensemble_entail(ensemble, premise, hypothesis, threshold, majority):
        per_model: Dict[str, Optional[float]] = {}
        votes = 0; n_models = 0
        for name, m in ensemble:
            if m is None:
                per_model[name] = None; continue
            n_models += 1
            p = entail(m, premise, hypothesis)
            per_model[name] = round(p, 4) if p is not None else None
            if p is not None and p >= threshold:
                votes += 1
        return {"votes": votes, "n_models": n_models,
                "passes": votes >= majority, "per_model": per_model}

    return {"get_single": get_single, "entail": entail,
            "get_ensemble": get_ensemble, "ensemble_entail": ensemble_entail}


# ─────────────────────────────────────────────────────────────────────────────
# MLM synonym generation
# ─────────────────────────────────────────────────────────────────────────────

class MLMSynonymProposer:
    """Wraps a BERT-style fill-mask pipeline for context-aware synonyms.

    The MLM proposes top-K alternatives for each masked content word; the
    caller is responsible for downstream gating (semantic / NLI / POS).
    """

    def __init__(self, model_name: str = DEFAULT_MLM_MODEL):
        try:
            from transformers import pipeline, AutoTokenizer
        except ImportError:
            sys.exit("ERROR: pip install transformers torch (required for adversarial generation)")
        logger.info(f"Loading MLM: {model_name}")
        self.fill = pipeline("fill-mask", model=model_name)
        self.mask_token = self.fill.tokenizer.mask_token
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)

    def propose(self, sentence: str, position: int, top_k: int) -> List[str]:
        """Mask the word at `position` (0-indexed by simple whitespace split)
        and return up to `top_k` distinct fill candidates."""
        words = sentence.split()
        if position < 0 or position >= len(words):
            return []
        original = words[position]
        words[position] = self.mask_token
        masked = " ".join(words)
        try:
            preds = self.fill(masked, top_k=top_k * 2)  # over-request, filter below
        except Exception as e:  # noqa: BLE001
            logger.debug(f"MLM proposal failed at pos {position}: {e}")
            return []

        out: List[str] = []
        seen = {original.lower(), original.strip(".,!?;:").lower()}
        for p in preds:
            tok = str(p.get("token_str", "")).strip()
            # Strip BERT WordPiece prefixes and surrounding punctuation.
            tok = tok.lstrip("##").strip()
            if not tok or len(tok) < 2:
                continue
            lo = tok.lower()
            if lo in seen:
                continue
            # Reject tokens that are punctuation-only or contain no letters.
            if not re.search(r"[a-zA-Z]", tok):
                continue
            seen.add(lo)
            out.append(tok)
            if len(out) >= top_k:
                break
        return out


# ─────────────────────────────────────────────────────────────────────────────
# SBERT helper (lightweight wrapper so we own a single embedder instance)
# ─────────────────────────────────────────────────────────────────────────────

class SBERTSimilarity:
    def __init__(self, model_name: str = DEFAULT_SBERT_MODEL):
        try:
            from sentence_transformers import SentenceTransformer, util
        except ImportError:
            sys.exit("ERROR: pip install sentence-transformers")
        logger.info(f"Loading SBERT: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._util = util

    def cos(self, a: str, b: str) -> float:
        emb = self.model.encode([a, b], convert_to_tensor=True)
        return float(self._util.cos_sim(emb[0], emb[1]).item())


# ─────────────────────────────────────────────────────────────────────────────
# Core adversarial generator
# ─────────────────────────────────────────────────────────────────────────────

class AdversarialTextFooler:
    """End-to-end TextFooler-style adversarial paraphrase pipeline.

    Filter stack mirrors PUBLICATION_SPEC §6 — same SBERT band semantics,
    bidirectional NLI gate (single or ensemble), and length ratio. The
    difference vs the normal paraphrase pipeline is the *generation*
    strategy: MLM word substitution instead of LLM rephrasing.
    """

    def __init__(
        self,
        mlm_model_name:        str  = DEFAULT_MLM_MODEL,
        sbert_model_name:      str  = DEFAULT_SBERT_MODEL,
        nli_model_name:        str  = DEFAULT_NLI_MODEL,
        enable_nli_ensemble:   bool = False,
        nli_ensemble_models:   Optional[List[str]] = None,
        nli_ensemble_majority: int  = 2,
        semantic_floor:        float = DEFAULT_SEMANTIC_FLOOR,
        nli_threshold:         float = DEFAULT_NLI_THRESHOLD,
        length_ratio_min:      float = DEFAULT_LENGTH_RATIO_MIN,
        length_ratio_max:      float = DEFAULT_LENGTH_RATIO_MAX,
        max_substitutions:     int  = DEFAULT_MAX_SUBSTITUTIONS,
        top_k_mlm:             int  = DEFAULT_TOP_K_MLM,
    ):
        self.mlm   = MLMSynonymProposer(mlm_model_name)
        self.sbert = SBERTSimilarity(sbert_model_name)
        self._nli_loaders = _wire_nli_helpers()
        self.nli_model_name        = nli_model_name
        self.enable_nli_ensemble   = bool(enable_nli_ensemble)
        self.nli_ensemble_models   = list(nli_ensemble_models or DEFAULT_NLI_ENSEMBLE)
        self.nli_ensemble_majority = int(nli_ensemble_majority)
        self.semantic_floor        = float(semantic_floor)
        self.nli_threshold         = float(nli_threshold)
        self.length_ratio_min      = float(length_ratio_min)
        self.length_ratio_max      = float(length_ratio_max)
        self.max_substitutions     = int(max_substitutions)
        self.top_k_mlm             = int(top_k_mlm)
        self._nli_single = None
        self._nli_ensemble = None

    # ── Bidirectional NLI (mirrors paraphrase_generator semantics) ─────────

    def _bidirectional_nli(self, base: str, cand: str) -> Dict[str, Any]:
        if self.enable_nli_ensemble:
            if self._nli_ensemble is None:
                self._nli_ensemble = self._nli_loaders["get_ensemble"](self.nli_ensemble_models)
            fwd = self._nli_loaders["ensemble_entail"](
                self._nli_ensemble, base, cand,
                self.nli_threshold, self.nli_ensemble_majority,
            )
            bwd = self._nli_loaders["ensemble_entail"](
                self._nli_ensemble, cand, base,
                self.nli_threshold, self.nli_ensemble_majority,
            )
            if fwd["n_models"] == 0 or bwd["n_models"] == 0:
                return {"passes": True, "p_fwd": None, "p_bwd": None,
                        "mode": "ensemble_bypass"}
            def _mean(audit):
                vals = [v for v in audit["per_model"].values() if v is not None]
                return sum(vals)/len(vals) if vals else None
            return {"passes": fwd["passes"] and bwd["passes"],
                    "p_fwd": _mean(fwd), "p_bwd": _mean(bwd),
                    "fwd_votes": fwd["votes"], "bwd_votes": bwd["votes"],
                    "mode": "ensemble"}
        # single-model path
        if self._nli_single is None:
            self._nli_single = self._nli_loaders["get_single"](self.nli_model_name)
        if self._nli_single is None:
            return {"passes": True, "p_fwd": None, "p_bwd": None, "mode": "single_bypass"}
        p_fwd = self._nli_loaders["entail"](self._nli_single, base, cand)
        p_bwd = self._nli_loaders["entail"](self._nli_single, cand, base)
        if p_fwd is None or p_bwd is None:
            return {"passes": True, "p_fwd": p_fwd, "p_bwd": p_bwd, "mode": "single_bypass"}
        return {"passes": (p_fwd >= self.nli_threshold and p_bwd >= self.nli_threshold),
                "p_fwd": p_fwd, "p_bwd": p_bwd, "mode": "single"}

    # ── Per-candidate gates ────────────────────────────────────────────────

    def _qualifies(self, base: str, perturbed: str) -> Dict[str, Any]:
        """Run all filter gates; return dict with passes + per-gate metadata."""
        report: Dict[str, Any] = {"passes": False, "rejection": ""}

        # Length ratio
        lr = _length_ratio(perturbed, base)
        report["length_ratio"] = round(lr, 4)
        if not (self.length_ratio_min <= lr <= self.length_ratio_max):
            report["rejection"] = "length_ratio"
            return report

        # SBERT semantic floor — adversarial paraphrases must still be paraphrases
        sim = self.sbert.cos(base, perturbed)
        report["sbert_sim"] = round(sim, 4)
        if sim < self.semantic_floor:
            report["rejection"] = "semantic_floor"
            return report

        # Token overlap (lenient — we want to allow adversarial near-paraphrases
        # that share many tokens with the base; the SBERT gate already prevents
        # gibberish, so we only cap pure copies).
        tov = _token_overlap(perturbed, base)
        report["token_overlap"] = round(tov, 4)
        if tov >= 0.999:
            report["rejection"] = "identical_to_base"
            return report

        # Bidirectional NLI
        nli = self._bidirectional_nli(base, perturbed)
        report["nli"] = nli
        if not nli["passes"]:
            report["rejection"] = "nli"
            return report

        report["passes"] = True
        return report

    # ── Main loop ──────────────────────────────────────────────────────────

    def generate(self, base: str) -> Dict[str, Any]:
        """Generate one adversarial paraphrase of ``base``.

        Returns the full result dict (see module docstring); the
        ``adversarial_paraphrase`` field equals ``base`` when no qualifying
        substitution was found.
        """
        words = base.split()
        # Content positions are eligible content words by whitespace index.
        content_positions = [i for i, w in enumerate(words) if _is_content_word(w)]
        used_positions: set = set()
        substitutions: List[Dict[str, Any]] = []
        current = base

        for iteration in range(self.max_substitutions):
            best: Optional[Dict[str, Any]] = None
            best_score = -1.0

            cur_words = current.split()
            for pos in content_positions:
                if pos in used_positions:
                    continue
                if pos >= len(cur_words):
                    continue
                original_token = cur_words[pos]

                synonyms = self.mlm.propose(current, pos, self.top_k_mlm)
                for syn in synonyms:
                    candidate_words = list(cur_words)
                    candidate_words[pos] = syn
                    candidate = " ".join(candidate_words)

                    report = self._qualifies(base, candidate)
                    if not report["passes"]:
                        continue

                    # Adversariality score: lower SBERT-to-BASE within the
                    # gate = more divergent = more adversarial.
                    adversariality = 1.0 - float(report["sbert_sim"])
                    if adversariality > best_score:
                        best_score = adversariality
                        best = {
                            "position":       pos,
                            "original":       original_token,
                            "substitute":     syn,
                            "adversariality": round(adversariality, 4),
                            "candidate":      candidate,
                            "report":         report,
                        }

            if best is None:
                logger.debug(f"[adv] no qualified substitution at iteration {iteration}")
                break

            # Apply this substitution.
            current = best["candidate"]
            used_positions.add(best["position"])
            substitutions.append({
                "position":       best["position"],
                "original":       best["original"],
                "substitute":     best["substitute"],
                "sbert_sim":      best["report"]["sbert_sim"],
                "nli_fwd":        best["report"].get("nli", {}).get("p_fwd"),
                "nli_bwd":        best["report"].get("nli", {}).get("p_bwd"),
                "adversariality": best["adversariality"],
            })

        # Final filter snapshot on the surviving adversarial paraphrase.
        if substitutions:
            final_report = self._qualifies(base, current)
        else:
            final_report = {"sbert_sim": None, "token_overlap": None,
                            "length_ratio": None, "nli": None, "passes": False}

        return {
            "base":                   base,
            "adversarial_paraphrase": current,
            "substitutions":          substitutions,
            "num_substitutions":      len(substitutions),
            "filter_metadata": {
                "sbert_sim":     final_report.get("sbert_sim"),
                "token_overlap": final_report.get("token_overlap"),
                "length_ratio":  final_report.get("length_ratio"),
                "nli_fwd":       (final_report.get("nli") or {}).get("p_fwd"),
                "nli_bwd":       (final_report.get("nli") or {}).get("p_bwd"),
            },
            "perturbation_family":    "adversarial_textfooler",
        }


# ─────────────────────────────────────────────────────────────────────────────
# I/O — load base prompts from either GenSens JSONL or plain text
# ─────────────────────────────────────────────────────────────────────────────

def _load_base_prompts_from_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Extract one (instance_id, base_text, task) row per GenSens record."""
    rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            base = rec.get("base_text") or rec.get("base") or ""
            if not base:
                continue
            rows.append({
                "instance_id": rec.get("instance_id", ""),
                "task":        rec.get("task", "unknown"),
                "base":        base,
            })
    return rows


def _load_base_prompts_from_text(txt_path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                out.append({
                    "instance_id": f"prompt_{i:04d}",
                    "task":        "unknown",
                    "base":        line,
                })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="GenSens JSONL with base_text records")
    src.add_argument("--input-prompts", help="Plain text file, one base prompt per line")
    p.add_argument("--output", required=True, help="Output JSONL path")

    p.add_argument("--mlm-model",   default=DEFAULT_MLM_MODEL)
    p.add_argument("--sbert-model", default=DEFAULT_SBERT_MODEL)
    p.add_argument("--nli-model",   default=DEFAULT_NLI_MODEL)
    p.add_argument("--enable-nli-ensemble", action="store_true")
    p.add_argument("--nli-ensemble-majority", type=int, default=2)

    p.add_argument("--semantic-floor",     type=float, default=DEFAULT_SEMANTIC_FLOOR,
                   help="Lower-bound SBERT cos(base, adversarial) (default 0.85)")
    p.add_argument("--nli-threshold",      type=float, default=DEFAULT_NLI_THRESHOLD)
    p.add_argument("--length-ratio-min",   type=float, default=DEFAULT_LENGTH_RATIO_MIN)
    p.add_argument("--length-ratio-max",   type=float, default=DEFAULT_LENGTH_RATIO_MAX)
    p.add_argument("--max-substitutions",  type=int,   default=DEFAULT_MAX_SUBSTITUTIONS,
                   help="Maximum number of word substitutions per prompt (default 3)")
    p.add_argument("--top-k-mlm",          type=int,   default=DEFAULT_TOP_K_MLM,
                   help="Top-K MLM predictions per masked position (default 8)")

    p.add_argument("--max-prompts",        type=int,   default=0,
                   help="Cap on number of base prompts processed (0 = no limit)")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load base prompts.
    if args.input:
        bases = _load_base_prompts_from_jsonl(Path(args.input))
    else:
        bases = _load_base_prompts_from_text(Path(args.input_prompts))
    if args.max_prompts > 0:
        bases = bases[: args.max_prompts]
    print(f"→ Loaded {len(bases)} base prompts")

    # Resume support — skip instance_ids already on disk.
    done_ids: set = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line).get("instance_id", ""))
                except Exception:
                    pass
        if done_ids:
            print(f"↻ Resuming: {len(done_ids)} already on disk")

    pending = [b for b in bases if b["instance_id"] not in done_ids]
    print(f"  {len(pending)} pending")

    gen = AdversarialTextFooler(
        mlm_model_name        = args.mlm_model,
        sbert_model_name      = args.sbert_model,
        nli_model_name        = args.nli_model,
        enable_nli_ensemble   = args.enable_nli_ensemble,
        nli_ensemble_majority = args.nli_ensemble_majority,
        semantic_floor        = args.semantic_floor,
        nli_threshold         = args.nli_threshold,
        length_ratio_min      = args.length_ratio_min,
        length_ratio_max      = args.length_ratio_max,
        max_substitutions     = args.max_substitutions,
        top_k_mlm             = args.top_k_mlm,
    )

    n_with_subs = 0
    n_total     = 0
    t0 = time.time()
    fout = open(out_path, "a", encoding="utf-8")
    try:
        for i, row in enumerate(pending, 1):
            t_start = time.time()
            result = gen.generate(row["base"])
            result["instance_id"] = row["instance_id"]
            result["task"]        = row["task"]
            result["generation_time_s"] = round(time.time() - t_start, 2)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            n_total += 1
            if result["num_substitutions"] > 0:
                n_with_subs += 1
            if i % 10 == 0 or i == len(pending):
                rate = n_with_subs / n_total if n_total else 0.0
                elapsed = time.time() - t0
                avg = elapsed / i
                print(f"  [{i}/{len(pending)}] success rate: {100*rate:.1f}% | "
                      f"avg {avg:.1f}s/prompt")
    finally:
        fout.close()

    print(f"\n✓ wrote {out_path}")
    print(f"  prompts processed: {n_total}")
    print(f"  with ≥1 substitution: {n_with_subs} ({100*n_with_subs/max(1,n_total):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
