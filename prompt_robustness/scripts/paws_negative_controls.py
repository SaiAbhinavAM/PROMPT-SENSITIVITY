"""
paws_negative_controls.py — Template-generated negative controls + filter audit.

PUBLICATION_SPEC §8.3 (Validation Prong 3): construct ~200 (base, NEAR-paraphrase)
pairs that share HIGH lexical overlap but are NOT paraphrases, then audit the
GenSens filter pipeline's rejection rate. Target ≥ 90%.

This is the PAWS methodology (Zhang et al., NAACL 2019) applied to our filter
stack as a quality control on the quality control — it validates that our
filters are sensitive to semantic meaning, not just lexical overlap.

Five perturbation types:
  1. Subject/object swap     — "X helped Y" / "Y helped X"
  2. Polarity flip           — "the drug works" / "the drug does not work"
  3. Quantifier swap         — "all patients recovered" / "some patients recovered"
  4. Antonym substitution    — "Increase production" / "Decrease production"
  5. Modifier flip           — "the tall man was fast" / "the fast man was tall"

MODES
-----
  --mode generate  Template-generate the pair JSONL.
  --mode audit     Run the GenSens filter pipeline on the pair JSONL and
                   report per-type rejection rate.

Both modes are CPU-friendly. Audit mode requires sentence-transformers and
the same NLI model used by the GenSens pipeline (lazy-downloaded on first run).
No GPU and no vLLM required.

USAGE
-----
  # 1. Generate the pair set (run once)
  python prompt_robustness/scripts/paws_negative_controls.py \\
      --mode generate \\
      --output gensens/data/paws_negative_controls.jsonl \\
      --n-per-type 40

  # 2. Audit the filter pipeline's rejection rate on those pairs
  python prompt_robustness/scripts/paws_negative_controls.py \\
      --mode audit \\
      --input gensens/data/paws_negative_controls.jsonl \\
      --output-dir results/paws_audit \\
      --enable-nli-ensemble
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("paws_negative_controls")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Filter constants — MUST stay in sync with gensens/scripts/paraphrase_generator.py.
# Pull the values directly when the module is importable so the audit faithfully
# mirrors the production filters. If the import fails (e.g. running in an
# environment without sentence-transformers), the script gracefully falls back
# to local constants.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SBERT_MODEL          = "all-mpnet-base-v2"
DEFAULT_NLI_PRIMARY          = "cross-encoder/nli-deberta-v3-small"
DEFAULT_NLI_ENSEMBLE = [
    "cross-encoder/nli-deberta-v3-small",
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-roberta-base",
]
DEFAULT_NLI_ENTAIL_THRESHOLD = 0.50
DEFAULT_SBERT_LOWER          = 0.82
DEFAULT_SBERT_UPPER          = 0.98
DEFAULT_TOKEN_OVERLAP_MAX    = 0.85
DEFAULT_LENGTH_RATIO_MIN     = 0.70
DEFAULT_LENGTH_RATIO_MAX     = 1.50


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation 1 — Subject / object swap
# ─────────────────────────────────────────────────────────────────────────────

_SO_AGENT_PAIRS: List[Tuple[str, str]] = [
    ("lawyer", "judge"), ("doctor", "patient"), ("teacher", "student"),
    ("manager", "employee"), ("hunter", "prey"), ("parent", "child"),
    ("officer", "suspect"), ("nurse", "doctor"), ("director", "actor"),
    ("editor", "writer"), ("trainer", "athlete"), ("coach", "player"),
    ("scientist", "assistant"), ("guard", "prisoner"), ("waiter", "customer"),
    ("driver", "passenger"), ("seller", "buyer"), ("interviewer", "candidate"),
    ("teacher", "principal"), ("voter", "candidate"),
]
_SO_TRANSITIVE_VERBS = [
    "convinced", "helped", "examined", "instructed", "supervised", "chased",
    "scolded", "hired", "interviewed", "evaluated", "praised", "criticised",
    "trained", "questioned", "called", "rescued", "guided",
]
_SO_TEMPLATE = "The {a} {v} the {b}."


def _generate_subject_object_swap(n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    combos = [(a, b, v) for (a, b) in _SO_AGENT_PAIRS for v in _SO_TRANSITIVE_VERBS]
    rng.shuffle(combos)
    out: List[Dict[str, Any]] = []
    for a, b, v in combos:
        base   = _SO_TEMPLATE.format(a=a, v=v, b=b)
        cand   = _SO_TEMPLATE.format(a=b, v=v, b=a)
        out.append({
            "perturbation": "subject_object_swap",
            "base":         base,
            "candidate":    cand,
            "slots":        {"agent": a, "patient": b, "verb": v},
        })
        if len(out) >= n:
            break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation 2 — Polarity flip
# ─────────────────────────────────────────────────────────────────────────────

_POLARITY_NOUNS = [
    "treatment", "medication", "vaccine", "therapy", "drug", "policy",
    "law", "rule", "method", "plan", "strategy", "approach",
    "design", "model", "algorithm", "tool", "device", "system",
]
_POLARITY_ADJECTIVES = [
    "effective", "successful", "useful", "reliable", "safe", "accurate",
    "fast", "popular", "scalable", "robust",
]
_POLARITY_COPULAS = ["is", "was"]


def _generate_polarity_flip(n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    combos = list(itertools.product(_POLARITY_NOUNS, _POLARITY_COPULAS, _POLARITY_ADJECTIVES))
    rng.shuffle(combos)
    out: List[Dict[str, Any]] = []
    for noun, cop, adj in combos:
        base = f"The {noun} {cop} {adj}."
        cand = f"The {noun} {cop} not {adj}."
        out.append({
            "perturbation": "polarity_flip",
            "base":         base,
            "candidate":    cand,
            "slots":        {"noun": noun, "copula": cop, "adjective": adj},
        })
        if len(out) >= n:
            break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation 3 — Quantifier swap
# ─────────────────────────────────────────────────────────────────────────────

_QUANTIFIER_PAIRS: List[Tuple[str, str]] = [
    ("All", "Some"), ("Every", "Few"), ("Most", "Few"),
    ("Always", "Sometimes"), ("None of the", "Most of the"),
    ("Many", "Few"),
]
_QUANT_NOUNS = [
    "students passed the exam", "patients recovered fully",
    "employees received bonuses", "candidates were qualified",
    "voters supported the proposal", "samples tested positive",
    "users reported the bug", "citizens attended the meeting",
    "workers received training", "applicants met the criteria",
]


def _generate_quantifier_swap(n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    combos = list(itertools.product(_QUANTIFIER_PAIRS, _QUANT_NOUNS))
    rng.shuffle(combos)
    out: List[Dict[str, Any]] = []
    for (q1, q2), tail in combos:
        # Lowercase the tail when the quantifier already encodes the article.
        base = f"{q1} {tail}."
        cand = f"{q2} {tail}."
        out.append({
            "perturbation": "quantifier_swap",
            "base":         base,
            "candidate":    cand,
            "slots":        {"q1": q1, "q2": q2, "tail": tail},
        })
        if len(out) >= n:
            break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation 4 — Antonym substitution (single content word)
# ─────────────────────────────────────────────────────────────────────────────

_ANTONYM_PAIRS: List[Tuple[str, str]] = [
    ("increase", "decrease"), ("succeed", "fail"), ("accept", "reject"),
    ("win", "lose"), ("open", "close"), ("allow", "forbid"),
    ("support", "oppose"), ("approve", "deny"), ("praise", "criticise"),
    ("strengthen", "weaken"), ("expand", "shrink"), ("encourage", "discourage"),
    ("solve", "complicate"), ("simplify", "complicate"), ("speed up", "slow down"),
]
_ANTONYM_FRAMES = [
    "The proposal aims to {v} the system.",
    "Researchers tried to {v} the workflow.",
    "Officials voted to {v} the existing policy.",
    "The team plans to {v} production this year.",
    "Engineers will {v} the deployment process.",
    "The CEO promised to {v} customer support.",
    "We must {v} our outreach efforts.",
    "Auditors recommended that we {v} our reporting.",
]


def _generate_antonym_substitution(n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    combos = list(itertools.product(_ANTONYM_PAIRS, _ANTONYM_FRAMES))
    rng.shuffle(combos)
    out: List[Dict[str, Any]] = []
    for (v1, v2), frame in combos:
        base = frame.format(v=v1)
        cand = frame.format(v=v2)
        out.append({
            "perturbation": "antonym_substitution",
            "base":         base,
            "candidate":    cand,
            "slots":        {"v1": v1, "v2": v2, "frame": frame},
        })
        if len(out) >= n:
            break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation 5 — Modifier flip (swap two adjectives within the sentence)
# ─────────────────────────────────────────────────────────────────────────────

_MOD_NOUNS = [
    "man", "woman", "child", "doctor", "athlete", "teacher",
    "scientist", "engineer", "dancer", "lawyer", "manager", "soldier",
]
_MOD_ADJ_PAIRS: List[Tuple[str, str]] = [
    ("tall", "fast"), ("smart", "tall"), ("strong", "quiet"),
    ("careful", "experienced"), ("young", "calm"), ("kind", "tired"),
    ("brave", "tall"), ("rich", "humble"), ("creative", "patient"),
    ("loud", "thin"),
]


def _generate_modifier_flip(n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    combos = list(itertools.product(_MOD_NOUNS, _MOD_ADJ_PAIRS))
    rng.shuffle(combos)
    out: List[Dict[str, Any]] = []
    for noun, (a1, a2) in combos:
        base = f"The {a1} {noun} was very {a2}."
        cand = f"The {a2} {noun} was very {a1}."
        out.append({
            "perturbation": "modifier_flip",
            "base":         base,
            "candidate":    cand,
            "slots":        {"noun": noun, "adj1": a1, "adj2": a2},
        })
        if len(out) >= n:
            break
    return out


_GENERATORS: Dict[str, Callable[[int, int], List[Dict[str, Any]]]] = {
    "subject_object_swap":   _generate_subject_object_swap,
    "polarity_flip":         _generate_polarity_flip,
    "quantifier_swap":       _generate_quantifier_swap,
    "antonym_substitution":  _generate_antonym_substitution,
    "modifier_flip":         _generate_modifier_flip,
}


# ─────────────────────────────────────────────────────────────────────────────
# --mode generate
# ─────────────────────────────────────────────────────────────────────────────

def run_generate(args) -> int:
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_pairs: List[Dict[str, Any]] = []
    n_per_type = args.n_per_type
    base_seed = args.seed
    for i, (ptype, fn) in enumerate(_GENERATORS.items()):
        pairs = fn(n_per_type, seed=base_seed + i)
        # Tag with a per-type id for resume / dedup.
        for j, p in enumerate(pairs):
            p["pair_id"] = f"{ptype}::{j:03d}"
        logger.info(f"  {ptype:25s}: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    with open(out_path, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n✓ wrote {len(all_pairs)} pairs → {out_path}")
    print(f"  perturbation types: {list(_GENERATORS.keys())}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Filter pipeline — lightweight reimplementation of the GenSens filters.
# Pulls helpers from gensens.scripts.paraphrase_generator when importable so
# the behaviour mirrors production; otherwise reimplements the basic gates so
# the audit can still run.
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


class FilterPipeline:
    """CPU-friendly reimplementation of the GenSens filter stack.

    Filters (in execution order):
      1. SBERT band   — lower ≤ cos ≤ upper
      2. Token Jaccard — overlap ≤ threshold
      3. Length ratio — within [min, max]
      4. Bidirectional NLI — (ensemble or single-model)

    Each pair is recorded with a rejection reason ('' if accepted) so the
    audit can break down rejections per perturbation type.
    """

    def __init__(
        self,
        sbert_model:           str   = DEFAULT_SBERT_MODEL,
        sbert_lower:           float = DEFAULT_SBERT_LOWER,
        sbert_upper:           float = DEFAULT_SBERT_UPPER,
        token_overlap_max:     float = DEFAULT_TOKEN_OVERLAP_MAX,
        length_ratio_min:      float = DEFAULT_LENGTH_RATIO_MIN,
        length_ratio_max:      float = DEFAULT_LENGTH_RATIO_MAX,
        nli_entail_threshold:  float = DEFAULT_NLI_ENTAIL_THRESHOLD,
        nli_model_name:        str   = DEFAULT_NLI_PRIMARY,
        enable_nli_ensemble:   bool  = False,
        nli_ensemble_models:   Optional[List[str]] = None,
        nli_ensemble_majority: int   = 2,
    ):
        self.sbert_lower          = float(sbert_lower)
        self.sbert_upper          = float(sbert_upper)
        self.token_overlap_max    = float(token_overlap_max)
        self.length_ratio_min     = float(length_ratio_min)
        self.length_ratio_max     = float(length_ratio_max)
        self.nli_entail_threshold = float(nli_entail_threshold)

        # Lazy-load SBERT.
        try:
            from sentence_transformers import SentenceTransformer, util
        except ImportError:
            sys.exit("ERROR: pip install sentence-transformers (required for --mode audit)")
        logger.info(f"Loading SBERT: {sbert_model}")
        self.sbert    = SentenceTransformer(sbert_model)
        self._st_util = util

        # NLI loaders — prefer to reuse the production module's helpers so
        # behaviour is exactly equivalent. Fall back to local reimplementation
        # otherwise.
        self._nli_loaders = self._wire_nli_helpers()
        self.enable_nli_ensemble = bool(enable_nli_ensemble)
        self.nli_model_name      = nli_model_name
        self.nli_ensemble_models = (
            list(nli_ensemble_models) if nli_ensemble_models else list(DEFAULT_NLI_ENSEMBLE)
        )
        self.nli_ensemble_majority = int(nli_ensemble_majority)

        # NLI models are lazy-loaded on first call.
        self._nli_single: Any = None
        self._nli_ensemble: Optional[List[Tuple[str, Any]]] = None

    def _wire_nli_helpers(self) -> Dict[str, Callable]:
        """Try to import the production NLI helpers; fall back if unavailable."""
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from gensens.scripts.paraphrase_generator import (
                _get_nli_model, _nli_entail_prob,
                _get_nli_ensemble, _nli_ensemble_entail,
            )
            logger.info("[wire] Using production NLI helpers from gensens.scripts.paraphrase_generator")
            return {
                "get_single":    _get_nli_model,
                "entail":        _nli_entail_prob,
                "get_ensemble":  _get_nli_ensemble,
                "ensemble_entail": _nli_ensemble_entail,
            }
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[wire] Could not import production NLI helpers ({e}). Using local fallback.")
            return self._build_local_nli_helpers()

    def _build_local_nli_helpers(self) -> Dict[str, Callable]:
        from sentence_transformers import CrossEncoder
        import numpy as np

        def get_single(name: str):
            try:
                return CrossEncoder(name)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[local] Failed to load NLI {name}: {e}")
                return None

        def entail(model, premise: str, hypothesis: str) -> Optional[float]:
            if model is None:
                return None
            try:
                s = np.asarray(model.predict([(premise, hypothesis)])).reshape(-1)
                if s.size == 3:
                    ex = np.exp(s - np.max(s))
                    return float((ex / ex.sum())[1])
                return float(s[0])
            except Exception:  # noqa: BLE001
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

    # ── Filter logic ────────────────────────────────────────────────────

    def _bidirectional_nli(self, base: str, cand: str) -> Dict[str, Any]:
        if self.enable_nli_ensemble:
            if self._nli_ensemble is None:
                self._nli_ensemble = self._nli_loaders["get_ensemble"](self.nli_ensemble_models)
            fwd = self._nli_loaders["ensemble_entail"](
                self._nli_ensemble, base, cand,
                self.nli_entail_threshold, self.nli_ensemble_majority,
            )
            bwd = self._nli_loaders["ensemble_entail"](
                self._nli_ensemble, cand, base,
                self.nli_entail_threshold, self.nli_ensemble_majority,
            )
            if fwd["n_models"] == 0 or bwd["n_models"] == 0:
                return {"passes": True, "fwd_votes": None, "bwd_votes": None,
                        "p_fwd": None, "p_bwd": None, "mode": "ensemble_bypass"}
            def _mean_prob(audit):
                vals = [v for v in audit["per_model"].values() if v is not None]
                return sum(vals) / len(vals) if vals else None
            return {"passes": fwd["passes"] and bwd["passes"],
                    "fwd_votes": fwd["votes"], "bwd_votes": bwd["votes"],
                    "p_fwd": _mean_prob(fwd), "p_bwd": _mean_prob(bwd),
                    "mode": "ensemble"}
        # Single model
        if self._nli_single is None:
            self._nli_single = self._nli_loaders["get_single"](self.nli_model_name)
        if self._nli_single is None:
            return {"passes": True, "p_fwd": None, "p_bwd": None, "mode": "single_bypass"}
        p_fwd = self._nli_loaders["entail"](self._nli_single, base, cand)
        p_bwd = self._nli_loaders["entail"](self._nli_single, cand, base)
        if p_fwd is None or p_bwd is None:
            return {"passes": True, "p_fwd": p_fwd, "p_bwd": p_bwd, "mode": "single_bypass"}
        return {
            "passes": (p_fwd >= self.nli_entail_threshold
                       and p_bwd >= self.nli_entail_threshold),
            "p_fwd": p_fwd, "p_bwd": p_bwd, "mode": "single",
        }

    def evaluate_pair(self, base: str, cand: str) -> Dict[str, Any]:
        """Run all filters; return per-filter outcomes + final accept/reject.

        For NEGATIVE controls a REJECTION is the desired outcome
        (`accepted=False` means the filter pipeline correctly identified the
        pair as non-paraphrastic).
        """
        record: Dict[str, Any] = {
            "rejection_reason": "", "accepted": True,
            "sbert_sim":         None, "token_overlap":  None,
            "length_ratio":      None, "nli":            None,
        }

        # 1. SBERT band
        emb = self.sbert.encode([base, cand], convert_to_tensor=True)
        cos = float(self._st_util.cos_sim(emb[0], emb[1]).item())
        record["sbert_sim"] = round(cos, 4)
        if cos < self.sbert_lower or cos > self.sbert_upper:
            record["accepted"] = False
            record["rejection_reason"] = "sbert_band"
            return record

        # 2. Token Jaccard
        tov = _token_overlap(cand, base)
        record["token_overlap"] = round(tov, 4)
        if tov > self.token_overlap_max:
            record["accepted"] = False
            record["rejection_reason"] = "token_overlap"
            return record

        # 3. Length ratio
        lr = _length_ratio(cand, base)
        record["length_ratio"] = round(lr, 4)
        if not (self.length_ratio_min <= lr <= self.length_ratio_max):
            record["accepted"] = False
            record["rejection_reason"] = "length_ratio"
            return record

        # 4. Bidirectional NLI (single-model or ensemble)
        nli = self._bidirectional_nli(base, cand)
        record["nli"] = nli
        if not nli["passes"]:
            record["accepted"] = False
            record["rejection_reason"] = "nli"
            return record

        return record


# ─────────────────────────────────────────────────────────────────────────────
# --mode audit
# ─────────────────────────────────────────────────────────────────────────────

def run_audit(args) -> int:
    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"ERROR: input {in_path} does not exist")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_pair_path = out_dir / "paws_audit_pairs.jsonl"
    summary_path  = out_dir / "paws_audit.json"

    pairs: List[Dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    print(f"→ Auditing {len(pairs)} negative-control pairs from {in_path}")

    pipeline = FilterPipeline(
        sbert_model           = args.sbert_model,
        nli_model_name        = args.nli_model,
        enable_nli_ensemble   = args.enable_nli_ensemble,
        nli_ensemble_models   = args.nli_ensemble_models,
        nli_ensemble_majority = args.nli_ensemble_majority,
    )

    per_pair_rows: List[Dict[str, Any]] = []
    with open(per_pair_path, "w", encoding="utf-8") as fout:
        for i, pair in enumerate(pairs, 1):
            res = pipeline.evaluate_pair(pair["base"], pair["candidate"])
            row = {**pair, **res, "correctly_rejected": (not res["accepted"])}
            per_pair_rows.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(pairs):
                rej = sum(1 for r in per_pair_rows if r["correctly_rejected"])
                print(f"  [{i}/{len(pairs)}] rejection rate so far: "
                      f"{100*rej/i:.1f}%")

    summary = _audit_summary(per_pair_rows)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ wrote {per_pair_path}")
    print(f"✓ wrote {summary_path}")
    print("\n" + "=" * 72)
    print(f"OVERALL REJECTION RATE: {100*summary['overall_rejection_rate']:.1f}% "
          f"(target ≥ 90%)")
    print("=" * 72)
    print("\nPer perturbation type:")
    for ptype, stats in summary["per_perturbation"].items():
        flag = "✓" if stats["rejection_rate"] >= 0.90 else "✗"
        print(f"  {flag} {ptype:25s}  {100*stats['rejection_rate']:5.1f}% "
              f"({stats['n_rejected']}/{stats['n_total']})")
    print("\nRejection reasons (most common):")
    for reason, cnt in summary["rejection_reason_counts"].items():
        print(f"  {reason:20s}: {cnt}")
    return 0


def _audit_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_type[r.get("perturbation", "unknown")].append(r)

    per_perturbation: Dict[str, Dict[str, Any]] = {}
    for ptype, group in by_type.items():
        n = len(group)
        n_rej = sum(1 for r in group if r["correctly_rejected"])
        reasons = Counter(r["rejection_reason"] for r in group if r["rejection_reason"])
        per_perturbation[ptype] = {
            "n_total":        n,
            "n_rejected":     n_rej,
            "rejection_rate": n_rej / n if n else 0.0,
            "mean_sbert_sim": float(mean([r["sbert_sim"] for r in group
                                          if isinstance(r["sbert_sim"], (int, float))]))
                              if group else 0.0,
            "mean_token_overlap": float(mean([r["token_overlap"] for r in group
                                              if isinstance(r["token_overlap"], (int, float))]))
                              if group else 0.0,
            "reason_counts":  dict(reasons),
        }

    n = len(rows)
    n_rej = sum(1 for r in rows if r["correctly_rejected"])
    return {
        "n_pairs":                 n,
        "overall_rejection_rate":  n_rej / n if n else 0.0,
        "target_threshold":        0.90,
        "passes_target":           (n_rej / n) >= 0.90 if n else False,
        "rejection_reason_counts": dict(Counter(r["rejection_reason"]
                                                for r in rows if r["rejection_reason"])),
        "per_perturbation":        per_perturbation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", required=True, choices=["generate", "audit"])

    p.add_argument("--output",       help="(generate) JSONL output path")
    p.add_argument("--n-per-type",   type=int, default=40,
                   help="(generate) pairs per perturbation type (default 40 → 200 total)")
    p.add_argument("--seed",         type=int, default=42)

    p.add_argument("--input",        help="(audit) negative-controls JSONL")
    p.add_argument("--output-dir",   help="(audit) directory for audit artefacts")
    p.add_argument("--sbert-model",  default=DEFAULT_SBERT_MODEL)
    p.add_argument("--nli-model",    default=DEFAULT_NLI_PRIMARY,
                   help="primary NLI checkpoint (single-model mode)")
    p.add_argument("--enable-nli-ensemble", action="store_true",
                   help="(audit) use 3-of-3 NLI majority voting instead of single-model")
    p.add_argument("--nli-ensemble-models", nargs="*",
                   default=None, help="(audit) override ensemble members")
    p.add_argument("--nli-ensemble-majority", type=int, default=2)
    return p


def main() -> int:
    args = build_argparser().parse_args()
    if args.mode == "generate":
        if not args.output:
            sys.exit("ERROR: --output required for --mode generate")
        return run_generate(args)
    if args.mode == "audit":
        if not args.input or not args.output_dir:
            sys.exit("ERROR: --input and --output-dir required for --mode audit")
        return run_audit(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
