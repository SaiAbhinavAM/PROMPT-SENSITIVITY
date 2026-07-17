"""
back_translation_family.py — round-trip translation as a 5th paraphrase family.

PUBLICATION_SPEC §7.2 (Tier-2): augment the GenSens prompt-engineering
strategies (§6.6 — 4 families) with a 5th family that uses neural
machine-translation round-trips through pivot languages.

Why a 5th family?
-----------------
The prompt-engineering strategies (lexical/syntactic/pragmatic/length) all
share a generation mechanism: an LLM rewriting under a strategy-conditioned
instruction. Round-trip translation produces genuine syntactic and lexical
variation that the LLM-rewrite strategies cannot — the source-side encoder
forces a different linguistic encoding, which the target-side decoder then
"hallucinates" a paraphrase from. This is the canonical PromptBench paraphrase
strategy (Zhu et al., NeurIPS 2023 D&B).

Pivot languages: DE, FR, RU (default). These are sufficiently different from
English to produce non-trivial structural variation. Helsinki-NLP/opus-mt
models are tiny (~70MB each) and CPU-friendly.

Mode
----
This module is INTENDED to be invoked from the main GenSens pipeline as an
augmentation pass:

    1. Run paraphrase_generator.py to produce a JSONL with the 4 standard
       families (lexical/syntactic/pragmatic/length).
    2. Run back_translation_family.py to ADD 1–2 back-translation variants
       per instance (slots 7–8 of K=8) and append them to the same JSONL.
    3. Re-run validate_dataset.py if downstream tools require it.

This separation lets the heavy NMT models load ONCE per run rather than
per-instance, which is the dominant cost.

USAGE
-----
    python gensens/scripts/back_translation_family.py \\
        --input gensens/data/gensens_summarization_200inst_8var.jsonl \\
        --output gensens/data/gensens_summarization_200inst_8var_bt.jsonl \\
        --pivot-langs de fr ru \\
        --max-per-instance 2 \\
        --enable-nli-ensemble

The output JSONL is the input JSONL with up to `--max-per-instance` extra
back-translation variants APPENDED to each record's `variants` list, each
tagged with `strategy="back_translation_<lang>"` and `strategy_family="back_translation"`.

DEPENDENCIES
------------
    transformers + torch         (Helsinki-NLP/opus-mt-* MT models)
    sentence-transformers        (SBERT semantic gate)
    cross-encoder NLI            (bidirectional NLI gate)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("back_translation_family")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Constants — match production filters (PUBLICATION_SPEC §6)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SBERT_MODEL          = "all-mpnet-base-v2"
DEFAULT_NLI_MODEL            = "cross-encoder/nli-deberta-v3-small"
DEFAULT_NLI_ENSEMBLE         = [
    "cross-encoder/nli-deberta-v3-small",
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-roberta-base",
]
DEFAULT_NLI_THRESHOLD        = 0.50
DEFAULT_SBERT_LOWER          = 0.82
DEFAULT_SBERT_UPPER          = 0.98
DEFAULT_SEMANTIC_DEDUP_THRESH = 0.95
DEFAULT_TOKEN_OVERLAP_MAX    = 0.85
DEFAULT_LENGTH_RATIO_MIN     = 0.70
DEFAULT_LENGTH_RATIO_MAX     = 1.50
DEFAULT_MIN_WORD_COUNT       = 5

# Pivot language → (EN→pivot model, pivot→EN model). The trio EN↔DE/FR/RU is
# the same one that lived in `config.yaml` `paraphrase.pivot_languages` as
# legacy. Re-enabling them now under the new gate stack.
PIVOT_MODELS: Dict[str, Tuple[str, str]] = {
    "de": ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"),
    "fr": ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"),
    "ru": ("Helsinki-NLP/opus-mt-en-ru", "Helsinki-NLP/opus-mt-ru-en"),
    "es": ("Helsinki-NLP/opus-mt-en-es", "Helsinki-NLP/opus-mt-es-en"),
    "zh": ("Helsinki-NLP/opus-mt-en-zh", "Helsinki-NLP/opus-mt-zh-en"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Filter helpers (kept inline so back-translation is self-contained)
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


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────────────────

def _wire_nli_helpers() -> Dict[str, Callable]:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from gensens.scripts.paraphrase_generator import (
            _get_nli_model, _nli_entail_prob,
            _get_nli_ensemble, _nli_ensemble_entail,
        )
        logger.info("[wire] Using production NLI helpers from paraphrase_generator")
        return {"get_single": _get_nli_model, "entail": _nli_entail_prob,
                "get_ensemble": _get_nli_ensemble, "ensemble_entail": _nli_ensemble_entail}
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
            logger.warning(f"Local NLI load failed: {e}")
            return None

    def entail(model, p, h):
        if model is None: return None
        try:
            s = np.asarray(model.predict([(p, h)])).reshape(-1)
            if s.size == 3:
                ex = np.exp(s - np.max(s))
                return float((ex / ex.sum())[1])
            return float(s[0])
        except Exception:
            return None

    def get_ensemble(names):
        return [(n, get_single(n)) for n in (names or DEFAULT_NLI_ENSEMBLE)]

    def ensemble_entail(ensemble, p, h, threshold, majority):
        per: Dict[str, Optional[float]] = {}
        votes = 0; n = 0
        for name, m in ensemble:
            if m is None: per[name] = None; continue
            n += 1
            prob = entail(m, p, h)
            per[name] = round(prob, 4) if prob is not None else None
            if prob is not None and prob >= threshold: votes += 1
        return {"votes": votes, "n_models": n, "passes": votes >= majority, "per_model": per}

    return {"get_single": get_single, "entail": entail,
            "get_ensemble": get_ensemble, "ensemble_entail": ensemble_entail}


class _MTModel:
    """Wraps one direction of a Helsinki-NLP/opus-mt model with a simple
    `translate(text) -> str` interface."""

    def __init__(self, model_id: str, max_length: int = 256):
        try:
            from transformers import MarianMTModel, MarianTokenizer
            import torch
        except ImportError:
            sys.exit("ERROR: pip install transformers torch sentencepiece (required for MT)")
        logger.info(f"Loading MT model: {model_id}")
        self.tokenizer = MarianTokenizer.from_pretrained(model_id)
        self.model     = MarianMTModel.from_pretrained(model_id).eval()
        self.max_length = max_length
        self._torch = torch
        # Device routing — prefer GPU when available.
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.device = "cuda"
        else:
            self.device = "cpu"

    def translate(self, text: str) -> str:
        with self._torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                    max_length=self.max_length).to(self.device)
            out = self.model.generate(
                **inputs, max_length=self.max_length, num_beams=4, early_stopping=True
            )
            return self.tokenizer.decode(out[0], skip_special_tokens=True)


class _SBERT:
    def __init__(self, model_name: str = DEFAULT_SBERT_MODEL):
        from sentence_transformers import SentenceTransformer, util
        logger.info(f"Loading SBERT: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._util = util

    def cos(self, a: str, b: str) -> float:
        emb = self.model.encode([a, b], convert_to_tensor=True)
        return float(self._util.cos_sim(emb[0], emb[1]).item())


# ─────────────────────────────────────────────────────────────────────────────
# Back-translation family generator
# ─────────────────────────────────────────────────────────────────────────────

class BackTranslationFamily:
    """Generate up to `max_per_instance` back-translation variants per instance.

    Applies the SAME filter stack as the prompt-engineering families:
      - SBERT band [lower, upper]
      - Token Jaccard ≤ max_overlap (vs base AND vs already-accepted variants)
      - Length ratio in [min, max]
      - SBERT semantic dedup (≤ semantic_dedup_threshold vs accepted)
      - Bidirectional NLI ≥ threshold (single-model or ensemble)
    """

    def __init__(
        self,
        pivot_langs:           List[str],
        sbert_model:           str   = DEFAULT_SBERT_MODEL,
        nli_model_name:        str   = DEFAULT_NLI_MODEL,
        enable_nli_ensemble:   bool  = False,
        nli_ensemble_models:   Optional[List[str]] = None,
        nli_ensemble_majority: int   = 2,
        nli_threshold:         float = DEFAULT_NLI_THRESHOLD,
        sbert_lower:           float = DEFAULT_SBERT_LOWER,
        sbert_upper:           float = DEFAULT_SBERT_UPPER,
        token_overlap_max:     float = DEFAULT_TOKEN_OVERLAP_MAX,
        semantic_dedup_thresh: float = DEFAULT_SEMANTIC_DEDUP_THRESH,
        length_ratio_min:      float = DEFAULT_LENGTH_RATIO_MIN,
        length_ratio_max:      float = DEFAULT_LENGTH_RATIO_MAX,
        min_word_count:        int   = DEFAULT_MIN_WORD_COUNT,
        max_per_instance:      int   = 2,
    ):
        invalid = [p for p in pivot_langs if p not in PIVOT_MODELS]
        if invalid:
            sys.exit(f"ERROR: unsupported pivot lang(s) {invalid}. "
                     f"Supported: {sorted(PIVOT_MODELS)}")
        self.pivot_langs = list(pivot_langs)
        self.max_per_instance = int(max_per_instance)
        self.sbert_lower = float(sbert_lower)
        self.sbert_upper = float(sbert_upper)
        self.token_overlap_max = float(token_overlap_max)
        self.semantic_dedup_thresh = float(semantic_dedup_thresh)
        self.length_ratio_min = float(length_ratio_min)
        self.length_ratio_max = float(length_ratio_max)
        self.min_word_count = int(min_word_count)
        self.nli_threshold = float(nli_threshold)

        self.sbert = _SBERT(sbert_model)
        self._nli_loaders = _wire_nli_helpers()
        self.nli_model_name        = nli_model_name
        self.enable_nli_ensemble   = bool(enable_nli_ensemble)
        self.nli_ensemble_models   = list(nli_ensemble_models or DEFAULT_NLI_ENSEMBLE)
        self.nli_ensemble_majority = int(nli_ensemble_majority)
        self._nli_single = None
        self._nli_ensemble = None

        # Load all MT pairs upfront — one load amortised across the whole run.
        self._mt: Dict[str, Tuple[_MTModel, _MTModel]] = {}
        for lang in self.pivot_langs:
            fwd_id, bwd_id = PIVOT_MODELS[lang]
            self._mt[lang] = (_MTModel(fwd_id), _MTModel(bwd_id))

    # ── NLI dispatch (mirrors paraphrase_generator) ────────────────────────

    def _bidirectional_nli(self, base: str, cand: str) -> Tuple[bool, Optional[float], Optional[float]]:
        if self.enable_nli_ensemble:
            if self._nli_ensemble is None:
                self._nli_ensemble = self._nli_loaders["get_ensemble"](self.nli_ensemble_models)
            fwd = self._nli_loaders["ensemble_entail"](
                self._nli_ensemble, base, cand, self.nli_threshold, self.nli_ensemble_majority)
            bwd = self._nli_loaders["ensemble_entail"](
                self._nli_ensemble, cand, base, self.nli_threshold, self.nli_ensemble_majority)
            if fwd["n_models"] == 0 or bwd["n_models"] == 0:
                return True, None, None
            def _mean(audit):
                vals = [v for v in audit["per_model"].values() if v is not None]
                return sum(vals) / len(vals) if vals else None
            return (fwd["passes"] and bwd["passes"]), _mean(fwd), _mean(bwd)
        if self._nli_single is None:
            self._nli_single = self._nli_loaders["get_single"](self.nli_model_name)
        if self._nli_single is None:
            return True, None, None
        p_fwd = self._nli_loaders["entail"](self._nli_single, base, cand)
        p_bwd = self._nli_loaders["entail"](self._nli_single, cand, base)
        if p_fwd is None or p_bwd is None:
            return True, p_fwd, p_bwd
        return (p_fwd >= self.nli_threshold and p_bwd >= self.nli_threshold), p_fwd, p_bwd

    # ── Per-pivot back-translation generation ──────────────────────────────

    def _round_trip(self, base: str, lang: str) -> str:
        fwd, bwd = self._mt[lang]
        pivot_text  = fwd.translate(base)
        back_text   = bwd.translate(pivot_text)
        return back_text.strip()

    # ── Filter ─────────────────────────────────────────────────────────────

    def _passes_filters(
        self,
        base: str,
        candidate: str,
        accepted_variants: List[Dict[str, Any]],
    ) -> Tuple[bool, Dict[str, Any], str]:
        """Return (passes, audit_dict, rejection_reason)."""
        report: Dict[str, Any] = {}
        if not candidate or len(candidate.split()) < self.min_word_count:
            return False, report, "empty_or_short"
        if candidate.strip().lower() == base.strip().lower():
            return False, report, "identical_to_base"

        # Length ratio
        lr = _length_ratio(candidate, base)
        report["length_ratio"] = round(lr, 4)
        if not (self.length_ratio_min <= lr <= self.length_ratio_max):
            return False, report, "length_ratio"

        # SBERT band
        sim = self.sbert.cos(base, candidate)
        report["sbert_similarity"] = round(sim, 4)
        if sim < self.sbert_lower or sim > self.sbert_upper:
            return False, report, "sbert_band"

        # Token Jaccard floor (vs base and accepted variants)
        tov = _token_overlap(candidate, base)
        report["token_overlap_to_base"] = round(tov, 4)
        if tov > self.token_overlap_max:
            return False, report, "token_overlap"
        for v in accepted_variants:
            if _token_overlap(candidate, v.get("paraphrased_text", "")) > self.token_overlap_max:
                return False, report, "token_overlap_accepted"

        # SBERT semantic dedup vs accepted
        max_cos = 0.0
        for v in accepted_variants:
            c = self.sbert.cos(candidate, v.get("paraphrased_text", ""))
            if c > max_cos:
                max_cos = c
            if c > self.semantic_dedup_thresh:
                report["max_cos_to_accepted"] = round(c, 4)
                return False, report, "semantic_dedup"
        report["max_cos_to_accepted"] = round(max_cos, 4)

        # Bidirectional NLI
        nli_ok, p_fwd, p_bwd = self._bidirectional_nli(base, candidate)
        report["nli_entail_fwd"] = round(p_fwd, 4) if p_fwd is not None else None
        report["nli_entail_bwd"] = round(p_bwd, 4) if p_bwd is not None else None
        report["nli_passed"]     = (p_fwd is not None and p_bwd is not None)
        if not nli_ok:
            return False, report, "nli"

        return True, report, ""

    # ── Process one instance ───────────────────────────────────────────────

    def augment_record(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """Add up to `max_per_instance` back-translation variants to ``rec``.

        rec must contain ``base_text`` and ``variants`` (list of variant
        dicts in the GenSens schema). Returns the updated record.
        """
        base = rec.get("base_text", "")
        if not base:
            return rec

        existing = rec.get("variants", []) or []
        added: List[Dict[str, Any]] = []
        next_idx = len(existing)
        n_target = min(self.max_per_instance, len(self.pivot_langs))

        for lang in self.pivot_langs:
            if len(added) >= n_target:
                break
            try:
                cand = self._round_trip(base, lang)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  [{rec.get('instance_id','?')}] back-translation via {lang} failed: {e}")
                continue

            accepted = existing + added
            ok, report, reason = self._passes_filters(base, cand, accepted)
            if not ok:
                logger.debug(f"  [{rec.get('instance_id','?')}] {lang} rejected: {reason}")
                continue

            variant = {
                "variant_idx":           next_idx,
                "paraphrased_text":      cand,
                "strategy":              f"back_translation_{lang}",
                "strategy_family":       "back_translation",
                **report,
            }
            added.append(variant)
            next_idx += 1

        if added:
            rec["variants"] = existing + added
            # Update count fields if downstream tools read them.
            if "n_variants_generated" in rec:
                rec["n_variants_generated"] = len(rec["variants"])

        return rec


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",  required=True,
                   help="GenSens JSONL produced by paraphrase_generator")
    p.add_argument("--output", required=True,
                   help="Output JSONL with back-translation variants appended")
    p.add_argument("--pivot-langs", nargs="+",
                   default=["de", "fr", "ru"],
                   choices=sorted(PIVOT_MODELS.keys()),
                   help="Pivot languages (default: de fr ru)")
    p.add_argument("--max-per-instance", type=int, default=2,
                   help="Maximum back-translation variants per instance (default 2)")

    p.add_argument("--sbert-model", default=DEFAULT_SBERT_MODEL)
    p.add_argument("--nli-model",   default=DEFAULT_NLI_MODEL)
    p.add_argument("--enable-nli-ensemble", action="store_true")
    p.add_argument("--nli-ensemble-majority", type=int, default=2)

    p.add_argument("--nli-threshold",      type=float, default=DEFAULT_NLI_THRESHOLD)
    p.add_argument("--sbert-lower",        type=float, default=DEFAULT_SBERT_LOWER)
    p.add_argument("--sbert-upper",        type=float, default=DEFAULT_SBERT_UPPER)
    p.add_argument("--token-overlap-max",  type=float, default=DEFAULT_TOKEN_OVERLAP_MAX)
    p.add_argument("--semantic-dedup",     type=float, default=DEFAULT_SEMANTIC_DEDUP_THRESH)
    p.add_argument("--length-ratio-min",   type=float, default=DEFAULT_LENGTH_RATIO_MIN)
    p.add_argument("--length-ratio-max",   type=float, default=DEFAULT_LENGTH_RATIO_MAX)
    return p


def main() -> int:
    args = build_argparser().parse_args()
    in_path  = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        sys.exit(f"ERROR: input {in_path} not found")

    records = _load_jsonl(in_path)
    print(f"→ Loaded {len(records)} records from {in_path}")

    augmenter = BackTranslationFamily(
        pivot_langs           = args.pivot_langs,
        sbert_model           = args.sbert_model,
        nli_model_name        = args.nli_model,
        enable_nli_ensemble   = args.enable_nli_ensemble,
        nli_ensemble_majority = args.nli_ensemble_majority,
        nli_threshold         = args.nli_threshold,
        sbert_lower           = args.sbert_lower,
        sbert_upper           = args.sbert_upper,
        token_overlap_max     = args.token_overlap_max,
        semantic_dedup_thresh = args.semantic_dedup,
        length_ratio_min      = args.length_ratio_min,
        length_ratio_max      = args.length_ratio_max,
        max_per_instance      = args.max_per_instance,
    )

    t0 = time.time()
    n_added_total = 0
    augmented_records: List[Dict[str, Any]] = []
    for i, rec in enumerate(records, 1):
        n_before = len(rec.get("variants", []) or [])
        rec = augmenter.augment_record(rec)
        n_after  = len(rec.get("variants", []) or [])
        n_added_total += (n_after - n_before)
        augmented_records.append(rec)
        if i % 10 == 0 or i == len(records):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(records)}] +{n_added_total} BT variants total | "
                  f"{elapsed:.1f}s elapsed")

    _write_jsonl(augmented_records, out_path)
    print(f"\n✓ wrote {out_path}")
    print(f"  back-translation variants added: {n_added_total}")
    avg = n_added_total / max(1, len(records))
    print(f"  average per instance: {avg:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
