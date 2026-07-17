"""
csv_io.py — CSV persistence for the evaluation pipeline.

Goal: run the expensive H100 generation ONCE, persist to CSV, then re-evaluate
(and re-aggregate the PRI methodology) cheaply or with no GPU at all.

Three layers:
  A. dataset variants     → dataset CSV         (one row per prompt variant)
  B. model responses      → responses.csv       (one row per model×instance×variant)
  C. metric components    → scored_samples.csv  (one row per model×instance)
From C, all composite scores (PRI/ORI/IFI/Final) are pure arithmetic — no models.

All CSVs are written/read with pandas (QUOTE_MINIMAL) so article text containing
commas, quotes and newlines round-trips losslessly.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# ── Schemas ──────────────────────────────────────────────────────────────────

DATASET_COLS = [
    "instance_id", "task", "variant_idx", "strategy",
    "base_text", "paraphrased_text", "sbert_similarity",
    "full_prompt", "input_text", "reference_output",
]

RESPONSES_COLS = [
    "model", "instance_id", "topic_label", "variant_idx", "strategy",
    "prompt", "response", "input_text", "reference_output",
    # 70B judge outputs stored per variant so results are auditable without re-running
    "judge_raw_output",   # exact text the judge model produced
    "judge_score",        # normalised 0–1 score parsed from raw output
    # IFI saturation fix: the subject model is only resident during the
    # generate-only phase. The score-from-CSV phase has no generation model,
    # so without these the evaluator must default ppl_var/bf to 0 → IFI
    # saturates at 1.0 for every row (FLAWS §2.2). Compute once per
    # (model, instance) during generation and replicate across the K variant
    # rows so the value survives a partial resume. Blank for legacy CSVs.
    "ppl_var_inst",       # per-(model, instance) normalised PPL variance
    "bf_inst",            # per-(model, instance) normalised branching factor
]

# Raw per-(model,instance) metric components. Composites are derived from these.
# Flaw §2.2 fix: `ppl_var_norm`, `bf_norm`, and `ifi_norm` are populated by
# `normalize_ifi_per_model` *after* all rows for a model are written, so they
# are blank during the live streaming write and only become non-zero once the
# per-model min/max is known. The intra-model normalization makes IFI a true
# within-model comparison signal instead of saturating at 1.0.
SCORED_COLS = [
    "model", "instance_id", "topic_label", "n_variants",
    "sms", "auc_e", "trd", "kpig", "ppl_var", "bf",
    "ppl_var_norm", "bf_norm", "ifi_norm",
    "cs", "hs", "faithfulness", "human_score",
    "avg_length", "avg_coverage", "usd",
    "trd_semantic", "kpig_advanced",
    "diagnostic_ori", "diagnostic_ifi", "diagnostic_pri", "diagnosis",
    "rouge1", "rouge2", "rougeL",
    # 70B raw outputs stored at instance level (JSON arrays) for full auditability
    "judge_raw_outputs",          # JSON: [{"raw_output": str, "score": float}, ...]
    "faithfulness_raw_outputs",   # JSON: [{"entail_prob/raw_label": ..., "score": float}, ...]
]


# ── Layer A: dataset variants ────────────────────────────────────────────────

def gensens_jsonl_to_csv(jsonl_path: str, csv_path: str) -> int:
    """Flatten a GenSens JSONL dataset to one CSV row per variant. Returns row count."""
    rows: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            # Canonical bridge keys first; legacy summarization aliases as fallback.
            input_text = meta.get("input_text", meta.get("article", ""))
            reference_output = meta.get("reference_output", meta.get("gold_summary", ""))
            for v in rec.get("variants", []) or []:
                rows.append({
                    "instance_id": rec.get("instance_id", ""),
                    "task": rec.get("task", ""),
                    "variant_idx": v.get("variant_idx", 0),
                    "strategy": v.get("strategy", ""),
                    "base_text": rec.get("base_text", ""),
                    "paraphrased_text": v.get("paraphrased_text", ""),
                    "sbert_similarity": v.get("sbert_similarity", 0.0),
                    "full_prompt": v.get("full_prompt", ""),
                    "input_text": input_text,
                    "reference_output": reference_output,
                })
    df = pd.DataFrame(rows, columns=DATASET_COLS)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return len(df)


# ── Layer B: model responses ─────────────────────────────────────────────────

def write_responses_csv(rows: List[Dict], csv_path: str) -> int:
    """Write per-model×instance×variant response rows. Returns row count."""
    df = pd.DataFrame(rows, columns=RESPONSES_COLS)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return len(df)


def responses_rows_from_result(result: Dict) -> List[Dict]:
    """Build response rows from one evaluate_sample result dict."""
    rows = []
    prompts = result.get("prompts", [])
    responses = result.get("responses", [])
    strategies = result.get("strategies") or [None] * len(prompts)
    judge_raw = result.get("judge_raw_outputs") or []   # per-variant 70B judge outputs
    for i, (p, r) in enumerate(zip(prompts, responses)):
        judge_entry = judge_raw[i] if i < len(judge_raw) else {}
        rows.append({
            "model":            result.get("model", ""),
            "instance_id":      result.get("instance_id", ""),
            "topic_label":      result.get("topic_label", result.get("task", "")),
            "variant_idx":      i,
            "strategy":         strategies[i] if i < len(strategies) else None,
            "prompt":           p,
            "response":         r,
            "input_text":       result.get("input_text", ""),
            "reference_output": result.get("reference_output", ""),
            "judge_raw_output": judge_entry.get("raw_output", ""),
            "judge_score":      judge_entry.get("score", ""),
        })
    return rows


def read_responses_grouped(csv_path: str) -> Dict[Tuple[str, str], Dict]:
    """Read responses.csv → {(model, instance_id): sample dict ready to score}.

    Each value has: input_text, reference_output, topic_label, prompt_variants,
    precomputed_responses, strategies — i.e. exactly what evaluate_sample needs
    to SCORE without re-generating. When the persisted CSV includes the
    intra-model diagnostics (ppl_var_inst, bf_inst), these are surfaced as
    precomputed_ppl_var / precomputed_bf so the score-from-CSV path can keep
    a non-saturated IFI even with no subject model loaded (FLAWS §2.2 fix).
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df["variant_idx"] = df["variant_idx"].astype(int)
    has_ppl = "ppl_var_inst" in df.columns
    has_bf  = "bf_inst" in df.columns

    def _opt_float(series) -> Optional[float]:
        """Extract a single precomputed metric value if present, else None."""
        if series is None or len(series) == 0:
            return None
        v = str(series.iloc[0]).strip()
        if v == "" or v.lower() == "nan":
            return None
        try:
            return float(v)
        except ValueError:
            return None

    grouped: Dict[Tuple[str, str], Dict] = {}
    for (model, inst), g in df.groupby(["model", "instance_id"], sort=False):
        g = g.sort_values("variant_idx")
        entry = {
            "model": model,
            "instance_id": inst,
            "topic_label": g["topic_label"].iloc[0],
            "input_text": g["input_text"].iloc[0],
            "reference_output": g["reference_output"].iloc[0],
            "prompt_variants": g["prompt"].tolist(),
            "precomputed_responses": g["response"].tolist(),
            "strategies": g["strategy"].tolist(),
        }
        if has_ppl:
            v = _opt_float(g["ppl_var_inst"])
            if v is not None:
                entry["precomputed_ppl_var"] = v
        if has_bf:
            v = _opt_float(g["bf_inst"])
            if v is not None:
                entry["precomputed_bf"] = v
        grouped[(model, inst)] = entry
    return grouped


# ── Layer C: scored components ───────────────────────────────────────────────

def scored_row_from_result(result: Dict) -> Dict:
    """Extract the raw metric components from one result dict (for reaggregation)."""
    m = result.get("metrics", {})
    rouge = result.get("rouge", {}) or {}
    return {
        "model": result.get("model", ""),
        "instance_id": result.get("instance_id", ""),
        "topic_label": result.get("topic_label", result.get("task", "")),
        "n_variants": len(result.get("responses", [])),
        "sms": m.get("sms", 0.0),
        "auc_e": m.get("auc_e", 0.0),
        "trd": m.get("trd", 0.0),
        "kpig": m.get("kpig", 0.0),
        "ppl_var": m.get("ppl_var", 0.0),
        "bf": m.get("bf", 0.0),
        # Per-model normalized intra-model diagnostics — filled in by
        # normalize_ifi_per_model() once all rows for a model are collected.
        "ppl_var_norm": "",
        "bf_norm": "",
        "ifi_norm": "",
        "cs": result.get("cs", 0.0),
        "hs": result.get("hs_score", 0.0),
        "faithfulness": result.get("faithfulness", 0.0),
        "human_score": result.get("human_score", 0.0),
        "avg_length": result.get("avg_length", 0.0),
        "avg_coverage": result.get("avg_coverage", 0.0),
        "usd": result.get("usd", 0.0),
        "trd_semantic": result.get("trd_semantic", 0.0),
        "kpig_advanced": result.get("kpig_advanced", 0.0),
        "diagnostic_ori": result.get("diagnostic_ori", 0.0),
        "diagnostic_ifi": result.get("diagnostic_ifi", 0.0),
        "diagnostic_pri": result.get("diagnostic_pri", 0.0),
        "diagnosis": result.get("diagnosis", ""),
        "rouge1": rouge.get("rouge1", 0.0),
        "rouge2": rouge.get("rouge2", 0.0),
        "rougeL": rouge.get("rougeL", 0.0),
        "judge_raw_outputs":        json.dumps(result.get("judge_raw_outputs") or []),
        "faithfulness_raw_outputs": json.dumps(result.get("faithfulness_raw_outputs") or []),
    }


def write_scored_csv(rows: List[Dict], csv_path: str) -> int:
    df = pd.DataFrame(rows, columns=SCORED_COLS)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return len(df)


# ── Validation ───────────────────────────────────────────────────────────────

def _validate(csv_path: str, required_cols: List[str], required_nonnull: List[str]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    p = Path(csv_path)
    if not p.is_file():
        return False, [f"file not found: {csv_path}"]
    df = pd.read_csv(csv_path, dtype=str)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        issues.append(f"missing columns: {missing}")
    extra = [c for c in df.columns if c not in required_cols]
    if extra:
        issues.append(f"unexpected columns: {extra}")
    if len(df) == 0:
        issues.append("zero rows")
    for c in required_nonnull:
        if c in df.columns and df[c].isna().any():
            issues.append(f"nulls in required column '{c}'")
    return (len(issues) == 0), issues


def validate_dataset_csv(csv_path: str) -> Tuple[bool, List[str]]:
    return _validate(csv_path, DATASET_COLS, ["instance_id", "full_prompt", "input_text"])


def validate_responses_csv(csv_path: str) -> Tuple[bool, List[str]]:
    ok, issues = _validate(csv_path, RESPONSES_COLS, ["model", "instance_id", "prompt", "response"])
    # Consistency: each (model,instance) should have contiguous 0..n-1 variant_idx.
    if ok:
        df = pd.read_csv(csv_path)
        for (mdl, inst), g in df.groupby(["model", "instance_id"]):
            idx = sorted(g["variant_idx"].tolist())
            if idx != list(range(len(idx))):
                issues.append(f"non-contiguous variant_idx for {mdl}/{inst}: {idx}")
    return (len(issues) == 0), issues


def validate_scored_csv(csv_path: str) -> Tuple[bool, List[str]]:
    return _validate(csv_path, SCORED_COLS, ["model", "instance_id"])


def normalize_ifi_per_model(scored_csv_path: str) -> None:
    """Fill in ppl_var_norm, bf_norm, ifi_norm by per-model min-max normalization.

    Implements the Flaw §2.2 fix: the raw `ppl_var` / `bf` numbers are on
    different scales for every model, so a global IFI (1 - (ppl + bf)/2)
    saturates near 1.0 and loses all discriminating power. Normalizing within
    a model rescales the diagnostic to a meaningful per-model spread.

    Reads the scored CSV, rewrites it in place with the three norm columns
    populated. Safe to call at the very end of a benchmark run.
    """
    if not os.path.exists(scored_csv_path):
        return
    df = pd.read_csv(scored_csv_path)
    if df.empty or "model" not in df.columns:
        return

    df["ppl_var_norm"] = 0.0
    df["bf_norm"] = 0.0
    df["ifi_norm"] = 1.0
    for model_name, g in df.groupby("model", sort=False):
        ppl_max = max(1e-3, float(g["ppl_var"].astype(float).max()))
        bf_max = max(1e-3, float(g["bf"].astype(float).max()))
        ppl_norm = g["ppl_var"].astype(float) / ppl_max
        bf_norm = g["bf"].astype(float) / bf_max
        ifi_norm = (1.0 - (ppl_norm + bf_norm) / 2.0).clip(lower=0.0, upper=1.0)
        df.loc[g.index, "ppl_var_norm"] = ppl_norm
        df.loc[g.index, "bf_norm"] = bf_norm
        df.loc[g.index, "ifi_norm"] = ifi_norm

    # Re-order columns to match the canonical SCORED_COLS schema.
    out_cols = [c for c in SCORED_COLS if c in df.columns]
    df = df[out_cols]
    df.to_csv(scored_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)


# ── Fault-tolerant incremental writing + resume ──────────────────────────────

class IncrementalCSVWriter:
    """Append rows one batch at a time, flushing + fsync after each write.

    Guarantees that whatever has been computed is already on disk — so if the
    process crashes (OOM, killed box, model error) the completed rows survive.
    Opens in append mode when the file already exists (resume), writing the
    header only for a fresh file.
    """

    def __init__(self, path: str, columns: List[str], resume: bool = True):
        self.path = path
        self.columns = columns
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        append = resume and file_exists
        if append:
            self._migrate_schema_if_needed()
        self._f = open(path, "a" if append else "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=columns, quoting=csv.QUOTE_MINIMAL)
        if not append:
            self._w.writeheader()
            self._flush()

    def _migrate_schema_if_needed(self) -> None:
        """Add newly introduced columns before appending to an existing CSV."""
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])

        if existing_header == self.columns:
            return

        missing = [c for c in self.columns if c not in existing_header]
        extra = [c for c in existing_header if c not in self.columns]
        if not missing and not extra:
            return

        df = pd.read_csv(self.path, dtype=str).fillna("")
        for col in missing:
            df[col] = ""
        df = df[[c for c in self.columns if c in df.columns]]
        df.to_csv(self.path, index=False, quoting=csv.QUOTE_MINIMAL)

    def write_rows(self, rows: List[Dict]) -> None:
        for r in rows:
            self._w.writerow({k: r.get(k, "") for k in self.columns})
        self._flush()

    def _flush(self) -> None:
        self._f.flush()
        os.fsync(self._f.fileno())   # durable across a crash / power loss

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _existing_keys(path: str, key_cols: List[str]) -> Set[Tuple]:
    """Return the set of already-written key tuples (for resume). Empty if no file."""
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return set()
    try:
        df = pd.read_csv(path, dtype=str, usecols=key_cols)
    except Exception:
        return set()
    return {tuple(row) for row in df[key_cols].itertuples(index=False, name=None)}


def existing_response_keys(path: str) -> Set[Tuple[str, str]]:
    """(model, instance_id) pairs already present in responses.csv."""
    return _existing_keys(path, ["model", "instance_id"])


def existing_scored_keys(path: str) -> Set[Tuple[str, str]]:
    """(model, instance_id) pairs already present in scored_samples.csv."""
    return _existing_keys(path, ["model", "instance_id"])
