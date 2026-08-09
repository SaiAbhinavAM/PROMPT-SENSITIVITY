#!/usr/bin/env python3
"""
compute_cell_metrics.py — per-cell SENSITIVITY (spread) and QUALITY (mean)
metrics for the crossed-design GenSens summarization benchmark.

A "cell" = one (article_id, seed_id) group of 6 variant outputs (variant_idx
-1..4). For each cell we hold the article fixed and measure how much the 6
outputs DIFFER (sensitivity = spread across the 6) versus how good they are
ON AVERAGE (quality = mean across the 6) — see method.md 2026-07-06 entry
for why these must never be collapsed into one number.

Heavy ML deps (sentence-transformers, transformers/torch, rouge_score,
bert_score) are imported lazily inside main() so `--help` and import-time
checks work on a machine without a GPU.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("compute_cell_metrics")

_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9]+\b")

# ── Instruction-echo / preamble stripping (method.md 2026-07-09 fix) ──────────
# Instruction-tuned Llama frequently prefixes summaries with a meta line that
# echoes the instruction — "Here is a summary of the article in 3-4 sentences:",
# "**Main Events and Key Details:**", "Here are the key points:" — and the exact
# wording VARIES across paraphrases (some outputs have it, some don't). That
# boilerplate variation is not summary-content sensitivity: it inflates SMS drift
# and dilutes the quality metrics (ROUGE / cosine / entity coverage). We strip a
# leading meta clause before computing the CONTENT/QUALITY metrics. The raw
# generation is preserved for the PPL/branching-factor pass (those measure the
# model's confidence over what it actually generated). Conservative by design:
# only a short (≤20-word) leading segment ending in a colon that is clearly a
# meta-announcement is removed, and never to the point of emptying the output.
_ANNOUNCE_RE = re.compile(
    r"^(here(?:'|’)?s|here is|here are|these are|below is|the following|"
    r"sure\b|certainly\b|of course\b|absolutely\b)", re.I)
_META_KW = ("summary", "summarize", "summarise", "synopsis", "rundown", "overview",
            "tl;dr", "recap", "key point", "key detail", "main point", "main event",
            "in summary", "significant detail", "essential point", "crucial", "takeaway",
            "significant and relevant", "most important")


def strip_preamble(text: str) -> str:
    """Remove a leading instruction-echo/preamble clause from a summary output.
    Returns the summary body; falls back to the trimmed original if stripping
    would empty it or the leading clause is not clearly a meta-announcement."""
    t = (text or "").strip()
    for _ in range(3):  # outputs occasionally stack a header + an announcement
        t2 = t.lstrip(" *#->\t").strip()
        m = re.match(r"^(.{0,180}?):(?:\s+|\s*\n+|\*+\s*)", t2)
        if not m:
            t = t2
            break
        prefix = m.group(1)
        pl = prefix.lower().strip(" *#-\t")
        is_meta = bool(_ANNOUNCE_RE.match(pl)) or any(k in pl for k in _META_KW)
        if is_meta and len(prefix.split()) <= 20:
            rest = t2[m.end():].strip(" *#->\t\n")
            if rest:
                t = rest
                continue
        t = t2
        break
    return t


def entity_coverage(output_text: str, gold_text: str) -> float:
    """|gold_entities ∩ output_entities| / |gold_entities|, capitalized-token
    based (per spec). 1.0 if the gold reference has no capitalized tokens
    (nothing to miss)."""
    gold_ents = set(_ENTITY_RE.findall(gold_text or ""))
    if not gold_ents:
        return 1.0
    out_ents = set(_ENTITY_RE.findall(output_text or ""))
    return len(gold_ents & out_ents) / len(gold_ents)


def group_into_cells(rows):
    cells = {}
    for row in rows:
        key = (row["article_id"], row["seed_id"])
        cells.setdefault(key, {})[row["variant_idx"]] = row
    return cells


def cosine_matrix(embs: np.ndarray) -> np.ndarray:
    norm = embs / np.clip(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12, None)
    return norm @ norm.T


def mean_pairwise_cosine(embs: np.ndarray) -> float:
    """Mean of the off-diagonal upper triangle of the pairwise cosine matrix."""
    n = embs.shape[0]
    if n < 2:
        return 1.0
    sim = cosine_matrix(embs)
    iu = np.triu_indices(n, k=1)
    return float(sim[iu].mean())


def truncate_premise_for_nli(tokenizer, premise: str, hypothesis: str, max_length: int) -> str:
    """Truncate the article (premise) so the NLI pair fits max_length while
    keeping the hypothesis (model output) whole — per spec, never truncate
    the output. Falls back to the full premise if the hypothesis alone
    already exceeds the budget (flagged by the caller via a warning)."""
    hyp_ids = tokenizer.encode(hypothesis, add_special_tokens=False)
    special_budget = 4  # CLS/SEP-style tokens for a pair encoding
    available = max_length - len(hyp_ids) - special_budget
    if available <= 0:
        return ""  # caller logs/flags this; NLI score will default to neutral
    prem_ids = tokenizer.encode(premise, add_special_tokens=False)
    if len(prem_ids) <= available:
        return premise
    truncated_ids = prem_ids[:available]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)


def batched_teacher_forced_metrics(model, tokenizer, device, prompt_texts, response_texts, max_model_len, batch_size=8):
    """Real per-token PPL and branching-factor (exp of mean softmax entropy)
    via a single teacher-forced HF forward pass per (prompt, response) pair
    — never faked. Returns (ppl_list, bf_list), with None entries for
    responses that were empty (no tokens to score)."""
    import torch

    n = len(prompt_texts)
    ppl_out = [None] * n
    bf_out = [None] * n

    idx = 0
    while idx < n:
        batch_idx = list(range(idx, min(idx + batch_size, n)))
        seqs = []
        resp_lens = []
        for i in batch_idx:
            prompt_ids = tokenizer(prompt_texts[i], add_special_tokens=False)["input_ids"]
            resp_ids = tokenizer(response_texts[i], add_special_tokens=False)["input_ids"]
            if len(resp_ids) == 0:
                seqs.append(None)
                resp_lens.append(0)
                continue
            total_len = len(prompt_ids) + len(resp_ids)
            if total_len > max_model_len:
                # Keep the response whole; truncate the prompt from the left.
                keep_prompt = max(0, max_model_len - len(resp_ids))
                prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
            seqs.append((prompt_ids, resp_ids))
            resp_lens.append(len(resp_ids))
        idx += batch_size

        real = [(i, s) for i, s in zip(batch_idx, seqs) if s is not None]
        if not real:
            continue

        max_total = max(len(p) + len(r) for _, (p, r) in real)
        input_ids = torch.full((len(real), max_total), tokenizer.pad_token_id or tokenizer.eos_token_id, dtype=torch.long)
        attn_mask = torch.zeros((len(real), max_total), dtype=torch.long)
        resp_start = [0] * len(real)
        resp_len = [0] * len(real)
        for row_i, (_, (p, r)) in enumerate(real):
            seq = p + r
            input_ids[row_i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn_mask[row_i, : len(seq)] = 1
            resp_start[row_i] = len(p)
            resp_len[row_i] = len(r)

        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits  # (B, T, V), model dtype (bf16)

            # MEMORY-CRITICAL: do NOT softmax the full (B, T, V) tensor — with V≈128k
            # that materializes several GB of fp32 and OOMs a 24GB card that already
            # holds the 16GB model. Instead slice to each row's RESPONSE positions
            # first (l ≤ max_tokens ≪ T), then softmax only that small (l, V) slice.
            # PERF: keep the per-row NLL/entropy scalars ON the GPU and transfer them
            # ONCE per batch (a single .cpu()), instead of an .item() per row which
            # forces a GPU sync ~18k times over the full run.
            batch_keys = []
            batch_nll = []
            batch_ent = []
            for row_i, (orig_i, _) in enumerate(real):
                s, l = resp_start[row_i], resp_len[row_i]
                if l == 0:
                    continue
                # Position t's logits predict token t+1, so response token at
                # absolute position (s+k) is predicted from logits at (s+k-1).
                row_logits = logits[row_i, s - 1: s - 1 + l, :].float()  # (l, V) — small
                row_logprobs = torch.log_softmax(row_logits, dim=-1)
                target_ids = input_ids[row_i, s: s + l]
                token_logprobs = row_logprobs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
                batch_keys.append(orig_i)
                batch_nll.append(-token_logprobs.mean())                             # GPU scalar
                batch_ent.append(-(row_logprobs.exp() * row_logprobs).sum(dim=-1).mean())  # GPU scalar
                del row_logits, row_logprobs

            if batch_keys:
                nll_cpu = torch.stack(batch_nll).cpu().numpy()   # single transfer/sync
                ent_cpu = torch.stack(batch_ent).cpu().numpy()
                for j, orig_i in enumerate(batch_keys):
                    ppl_out[orig_i] = float(np.exp(min(float(nll_cpu[j]), 50.0)))  # clamp to avoid overflow
                    bf_out[orig_i] = float(np.exp(float(ent_cpu[j])))

        del logits, input_ids, attn_mask
        if device == "cuda":
            torch.cuda.empty_cache()

    return ppl_out, bf_out


def _empty_gpu_cache(log=None):
    """Reclaim freed GPU memory (call AFTER `del`-ing the scorer in the caller's
    own scope — deleting a function parameter would not drop the caller's
    binding). Important when Phase 2 stacks SBERT + faithfulness + BERTScore +
    (optionally) the 8B PPL pass on one card: freeing each scorer as we finish
    keeps headroom for the 8B."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if log is not None:
                free_b, total_b = torch.cuda.mem_get_info()
                log.info(f"GPU free after releasing scorer: {free_b/1e9:.1f} / {total_b/1e9:.1f} GiB")
    except Exception:
        pass


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str):
    """Lightweight, dependency-free sentence splitter (no nltk punkt download to
    fail on a fresh box). Good enough to feed MiniCheck per-sentence."""
    t = (text or "").strip()
    if not t:
        return []
    return [p.strip() for p in _SENT_SPLIT_RE.split(t) if p.strip()]


def compute_faithfulness_minicheck(model_name, docs, outputs, sentence_level, cache_dir, warnings, log):
    """Per-output faithfulness in [0,1] via MiniCheck (support probability of the
    summary given the article). Long articles are chunked by MiniCheck itself, so
    no premise truncation is needed. Multi-sentence summaries are scored
    sentence-by-sentence (MiniCheck's recommended usage) and averaged. Returns a
    flat list aligned with `outputs`. Never fakes: raises if the package/model
    cannot load so the caller can fall back to the NLI backend or abort."""
    from minicheck.minicheck import MiniCheck  # lazy: optional heavy dep

    scorer = MiniCheck(model_name=model_name, cache_dir=cache_dir)
    n = len(outputs)
    pair_docs, pair_claims, owner = [], [], []
    for i, (doc, out) in enumerate(zip(docs, outputs)):
        if not (out or "").strip():
            continue  # empty generation -> no pair -> neutral 0.5 via the acc fallback below
        sents = split_sentences(out) if sentence_level else [out.strip()]
        for s in sents:
            pair_docs.append(doc)
            pair_claims.append(s)
            owner.append(i)
    log.info(f"MiniCheck[{model_name}] scoring {len(pair_claims)} (doc, claim) pairs "
             f"(sentence_level={sentence_level}) over {n} outputs ...")
    _, raw_prob, _, _ = scorer.score(docs=pair_docs, claims=pair_claims)
    from collections import defaultdict
    acc = defaultdict(list)
    for p, o in zip(raw_prob, owner):
        if p is not None:
            acc[o].append(float(p))
    # An output with no scorable sentence (empty generation) defaults to neutral 0.5.
    faith = [float(np.mean(acc[i])) if acc.get(i) else 0.5 for i in range(n)]
    del scorer  # release MiniCheck (in this scope) before the PPL pass loads the 8B
    _empty_gpu_cache(log)
    return faith


def compute_faithfulness_nli(nli_model, nli_max_length, premise_max_chars, docs, outputs, warnings, log):
    """Fallback faithfulness: 3-way NLI cross-encoder, P(entail)+0.5·P(neutral) of
    article→output, article truncated (never the output) to the NLI window. Returns
    a flat list aligned with `outputs`."""
    from sentence_transformers import CrossEncoder

    nli = CrossEncoder(nli_model, max_length=nli_max_length)
    id2label = {int(k): v.lower() for k, v in nli.config.id2label.items()}
    pairs = []
    empty_premise_flags = 0
    for doc, out in zip(docs, outputs):
        premise = truncate_premise_for_nli(nli.tokenizer, (doc or "")[:premise_max_chars], out, nli_max_length)
        if not premise:
            empty_premise_flags += 1
        pairs.append((premise, out))
    if empty_premise_flags:
        warnings.append(f"{empty_premise_flags} NLI pairs had an output too long for any premise context; scored neutral (0.5).")
    log.info(f"NLI[{nli_model}] scoring {len(pairs)} (article, output) pairs ...")
    logits = nli.predict(pairs, batch_size=8, show_progress_bar=True, apply_softmax=False)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    entail_idx = next(i for i, lbl in id2label.items() if "entail" in lbl)
    neutral_idx = next(i for i, lbl in id2label.items() if "neutral" in lbl)
    faith = (probs[:, entail_idx] + 0.5 * probs[:, neutral_idx]).tolist()
    for i, (premise, _) in enumerate(pairs):
        if not premise:
            faith[i] = 0.5
    del nli  # release the NLI model (in this scope) before the PPL pass loads the 8B
    _empty_gpu_cache(log)
    return faith


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                     help="Must match the generation model — PPL/entropy are model-relative.")
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    # ── Faithfulness backend (method.md 2026-08-09) ──────────────────────────
    # 'minicheck' (DEFAULT): MiniCheck (EMNLP 2024), a dedicated fact-checking
    # model that is SOTA on the LLM-AggreFact leaderboard, deterministic, and
    # CHUNKS long documents internally — so it needs no article truncation and
    # produces no "output too long → default neutral" fallbacks. Multi-sentence
    # summaries are scored per-sentence (MiniCheck's recommended usage) and
    # averaged. 'nli': the previous NLI-cross-encoder path, kept as a
    # zero-extra-dependency fallback (works if the minicheck package can't be
    # installed on the box).
    ap.add_argument("--faithfulness_backend", choices=["minicheck", "nli"], default="minicheck")
    ap.add_argument("--faithfulness_model", default="deberta-v3-large",
                     help="MiniCheck model_name: roberta-large | deberta-v3-large | flan-t5-large | "
                          "Bespoke-MiniCheck-7B. Only used when --faithfulness_backend=minicheck.")
    ap.add_argument("--faith_whole_summary", action="store_true",
                     help="Score each summary as ONE claim instead of per-sentence. Per-sentence "
                          "(default) is MiniCheck's recommended usage and scores more faithfully.")
    ap.add_argument("--minicheck_cache_dir", default=None, help="Where MiniCheck caches its weights.")
    # NLI fallback backend knobs (unused when backend=minicheck). Long-context NLI
    # (1280 tokens) lets the article premise fit alongside longer summaries. Label
    # order is matched by NAME, so any 3-way NLI model id works here.
    ap.add_argument("--nli_model", default="tasksource/deberta-base-long-nli")
    ap.add_argument("--nli_max_length", type=int, default=1280)
    ap.add_argument("--nli_premise_max_chars", type=int, default=8000,
                     help="Cheap pre-truncation of the article before the doc/premise is scored.")
    # ── Correctness-vs-gold now uses BERTScore (method.md 2026-08-09) ────────
    ap.add_argument("--bertscore_model", default="roberta-large",
                     help="BERTScore backbone for correctness-vs-gold. 'roberta-large' (default) or a "
                          "stronger 'microsoft/deberta-xlarge-mnli' if GPU budget allows.")
    ap.add_argument("--no_bertscore_rescale", action="store_true",
                     help="Disable rescale_with_baseline. Rescaling (default ON) spreads BERTScore over "
                          "a usable [0,1] range instead of the compressed ~0.85–0.95 band.")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--ppl_batch_size", type=int, default=4,
                     help="Batch size for the teacher-forced PPL/entropy pass. 4 is proven safe at "
                          "30-article scale; the (batch, seq_len, vocab) logits tensor is the memory "
                          "limit. Speed comes from the batched GPU->CPU transfer, not the batch size.")
    ap.add_argument("--skip_ppl_entropy", action="store_true",
                     help="Mark ppl_var/pc_stab_var as not-computed instead of running the HF "
                          "forward pass (e.g. if GPU time/memory is prohibitive). Never faked.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else common.RESULTS_DIR_DEFAULT
    responses_path = results_dir / "responses_crossed.jsonl"
    out_path = results_dir / "cell_metrics.jsonl"

    rows = common.read_jsonl(responses_path)
    if not rows:
        raise FileNotFoundError(f"No responses found at {responses_path}. Run run_inference_crossed.py first.")
    log.info(f"Loaded {len(rows)} response rows.")

    # Expected variant_idx set per seed — usually {-1,0,1,2,3,4}, but seeds with a
    # known upstream gap (e.g. B12, see common.py) legitimately have fewer. A cell
    # is "complete" when it has ALL of its seed's expected variants, not a flat 6.
    seeds = common.load_seed_prompts()
    expected_variants_by_seed = {s["seed_id"]: {v["variant_idx"] for v in s["variants"]} for s in seeds}

    cells = group_into_cells(rows)
    complete_cells = {}
    for k, v in cells.items():
        _article_id, seed_id = k
        expected = expected_variants_by_seed.get(seed_id)
        if expected is not None and set(v.keys()) == expected:
            complete_cells[k] = v
    incomplete = len(cells) - len(complete_cells)
    if incomplete:
        log.warning(f"{incomplete} cells are missing variants relative to their seed's expected set (inference incomplete) — skipping them.")
    log.info(f"{len(complete_cells)} complete cells to score.")

    cell_keys = sorted(complete_cells.keys())
    cell_rows = [[complete_cells[k][vi] for vi in sorted(complete_cells[k].keys())] for k in cell_keys]

    # Strip instruction-echo preamble for the CONTENT/QUALITY metrics; keep the
    # raw generation for the PPL/branching-factor pass. Store the cleaned text on
    # each row as "output_clean" so the per-cell assembly reads a single source.
    warnings = []
    n_cleaned = 0
    for cell in cell_rows:
        for r in cell:
            r["output_clean"] = strip_preamble(r["output"])
            if r["output_clean"] != (r["output"] or "").strip():
                n_cleaned += 1
    flat_outputs = [r["output"] for cell in cell_rows for r in cell]          # raw (for PPL)
    flat_outputs_clean = [r["output_clean"] for cell in cell_rows for r in cell]  # cleaned (content metrics)
    flat_gold = [r["gold_summary"] for cell in cell_rows for r in cell]
    log.info(f"Preamble stripped from {n_cleaned}/{len(flat_outputs)} outputs "
             f"({100*n_cleaned/max(len(flat_outputs),1):.0f}%) for content/quality metrics.")
    warnings.append(f"Instruction-echo preamble stripped from {n_cleaned}/{len(flat_outputs)} outputs "
                    f"before SMS/CS/ROUGE/BERTScore/faithfulness; PPL/BF use raw generation.")

    # ── SMS: SBERT embeddings ────────────────────────────────────────────
    log.info(f"Embedding {len(flat_outputs_clean)} (cleaned) outputs with {args.embedder} ...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(args.embedder)
    output_embs = np.asarray(embedder.encode(flat_outputs_clean, batch_size=64, show_progress_bar=True, convert_to_numpy=True))
    # (Gold embeddings are no longer needed: correctness-vs-gold now uses BERTScore,
    # and SBERT is used only for output-to-output sms_drift.)

    # ── ROUGE-L, BERTScore vs gold (on cleaned outputs) ──────────────────
    log.info("Computing ROUGE-L ...")
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL_scores = [scorer.score(g, o)["rougeL"].fmeasure for g, o in zip(flat_gold, flat_outputs_clean)]

    log.info(f"Computing BERTScore [{args.bertscore_model}, rescale={not args.no_bertscore_rescale}] over "
             f"{len(flat_outputs_clean)} pairs — now feeds correctness-vs-gold (its designed use) ...")
    from bert_score import score as bertscore_fn
    _, _, bertscore_f1 = bertscore_fn(
        flat_outputs_clean, flat_gold, lang="en",
        model_type=args.bertscore_model,
        rescale_with_baseline=not args.no_bertscore_rescale,
        verbose=False, batch_size=64,
    )
    bertscore_scores = [float(x) for x in bertscore_f1]

    # ── Faithfulness (article → output): MiniCheck (default) or NLI fallback ──
    # Free the SBERT embedder + reclaim GPU cache before loading the faithfulness
    # model. faith_docs is aligned with flat_outputs_clean (cell-then-variant order).
    import gc as _gc
    import torch as _torch
    try:
        del embedder
    except Exception:
        pass
    _gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()

    article_lookup = {a["article_id"]: a["text"] for a in common.read_jsonl(results_dir / "articles_sample.jsonl")}
    faith_docs = [article_lookup.get(key[0], "") for key, cell in zip(cell_keys, cell_rows) for _ in cell]

    if args.faithfulness_backend == "minicheck":
        try:
            faith_scores = compute_faithfulness_minicheck(
                args.faithfulness_model, faith_docs, flat_outputs_clean,
                sentence_level=not args.faith_whole_summary,
                cache_dir=args.minicheck_cache_dir, warnings=warnings, log=log,
            )
            warnings.append(f"Faithfulness: MiniCheck[{args.faithfulness_model}], "
                            f"{'per-sentence' if not args.faith_whole_summary else 'whole-summary'}.")
        except Exception as e:
            log.warning(f"MiniCheck backend failed ({type(e).__name__}: {str(e)[:160]}); falling back to "
                        f"NLI[{args.nli_model}]. Install with: "
                        f"pip install 'minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main'")
            warnings.append(f"MiniCheck unavailable ({type(e).__name__}); fell back to NLI[{args.nli_model}].")
            faith_scores = compute_faithfulness_nli(
                args.nli_model, args.nli_max_length, args.nli_premise_max_chars,
                faith_docs, flat_outputs_clean, warnings, log,
            )
    else:
        faith_scores = compute_faithfulness_nli(
            args.nli_model, args.nli_max_length, args.nli_premise_max_chars,
            faith_docs, flat_outputs_clean, warnings, log,
        )

    # ── PPL / branching factor: real HF forward pass, or not-computed ──
    if args.skip_ppl_entropy:
        log.warning("--skip_ppl_entropy set: ppl_var/pc_stab_var will be marked not-computed (None).")
        ppl_scores = [None] * len(flat_outputs)
        bf_scores = [None] * len(flat_outputs)
        warnings.append("PPL_var/PC_stab_var not computed (--skip_ppl_entropy).")
    else:
        import gc
        import torch

        # Free the scorer models still resident on the GPU (SBERT embedder, NLI
        # cross-encoder, and BERTScore's cached roberta-large) BEFORE loading the
        # 16GB Llama-8B — otherwise all of them plus the 8B co-reside on the 24GB
        # card and the teacher-forced pass OOMs. Their outputs are already computed.
        try:
            del embedder
        except Exception:
            pass
        try:
            del nli
        except Exception:
            pass
        # BERTScore caches roberta-large in a module global with no public free
        # API; gc.collect() below drops our references and empty_cache() reclaims
        # what it can. The slice-before-softmax PPL path keeps headroom small
        # enough that a residual BERTScore cache is not fatal.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_b, total_b = torch.cuda.mem_get_info()
            log.info(f"GPU free after releasing scorer models: {free_b/1e9:.1f} / {total_b/1e9:.1f} GiB")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        log.info(f"Loading {args.model} via HF transformers for teacher-forced PPL/entropy ...")
        hf_tok = AutoTokenizer.from_pretrained(args.model)
        if hf_tok.pad_token_id is None:
            hf_tok.pad_token = hf_tok.eos_token
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            warnings.append("No CUDA device found for PPL/entropy pass — running on CPU (will be very slow).")
        hf_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", trust_remote_code=True).to(device)
        hf_model.eval()

        prompt_texts = []
        for cell in cell_rows:
            for r in cell:
                prompt_texts.append(hf_tok.apply_chat_template(
                    [{"role": "user", "content": r["full_prompt"]}],
                    tokenize=False, add_generation_prompt=True,
                ))
        ppl_scores, bf_scores = batched_teacher_forced_metrics(
            hf_model, hf_tok, device, prompt_texts, flat_outputs, args.max_model_len, args.ppl_batch_size,
        )
        n_none = sum(1 for p in ppl_scores if p is None)
        if n_none:
            warnings.append(f"{n_none}/{len(ppl_scores)} outputs had no scorable tokens (empty generation) — PPL/BF left as None for those.")

    # ── Assemble per-cell rows ───────────────────────────────────────────
    log.info("Assembling per-cell metrics ...")
    out_rows = []
    flat_i = 0
    for key, cell in zip(cell_keys, cell_rows):
        article_id, seed_id = key
        pool = cell[0]["pool"]
        dimension = cell[0]["dimension"]

        idxs = list(range(flat_i, flat_i + len(cell)))
        flat_i += len(cell)

        embs = output_embs[idxs]
        sms_similarity = mean_pairwise_cosine(embs)
        sms_drift = 1.0 - sms_similarity

        gold = cell[0]["gold_summary"]
        # Correctness vs gold = 0.5·BERTScore-F1 + 0.5·entity coverage. BERTScore
        # (token-level contextual match to the reference) is BERTScore's designed
        # purpose; SBERT is reserved for output-to-output sms_drift above.
        cs_list = []
        for pos in range(len(cell)):
            bsc = max(0.0, min(1.0, float(bertscore_scores[idxs[pos]])))
            cov = entity_coverage(cell[pos]["output_clean"], gold)
            cs_list.append(0.5 * bsc + 0.5 * cov)
        cs_var = float(np.var(cs_list))
        cs_mean = float(np.mean(cs_list))

        faith_list = [faith_scores[i] for i in idxs]
        faith_var = float(np.var(faith_list))
        faith_mean = float(np.mean(faith_list))

        ppl_list = [ppl_scores[i] for i in idxs]
        if all(p is not None for p in ppl_list):
            mean_ppl = float(np.mean(ppl_list))
            std_ppl = float(np.std(ppl_list))
            ppl_var = min(std_ppl / mean_ppl, 5.0) if mean_ppl > 0 else None
        else:
            ppl_var = None

        bf_list = [bf_scores[i] for i in idxs]
        # Store BOTH the spread (variance) and the LEVEL (mean) of the branching
        # factor so aggregate_scores.py can level-normalize it (pc_stab_cv =
        # std/mean) the same way ppl_var / cs_cv / faith_cv are — otherwise raw
        # variance of an unbounded quantity is entangled with its magnitude
        # (method.md 2026-08-09). Runs generated before this field existed fall
        # back to raw variance in aggregate_scores, so their numbers are unchanged.
        if all(b is not None for b in bf_list):
            pc_stab_var = float(np.var(bf_list))
            pc_stab_mean = float(np.mean(bf_list))
        else:
            pc_stab_var = None
            pc_stab_mean = None

        rouge_list = [rougeL_scores[i] for i in idxs]
        rougeL_var = float(np.var(rouge_list))
        rougeL_mean = float(np.mean(rouge_list))

        bert_list = [bertscore_scores[i] for i in idxs]
        bertscore_var = float(np.var(bert_list))
        bertscore_mean = float(np.mean(bert_list))

        lens = [len(r["output_clean"].split()) for r in cell]
        mean_output_len = float(np.mean(lens))
        if mean_output_len < 12:
            warnings.append(f"cell ({article_id},{seed_id}): mean_output_len={mean_output_len:.1f} < 12 words.")

        out_rows.append({
            "article_id": article_id,
            "seed_id": seed_id,
            "pool": pool,
            "dimension": dimension,
            "sms_drift": sms_drift,
            "cs_var": cs_var,
            "faith_var": faith_var,
            "ppl_var": ppl_var,
            "pc_stab_var": pc_stab_var,
            "pc_stab_mean": pc_stab_mean,
            "rougeL_var": rougeL_var,
            "bertscore_var": bertscore_var,
            "sms_similarity": sms_similarity,
            "cs_mean": cs_mean,
            "faith_mean": faith_mean,
            "rougeL_mean": rougeL_mean,
            "bertscore_mean": bertscore_mean,
            "mean_output_len": mean_output_len,
            "n_variants": len(cell),
        })

    common.write_jsonl(out_path, out_rows)
    log.info(f"Wrote {len(out_rows)} per-cell rows to {out_path}.")

    warnings_path = results_dir / "compute_cell_metrics_warnings.txt"
    with open(warnings_path, "w") as f:
        f.write("\n".join(warnings))
    log.info(f"{len(warnings)} warnings written to {warnings_path}.")


if __name__ == "__main__":
    main()
