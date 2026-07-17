"""
generate_report.py — generate a PDF methodology report for the mentor.
Run: python generate_report.py
Output: prompt_sensitivity_methodology_report.pdf
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus.flowables import Flowable
import datetime

# ── Colour palette ─────────────────────────────────────────────────────────────
BRAND_DARK   = HexColor("#1a2340")   # navy
BRAND_MID    = HexColor("#2e4a8c")   # medium blue
BRAND_ACCENT = HexColor("#4a90d9")   # sky blue
BRAND_LIGHT  = HexColor("#e8f0fb")   # pale blue fill
BRAND_GOLD   = HexColor("#f0a500")   # gold accent
GREY_DARK    = HexColor("#333333")
GREY_MID     = HexColor("#555555")
GREY_LIGHT   = HexColor("#f5f5f5")
RULE_COLOR   = HexColor("#c0cce0")

OUTPUT_FILE = "prompt_sensitivity_methodology_report.pdf"
DATE_STR = datetime.date.today().strftime("%B %d, %Y")

# ── Document setup ──────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT_FILE,
    pagesize=A4,
    leftMargin=2.2*cm, rightMargin=2.2*cm,
    topMargin=2.5*cm, bottomMargin=2.5*cm,
    title="Adaptive Prompt Sensitivity in Complex LLM Tasks — Methodology Report",
    author="SAI ABHINAV A M",
)

W = A4[0] - 4.4*cm   # usable width

# ── Style definitions ───────────────────────────────────────────────────────────
base_styles = getSampleStyleSheet()

def make_style(name, **kwargs):
    defaults = dict(fontName="Helvetica", fontSize=10, leading=14, textColor=GREY_DARK,
                    spaceAfter=4, spaceBefore=0, leftIndent=0, rightIndent=0)
    defaults.update(kwargs)
    return ParagraphStyle(name, **defaults)

S = {
    "cover_title": make_style("cover_title",
        fontName="Helvetica-Bold", fontSize=26, leading=32,
        textColor=BRAND_DARK, alignment=TA_CENTER, spaceAfter=8),
    "cover_subtitle": make_style("cover_subtitle",
        fontName="Helvetica", fontSize=14, leading=20,
        textColor=BRAND_MID, alignment=TA_CENTER, spaceAfter=6),
    "cover_meta": make_style("cover_meta",
        fontName="Helvetica", fontSize=11, leading=16,
        textColor=GREY_MID, alignment=TA_CENTER, spaceAfter=4),

    "h1": make_style("h1",
        fontName="Helvetica-Bold", fontSize=16, leading=20,
        textColor=BRAND_DARK, spaceBefore=20, spaceAfter=6),
    "h2": make_style("h2",
        fontName="Helvetica-Bold", fontSize=13, leading=17,
        textColor=BRAND_MID, spaceBefore=14, spaceAfter=4),
    "h3": make_style("h3",
        fontName="Helvetica-Bold", fontSize=11, leading=15,
        textColor=GREY_DARK, spaceBefore=10, spaceAfter=3),

    "body": make_style("body",
        fontSize=10, leading=15, alignment=TA_JUSTIFY, spaceAfter=6),
    "body_tight": make_style("body_tight",
        fontSize=10, leading=14, spaceAfter=3),

    "bullet": make_style("bullet",
        fontSize=9.5, leading=14, leftIndent=16, spaceAfter=3,
        bulletIndent=6),
    "sub_bullet": make_style("sub_bullet",
        fontSize=9, leading=13, leftIndent=30, spaceAfter=2,
        textColor=GREY_MID),

    "code": make_style("code",
        fontName="Courier", fontSize=8.5, leading=12, leftIndent=14,
        textColor=HexColor("#1a1a6e"), spaceAfter=2),
    "formula": make_style("formula",
        fontName="Courier-Bold", fontSize=9, leading=13, leftIndent=14,
        textColor=BRAND_DARK, spaceAfter=3),

    "caption": make_style("caption",
        fontSize=8.5, textColor=GREY_MID, alignment=TA_CENTER, spaceAfter=4),
    "callout": make_style("callout",
        fontSize=9.5, leading=14, leftIndent=10, rightIndent=10,
        textColor=BRAND_DARK, spaceAfter=4),
    "phase_label": make_style("phase_label",
        fontName="Helvetica-Bold", fontSize=9, textColor=white,
        alignment=TA_CENTER),
}


# ── Helpers ─────────────────────────────────────────────────────────────────────
def rule(color=RULE_COLOR, thickness=0.8, spaceB=6, spaceA=6):
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceAfter=spaceA, spaceBefore=spaceB)

def gap(h=6):
    return Spacer(1, h)

def h1(text): return Paragraph(text, S["h1"])
def h2(text): return Paragraph(text, S["h2"])
def h3(text): return Paragraph(text, S["h3"])
def body(text): return Paragraph(text, S["body"])
def bt(text):  return Paragraph(text, S["body_tight"])
def bullet(text): return Paragraph(f"• &nbsp; {text}", S["bullet"])
def sub_bullet(text): return Paragraph(f"◦ &nbsp; {text}", S["sub_bullet"])
def code(text): return Paragraph(text, S["code"])
def formula(text): return Paragraph(text, S["formula"])


def section_box(label, color=BRAND_MID):
    """Coloured phase label badge."""
    data = [[Paragraph(label, S["phase_label"])]]
    t = Table(data, colWidths=[W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), color),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
    ]))
    return t


def info_box(text, bg=BRAND_LIGHT, border=BRAND_ACCENT):
    """Highlighted callout box."""
    data = [[Paragraph(text, S["callout"])]]
    t = Table(data, colWidths=[W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), bg),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LINECOLOR", (0,0), (-1,-1), border),
        ("LINEBEFORE", (0,0), (0,-1), 3, border),
    ]))
    return t


def formula_box(rows):
    """Two-column formula table: [Name, Formula]."""
    header = [
        Paragraph("<b>Component / Score</b>", S["body_tight"]),
        Paragraph("<b>Formula / Value</b>", S["body_tight"]),
    ]
    table_data = [header] + [
        [Paragraph(r[0], S["body_tight"]), Paragraph(r[1], S["formula"])]
        for r in rows
    ]
    col_w = [W * 0.32, W * 0.68]
    t = Table(table_data, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), BRAND_DARK),
        ("TEXTCOLOR", (0,0), (-1,0), white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, GREY_LIGHT]),
        ("GRID", (0,0), (-1,-1), 0.4, RULE_COLOR),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 7),
        ("RIGHTPADDING", (0,0), (-1,-1), 7),
    ]))
    return t


def threshold_table(rows):
    """Three-column table: [Parameter, Value, Location]."""
    header = [
        Paragraph("<b>Parameter</b>", S["body_tight"]),
        Paragraph("<b>Value</b>", S["body_tight"]),
        Paragraph("<b>Location</b>", S["body_tight"]),
    ]
    table_data = [header] + [
        [Paragraph(r[0], S["body_tight"]),
         Paragraph(f"<b>{r[1]}</b>", S["formula"]),
         Paragraph(r[2], S["caption"])]
        for r in rows
    ]
    col_w = [W * 0.42, W * 0.22, W * 0.36]
    t = Table(table_data, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), BRAND_DARK),
        ("TEXTCOLOR", (0,0), (-1,0), white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, GREY_LIGHT]),
        ("GRID", (0,0), (-1,-1), 0.4, RULE_COLOR),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 7),
        ("RIGHTPADDING", (0,0), (-1,-1), 7),
    ]))
    return t


# ─────────────────────────────────────────────────────────────────────────────
# BUILD STORY
# ─────────────────────────────────────────────────────────────────────────────
story = []

# ════════════════════ COVER PAGE ════════════════════════════════════════════
story += [
    gap(50),
    Paragraph("Adaptive Prompt Sensitivity<br/>in Complex LLM Tasks", S["cover_title"]),
    gap(12),
    rule(BRAND_ACCENT, thickness=2, spaceB=0, spaceA=0),
    gap(8),
    Paragraph("Research Methodology Report", S["cover_subtitle"]),
    gap(30),
    Paragraph("Prepared by: <b>Sai Abhinav A M</b>", S["cover_meta"]),
    Paragraph(f"Date: <b>{DATE_STR}</b>", S["cover_meta"]),
    Paragraph("Course / Project: Prompt Sensitivity Research Pipeline", S["cover_meta"]),
    gap(40),
    info_box(
        "<b>Abstract.</b>  This report presents the complete methodology of an end-to-end "
        "research pipeline that investigates and mitigates <b>prompt sensitivity</b> in large "
        "language models — the phenomenon where semantically equivalent but lexically different "
        "prompts produce inconsistent outputs. The pipeline spans four phases: (1) benchmark "
        "dataset generation using GenSens, (2) quantitative robustness evaluation via the PRI "
        "Benchmark, (3) an inference-time mitigation technique called LL-PIRC (Logit-Lens "
        "Paraphrase-Invariant Residual Clamping), and (4) rigorous statistical validation."
    ),
    PageBreak(),
]

# ════════════════════ TABLE OF CONTENTS ════════════════════════════════════
story += [
    h1("Table of Contents"),
    rule(),
    gap(4),
]
toc_items = [
    ("1.", "Problem Statement & Motivation"),
    ("2.", "Pipeline Overview"),
    ("3.", "Phase 1 — GenSens: Benchmark Dataset Generation"),
    ("4.", "Phase 2 — PRI Benchmark: Robustness Evaluation"),
    ("5.", "Phase 3 — LL-PIRC: Inference-Time Mitigation"),
    ("6.", "Phase 4 — Statistical Evaluation & Validation"),
    ("7.", "Key Formulas Reference"),
    ("8.", "Key Thresholds & Constants"),
    ("9.", "Model Configuration"),
    ("10.", "Methodology Evolution & Design Decisions"),
]
for num, title in toc_items:
    story.append(bt(f"<b>{num}</b>&nbsp;&nbsp;&nbsp;{title}"))
    story.append(gap(2))

story.append(PageBreak())

# ════════════════════ 1. PROBLEM STATEMENT ══════════════════════════════════
story += [
    section_box("SECTION 1  ·  Problem Statement & Motivation"),
    gap(6),
    h1("1. Problem Statement & Motivation"),
    rule(),
    body(
        "Large language models (LLMs) are increasingly deployed in real-world applications "
        "that depend on reliable, consistent outputs. However, a well-documented phenomenon "
        "called <b>prompt sensitivity</b> causes LLMs to produce qualitatively different "
        "responses when presented with semantically equivalent but lexically varied prompts. "
        "For example, 'Summarise the article in three sentences.' and "
        "'Provide a brief three-sentence summary of the article.' should elicit nearly "
        "identical responses — yet in practice they often do not."
    ),
    gap(4),
    h2("Why This Matters"),
    bullet("Reproducibility: scientific or engineering workflows require consistent model behaviour across prompt phrasings."),
    bullet("Fairness: users who phrase the same query differently should not receive systematically worse answers."),
    bullet("Reliability: downstream pipelines built on LLM outputs fail unpredictably when sensitivity is high."),
    bullet("Evaluation validity: benchmarks that use a single prompt template may over- or under-estimate true model capability."),
    gap(4),
    h2("Research Questions"),
    bullet("RQ1: How can prompt sensitivity be quantified rigorously across multiple semantic axes?"),
    bullet("RQ2: Is it possible to detect the model-internal layer responsible for sensitivity?"),
    bullet("RQ3: Can an inference-time intervention reduce sensitivity without retraining the model?"),
    bullet("RQ4: Do the sensitivity reduction gains hold up under rigorous statistical testing?"),
    gap(8),
    info_box(
        "<b>Scope.</b>  The pipeline operates on four task domains: <b>Summarization</b> "
        "(CNN/DailyMail news articles), <b>Creative generation</b> (article expansion from "
        "highlights), <b>Dialogue understanding</b> (DREAM multi-turn QA), and "
        "<b>Explanatory QA</b> (ELI5 question–answer pairs). Evaluation is conducted on "
        "3×7–8B instruction-tuned models and one 70B AWQ-quantised model."
    ),
    PageBreak(),
]

# ════════════════════ 2. PIPELINE OVERVIEW ══════════════════════════════════
story += [
    section_box("SECTION 2  ·  Pipeline Overview"),
    gap(6),
    h1("2. Pipeline Overview"),
    rule(),
    body(
        "The pipeline consists of four sequential phases. Each phase produces durable "
        "artefacts (JSONL datasets, CSV result files) that downstream phases consume, "
        "enabling crash-safe, resumable execution and separation of expensive GPU steps."
    ),
    gap(8),
]

phase_data = [
    [Paragraph("<b>Phase</b>", S["body_tight"]),
     Paragraph("<b>Name</b>", S["body_tight"]),
     Paragraph("<b>Primary Output</b>", S["body_tight"]),
     Paragraph("<b>Key Models</b>", S["body_tight"])],
    [bt("Phase 1"), bt("GenSens Dataset Generation"),
     bt("gensens_<task>_Ninst_Kvar.jsonl"), bt("LLaMA-3.1-70B-AWQ (generator)\nSBERT all-mpnet-base-v2 (filter)")],
    [bt("Phase 2"), bt("PRI Benchmark Evaluation"),
     bt("responses.csv\nscored_samples.csv"), bt("3×8B + 1×70B subject models\nBAAI/bge-large-en-v1.5 (embedder)\nLLaMA-3.1-70B (judge/NLI)")],
    [bt("Phase 3"), bt("LL-PIRC Intervention"),
     bt("pirc.jsonl / pirc.json"), bt("LLaMA-3.1-8B-Instruct\n(HF hooks required)")],
    [bt("Phase 4"), bt("Statistical Validation"),
     bt("eval_summary.json\nplots/"), bt("CPU-only (scipy, numpy)")],
]
phase_table = Table(phase_data, colWidths=[W*0.12, W*0.25, W*0.30, W*0.33])
phase_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), BRAND_DARK),
    ("TEXTCOLOR", (0,0), (-1,0), white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,0), 9),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [BRAND_LIGHT, white]),
    ("GRID", (0,0), (-1,-1), 0.5, RULE_COLOR),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("TOPPADDING", (0,0), (-1,-1), 6),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ("LEFTPADDING", (0,0), (-1,-1), 7),
    ("RIGHTPADDING", (0,0), (-1,-1), 7),
    ("BACKGROUND", (0,1), (0,1), BRAND_GOLD),
    ("BACKGROUND", (0,2), (0,2), BRAND_MID),
    ("BACKGROUND", (0,3), (0,3), HexColor("#2e7d32")),
    ("BACKGROUND", (0,4), (0,4), HexColor("#6a1b9a")),
    ("TEXTCOLOR", (0,1), (0,-1), white),
]))
story.append(phase_table)
story.append(PageBreak())

# ════════════════════ 3. PHASE 1 — GENSENS ══════════════════════════════════
story += [
    section_box("SECTION 3  ·  Phase 1 — GenSens: Benchmark Dataset Generation", BRAND_GOLD),
    gap(6),
    h1("3. Phase 1 — GenSens: Benchmark Dataset Generation"),
    rule(),
    body(
        "GenSens is a controlled benchmark dataset generator that produces sets of "
        "<b>semantically equivalent but lexically diverse</b> prompt paraphrases. "
        "Each dataset instance consists of one base prompt and K=8 filtered paraphrase "
        "variants, all paired with a gold reference output."
    ),
    gap(4),
    h2("3.1  Task Domains"),
]

task_data = [
    [Paragraph("<b>Task</b>", S["body_tight"]),
     Paragraph("<b>Source Dataset</b>", S["body_tight"]),
     Paragraph("<b>What is Paraphrased</b>", S["body_tight"]),
     Paragraph("<b>Gold Reference</b>", S["body_tight"])],
    [bt("Summarization"), bt("CNN/DailyMail 3.0.0"), bt("Summarisation instruction"), bt("Highlights")],
    [bt("Creative"), bt("CNN/DailyMail 3.0.0"), bt("News highlights (premise)"), bt("Full article")],
    [bt("Dialogue"), bt("DREAM"), bt("Comprehension question"), bt("Correct answer choice")],
    [bt("QA"), bt("ELI5 (sentence-transformers)"), bt("Explanatory question"), bt("Answer/explanation")],
]
task_table = Table(task_data, colWidths=[W*0.18, W*0.25, W*0.30, W*0.27])
task_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), BRAND_DARK),
    ("TEXTCOLOR", (0,0), (-1,0), white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,0), 9),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, GREY_LIGHT]),
    ("GRID", (0,0), (-1,-1), 0.5, RULE_COLOR),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("LEFTPADDING", (0,0), (-1,-1), 7),
    ("RIGHTPADDING", (0,0), (-1,-1), 7),
]))
story.append(task_table)
story += [
    gap(8),
    h2("3.2  Paraphrase Generation Strategy"),
    body(
        "The generator employs LLaMA-3.1-70B-Instruct-AWQ-INT4 (via vLLM) with "
        "<b>16 distinct paraphrase strategies</b>, including: synonym substitution, "
        "passive/active voice transformation, sentence reordering, role-prefix injection, "
        "hedged phrasing, formality shift, and question-form conversion. A "
        "<b>per-task strategy whitelist</b> restricts strategies to those that preserve "
        "intent for each task domain."
    ),
    h2("3.3  Multi-Stage Quality Filtering"),
    bullet("<b>SBERT similarity band:</b> Candidates must satisfy 0.82 ≤ cosine(base, candidate) ≤ 0.98 using all-mpnet-base-v2. The upper bound rejects trivial restatements; the lower bound rejects meaning-drift."),
    bullet("<b>Lexical divergence floor:</b> Token Jaccard overlap between candidate and base (or any already-accepted variant) must be ≤ 0.85 to ensure lexical diversity."),
    bullet("<b>Per-strategy cap:</b> At most 2 variants per strategy family, ensuring the K=8 set spans multiple strategy types."),
    bullet("<b>Bidirectional NLI gate (QA/Dialogue only):</b> Both P(entail | base→candidate) ≥ 0.50 AND P(entail | candidate→base) ≥ 0.50, verified with cross-encoder/nli-deberta-v3-small. Prevents passive–active swaps that change the answer."),
    bullet("<b>Per-task SBERT threshold:</b> Dialogue uses a relaxed threshold of 0.78 (vs. default 0.82) to recover variant density for short conversational utterances."),
    gap(4),
    h2("3.4  Summarization Instruction Pool"),
    body(
        "For summarisation tasks, a pool of 100 unique instructions is pre-generated using "
        "Qwen2.5-7B-Instruct across four complexity levels (simple ≤12 words → complex 35–60 words). "
        "Each dataset instance is assigned a unique instruction (level-sorted), creating a "
        "complexity gradient that tests model sensitivity to instruction elaboration."
    ),
    h2("3.5  Adversarial Paraphrase Subset"),
    body(
        "Five deterministic perturbation families generate an adversarial subset to stress-test "
        "robustness beyond easy LLM rewrites: <b>typo</b> (8% character swaps), "
        "<b>sentence reorder</b>, <b>double negation</b> (¬¬X ≡ X), "
        "<b>hedged phrasing</b>, and <b>formality shift</b>. "
        "PRI is reported separately on the easy GenSens subset and the adversarial subset."
    ),
    h2("3.6  Dataset Scale"),
    bullet("Target: 200 instances × 4 tasks = 800 samples"),
    bullet("K = 8 paraphrase variants per instance"),
    bullet("All instances carry a non-empty gold reference_output"),
    bullet("Output format: JSONL with per-variant strategy tags, SBERT scores, and NLI audit fields"),
    PageBreak(),
]

# ════════════════════ 4. PHASE 2 — PRI BENCHMARK ════════════════════════════
story += [
    section_box("SECTION 4  ·  Phase 2 — PRI Benchmark: Robustness Evaluation", BRAND_MID),
    gap(6),
    h1("4. Phase 2 — PRI Benchmark: Robustness Evaluation"),
    rule(),
    body(
        "The PRI (Prompt Robustness Index) Benchmark evaluates how consistently each subject "
        "model responds across the K=8 paraphrase variants of every prompt. "
        "The pipeline is divided into a <b>generate-only</b> phase (one subject model at a time) "
        "and a <b>score-from-CSV</b> phase (embedder + judge), ensuring no two large models "
        "co-reside on the same 80 GB GPU card."
    ),
    gap(4),
    h2("4.1  Subject Models Under Evaluation"),
    bullet("meta-llama/Llama-3.1-8B-Instruct"),
    bullet("Qwen/Qwen2.5-7B-Instruct"),
    bullet("mistralai/Mistral-7B-Instruct-v0.3"),
    bullet("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 (AWQ Marlin quantisation)"),
    body("All models use greedy decoding (do_sample=False, seed=42) for reproducibility. "
         "Chat-template formatting is applied for all instruct-tuned models."),
    gap(4),
    h2("4.2  Core Metric Suite"),
    gap(4),
]

metric_data = [
    [Paragraph("<b>Metric</b>", S["body_tight"]),
     Paragraph("<b>Abbreviation</b>", S["body_tight"]),
     Paragraph("<b>What It Measures</b>", S["body_tight"]),
     Paragraph("<b>Range</b>", S["body_tight"])],
    [bt("Semantic Manifold Stability"), bt("SMS"), bt("Mean cosine similarity minus embedding variance penalty"), bt("↑ [0,1]")],
    [bt("Performance Elasticity"), bt("AUC-E"), bt("ROUGE-L stability across variant indices (AUC of performance curve)"), bt("↑ [0,1]")],
    [bt("Thematic Robustness Drift"), bt("TRD"), bt("Embedding-space semantic drift across variants"), bt("↓ [0,1]")],
    [bt("Key Point Info Gain"), bt("KPIG"), bt("Fraction of reference key-facts covered across variants"), bt("↑ [0,1]")],
    [bt("PPL Variance"), bt("PPL_var"), bt("Perplexity variance across variants (intra-model diagnostic)"), bt("↓ [0,1]")],
    [bt("Branching Factor"), bt("BF"), bt("Token-level entropy variance (intra-model diagnostic)"), bt("↓ [0,1]")],
    [bt("Correctness Score"), bt("CS"), bt("0.40·semantic + 0.20·length_adequacy + 0.40·reference coverage"), bt("↑ [0,1]")],
    [bt("Faithfulness"), bt("Faith."), bt("NLI entailment P(entail)+0.5·P(neutral), source→response"), bt("↑ [0,1]")],
    [bt("Hallucination Score"), bt("HS"), bt("Fraction of ungrounded content words (diagnostic only)"), bt("↓ [0,1]")],
]
metric_table = Table(metric_data, colWidths=[W*0.26, W*0.12, W*0.47, W*0.15])
metric_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), BRAND_DARK),
    ("TEXTCOLOR", (0,0), (-1,0), white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,0), 9),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, GREY_LIGHT]),
    ("GRID", (0,0), (-1,-1), 0.5, RULE_COLOR),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("LEFTPADDING", (0,0), (-1,-1), 7),
    ("RIGHTPADDING", (0,0), (-1,-1), 7),
]))
story.append(metric_table)

story += [
    gap(8),
    h2("4.3  Composite Scores"),
    h3("Prompt Robustness Index (PRI)  — Primary Ranking Score"),
    info_box(
        "<b>PRI = 0.40 × SMS  +  0.35 × CS  +  0.25 × Faithfulness</b><br/>"
        "Three non-collinear axes: semantic consistency (SMS), reference quality (CS), "
        "and NLI-grounded faithfulness. A short-output gate (avg_len &lt; 12 tokens) "
        "applies a ×0.85 penalty."
    ),
    gap(4),
    h3("Observable Robustness Index (ORI)  — Cross-Model Ranking"),
    formula("ORI = (SMS + AUC-E + (1 − TRD) + KPIG) / 4"),
    body("Arithmetic mean over the four external-consistency axes. Used for cross-model ranking tables."),
    h3("Intrinsic Fidelity Index (IFI)  — Intra-Model Diagnostic"),
    formula("IFI = 1 − (PPL_var + BF) / 2"),
    body("Intra-model diagnostic measuring internal generation stability. PPL/BF are architecture-dependent and not cross-model comparable."),
    h3("Final Score"),
    formula("Final_Score = 0.60 × PRI + 0.40 × Human_Score"),
    gap(6),
    h2("4.4  Diagnostic Dual-Pillar Score (Publication Appendix)"),
    body(
        "A separate harmonic diagnostic stack is used for causal diagnosis and publication "
        "appendix tables. It is intentionally strict: if any pillar approaches zero, the "
        "composite also approaches zero."
    ),
    formula("Diagnostic_ORI = WHM(SMS, AUC-E, KPIG, 1−TRD)  weights: 0.40 / 0.30 / 0.20 / 0.10"),
    formula("Diagnostic_IFI = WHM(1−PPL_var, 1−BF [, 1−PC_stab])  weights: 0.50 / 0.30 / 0.20"),
    formula("Diagnostic_PRI = HM(Diagnostic_ORI, Diagnostic_IFI)  weights: 0.50 / 0.50"),
    body(
        "The TRD weight is demoted to 0.10 because TRD and SMS have Pearson r = −0.98 "
        "(nearly collinear), so equal weighting would double-count consistency. "
        "The weighted harmonic mean (WHM) enforces the 'any zero → composite zero' property."
    ),
    gap(4),
    h2("4.5  2×2 Diagnosis Matrix"),
]

diag_data = [
    [Paragraph("", S["body_tight"]),
     Paragraph("<b>IFI ≥ 0.70 (Internally Stable)</b>", S["body_tight"]),
     Paragraph("<b>IFI &lt; 0.70 (Internally Fragile)</b>", S["body_tight"])],
    [Paragraph("<b>ORI ≥ 0.70</b>", S["body_tight"]),
     Paragraph("✓ ROBUST", S["body_tight"]),
     Paragraph("Externally Stable / Internally Fragile", S["body_tight"])],
    [Paragraph("<b>ORI &lt; 0.70</b>", S["body_tight"]),
     Paragraph("Internally Stable / Output-Sensitive", S["body_tight"]),
     Paragraph("✗ FRAGILE", S["body_tight"])],
]
diag_table = Table(diag_data, colWidths=[W*0.22, W*0.39, W*0.39])
diag_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), BRAND_DARK),
    ("TEXTCOLOR", (0,0), (-1,0), white),
    ("BACKGROUND", (1,1), (1,1), HexColor("#c8e6c9")),
    ("BACKGROUND", (2,2), (2,2), HexColor("#ffcdd2")),
    ("GRID", (0,0), (-1,-1), 0.6, RULE_COLOR),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 7),
    ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ("LEFTPADDING", (0,0), (-1,-1), 7),
    ("RIGHTPADDING", (0,0), (-1,-1), 7),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
]))
story.append(diag_table)

story += [
    gap(6),
    h2("4.6  Scoring Infrastructure"),
    bullet("<b>Embedder:</b> BAAI/bge-large-en-v1.5 (335M, MTEB ~64) — drives SMS, CS semantic component, Semantic TRD, and advanced KPIG."),
    bullet("<b>LLM Judge:</b> LLaMA-3.1-70B-AWQ (independent of subjects) — provides Human_Score and Faithfulness-LLM-as-NLI."),
    bullet("<b>Incremental CSV writing:</b> every row is fsync'd to disk immediately — crash-safe resume."),
    bullet("<b>Disk caching:</b> MD5-keyed CacheManager avoids redundant model calls on resume."),
    bullet("<b>IFI normalization:</b> PPL_var and BF are min-max normalised per model before computing IFI to prevent saturation artefacts."),
    PageBreak(),
]

# ════════════════════ 5. PHASE 3 — LL-PIRC ══════════════════════════════════
story += [
    section_box("SECTION 5  ·  Phase 3 — LL-PIRC: Inference-Time Mitigation", HexColor("#2e7d32")),
    gap(6),
    h1("5. Phase 3 — LL-PIRC: Inference-Time Mitigation"),
    rule(),
    body(
        "LL-PIRC (<b>Logit-Lens Paraphrase-Invariant Residual Clamping</b>) is a "
        "<i>training-free</i> inference-time intervention that reduces prompt sensitivity "
        "by steering the model's internal hidden states toward a cross-paraphrase consensus "
        "representation at the layer most responsible for sensitivity."
    ),
    gap(4),
    h2("5.1  Step 1 — Logit Lens: Per-Layer PPL Extraction"),
    body(
        "For each paraphrase variant k and each transformer layer ℓ, the Logit Lens "
        "projects the intermediate hidden state h^(ℓ) through the final LayerNorm and "
        "language-model head to obtain vocabulary logits. Per-token perplexity is then "
        "computed from these intermediate logits:"
    ),
    formula("logits^(ℓ) = lm_head( LayerNorm( h^(ℓ) ) )"),
    formula("PPL^(ℓ)_k = exp( -mean_t [ log P_t^(ℓ) ] )"),
    body("Hidden states are cast to float32 for numerical stability during the Logit Lens computation."),
    gap(4),
    h2("5.2  Step 2 — Sensitive Layer Detection (ℓ*)"),
    body(
        "For each layer ℓ, a sensitivity signal S(ℓ) measures how much the layer's "
        "intermediate PPL varies across the K paraphrase variants:"
    ),
    formula("S(ℓ) = Var_k [ mean_token_PPL at layer ℓ ]"),
    body("The <b>sensitive layer ℓ*</b> is identified as the layer with the steepest upward jump in this curve:"),
    formula("ℓ* = argmax_ℓ [ S(ℓ) − S(ℓ−1) ]"),
    bullet("Layer scan range: 25% to 100% of total transformer depth (early layers carry syntax, not prompt intent)."),
    bullet("Fallback (Z-score method): first ℓ where S(ℓ) > mean_S + 2.0 × std_S, if inflection detection fails."),
    bullet("Ablation knob: --ell-star L forces ℓ* to a fixed layer for sensitivity sweeps."),
    gap(4),
    h2("5.3  Step 3 — Anchor Token Identification"),
    body(
        "At layer ℓ*, anchor tokens are the subset of input tokens whose hidden states are "
        "most stable across paraphrase variants — i.e., the tokens that reliably encode "
        "content regardless of surface phrasing."
    ),
    formula("Anchor score_t = rank(mean_ppl_t) + rank(var_ppl_t)   [lower = more stable]"),
    bullet("Selection: bottom 30% by anchor score (percentile-based, model-agnostic)."),
    bullet("Pre-filtering: punctuation and stopwords are excluded — only content words are anchor candidates."),
    bullet("Guarantee: at least 1 anchor is always selected; at most ⌊min_len / 2⌋ anchors."),
    bullet("NaN/Inf handling: non-finite PPL positions are hard-excluded before ranking."),
    gap(4),
    h2("5.4  Step 4 — Residual Clamping"),
    body(
        "At inference time, a PyTorch forward hook is registered on transformer layer ℓ*. "
        "The hook fires <b>only during the prefill pass</b> (the single forward pass over "
        "the full prompt), then disables itself for all subsequent decode steps:"
    ),
    formula("h_clamped[anchors] = α × mean_h[anchors]  +  (1 − α) × h_original[anchors]"),
    bullet("mean_h[anchors] = arithmetic mean of anchor hidden states across all K paraphrase variants."),
    bullet("α = 1.0 (default, full replacement): complete substitution with consensus representation."),
    bullet("α < 1.0 (soft clamping): convex interpolation preserving some prompt-specific signal."),
    bullet("Prefill-only constraint: prevents degenerate autoregressive loops that occur when clamping fires on single-token decode steps."),
    bullet("Ablation knob: --alpha A overrides α from the CLI for systematic sensitivity sweeps."),
    gap(4),
    h2("5.5  Evaluation Protocol"),
    body(
        "One PIRC-stabilised output is generated per paraphrase variant, yielding K outputs "
        "per article. ROUGE-L variance across these K outputs is the primary mitigation metric."
    ),
    formula("PIRC Variance = Var( ROUGE-L scores across K PIRC outputs )"),
    bullet("Fixed preregistered τ / α (dev_tune_articles = 0) to avoid oracle overfitting."),
    bullet("Baseline: greedy generation without clamping, under identical prompt variants."),
    bullet("No-training baselines for comparison: temperature smoothing, self-consistency voting, system-prompt stabilisation, in-context learning exemplars."),
    gap(4),
    h2("5.6  Supported Architectures"),
    bullet("LLaMA-style: forward hook on model.model.layers[ℓ*], final norm = model.model.norm."),
    bullet("GPT-2-style: forward hook on model.transformer.h[ℓ*], final norm = model.transformer.ln_f."),
    bullet("Hook handles both tuple and plain Tensor outputs (transformers v5.x compatibility)."),
    PageBreak(),
]

# ════════════════════ 6. PHASE 4 — STATISTICAL EVALUATION ═══════════════════
story += [
    section_box("SECTION 6  ·  Phase 4 — Statistical Evaluation & Validation", HexColor("#6a1b9a")),
    gap(6),
    h1("6. Phase 4 — Statistical Evaluation & Validation"),
    rule(),
    body(
        "Phase 4 rigorously validates the LL-PIRC intervention using non-parametric "
        "hypothesis tests, bootstrap confidence intervals, and paired effect sizes."
    ),
    gap(4),
    h2("6.1  Variance Reduction"),
    formula("Variance Reduction = 1 − mean( var_pirc ) / mean( var_baseline )"),
    body("Positive values indicate that PIRC outputs are more consistent across paraphrase variants than the unmodified baseline."),
    gap(4),
    h2("6.2  Hypothesis Tests"),
    bullet("<b>Wilcoxon Signed-Rank (Variance):</b> One-sided test — H₁: baseline ROUGE-L variance > PIRC ROUGE-L variance. Significance level α = 0.01."),
    bullet("<b>Wilcoxon Signed-Rank (Quality):</b> Two-sided test — verifies PIRC does not degrade mean ROUGE-L quality. Significance level α = 0.01."),
    body("The non-parametric Wilcoxon test is used because ROUGE-L distributions are not guaranteed to be normal, especially on small article samples."),
    gap(4),
    h2("6.3  Bootstrap Confidence Intervals"),
    formula("95% Bootstrap CI on mean paired difference (PIRC − Baseline)"),
    bullet("10,000 resamples with replacement, seed = 42 for reproducibility."),
    bullet("Applied to both variance reduction and ROUGE-L quality change."),
    gap(4),
    h2("6.4  Effect Sizes"),
    formula("Cohen's dz = mean(diff) / std(diff)    [paired, per-article differences]"),
    bullet("Computed for both ROUGE-L variance reduction and ROUGE-L mean quality change."),
    bullet("Interpretation: |dz| < 0.2 = negligible, 0.2–0.5 = small, 0.5–0.8 = medium, > 0.8 = large."),
    gap(4),
    h2("6.5  Sensitive Layer Analysis"),
    bullet("ℓ* distribution statistics: mean, std, min, max, coefficient of variation across articles."),
    bullet("S(ℓ) sensitivity curves plotted for 5 sample articles."),
    bullet("Distribution analysis validates that ℓ* is consistently identifiable across the dataset."),
    gap(4),
    h2("6.6  Deterministic Reproducibility"),
    bullet("Global seed: Python random, NumPy, PyTorch CPU/CUDA seeded to 42 at pipeline entry."),
    bullet("cuDNN determinism: torch.use_deterministic_algorithms(warn_only=True), PYTHONHASHSEED=42."),
    bullet("Greedy decoding (do_sample=False) eliminates sampling noise from cross-prompt output differences."),
    PageBreak(),
]

# ════════════════════ 7. KEY FORMULAS ═══════════════════════════════════════
story += [
    section_box("SECTION 7  ·  Key Formulas Reference"),
    gap(6),
    h1("7. Key Formulas Reference"),
    rule(),
    gap(4),
    formula_box([
        ("PRI (primary ranking)", "0.40×SMS + 0.35×CS + 0.25×Faithfulness"),
        ("Faithfulness", "mean_k [ P(entail|src→resp_k) + 0.5·P(neutral|...) ]"),
        ("Final Score", "0.60×PRI + 0.40×Human_Score"),
        ("SMS", "mean(cosine_sim) − 0.5 × Var(L2-normed embeddings)"),
        ("TRD", "Var(embedding distances) across variants  [lower = better]"),
        ("KPIG", "mean_k [ |facts_k ∩ all_facts| / |all_facts| ]"),
        ("CS", "0.40×semantic + 0.20×length_adequacy + 0.40×coverage"),
        ("ORI (cross-model ranking)", "(SMS + AUC-E + (1−TRD) + KPIG) / 4"),
        ("IFI (intra-model diag.)", "1 − (PPL_var + BF) / 2"),
        ("Diagnostic_ORI", "WHM(SMS, AUC-E, KPIG, 1−TRD)  w=0.40/0.30/0.20/0.10"),
        ("Diagnostic_IFI", "WHM(1−PPL_var, 1−BF [,1−PC_stab])  w=0.50/0.30/0.20"),
        ("Diagnostic_PRI", "HM(Diagnostic_ORI, Diagnostic_IFI)  w=0.50/0.50"),
        ("S(ℓ) sensitivity", "Var_k [ mean_token_PPL at layer ℓ ]"),
        ("ℓ* sensitive layer", "argmax_ℓ [ S(ℓ) − S(ℓ−1) ]"),
        ("Anchor score_t", "rank(mean_ppl_t) + rank(var_ppl_t)  → bottom 30%"),
        ("PIRC clamping", "h[anchors] = α×mean_h + (1−α)×h_original"),
        ("Variance Reduction", "1 − mean(var_pirc) / mean(var_baseline)"),
        ("Cohen's dz", "mean(diff) / std(diff)"),
    ]),
    PageBreak(),
]

# ════════════════════ 8. THRESHOLDS ═════════════════════════════════════════
story += [
    section_box("SECTION 8  ·  Key Thresholds & Constants"),
    gap(6),
    h1("8. Key Thresholds & Constants"),
    rule(),
    gap(4),
    threshold_table([
        ("SBERT similarity gate (lower)", "≥ 0.82", "paraphrase_generator.py"),
        ("SBERT similarity gate (upper)", "≤ 0.98", "paraphrase_generator.py"),
        ("SBERT threshold — dialogue task", "≥ 0.78", "paraphrase_generator.py"),
        ("Token Jaccard overlap ceiling", "≤ 0.85", "paraphrase_generator.py"),
        ("NLI bidirectional entailment (QA/Dial.)", "≥ 0.50", "paraphrase_generator.py"),
        ("Max variants per strategy", "2", "paraphrase_generator.py"),
        ("Paraphrase variants per instance (K)", "8", "config.yaml"),
        ("Hallucination penalty threshold", "HS > 0.5 → PRI × 0.6", "evaluator.py"),
        ("Short output penalty", "avg_len < 12 → PRI × 0.85", "scores.py"),
        ("Entity coverage penalty", "coverage < 0.20 → CS × 0.60", "correctness_metric.py"),
        ("CS composition", "0.40·sem + 0.20·len + 0.40·cov", "correctness_metric.py"),
        ("Anchor percentile (bottom)", "30%", "anchor_tokens.py"),
        ("Layer scan start fraction", "25% of depth", "sensitive_layer.py"),
        ("Z-score fallback threshold", "2.0", "sensitive_layer.py"),
        ("PIRC default α (clamping strength)", "1.0 (full replacement)", "config.yaml"),
        ("Wilcoxon significance level", "α = 0.01", "evaluate.py"),
        ("Bootstrap resamples", "10,000 (seed=42)", "evaluate.py"),
        ("Diagnosis dual-pillar threshold", "0.70 (both ORI and IFI)", "scores.py"),
        ("Diagnostic ORI weights", "SMS=0.40/AUC-E=0.30/KPIG=0.20/TRD=0.10", "scores.py"),
        ("Diagnostic IFI weights", "PPL_var=0.50/BF=0.30/PC_stab=0.20", "scores.py"),
        ("PRI weights", "Consistency=0.40/CS=0.35/Faith.=0.25", "scores.py"),
        ("Final Score weights", "PRI=0.60/Human=0.40", "scores.py"),
        ("Dynamic weighting cap", "trust_human ≤ 0.70", "evaluator.py"),
        ("max_new_tokens (generation)", "200", "config.yaml"),
    ]),
    PageBreak(),
]

# ════════════════════ 9. MODEL CONFIGURATION ════════════════════════════════
story += [
    section_box("SECTION 9  ·  Model Configuration"),
    gap(6),
    h1("9. Model Configuration"),
    rule(),
    gap(4),
]

model_cfg = [
    [Paragraph("<b>Role</b>", S["body_tight"]),
     Paragraph("<b>Model</b>", S["body_tight"]),
     Paragraph("<b>Quantization</b>", S["body_tight"])],
    [bt("Paraphrase Generator"), bt("meta-llama/Llama-3.1-70B-Instruct-AWQ-INT4"), bt("AWQ Marlin INT4")],
    [bt("Subject #1"), bt("meta-llama/Llama-3.1-8B-Instruct"), bt("bf16")],
    [bt("Subject #2"), bt("Qwen/Qwen2.5-7B-Instruct"), bt("bf16")],
    [bt("Subject #3"), bt("mistralai/Mistral-7B-Instruct-v0.3"), bt("bf16")],
    [bt("Subject #4 / 70B"), bt("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"), bt("AWQ Marlin INT4")],
    [bt("LLM Judge"), bt("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"), bt("AWQ Marlin INT4")],
    [bt("Faithfulness NLI"), bt("(same as judge — LLM-as-NLI path)"), bt("AWQ Marlin INT4")],
    [bt("Embedder (SMS/CS/TRD/KPIG)"), bt("BAAI/bge-large-en-v1.5"), bt("fp32 / bf16")],
    [bt("SBERT Filter (GenSens)"), bt("sentence-transformers/all-mpnet-base-v2"), bt("fp32")],
    [bt("PIRC Target"), bt("meta-llama/Llama-3.1-8B-Instruct (HF hooks)"), bt("fp16 device_map=auto")],
]
mcfg_table = Table(model_cfg, colWidths=[W*0.25, W*0.50, W*0.25])
mcfg_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), BRAND_DARK),
    ("TEXTCOLOR", (0,0), (-1,0), white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,0), 9),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, GREY_LIGHT]),
    ("GRID", (0,0), (-1,-1), 0.5, RULE_COLOR),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("LEFTPADDING", (0,0), (-1,-1), 7),
    ("RIGHTPADDING", (0,0), (-1,-1), 7),
]))
story.append(mcfg_table)
story += [
    gap(8),
    info_box(
        "<b>GPU Memory Strategy.</b>  The 80 GB A100/H100 card holds only one large (70B) "
        "model at a time. The pipeline uses a <b>generate-only → score-from-CSV</b> split: "
        "each subject model generates responses independently (no judge loaded), then the "
        "judge/NLI/embedder score from the saved responses.csv in a separate process. "
        "This prevents OOM and eliminates self-evaluation bias."
    ),
    PageBreak(),
]

# ════════════════════ 10. METHODOLOGY EVOLUTION ══════════════════════════════
story += [
    section_box("SECTION 10  ·  Methodology Evolution & Key Design Decisions"),
    gap(6),
    h1("10. Methodology Evolution & Key Design Decisions"),
    rule(),
    body(
        "The methodology has undergone systematic refinement through six phases of "
        "audit and fix cycles. The following summarises the most significant design "
        "decisions and the empirical evidence that motivated them."
    ),
    gap(6),
    h2("10.1  De-collinearising PRI (R3 — most significant change)"),
    body(
        "The original PRI formula (Consistency + CS + exp(−HS)) was empirically "
        "found to have SMS⟷PRI Pearson r = 0.949 — PRI was effectively a rescaled SMS. "
        "Three 'metrics' measured the same underlying signal. The fix introduced "
        "<b>NLI-based Faithfulness</b> as a genuinely independent third axis, replacing "
        "SMS-Wasserstein (r ≈ 0.98 with SMS) and cosine-KPIG (also collinear with SMS)."
    ),
    h2("10.2  Prefill-Only PIRC Clamping (Critical Bug Fix)"),
    body(
        "An early PIRC implementation applied the forward hook on every model forward "
        "call, including each single-token decode step. This overwrote newly generated "
        "token hidden states with the first anchor's consensus, collapsing autoregressive "
        "generation into degenerate repetition ('Here://://://...'). All previous PIRC "
        "results before this fix were invalid. The corrected implementation uses a "
        "<b>clamp_state latch</b> that disables the hook after the prefill pass."
    ),
    h2("10.3  Percentile Anchor Selection (Robustness Fix)"),
    body(
        "The original absolute-threshold anchor selection (mean_ppl < τ AND var_ppl < τ_var) "
        "was architecture-dependent: GPT-2 baselines operate at PPL ~50–100 while LLaMA "
        "operates at PPL ~5–15. A poorly tuned τ silently produced zero anchors, making "
        "PIRC a no-op with no warning. The percentile-based method is self-calibrating "
        "and always returns at least one anchor."
    ),
    h2("10.4  IFI Saturation and Normalisation"),
    body(
        "PPL_var and BF computed on raw model logits differed by orders of magnitude "
        "across seq2seq (encoder–decoder) and causal architectures, causing IFI to "
        "saturate at 1.0 for some models. This collapsed Diagnostic_PRI = HM(ORI, IFI) "
        "into Diagnostic_ORI. The fix normalises PPL_var and BF per-model (min-max) "
        "before computing IFI."
    ),
    h2("10.5  TRD Demotion in Diagnostic_ORI"),
    body(
        "TRD and SMS show Pearson r = −0.98 (almost perfectly collinear) in the "
        "5-instance pilot. Equal weighting in the harmonic mean let a TRD-only failure "
        "(SMS/AUC-E/KPIG=0.85, TRD=0.9) drop the composite to ~0.30 even though three "
        "of four axes agreed. The empirically motivated weights (SMS=0.40, AUC-E=0.30, "
        "KPIG=0.20, TRD=0.10) demote TRD to a tie-breaker while preserving the "
        "harmonic-mean 'any zero → zero' diagnostic property."
    ),
    h2("10.6  GenSens Pipeline Connections (R1)"),
    body(
        "Phase 1 generated SBERT-filtered paraphrase variants, but Phase 2 discarded "
        "them and re-wrapped articles in 9 near-synonymous templates. Phases 1 and 2 "
        "were completely disconnected — prompt sensitivity was never measured on the "
        "generated paraphrases. R1 wired the GenSens output directly into the evaluator, "
        "making the benchmark measure what it claims to measure."
    ),
    gap(8),
    info_box(
        "<b>Full Changelog.</b>  A complete record of every methodology change — "
        "including old values, new values, reasoning, and re-run requirements — "
        "is maintained in <b>method.md</b> at the repository root (newest entries first). "
        "This document represents the current state of the methodology as of "
        f"<b>{DATE_STR}</b>."
    ),
    gap(10),
    rule(BRAND_ACCENT, thickness=1.5),
    gap(6),
    Paragraph(
        f"<i>Report generated on {DATE_STR}.  "
        "Project: Adaptive Prompt Sensitivity in Complex LLM Tasks.  "
        "Author: Sai Abhinav A M.</i>",
        S["caption"]
    ),
]


# ── Page numbering callback ──────────────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    page_num = doc.page
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY_MID)
    canvas.drawString(2.2*cm, 1.4*cm, "Prompt Sensitivity — Methodology Report")
    canvas.drawRightString(A4[0] - 2.2*cm, 1.4*cm, f"Page {page_num}")
    canvas.restoreState()


# ── Build ───────────────────────────────────────────────────────────────────
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"PDF written → {OUTPUT_FILE}")
