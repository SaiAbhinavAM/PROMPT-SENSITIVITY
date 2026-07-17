"""
generate_status_report.py — project status report PDF with actual results.
Run: python generate_status_report.py
Output: prompt_sensitivity_status_report.pdf
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import datetime

# ── Colours ────────────────────────────────────────────────────────────────
BRAND_DARK   = HexColor("#1a2340")
BRAND_MID    = HexColor("#2e4a8c")
BRAND_ACCENT = HexColor("#4a90d9")
BRAND_LIGHT  = HexColor("#e8f0fb")
BRAND_GOLD   = HexColor("#f0a500")
GREEN_DARK   = HexColor("#1b5e20")
GREEN_MID    = HexColor("#2e7d32")
GREEN_LIGHT  = HexColor("#c8e6c9")
AMBER_DARK   = HexColor("#e65100")
AMBER_LIGHT  = HexColor("#ffe0b2")
RED_DARK     = HexColor("#b71c1c")
RED_LIGHT    = HexColor("#ffcdd2")
GREY_DARK    = HexColor("#333333")
GREY_MID     = HexColor("#555555")
GREY_LIGHT   = HexColor("#f5f5f5")
RULE_COLOR   = HexColor("#c0cce0")
PURPLE_DARK  = HexColor("#4a148c")

OUTPUT_FILE = "prompt_sensitivity_status_report.pdf"
DATE_STR = datetime.date.today().strftime("%B %d, %Y")

doc = SimpleDocTemplate(
    OUTPUT_FILE, pagesize=A4,
    leftMargin=2.2*cm, rightMargin=2.2*cm,
    topMargin=2.5*cm, bottomMargin=2.5*cm,
    title="Prompt Sensitivity — Project Status Report",
    author="SAI ABHINAV A M",
)
W = A4[0] - 4.4*cm

# ── Styles ─────────────────────────────────────────────────────────────────
def ms(name, **kw):
    d = dict(fontName="Helvetica", fontSize=10, leading=14, textColor=GREY_DARK,
             spaceAfter=4, spaceBefore=0, leftIndent=0, rightIndent=0)
    d.update(kw)
    return ParagraphStyle(name, **d)

S = {
    "cover_title": ms("ct", fontName="Helvetica-Bold", fontSize=24, leading=30,
                      textColor=BRAND_DARK, alignment=TA_CENTER, spaceAfter=8),
    "cover_sub":   ms("cs", fontSize=13, leading=18, textColor=BRAND_MID,
                      alignment=TA_CENTER, spaceAfter=6),
    "cover_meta":  ms("cm", fontSize=11, textColor=GREY_MID,
                      alignment=TA_CENTER, spaceAfter=4),
    "h1": ms("h1", fontName="Helvetica-Bold", fontSize=15, leading=19,
             textColor=BRAND_DARK, spaceBefore=18, spaceAfter=6),
    "h2": ms("h2", fontName="Helvetica-Bold", fontSize=12, leading=16,
             textColor=BRAND_MID, spaceBefore=12, spaceAfter=4),
    "h3": ms("h3", fontName="Helvetica-Bold", fontSize=10.5, leading=14,
             textColor=GREY_DARK, spaceBefore=8, spaceAfter=3),
    "body": ms("body", fontSize=10, leading=15, alignment=TA_JUSTIFY, spaceAfter=6),
    "bt":   ms("bt",   fontSize=9.5, leading=13, spaceAfter=3),
    "bullet": ms("bul", fontSize=9.5, leading=14, leftIndent=14, spaceAfter=3),
    "sub_b":  ms("sb",  fontSize=9,   leading=12, leftIndent=28, spaceAfter=2, textColor=GREY_MID),
    "mono":   ms("mono", fontName="Courier", fontSize=8.5, leading=12, spaceAfter=2),
    "caption":ms("cap", fontSize=8.5, textColor=GREY_MID, alignment=TA_CENTER, spaceAfter=4),
    "callout":ms("call", fontSize=9.5, leading=14, leftIndent=10, rightIndent=10,
                 textColor=BRAND_DARK, spaceAfter=4),
    "badge_w": ms("bw", fontName="Helvetica-Bold", fontSize=9, textColor=white,
                  alignment=TA_CENTER),
    "badge_d": ms("bd", fontName="Helvetica-Bold", fontSize=9, textColor=GREY_DARK,
                  alignment=TA_CENTER),
    "num_big": ms("nb", fontName="Helvetica-Bold", fontSize=20, leading=24,
                  textColor=BRAND_DARK, alignment=TA_CENTER),
    "num_lbl": ms("nl", fontSize=8.5, textColor=GREY_MID, alignment=TA_CENTER),
}

def rule(c=RULE_COLOR, t=0.8, b=6, a=6):
    return HRFlowable(width="100%", thickness=t, color=c, spaceAfter=a, spaceBefore=b)

def gap(h=6): return Spacer(1, h)
def h1(t): return Paragraph(t, S["h1"])
def h2(t): return Paragraph(t, S["h2"])
def h3(t): return Paragraph(t, S["h3"])
def body(t): return Paragraph(t, S["body"])
def bt(t):   return Paragraph(t, S["bt"])
def bul(t):  return Paragraph(f"• &nbsp;{t}", S["bullet"])
def sbul(t): return Paragraph(f"◦ &nbsp;{t}", S["sub_b"])
def mono(t): return Paragraph(t, S["mono"])

def section_box(label, color=BRAND_MID):
    t = Table([[Paragraph(label, S["badge_w"])]], colWidths=[W])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),color),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
    ]))
    return t

def info_box(text, bg=BRAND_LIGHT, border=BRAND_ACCENT):
    t = Table([[Paragraph(text, S["callout"])]], colWidths=[W])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),bg),
        ("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LINEBEFORE",(0,0),(0,-1),3,border),
    ]))
    return t

def warn_box(text):  return info_box(text, bg=AMBER_LIGHT, border=AMBER_DARK)
def crit_box(text):  return info_box(text, bg=RED_LIGHT, border=RED_DARK)
def ok_box(text):    return info_box(text, bg=GREEN_LIGHT, border=GREEN_MID)

def kpi_table(items):
    """items = [(label, value, sub), ...]"""
    cells = [[Paragraph(v, S["num_big"]), Paragraph(l, S["num_lbl"]),
              Paragraph(s, S["num_lbl"])]
             for l, v, s in items]
    row = [cells[i][0] for i in range(len(items))]
    row2 = [cells[i][1] for i in range(len(items))]
    row3 = [cells[i][2] for i in range(len(items))]
    cw = [W/len(items)] * len(items)
    t = Table([row, row2, row3], colWidths=cw)
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),BRAND_LIGHT),
        ("GRID",(0,0),(-1,-1),0.4,RULE_COLOR),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    return t

def data_table(headers, rows_data, col_widths=None, row_colors=None):
    header_row = [Paragraph(f"<b>{h}</b>", S["bt"]) for h in headers]
    body_rows  = [[Paragraph(str(c), S["bt"]) for c in r] for r in rows_data]
    all_rows = [header_row] + body_rows
    if col_widths is None:
        col_widths = [W/len(headers)]*len(headers)
    style = [
        ("BACKGROUND",(0,0),(-1,0),BRAND_DARK),
        ("TEXTCOLOR",(0,0),(-1,0),white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,0),9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[white,GREY_LIGHT]),
        ("GRID",(0,0),(-1,-1),0.4,RULE_COLOR),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("FONTSIZE",(0,1),(-1,-1),9),
    ]
    if row_colors:
        for ri, color in row_colors.items():
            style.append(("BACKGROUND",(0,ri+1),(-1,ri+1),color))
    t = Table(all_rows, colWidths=col_widths)
    t.setStyle(TableStyle(style))
    return t

def status_badge(status):
    if status == "COMPLETE":
        bg, fg = GREEN_MID, white
    elif status == "PARTIAL":
        bg, fg = BRAND_GOLD, GREY_DARK
    elif status == "BUGGY":
        bg, fg = RED_DARK, white
    else:
        bg, fg = GREY_MID, white
    p = ParagraphStyle("b", fontName="Helvetica-Bold", fontSize=8,
                       textColor=fg, alignment=TA_CENTER)
    t = Table([[Paragraph(status, p)]], colWidths=[2.0*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),bg),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
    ]))
    return t

# ════════════════════════════════════════════════════════════════════════════
story = []

# ── COVER ───────────────────────────────────────────────────────────────────
story += [
    gap(40),
    Paragraph("Prompt Sensitivity Research Pipeline", S["cover_title"]),
    gap(8),
    rule(BRAND_ACCENT, t=2, b=0, a=0),
    gap(8),
    Paragraph("Project Status Report — Pilot Run Results", S["cover_sub"]),
    gap(24),
    Paragraph("Prepared by: <b>Sai Abhinav A M</b>", S["cover_meta"]),
    Paragraph(f"Report Date: <b>{DATE_STR}</b>", S["cover_meta"]),
    Paragraph("Run Scale: <b>5 instances × 4 tasks · Pilot (n=5)</b>", S["cover_meta"]),
    Paragraph("Primary Model Under Test: <b>meta-llama/Llama-3.1-8B-Instruct</b>", S["cover_meta"]),
    gap(30),
    info_box(
        "<b>Purpose of this Report.</b>  This document presents the actual numerical results "
        "from the first end-to-end pilot run of the Prompt Sensitivity pipeline "
        "(5 instances × 4 tasks). It covers the status of all four phases, reproduces "
        "key metric tables, flags critical issues discovered and resolved, and outlines "
        "what must be done to reach the full 200-instance scale for publication."
    ),
    PageBreak(),
]

# ── TABLE OF CONTENTS ───────────────────────────────────────────────────────
story += [h1("Table of Contents"), rule(), gap(4)]
for n, title in [
    ("1.", "Executive Summary & KPIs"),
    ("2.", "Phase Completion Status"),
    ("3.", "Phase 1 — GenSens Dataset Results"),
    ("4.", "Phase 2 — PRI Benchmark Results"),
    ("5.", "Phase 3 — LL-PIRC Results"),
    ("6.", "Phase 4 — Statistical Evaluation"),
    ("7.", "Critical Issues & Resolutions"),
    ("8.", "Next Steps to Full Scale"),
]:
    story.append(bt(f"<b>{n}</b>&nbsp;&nbsp;&nbsp;{title}"))
    story.append(gap(2))
story.append(PageBreak())

# ════════════════════ 1. EXECUTIVE SUMMARY ══════════════════════════════════
story += [
    section_box("SECTION 1  ·  Executive Summary & KPIs"),
    gap(6),
    h1("1. Executive Summary & Key Performance Indicators"),
    rule(),
    body(
        "The full pipeline has been executed end-to-end on a pilot scale of <b>5 instances per "
        "task (4 tasks)</b> using LLaMA-3.1-8B-Instruct as the primary subject and three "
        "additional models (Qwen2.5-7B, Mistral-7B, LLaMA-3.1-70B-AWQ) for Phase 2 comparison. "
        "A critical PIRC bug (degenerate 'Here://://...' outputs) was identified and fixed. "
        "The corrected pilot demonstrates <b>37.5% variance reduction</b> with negligible "
        "quality degradation (−0.35% ROUGE-L), though statistical significance requires "
        "scale-up to n≥50 articles."
    ),
    gap(6),
    kpi_table([
        ("Instances scored (Phase 2)", "76", "5 inst × 4 tasks × 4 models"),
        ("Mean PRI (best model)", "0.787", "LLaMA-3.1-70B-AWQ"),
        ("PIRC var. reduction", "37.5%", "α=1.0, ℓ*=9 (corrected run)"),
        ("ROUGE quality change", "−0.35%", "Quality preserved"),
        ("Wilcoxon p (variance)", "0.156", "n=5 — not yet significant"),
        ("Effect size dz", "0.588", "Medium effect (|dz|>0.5)"),
    ]),
    gap(8),
    warn_box(
        "<b>Scale Warning.</b>  All results are from a <b>5-article pilot</b>. "
        "Wilcoxon signed-rank tests require n≥50 paired observations to achieve "
        "power ≥0.80 at α=0.01. The current n=5 is insufficient for publication. "
        "All conclusions below are preliminary trends, not confirmed findings."
    ),
    PageBreak(),
]

# ════════════════════ 2. PHASE COMPLETION STATUS ════════════════════════════
story += [
    section_box("SECTION 2  ·  Phase Completion Status"),
    gap(6),
    h1("2. Phase Completion Status"),
    rule(),
    gap(4),
]

status_data = [
    ("Phase 1", "GenSens Dataset Generation",
     "5 inst × 4 tasks (pilot)\nTarget: 200 inst × 4 tasks",
     "PARTIAL",
     GREEN_LIGHT),
    ("Phase 2", "PRI Benchmark (4 models × 4 tasks)",
     "76 samples scored\nAll 4 models, all 4 tasks",
     "PARTIAL",
     GREEN_LIGHT),
    ("Phase 3", "LL-PIRC Intervention (corrected)",
     "Prefill-only bug fixed\nGPU snapshot: corrected results",
     "PARTIAL",
     GREEN_LIGHT),
    ("Phase 4", "Statistical Evaluation",
     "Bootstrap CI + Wilcoxon done\nNot significant at n=5",
     "PARTIAL",
     AMBER_LIGHT),
    ("Infra", "Reproducibility & Crash-safety",
     "JSONL checkpointing, seeds,\npinned requirements",
     "COMPLETE",
     GREEN_LIGHT),
    ("Audit", "FLAWS_AND_FIXES.pdf addressed",
     "All 6 audit phases implemented\n(§2–§6 code changes committed)",
     "COMPLETE",
     GREEN_LIGHT),
]

phase_rows = []
for ph, name, detail, status, color in status_data:
    phase_rows.append([
        Paragraph(f"<b>{ph}</b>", S["bt"]),
        Paragraph(name, S["bt"]),
        Paragraph(detail, S["bt"]),
        status_badge(status),
    ])
ph_header = [Paragraph(f"<b>{h}</b>", S["bt"]) for h in ["Phase","Description","Current State","Status"]]
ph_table = Table([ph_header]+phase_rows, colWidths=[W*0.11, W*0.33, W*0.38, W*0.18])
ph_style = [
    ("BACKGROUND",(0,0),(-1,0),BRAND_DARK),
    ("TEXTCOLOR",(0,0),(-1,0),white),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
    ("FONTSIZE",(0,0),(-1,0),9),
    ("GRID",(0,0),(-1,-1),0.4,RULE_COLOR),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
    ("FONTSIZE",(0,1),(-1,-1),9),
]
for i, (_, _, _, _, color) in enumerate(status_data):
    ph_style.append(("BACKGROUND",(0,i+1),(2,i+1),color))
ph_table.setStyle(TableStyle(ph_style))
story.append(ph_table)
story.append(PageBreak())

# ════════════════════ 3. PHASE 1 — GENSENS ══════════════════════════════════
story += [
    section_box("SECTION 3  ·  Phase 1 — GenSens Dataset Results", BRAND_GOLD),
    gap(6),
    h1("3. Phase 1 — GenSens Dataset Results"),
    rule(),
    body(
        "The GenSens pipeline successfully generated and validated a multi-task pilot dataset. "
        "The GPU-executed run (stored in <b>gpu_snapshot_5inst/gensens_data/</b>) covers "
        "all four task domains with 5 instances each."
    ),
    gap(4),
    h2("3.1  Dataset Statistics (GPU Pilot Run — 5 instances per task)"),
    gap(4),
]

gensens_rows = [
    ["Summarization", "CNN/DailyMail 3.0.0", "5", "7.2", "0.92 ± 0.04", "LLM strategy pool"],
    ["Creative", "CNN/DailyMail 3.0.0", "5", "7.0", "~0.91", "Highlights paraphrase"],
    ["Dialogue", "DREAM", "5", "2.8", "~0.89", "Low yield (NLI gate)"],
    ["QA", "ELI5 (sent-trans.)", "5", "4.6", "~0.90", "6-retry loop"],
    ["<b>Total</b>", "4 datasets", "<b>20</b>", "<b>5.4 avg</b>", "—", "—"],
]
story.append(data_table(
    ["Task","Source Dataset","Instances","Avg Variants","SBERT Sim (mean±std)","Notes"],
    gensens_rows,
    col_widths=[W*0.16, W*0.22, W*0.11, W*0.13, W*0.22, W*0.16],
))
story += [
    gap(8),
    h2("3.2  Key Observations"),
    bul("<b>Dialogue yield is low (avg 2.8 variants)</b>: The bidirectional NLI gate (τ=0.50) and the strategy whitelist (9 strategies for dialogue) are more restrictive than for summarization. Recommend raising max_retries or using the relaxed threshold (0.78) for dialogue."),
    bul("<b>SBERT similarity range</b>: Mean ~0.92 across tasks, well within the [0.82, 0.98] target band. No quality issues flagged by the validator."),
    bul("<b>Instruction pool</b>: 100-entry summarization pool generated (summ_instruction_pool.jsonl). Complexity gradient (simple→complex) confirmed present."),
    bul("<b>Scale gap</b>: Target is 200 instances × 4 tasks = 800 samples. Current pilot is 20 instances — 2.5% of target scale."),
    gap(6),
    warn_box(
        "<b>Scale-Up Required.</b>  The full 200-instance × 4-task dataset must be generated "
        "on the A100/H100 instance using the LLaMA-3.1-70B-AWQ generator and the hardened "
        "<tt>run_on_gpu.sh</tt> script before publication metrics can be reported."
    ),
    PageBreak(),
]

# ════════════════════ 4. PHASE 2 — PRI BENCHMARK ════════════════════════════
story += [
    section_box("SECTION 4  ·  Phase 2 — PRI Benchmark Results", BRAND_MID),
    gap(6),
    h1("4. Phase 2 — PRI Benchmark Results"),
    rule(),
    body(
        "The PRI benchmark was run on all 4 subject models × 4 tasks × 5 instances = "
        "<b>76 scored samples</b>. Results are stored in "
        "<tt>results_5inst_alpha1.0/scored_samples.csv</tt>."
    ),
    gap(4),
    h2("4.1  Per-Model PRI Rankings"),
    gap(4),
]

model_rows = [
    ["LLaMA-3.1-70B-AWQ", "0.898", "0.741", "0.030", "0.905", "0.600", "0.867", "0.787", "1st"],
    ["LLaMA-3.1-8B-Instruct", "0.893", "0.758", "0.032", "0.897", "0.579", "0.870", "0.784", "2nd"],
    ["Qwen2.5-7B-Instruct", "0.897", "0.848", "0.031", "0.923", "0.595", "0.834", "0.775", "3rd"],
    ["Mistral-7B-v0.3", "0.880", "0.826", "0.035", "0.852", "0.610", "0.791", "0.757", "4th"],
]
story.append(data_table(
    ["Model","SMS","AUC-E","TRD","Faithfulness","CS","Human Score","PRI","Rank"],
    model_rows,
    col_widths=[W*0.23, W*0.07, W*0.07, W*0.07, W*0.12, W*0.07, W*0.12, W*0.08, W*0.07],
))
story += [
    gap(4),
    Paragraph(
        "<i>PRI = 0.40×SMS + 0.35×CS + 0.25×Faithfulness. All values are means over "
        "5 instances × 4 tasks = 19 scored rows per model (some dialogue instances had "
        "K=1 variant, contributing reduced metrics).</i>",
        S["caption"]
    ),
    gap(8),
    h2("4.2  Per-Task Metric Breakdown (All 4 Models Combined)"),
    gap(4),
]

task_rows = [
    ["Summarization", "20", "0.892", "0.844", "0.031", "0.838", "0.173", "0/20 (0%)"],
    ["Creative",      "20", "0.887", "0.930", "0.033", "0.944", "0.114", "0/20 (0%)"],
    ["Dialogue",      "16", "0.924", "0.662", "0.016", "0.863", "0.594", "5/16 (31%)"],
    ["QA",            "20", "0.871", "0.711", "0.051", "0.926", "0.116", "0/20 (0%)"],
]
story.append(data_table(
    ["Task","n","SMS","AUC-E","TRD","Faithfulness","ROUGE-L","Robust Count"],
    task_rows,
    col_widths=[W*0.16, W*0.06, W*0.09, W*0.09, W*0.09, W*0.13, W*0.11, W*0.18],
    row_colors={2: GREEN_LIGHT},
))
story += [
    gap(8),
    h2("4.3  Diagnosis Distribution"),
    gap(4),
]

diag_data_rows = [
    ["Internally Stable / Output-Sensitive", "71", "93.4%",
     "IFI=1.0 (saturated) but ORI<0.70. All models show strong internal stability but "
     "output-level consistency is moderate. ORI limited mainly by AUC-E and KPIG on "
     "non-dialogue tasks."],
    ["Robust (ORI≥0.70 ∧ IFI≥0.70)", "5", "6.6%",
     "All 5 Robust cases are Dialogue samples where KPIG=1.0 (single-answer format "
     "gives full key-point coverage) and AUC-E is higher."],
    ["Externally Stable / Internally Fragile", "0", "0%", "PPL_var=0 for all samples — IFI always 1.0."],
    ["Fragile (ORI<0.70 ∧ IFI<0.70)", "0", "0%", "PPL_var=0 for all samples — IFI always 1.0."],
]
story.append(data_table(
    ["Diagnosis","Count","Fraction","Explanation"],
    diag_data_rows,
    col_widths=[W*0.30, W*0.08, W*0.10, W*0.52],
    row_colors={0: AMBER_LIGHT},
))
story += [
    gap(6),
    crit_box(
        "<b>IFI Saturation Issue.</b>  PPL_var = 0.0 and BF = 0.0 for <i>every</i> scored "
        "sample, which forces IFI = 1.0 for all rows. This is the saturation problem identified "
        "in FLAWS §2.2: the IFI normalization step (per-model min-max) collapses when all values "
        "are identical (min == max == 0). As a result, the two categories "
        "'Externally Stable/Internally Fragile' and 'Fragile' are unreachable. "
        "The fix (<tt>csv_io.normalize_ifi_per_model()</tt>) requires non-zero variance across "
        "the sample pool — needs larger scale or explicit IFI computation during generation."
    ),
    gap(6),
    h2("4.4  Key Metric Observations"),
    bul("<b>SMS is uniformly high</b> (0.88–0.92 across all models and tasks). The d1/d2/d3 synthetic paraphrase templates produce very similar outputs — consistent with the finding from the R1 rectification that the real GenSens paraphrases are needed to stress-test sensitivity."),
    bul("<b>AUC-E varies by task</b>: Creative (0.930) and Summarization (0.844) are high; Dialogue (0.662) is lower because the model sometimes gives the wrong answer choice, depressing performance across variant indices."),
    bul("<b>KPIG is low for generation tasks</b> (0.22–0.26 for Creative/QA). This is expected: open-ended generation explores many different key points, and KPIG measures reference coverage. Dialogue achieves KPIG≈0.65 because single-answer outputs either match the reference or not."),
    bul("<b>Faithfulness is high</b> (0.85–0.94). The 70B LLM-as-NLI judge rarely flags contradiction, suggesting the pilot articles are straightforward to summarise faithfully."),
    bul("<b>Human Score spread</b>: Dialogue scores are highest (mean ~0.75–1.0 per variant) while QA and Creative are more varied (0.2–1.0 per variant), reflecting the judge's difficulty rating generation tasks."),
    PageBreak(),
]

# ════════════════════ 5. PHASE 3 — LL-PIRC ══════════════════════════════════
story += [
    section_box("SECTION 5  ·  Phase 3 — LL-PIRC Results", GREEN_MID),
    gap(6),
    h1("5. Phase 3 — LL-PIRC Results"),
    rule(),
    h2("5.1  Bug History & Corrected Run"),
    body(
        "Two PIRC result sets exist in the repository. The <b>buggy runs</b> "
        "(<tt>results_5inst_alpha1.0/pirc.json</tt>) show the pre-fix behaviour "
        "where the hook fired on every decode step, not just prefill, collapsing all "
        "outputs into the degenerate pattern <tt>'Here://://://...'</tt>. "
        "The <b>corrected run</b> (<tt>gpu_snapshot_5inst/results/eval_summary.json</tt>) "
        "uses the prefill-only latch fix and produces valid summaries."
    ),
    gap(4),
]

bug_compare = [
    ["Run", "PIRC Output Quality", "Mean ROUGE-L", "Var Reduction", "Interpretation"],
    ["Buggy (α=1.0, pre-fix)",
     "Degenerate — 'Here://://://...'",
     "0.012 (near zero)",
     "47% (artificial)",
     "Fake reduction: PIRC ROUGE≈0 vs baseline≈0.23"],
    ["Corrected (GPU snapshot)",
     "Valid summaries",
     "0.226 (−0.35%)",
     "37.5% (real)",
     "Genuine variance reduction, quality preserved"],
]
story.append(data_table(
    bug_compare[0], bug_compare[1:],
    col_widths=[W*0.20, W*0.25, W*0.16, W*0.16, W*0.23],
    row_colors={0: RED_LIGHT, 1: GREEN_LIGHT},
))
story += [
    gap(8),
    h2("5.2  Corrected PIRC Results (GPU Snapshot, α=1.0, τ=3.0)"),
    gap(4),
]

pirc_rows = [
    ["Article 0", "0.2486", "0.0007", "0.0000", "0.0007", "100.0%",
     "Full reduction (PIRC var=0)"],
    ["Article 1", "0.2529", "0.0010", "0.0227", "0.0005", "50.7%", "Partial reduction"],
    ["Article 2", "0.1629", "0.0008", "0.0139", "0.0006", "26.9%", "Partial reduction"],
    ["Article 3", "0.2755", "0.0005", "0.0000", "0.0000", "100.0%",
     "Full reduction (PIRC var=0)"],
    ["Article 4", "0.1947", "0.00005","0.0233", "0.0005", "−10.9%",
     "Regression — PIRC introduces variance"],
    ["<b>Mean</b>","<b>0.2269</b>","<b>0.000618</b>","<b>0.0120</b>","<b>0.000386</b>",
     "<b>37.5%</b>","GPU snapshot corrected run"],
]
story.append(data_table(
    ["Article","Baseline ROUGE","Baseline Var","PIRC ROUGE","PIRC Var","Var Reduction","Note"],
    pirc_rows,
    col_widths=[W*0.11, W*0.14, W*0.13, W*0.12, W*0.12, W*0.13, W*0.25],
    row_colors={4: AMBER_LIGHT},  # regression article
))
story += [
    gap(6),
    warn_box(
        "<b>Article 4 Regression.</b>  Article 4 (Duke University noose story) shows "
        "<i>negative</i> variance reduction (−10.9%). The baseline ROUGE variance for this "
        "article is already extremely low (4.56×10⁻⁵), so any variance introduced by PIRC "
        "looks large by comparison. The sensitive layer ℓ*=9 is the same as all other articles, "
        "suggesting the S-curve NaN problem (see Section 7) prevented proper ℓ* selection."
    ),
    gap(8),
    h2("5.3  Sensitive Layer (ℓ*) Analysis"),
    gap(4),
]

lstar_rows = [
    ["All 5 articles (corrected run)", "9", "9", "9", "0.0", "0%", "Constant — no variation"],
    ["All 5 articles (buggy run)", "9", "9", "9", "0.0", "0%", "Same constant — NaN S-curves"],
]
story.append(data_table(
    ["Sample","ℓ* Mean","ℓ* Min","ℓ* Max","ℓ* Std","CV","Interpretation"],
    lstar_rows,
    col_widths=[W*0.28, W*0.10, W*0.10, W*0.10, W*0.10, W*0.08, W*0.24],
))
story += [
    gap(6),
    crit_box(
        "<b>S-Curve NaN Problem.</b>  The Logit Lens sensitivity curves S(ℓ) are <b>NaN "
        "for every layer</b> in both the corrected and buggy PIRC runs. This means the "
        "inflection-point method (argmax_ℓ [S(ℓ)−S(ℓ−1)]) cannot operate on real data — "
        "ℓ*=9 is likely the fallback value (25% scan start of 32 layers ≈ layer 8). "
        "The variance reduction result therefore does NOT demonstrate that the logit lens "
        "correctly identified the sensitive layer. Root cause: likely a Logit Lens "
        "numerical precision issue or hidden-state shape mismatch under the current "
        "transformers version. This is the most critical unresolved issue."
    ),
    gap(6),
    h2("5.4  Anchor Token Statistics"),
    bul("Anchor fraction: consistently ~30% (percentile-based method, as designed)."),
    bul("Anchor counts per article: 76–138 tokens (varies with prompt length)."),
    bul("NaN/Inf exclusions: 0 positions excluded (all PPL values finite at layer 9)."),
    bul("Generation: valid text produced in corrected run — prefill-only latch confirmed working."),
    PageBreak(),
]

# ════════════════════ 6. PHASE 4 — STATISTICAL EVALUATION ═══════════════════
story += [
    section_box("SECTION 6  ·  Phase 4 — Statistical Evaluation", PURPLE_DARK),
    gap(6),
    h1("6. Phase 4 — Statistical Evaluation"),
    rule(),
    gap(4),
    h2("6.1  Summary Statistics"),
    gap(4),
]

stat_rows = [
    ["Baseline mean ROUGE-L", "0.2269", "—"],
    ["PIRC mean ROUGE-L", "0.2261", "Quality preserved"],
    ["ROUGE change", "−0.0008 (−0.35%)", "95% CI: [−0.0108, +0.0119]"],
    ["Baseline mean variance", "6.18×10⁻⁴", "—"],
    ["PIRC mean variance", "3.86×10⁻⁴", "37.5% reduction"],
    ["Variance reduction 95% CI", "[−6.86×10⁻⁵, +5.39×10⁻⁴]", "CI straddles zero"],
    ["Wilcoxon variance (1-sided)", "W=12.0,  p=0.156", "NOT significant at α=0.01"],
    ["Wilcoxon ROUGE (2-sided)", "W=6.0,   p=0.813", "NOT significant at α=0.01"],
    ["Effect size dz (variance)", "0.588", "Medium effect (0.5–0.8)"],
    ["Effect size dz (ROUGE)", "−0.054", "Negligible quality change"],
    ["ℓ* distribution", "mean=9, std=0, CV=0%", "Constant — NaN S-curves"],
    ["Bootstrap resamples", "10,000 (seed=42)", "Reproducible"],
    ["Sample size n", "5 articles", "Insufficient for α=0.01 power"],
]
story.append(data_table(
    ["Metric","Value","Interpretation"],
    stat_rows,
    col_widths=[W*0.38, W*0.30, W*0.32],
    row_colors={6: AMBER_LIGHT, 7: AMBER_LIGHT},
))
story += [
    gap(8),
    h2("6.2  Power Analysis"),
    body(
        "With n=5 paired observations and a medium effect size (dz≈0.59), the estimated "
        "statistical power for a one-sided Wilcoxon test at α=0.01 is approximately "
        "<b>15–20%</b> — far below the conventional 80% threshold. The confidence intervals "
        "for variance reduction straddle zero ([−6.86×10⁻⁵, +5.39×10⁻⁴]), "
        "meaning we cannot yet rule out that PIRC has no effect."
    ),
    bul("To achieve 80% power at α=0.01 with dz≈0.60: need approximately <b>n≈50</b> articles."),
    bul("To achieve 90% power at α=0.01 with dz≈0.60: need approximately <b>n≈70</b> articles."),
    bul("Publication target: 200 articles per task — this will comfortably exceed all power requirements."),
    gap(6),
    ok_box(
        "<b>Positive Signal.</b>  Despite the small n, the pilot shows a consistent directional "
        "trend: 4 out of 5 articles show variance reduction under PIRC (80% hit rate), "
        "and quality is maintained (ROUGE changes are within noise). "
        "The medium effect size (dz=0.588) is a promising indicator for the full-scale run."
    ),
    PageBreak(),
]

# ════════════════════ 7. CRITICAL ISSUES ════════════════════════════════════
story += [
    section_box("SECTION 7  ·  Critical Issues & Resolutions"),
    gap(6),
    h1("7. Critical Issues & Resolutions"),
    rule(),
    gap(4),
]

issues = [
    (
        "CRITICAL — RESOLVED",
        GREEN_MID, GREEN_LIGHT,
        "PIRC Prefill-Only Bug",
        "The PIRC forward hook fired on every decode step, not just prefill. During "
        "autoregressive generation each new token's hidden state was overwritten with "
        "the first anchor's consensus, collapsing all outputs to 'Here://://://...'.",
        "Added a clamp_state['prefilled'] latch that disables the hook after the first "
        "forward pass. Corrected results in gpu_snapshot_5inst/results/.",
        "All PIRC results from before this fix are INVALID."
    ),
    (
        "CRITICAL — OPEN",
        RED_DARK, RED_LIGHT,
        "Logit Lens S-Curve NaN Values",
        "S(ℓ) = NaN for ALL layers across all 5 articles in both runs. The inflection "
        "method cannot determine ℓ* from real data. ℓ*=9 appears to be a default/fallback "
        "value (≈25% of 32 layers = layer 8), not empirically detected.",
        "Debug logit_lens.py — check hidden-state dtype, shape, and lm_head compatibility "
        "with current transformers version. Possibly need float32 casting before lm_head "
        "projection or a model-specific norm lookup.",
        "The 37.5% variance reduction claim cannot be attributed to ℓ* detection — it "
        "reflects clamping at an arbitrary layer. LL-PIRC mechanistic story is at risk."
    ),
    (
        "MEDIUM — OPEN",
        AMBER_DARK, AMBER_LIGHT,
        "IFI Saturation (PPL_var = BF = 0)",
        "PPL_var and BF are 0.0 for every scored sample, making IFI=1.0 constant. "
        "The 'Externally Stable / Internally Fragile' and 'Fragile' diagnosis categories "
        "are unreachable. This is the saturation issue from FLAWS §2.2.",
        "The per-model IFI normalization requires non-zero variance. Investigate whether "
        "PPL and BF are being computed correctly during the generate-only phase, or whether "
        "they need to be computed from the IFI-specific experiment runner.",
        "Diagnostic_IFI and the 2×2 diagnosis matrix are unreliable at current scale."
    ),
    (
        "MEDIUM — RESOLVED",
        GREEN_MID, GREEN_LIGHT,
        "Anchor Token NaN/Inf Pool",
        "The anchor selector could previously select positions with NaN/Inf PPL values "
        "as 'most stable' tokens (sorting artifact), causing PIRC to clamp on the worst tokens.",
        "_sanitize_for_ranking() in anchor_tokens.py now masks non-finite positions with "
        "dtype-max sentinel before ranking, hard-excluding them from the candidate pool.",
        "PIRC clamping now always uses the most stable finite positions."
    ),
    (
        "LOW — RESOLVED",
        GREEN_MID, GREEN_LIGHT,
        "Dialogue Variant Yield",
        "Dialogue task produced only 2.8 variants per instance (below target K=8) due to "
        "the strict bidirectional NLI gate and small strategy whitelist (9 strategies).",
        "task_thresholds={\"dialogue\": 0.78} applied by default. Further improvement "
        "possible by raising max_per_strategy from 2→3 for dialogue.",
        "Dialogue datasets are viable but will have lower K than other tasks."
    ),
]

for severity, sev_color, bg_color, title, problem, fix, impact in issues:
    story.append(KeepTogether([
        Table([[Paragraph(severity, S["badge_w"])]], colWidths=[W]),
        gap(2),
    ]))
    sev_tbl = Table([[Paragraph(severity, S["badge_w"])]], colWidths=[W])
    sev_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),sev_color),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]))

    detail_data = [
        [Paragraph("<b>Problem</b>",S["bt"]), Paragraph(problem, S["bt"])],
        [Paragraph("<b>Fix / Status</b>",S["bt"]), Paragraph(fix, S["bt"])],
        [Paragraph("<b>Impact</b>",S["bt"]), Paragraph(impact, S["bt"])],
    ]
    detail_tbl = Table(detail_data, colWidths=[W*0.18, W*0.82])
    detail_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),bg_color),
        ("GRID",(0,0),(-1,-1),0.4,RULE_COLOR),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("FONTSIZE",(0,0),(-1,-1),9),
    ]))
    title_tbl = Table([[Paragraph(f"<b>{title}</b>", S["bt"])]], colWidths=[W])
    title_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),sev_color),
        ("TEXTCOLOR",(0,0),(-1,-1),white),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
    ]))
    story += [gap(4), title_tbl, detail_tbl, gap(6)]

story.append(PageBreak())

# ════════════════════ 8. NEXT STEPS ═════════════════════════════════════════
story += [
    section_box("SECTION 8  ·  Next Steps to Full Scale"),
    gap(6),
    h1("8. Next Steps to Full Scale"),
    rule(),
    gap(4),
    h2("8.1  Immediate Blockers (Must Fix Before Scale-Up)"),
    gap(4),
]

blockers = [
    ("#1 CRITICAL", "Fix Logit Lens S-Curve NaN",
     "Debug logit_lens.py — verify float32 casting, lm_head hook, and per-layer "
     "PPL computation. Validate with a unit test on GPT-2 before running on LLaMA-8B.",
     "Without this, ℓ* is not empirically detected and the LL-PIRC mechanistic claim fails."),
    ("#2 MEDIUM", "Fix IFI PPL/BF Computation",
     "Verify that ppl_var and bf are non-zero for diverse outputs. Check that the "
     "per-model IFI normalization (csv_io.normalize_ifi_per_model) runs correctly.",
     "Without this, the dual-pillar diagnosis matrix only produces one category."),
    ("#3 LOW", "Increase Dialogue Variant Yield",
     "Raise max_per_strategy for dialogue from 2→3 and run with task_thresholds={dialogue:0.78}.",
     "Dialogue at K=2.8 will produce unreliable AUC-E/SMS estimates (too few variants)."),
]
blocker_data = [[Paragraph(f"<b>{b}</b>",S["bt"]), Paragraph(t,S["bt"]),
                 Paragraph(d,S["bt"]), Paragraph(i,S["bt"])]
                for b, t, d, i in blockers]
bl_tbl = Table(
    [[ Paragraph("<b>Priority</b>",S["bt"]), Paragraph("<b>Task</b>",S["bt"]),
       Paragraph("<b>Action</b>",S["bt"]), Paragraph("<b>Why Blocking</b>",S["bt"])]
    ] + blocker_data,
    colWidths=[W*0.12, W*0.20, W*0.38, W*0.30]
)
bl_tbl.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),BRAND_DARK),("TEXTCOLOR",(0,0),(-1,0),white),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),9),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[RED_LIGHT,AMBER_LIGHT,BRAND_LIGHT]),
    ("GRID",(0,0),(-1,-1),0.4,RULE_COLOR),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
    ("FONTSIZE",(0,1),(-1,-1),9),
]))
story.append(bl_tbl)
story += [
    gap(10),
    h2("8.2  Scale-Up Roadmap"),
    gap(4),
]

roadmap = [
    ["Step","Action","Target","Dependency"],
    ["A","Fix Logit Lens NaN (see §8.1 #1)","Before any GPU run","None"],
    ["B","Fix IFI computation (see §8.1 #2)","Before Phase 2 scale-up","None"],
    ["C","Generate 200-inst × 4-task GenSens dataset on A100","gensens_all_200inst_8var.jsonl","A, B"],
    ["D","Run Phase 2 PRI benchmark on all 4 models","scored_samples.csv (n=800)","C"],
    ["E","Run Phase 3 PIRC on LLaMA-3.1-8B-Instruct","pirc.json (n=200)","C, A"],
    ["F","Run Phase 4 statistical evaluation","eval_summary.json (n=200)","D, E"],
    ["G","ℓ* ablation sweep (--ell-star 6,9,12,18,24,28)","ablation_ell_star.json","E"],
    ["H","α ablation sweep (--alpha 0.25,0.5,0.75,1.0)","ablation_alpha.json","E"],
    ["I","Mitigation baselines comparison","baselines_comparison.json","D"],
    ["J","Cross-embedder ablation (BAAI, gte-Qwen2, e5-mistral)","embedding_ablation.json","D"],
    ["K","PRI weight ablation (equal vs default vs learned)","pri_weight_ablation.json","D"],
]
story.append(data_table(
    roadmap[0], roadmap[1:],
    col_widths=[W*0.06, W*0.40, W*0.27, W*0.27],
))
story += [
    gap(8),
    h2("8.3  Publication Readiness Checklist"),
    bul("[ ] Logit Lens NaN fixed and validated on unit test"),
    bul("[ ] IFI non-zero (PPL/BF computed correctly for diverse outputs)"),
    bul("[ ] Full 200-instance × 4-task GenSens dataset generated on A100"),
    bul("[ ] Phase 2 run on 200 instances (Wilcoxon power ≥80% at α=0.01)"),
    bul("[ ] Phase 3 PIRC run with verified ℓ* detection"),
    bul("[ ] ℓ* ablation sweep confirming ℓ*=9 is not just a default"),
    bul("[ ] α ablation confirming monotonic variance reduction trend"),
    bul("[ ] Mitigation baselines showing PIRC > temperature smoothing etc."),
    bul("[ ] Cross-embedder Spearman ρ > 0.70 for ranking stability"),
    bul("[ ] PRI weight ablation with equal-weight comparison"),
    bul("[ ] Adversarial paraphrase subset scored and reported separately"),
    bul("[ ] Docker image or pinned-requirements confirmed reproducible"),
    gap(10),
    rule(BRAND_ACCENT, t=1.5),
    gap(6),
    Paragraph(
        f"<i>Status report generated on {DATE_STR}. "
        "Results from pilot run: 5 instances × 4 tasks × 4 models = 76 scored samples. "
        "Author: Sai Abhinav A M.</i>",
        S["caption"]
    ),
]


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY_MID)
    canvas.drawString(2.2*cm, 1.4*cm, "Prompt Sensitivity — Project Status Report")
    canvas.drawRightString(A4[0]-2.2*cm, 1.4*cm, f"Page {doc.page}")
    canvas.restoreState()

doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"PDF written → {OUTPUT_FILE}")
