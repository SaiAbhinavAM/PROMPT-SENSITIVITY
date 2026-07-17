"""
export_summ_paraphrases_pdf.py — render gensens_summ_50seed_5para.jsonl as a PDF.

Input:  gensens/data/gensens_summ_50seed_5para.jsonl, gensens_summ_stats.json
Output: gensens/data/gensens_summ_50seed_5para.pdf

Usage:
    python gensens/scripts/export_summ_paraphrases_pdf.py
"""

import json
import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak,
)

DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "gensens_summ_50seed_5para.jsonl"
OUTPUT_FILE = DATA_DIR / "gensens_summ_50seed_5para.pdf"

# ── Palette (matches generate_report.py) ────────────────────────────────────
BRAND_DARK   = HexColor("#1a2340")
BRAND_MID    = HexColor("#2e4a8c")
BRAND_ACCENT = HexColor("#4a90d9")
BRAND_LIGHT  = HexColor("#e8f0fb")
GREY_DARK    = HexColor("#333333")
GREY_MID     = HexColor("#555555")
GREY_LIGHT   = HexColor("#f5f5f5")
RULE_COLOR   = HexColor("#c0cce0")

DATE_STR = datetime.date.today().strftime("%B %d, %Y")

doc = SimpleDocTemplate(
    str(OUTPUT_FILE),
    pagesize=A4,
    leftMargin=2.0*cm, rightMargin=2.0*cm,
    topMargin=2.2*cm, bottomMargin=2.2*cm,
    title="GenSens Summarization Paraphrases — 50 Seeds x 5 Variants",
    author="GenSens pipeline",
)
W = A4[0] - 4.0*cm

base_styles = getSampleStyleSheet()


def make_style(name, **kwargs):
    defaults = dict(fontName="Helvetica", fontSize=9.5, leading=13, textColor=GREY_DARK,
                     spaceAfter=3, spaceBefore=0)
    defaults.update(kwargs)
    return ParagraphStyle(name, **defaults)


S = {
    "cover_title": make_style("cover_title", fontName="Helvetica-Bold", fontSize=22,
                               leading=28, textColor=BRAND_DARK, alignment=TA_CENTER, spaceAfter=8),
    "cover_subtitle": make_style("cover_subtitle", fontSize=12, leading=17,
                                  textColor=BRAND_MID, alignment=TA_CENTER, spaceAfter=6),
    "cover_meta": make_style("cover_meta", fontSize=10, leading=15,
                              textColor=GREY_MID, alignment=TA_CENTER),
    "h1": make_style("h1", fontName="Helvetica-Bold", fontSize=15, leading=19,
                      textColor=BRAND_DARK, spaceBefore=16, spaceAfter=6),
    "h3": make_style("h3", fontName="Helvetica-Bold", fontSize=10.5, leading=14,
                      textColor=white, spaceBefore=0, spaceAfter=0),
    "body": make_style("body", fontSize=9.5, leading=14, alignment=TA_JUSTIFY),
    "base_text": make_style("base_text", fontName="Helvetica-Oblique", fontSize=9.5,
                             leading=13, textColor=GREY_DARK, spaceAfter=6),
    "cell": make_style("cell", fontSize=8.5, leading=11.5),
    "cell_hdr": make_style("cell_hdr", fontSize=8.5, leading=11, fontName="Helvetica-Bold",
                            textColor=white),
    "caption": make_style("caption", fontSize=8, textColor=GREY_MID, alignment=TA_CENTER),
}


def rule(color=RULE_COLOR, thickness=0.8, spaceB=6, spaceA=6):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=spaceA, spaceBefore=spaceB)


def gap(h=6):
    return Spacer(1, h)


def kv_table(rows):
    data = [[Paragraph(f"<b>{k}</b>", S["cell"]), Paragraph(str(v), S["cell"])] for k, v in rows]
    t = Table(data, colWidths=[W * 0.38, W * 0.62])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), BRAND_LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def seed_header(prompt_id, pool, dimension, complexity):
    label = f"{prompt_id}  ·  {dimension}   (Pool {pool}, complexity {complexity})"
    t = Table([[Paragraph(label, S["h3"])]], colWidths=[W])
    color = BRAND_MID if pool == "A" else HexColor("#6a1b9a")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def variants_table(variants):
    header = [
        Paragraph("#", S["cell_hdr"]),
        Paragraph("Strategy", S["cell_hdr"]),
        Paragraph("Family", S["cell_hdr"]),
        Paragraph("Sim", S["cell_hdr"]),
        Paragraph("Paraphrased Instruction", S["cell_hdr"]),
    ]
    rows = [header]
    for v in variants:
        rows.append([
            Paragraph(str(v.get("variant_idx", "")), S["cell"]),
            Paragraph(v.get("strategy", ""), S["cell"]),
            Paragraph(v.get("strategy_family", ""), S["cell"]),
            Paragraph(f"{v.get('sbert_similarity', 0.0):.3f}", S["cell"]),
            Paragraph(v.get("paraphrased_text", ""), S["cell"]),
        ])
    col_w = [W * 0.05, W * 0.16, W * 0.12, W * 0.08, W * 0.59]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_DARK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, GREY_LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def load_records():
    records = []
    with open(INPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_stats(records):
    """Compute pool stats directly from the JSONL — do not trust
    gensens_summ_stats.json, which is written once by generate_summ_paraphrases.py
    and goes stale after regenerate_flagged.py updates individual seeds in place."""
    import numpy as np

    def pool_stats(pool):
        recs = [r for r in records if r["pool"] == pool]
        ns = [r.get("n_variants_generated", 0) for r in recs]
        sims = [
            v.get("sbert_similarity", 0.0)
            for r in recs for v in r.get("variants", [])
        ]
        return {
            "count": len(recs),
            "mean_n": float(np.mean(ns)) if ns else 0.0,
            "mean_sim": float(np.mean(sims)) if sims else 0.0,
        }

    flagged = [r["prompt_id"] for r in records if r.get("n_variants_generated", 0) < 5]
    return {
        "total_seeds": len(records),
        "n_pool_a": sum(1 for r in records if r["pool"] == "A"),
        "n_pool_b": sum(1 for r in records if r["pool"] == "B"),
        "pool_a": pool_stats("A"),
        "pool_b": pool_stats("B"),
        "flagged_seeds": flagged,
    }


def build():
    records = load_records()
    stats = compute_stats(records)
    total_variants = sum(r.get("n_variants_generated", 0) for r in records)

    story = []

    # ── Cover ────────────────────────────────────────────────────────────────
    story += [
        gap(40),
        Paragraph("GenSens Summarization Paraphrases", S["cover_title"]),
        gap(8),
        rule(BRAND_ACCENT, thickness=2, spaceB=0, spaceA=0),
        gap(8),
        Paragraph("50 seed instructions x up to 5 paraphrase variants each", S["cover_subtitle"]),
        gap(20),
        Paragraph(f"Generated: <b>{DATE_STR}</b>", S["cover_meta"]),
        Paragraph("Generator: <b>meta-llama/Meta-Llama-3.1-8B-Instruct</b> (vLLM, bf16)", S["cover_meta"]),
        Paragraph("Filter: SBERT all-mpnet-base-v2 (0.82 &le; cosine &le; 0.92) + bidirectional NLI gate", S["cover_meta"]),
        gap(24),
        kv_table([
            ("Total seeds", stats["total_seeds"]),
            ("Pool A (free-form)", f"{stats['n_pool_a']} seeds — mean {stats['pool_a']['mean_n']:.1f} variants/seed, mean sim {stats['pool_a']['mean_sim']:.3f}"),
            ("Pool B (constraint-preserving)", f"{stats['n_pool_b']} seeds — mean {stats['pool_b']['mean_n']:.1f} variants/seed, mean sim {stats['pool_b']['mean_sim']:.3f}"),
            ("Total variants generated", total_variants),
            ("Seeds below target (&lt;5 variants)", ", ".join(stats.get("flagged_seeds", [])) or "none"),
        ]),
        PageBreak(),
    ]

    # ── Per-seed sections ────────────────────────────────────────────────────
    story += [Paragraph("Seed Prompts &amp; Generated Paraphrases", S["h1"]), rule(), gap(4)]

    for rec in records:
        block = [
            seed_header(rec["prompt_id"], rec["pool"], rec["dimension"], rec["complexity"]),
            gap(4),
            Paragraph(f"<b>Base:</b> {rec['base_text']}", S["base_text"]),
            variants_table(rec.get("variants", [])),
        ]
        story.append(KeepTogether(block))
        story.append(gap(10))

    def on_page(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GREY_MID)
        canvas.drawString(2.0*cm, 1.3*cm, "GenSens Summarization Paraphrases")
        canvas.drawRightString(A4[0] - 2.0*cm, 1.3*cm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"PDF written -> {OUTPUT_FILE}")


if __name__ == "__main__":
    build()
