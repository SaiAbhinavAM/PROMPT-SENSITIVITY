"""
table_utils.py — Publication-quality terminal table rendering for the evaluation framework.

Uses `rich` for colored, formatted tables with visual indicators.
Falls back to `tabulate` if `rich` is unavailable, and to plain-text if neither is installed.

Renders 5 structured tables:
    1. Model Performance Summary
    2. Advanced Metrics
    3. ROUGE Scores
    4. Correlation Analysis
    5. Stability Metrics
"""

import json
import os
from typing import List, Dict, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Rich detection — used for colored output; graceful fallback to tabulate
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Shared console instance (only created if rich is available)
_console = Console() if HAS_RICH else None


# =============================================================================
# Color helpers
# =============================================================================

def _score_color(value: float, invert: bool = False) -> str:
    """
    Return a rich color tag based on score quality.

    Normal mode (higher = better):
        >= 0.7 → green
        >= 0.4 → yellow
        <  0.4 → red

    Inverted mode (lower = better, e.g., HS, TRD):
        <= 0.1 → green
        <= 0.3 → yellow
        >  0.3 → red

    Args:
        value: The metric value (expected [0, 1]).
        invert: If True, lower values are better.

    Returns:
        Rich color string like "green", "yellow", or "red".
    """
    if invert:
        if value <= 0.1:
            return "green"
        elif value <= 0.3:
            return "yellow"
        else:
            return "red"
    else:
        if value >= 0.7:
            return "green"
        elif value >= 0.4:
            return "yellow"
        else:
            return "red"


def _stability_label(std: float) -> tuple:
    """
    Classify metric stability based on standard deviation.

    Returns:
        (label, color) tuple.
    """
    if std < 0.1:
        return ("Stable", "green")
    elif std <= 0.2:
        return ("Moderate", "yellow")
    else:
        return ("Unstable", "red")


def _fmt(value, decimals: int = 3) -> str:
    """Format a numeric value to fixed decimal places."""
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return str(value)


# =============================================================================
# TABLE 1: Model Performance Summary
# =============================================================================

def render_performance_table(results_df: pd.DataFrame) -> None:
    """
    Render TABLE 1 — Model Performance Summary.

    Columns: Rank, Model, PRI, CS, HS, Consistency, Human Score, Final Score
    Sorted by Final Score descending. Best model row highlighted.
    """
    if results_df.empty:
        print("  (no results)")
        return

    df = results_df.copy()
    df["Rank"] = df["Final_Score"].rank(ascending=False).astype(int)
    df = df.sort_values("Final_Score", ascending=False)

    if HAS_RICH:
        table = Table(
            title="Model Performance Summary",
            box=box.ROUNDED,
            show_lines=True,
            title_style="bold cyan",
            header_style="bold white on dark_blue",
            pad_edge=True,
        )
        table.add_column("Rank", justify="center", style="bold", width=6)
        table.add_column("Model", justify="left", min_width=20)
        table.add_column("PRI", justify="center", width=8)
        table.add_column("ORI", justify="center", width=8)
        table.add_column("IFI", justify="center", width=8)
        table.add_column("CS", justify="center", width=8)
        table.add_column("HS", justify="center", width=8)
        table.add_column("Consistency", justify="center", width=12)
        table.add_column("Human Score", justify="center", width=12)
        table.add_column("Final Score", justify="center", width=12)

        best_model = df.iloc[0]["Model"] if len(df) > 0 else None

        for _, row in df.iterrows():
            is_best = (row["Model"] == best_model)
            row_style = "bold on grey15" if is_best else ""

            pri_color = _score_color(row["PRI"])
            ori_color = _score_color(row.get("ORI", 0))
            ifi_color = _score_color(row.get("IFI", 0))
            cs_color = _score_color(row["CS"])
            hs_color = _score_color(row["HS"], invert=True)
            cons_color = _score_color(row["Consistency"])
            human_color = _score_color(row["Human_Score"])
            final_color = _score_color(row["Final_Score"])

            rank_val = f"{'🥇' if is_best else ''} {int(row['Rank'])}"

            table.add_row(
                rank_val,
                row["Model"],
                f"[{pri_color}]{_fmt(row['PRI'])}[/]",
                f"[{ori_color}]{_fmt(row.get('ORI', 0))}[/]",
                f"[{ifi_color}]{_fmt(row.get('IFI', 0))}[/]",
                f"[{cs_color}]{_fmt(row['CS'])}[/]",
                f"[{hs_color}]{_fmt(row['HS'])}[/]",
                f"[{cons_color}]{_fmt(row['Consistency'])}[/]",
                f"[{human_color}]{_fmt(row['Human_Score'])}[/]",
                f"[{final_color}]{_fmt(row['Final_Score'])}[/]",
                style=row_style,
            )

        _console.print()
        _console.print(table)
        _console.print()

    elif HAS_TABULATE:
        headers = ["Rank", "Model", "PRI", "ORI", "IFI", "CS", "HS", "Consistency", "Human Score", "Final Score"]
        rows = []
        for _, row in df.iterrows():
            rows.append([
                int(row["Rank"]), row["Model"],
                _fmt(row["PRI"]), _fmt(row.get("ORI", 0)), _fmt(row.get("IFI", 0)),
                _fmt(row["CS"]), _fmt(row["HS"]),
                _fmt(row["Consistency"]), _fmt(row["Human_Score"]),
                _fmt(row["Final_Score"]),
            ])
        print("\n=== Model Performance Summary ===")
        print(tabulate(rows, headers=headers, tablefmt="grid", stralign="center"))
        print()

    else:
        _fallback_print("Model Performance Summary", df, [
            "Rank", "Model", "PRI", "ORI", "IFI", "CS", "HS", "Consistency", "Human_Score", "Final_Score"
        ])


# =============================================================================
# TABLE 2: Advanced Metrics
# =============================================================================

def render_advanced_metrics_table(results_df: pd.DataFrame) -> None:
    """
    Render TABLE 2 — Advanced Metrics.

    Columns: Model, Faithfulness, TRD_Semantic, KPIG_Advanced, USD
    Adds ⚠ indicator if KPIG_Advanced > 0.95 (saturation warning).
    """
    adv_cols = ["Faithfulness", "TRD_Semantic", "KPIG_Advanced", "USD"]
    available = [c for c in adv_cols if c in results_df.columns]
    if not available:
        return

    df = results_df.copy()

    if HAS_RICH:
        table = Table(
            title="Advanced Metrics",
            box=box.ROUNDED,
            show_lines=True,
            title_style="bold cyan",
            header_style="bold white on dark_blue",
        )
        table.add_column("Model", justify="left", min_width=20)
        table.add_column("Faithfulness", justify="center", width=14)
        table.add_column("TRD_Semantic", justify="center", width=14)
        table.add_column("KPIG_Advanced", justify="center", width=15)
        table.add_column("USD", justify="center", width=8)

        for _, row in df.iterrows():
            faith = row.get("Faithfulness", 0)
            trd_s = row.get("TRD_Semantic", 0)
            kpig_a = row.get("KPIG_Advanced", 0)
            usd = row.get("USD", 0)

            # Saturation warning
            kpig_str = _fmt(kpig_a)
            if kpig_a > 0.95:
                kpig_str += " ⚠"
                kpig_color = "red"
            else:
                kpig_color = _score_color(kpig_a)

            table.add_row(
                row["Model"],
                f"[{_score_color(faith)}]{_fmt(faith)}[/]",
                f"[{_score_color(trd_s, invert=True)}]{_fmt(trd_s)}[/]",
                f"[{kpig_color}]{kpig_str}[/]",
                f"[{_score_color(usd, invert=True)}]{_fmt(usd)}[/]",
            )

        _console.print(table)
        _console.print()

    elif HAS_TABULATE:
        headers = ["Model", "Faithfulness", "TRD_Semantic", "KPIG_Advanced", "USD"]
        rows = []
        for _, row in df.iterrows():
            kpig_val = row.get("KPIG_Advanced", 0)
            kpig_str = _fmt(kpig_val)
            if kpig_val > 0.95:
                kpig_str += " ⚠"
            rows.append([
                row["Model"],
                _fmt(row.get("Faithfulness", 0)),
                _fmt(row.get("TRD_Semantic", 0)),
                kpig_str,
                _fmt(row.get("USD", 0)),
            ])
        print("=== Advanced Metrics ===")
        print(tabulate(rows, headers=headers, tablefmt="grid", stralign="center"))
        print()

    else:
        _fallback_print("Advanced Metrics", df, ["Model"] + available)


# =============================================================================
# TABLE 3: ROUGE Scores
# =============================================================================

def render_rouge_table(results_df: pd.DataFrame) -> None:
    """
    Render TABLE 3 — ROUGE Scores.

    Columns: Model, ROUGE-1, ROUGE-2, ROUGE-L
    """
    rouge_cols = ["ROUGE_1", "ROUGE_2", "ROUGE_L"]
    available = [c for c in rouge_cols if c in results_df.columns]
    if not available:
        return

    df = results_df.copy()

    if HAS_RICH:
        table = Table(
            title="ROUGE Scores",
            box=box.ROUNDED,
            show_lines=True,
            title_style="bold cyan",
            header_style="bold white on dark_blue",
        )
        table.add_column("Model", justify="left", min_width=20)
        table.add_column("ROUGE-1", justify="center", width=10)
        table.add_column("ROUGE-2", justify="center", width=10)
        table.add_column("ROUGE-L", justify="center", width=10)

        for _, row in df.iterrows():
            r1 = row.get("ROUGE_1", 0)
            r2 = row.get("ROUGE_2", 0)
            rl = row.get("ROUGE_L", 0)

            table.add_row(
                row["Model"],
                f"[{_score_color(r1)}]{_fmt(r1)}[/]",
                f"[{_score_color(r2)}]{_fmt(r2)}[/]",
                f"[{_score_color(rl)}]{_fmt(rl)}[/]",
            )

        _console.print(table)
        _console.print()

    elif HAS_TABULATE:
        headers = ["Model", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
        rows = []
        for _, row in df.iterrows():
            rows.append([
                row["Model"],
                _fmt(row.get("ROUGE_1", 0)),
                _fmt(row.get("ROUGE_2", 0)),
                _fmt(row.get("ROUGE_L", 0)),
            ])
        print("=== ROUGE Scores ===")
        print(tabulate(rows, headers=headers, tablefmt="grid", stralign="center"))
        print()

    else:
        _fallback_print("ROUGE Scores", df, ["Model"] + available)


# =============================================================================
# TABLE 4: Correlation Analysis
# =============================================================================

def render_correlation_table(
    correlations: Optional[Dict] = None,
    correlations_path: str = "results/correlations.json",
) -> None:
    """
    Render TABLE 4 — Correlation Analysis.

    Columns: Metric Pair, Pearson, Spearman
    Reads from correlations dict or correlations.json file.
    """
    # Load correlations
    if correlations is None:
        if os.path.exists(correlations_path):
            with open(correlations_path, "r") as f:
                correlations = json.load(f)
        else:
            print("  (correlations.json not found)")
            return

    corr_data = correlations.get("correlations", {})
    if not corr_data:
        print("  (no correlation data)")
        return

    # Friendly names for metric pairs
    friendly_names = {
        "PRI_vs_Human_Score": "PRI vs Human Score",
        "SMS_vs_Human_Score": "SMS vs Human Score",
        "CS_vs_Human_Score": "CS vs Human Score",
        "Faithfulness_vs_Human_Score": "Faithfulness vs Human Score",
        "TRD_vs_Human_Score": "TRD vs Human Score",
        "TRD_Semantic_vs_Human_Score": "TRD Semantic vs Human Score",
        "KPIG_Advanced_vs_Human_Score": "KPIG Advanced vs Human Score",
    }

    if HAS_RICH:
        table = Table(
            title="Correlation Analysis",
            box=box.ROUNDED,
            show_lines=True,
            title_style="bold cyan",
            header_style="bold white on dark_blue",
        )
        table.add_column("Metric Pair", justify="left", min_width=30)
        table.add_column("Pearson r", justify="center", width=12)
        table.add_column("Spearman ρ", justify="center", width=12)

        for key, data in corr_data.items():
            name = friendly_names.get(key, key.replace("_", " "))
            pearson = data.get("pearson_r", 0)
            spearman = data.get("spearman_rho", 0)

            # Color based on correlation strength
            p_color = "green" if abs(pearson) >= 0.5 else ("yellow" if abs(pearson) >= 0.3 else "red")
            s_color = "green" if abs(spearman) >= 0.5 else ("yellow" if abs(spearman) >= 0.3 else "red")

            table.add_row(
                name,
                f"[{p_color}]{_fmt(pearson)}[/]",
                f"[{s_color}]{_fmt(spearman)}[/]",
            )

        _console.print(table)
        _console.print()

    elif HAS_TABULATE:
        headers = ["Metric Pair", "Pearson r", "Spearman ρ"]
        rows = []
        for key, data in corr_data.items():
            name = friendly_names.get(key, key.replace("_", " "))
            rows.append([
                name,
                _fmt(data.get("pearson_r", 0)),
                _fmt(data.get("spearman_rho", 0)),
            ])
        print("=== Correlation Analysis ===")
        print(tabulate(rows, headers=headers, tablefmt="grid", stralign="center"))
        print()

    else:
        print("\n=== Correlation Analysis ===")
        for key, data in corr_data.items():
            name = friendly_names.get(key, key)
            print(f"  {name}: Pearson={_fmt(data.get('pearson_r', 0))}, "
                  f"Spearman={_fmt(data.get('spearman_rho', 0))}")
        print()


# =============================================================================
# TABLE 5: Stability Metrics
# =============================================================================

def render_stability_table(sample_df: pd.DataFrame) -> None:
    """
    Render TABLE 5 — Stability Metrics (Mean, Std, Stability Label).

    Rules:
        Std < 0.1  → "Stable" (green)
        0.1–0.2    → "Moderate" (yellow)
        > 0.2      → "Unstable" (red)
    """
    if sample_df.empty:
        return

    # Collect all numeric metric columns
    metric_cols = [c for c in sample_df.columns if sample_df[c].dtype in ('float64', 'float32', 'int64')]
    if not metric_cols:
        return

    stats = []
    for col in metric_cols:
        mean_val = sample_df[col].mean()
        std_val = sample_df[col].std()
        label, color = _stability_label(std_val)
        stats.append((col, mean_val, std_val, label, color))

    if HAS_RICH:
        table = Table(
            title="Stability Metrics",
            box=box.ROUNDED,
            show_lines=True,
            title_style="bold cyan",
            header_style="bold white on dark_blue",
        )
        table.add_column("Metric", justify="left", min_width=18)
        table.add_column("Mean", justify="center", width=10)
        table.add_column("Std Dev", justify="center", width=10)
        table.add_column("Stability", justify="center", width=12)

        for col, mean_val, std_val, label, color in stats:
            table.add_row(
                col,
                _fmt(mean_val),
                _fmt(std_val),
                f"[{color}]{label}[/]",
            )

        _console.print(table)
        _console.print()

    elif HAS_TABULATE:
        headers = ["Metric", "Mean", "Std Dev", "Stability"]
        rows = []
        for col, mean_val, std_val, label, _ in stats:
            rows.append([col, _fmt(mean_val), _fmt(std_val), label])
        print("=== Stability Metrics ===")
        print(tabulate(rows, headers=headers, tablefmt="grid", stralign="center"))
        print()

    else:
        print("\n=== Stability Metrics ===")
        for col, mean_val, std_val, label, _ in stats:
            print(f"  {col:<20} Mean={_fmt(mean_val)}  Std={_fmt(std_val)}  [{label}]")
        print()


# =============================================================================
# Master renderer — calls all 5 tables in sequence
# =============================================================================

def render_all_tables(
    results_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    correlations: Optional[Dict] = None,
    correlations_path: str = "results/correlations.json",
) -> None:
    """
    Render all 5 publication-quality tables to the terminal.

    Call this from main.py after the benchmark completes.

    Args:
        results_df: Aggregated per-model benchmark DataFrame.
        sample_df: Per-sample metrics DataFrame (for stability analysis).
        correlations: Pre-loaded correlations dict (or None to load from file).
        correlations_path: Path to correlations.json.
    """
    if HAS_RICH:
        _console.print()
        _console.print(
            Panel(
                "[bold green]✅ Benchmark Complete — Results Summary[/]",
                style="bold white on dark_blue",
                expand=False,
            )
        )
        _console.print()

    # TABLE 1: Performance Summary
    render_performance_table(results_df)

    # TABLE 2: Advanced Metrics (only if columns exist)
    adv_cols = ["SMS_Wasserstein", "TRD_Semantic", "KPIG_Advanced", "USD"]
    if any(c in results_df.columns for c in adv_cols):
        render_advanced_metrics_table(results_df)

    # TABLE 3: ROUGE Scores (only if columns exist)
    rouge_cols = ["ROUGE_1", "ROUGE_2", "ROUGE_L"]
    if any(c in results_df.columns for c in rouge_cols):
        render_rouge_table(results_df)

    # TABLE 4: Correlation Analysis
    render_correlation_table(correlations, correlations_path)

    # TABLE 5: Stability Metrics
    render_stability_table(sample_df)


# =============================================================================
# Fallback for environments without rich or tabulate
# =============================================================================

def _fallback_print(title: str, df: pd.DataFrame, cols: List[str]) -> None:
    """Plain-text table rendering when neither rich nor tabulate is available."""
    available = [c for c in cols if c in df.columns]
    if not available:
        return
    print(f"\n=== {title} ===")
    print(df[available].round(3).to_string(index=False))
    print()
