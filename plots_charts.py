#!/usr/bin/env python
"""

Run after:  `python run_pipeline.py`
Usage:      `python generate_metrics_and_plots.py`
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend for saving PNGs
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
PLOTS_DIR = Path("plots")
REPORT_DIR = Path("reports")  # metrics here, plots in plots/

CLEANED_V2 = Path("mainpipe_cleaned_v2.parquet")
CLEANED_V4 = Path("mainpipe_cleaned_v4.parquet")
SCORED_V6 = Path("mainpipe_scored_v6.parquet")
TOKENISED_V7 = Path("mainpipe_tokenised_v7.parquet")

DROPPED_V2 = Path("mainpipe_dropped_v2.parquet")
DROPPED_V4 = Path("mainpipe_dropped_v4.parquet")
DROPPED_V5 = Path("mainpipe_dropped_v5.parquet")

METRICS_JSON = REPORT_DIR / "metrics_summary.json"
DROP_COUNTS_CSV = REPORT_DIR / "drop_reason_counts.csv"
PII_STATS_CSV = REPORT_DIR / "pii_stats.csv"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def safe_read_parquet(path: Path) -> pd.DataFrame:
    if path.exists():
        logging.info("Reading %s", path)
        return pd.read_parquet(path)
    logging.warning("File not found: %s (skipping)", path)
    return pd.DataFrame()


def save_histogram(series: pd.Series, title: str, xlabel: str, out_path: Path, bins: int = 50) -> None:
    if series is None:
        logging.warning("Series is None for %s, not plotting %s", title, out_path)
        return

    series = series.dropna()
    if series.empty:
        logging.warning("No data for %s, not plotting %s", title, out_path)
        return

    plt.figure(figsize=(8, 5))
    plt.hist(series, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logging.info("Saved plot: %s", out_path)


def save_bar_counts(series: pd.Series, title: str, xlabel: str, out_path: Path, top_n: int = 30) -> None:
    if series is None:
        logging.warning("Series is None for %s, not plotting %s", title, out_path)
        return

    series = series.dropna()
    if series.empty:
        logging.warning("No data for %s, not plotting %s", title, out_path)
        return

    counts = series.value_counts().head(top_n)

    plt.figure(figsize=(10, 6))
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logging.info("Saved plot: %s", out_path)


def series_describe_for_json(series: pd.Series) -> dict:
    if series is None:
        return {}
    series = series.dropna()
    if series.empty:
        return {}
    desc = series.describe()
    return {k: float(v) for k, v in desc.to_dict().items()}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    ensure_dirs()

    # Load dataframes
    df_clean_v2 = safe_read_parquet(CLEANED_V2)
    df_clean_v4 = safe_read_parquet(CLEANED_V4)
    df_scored_v6 = safe_read_parquet(SCORED_V6)
    df_tok_v7 = safe_read_parquet(TOKENISED_V7)

    df_dropped_v2 = safe_read_parquet(DROPPED_V2)
    df_dropped_v4 = safe_read_parquet(DROPPED_V4)
    df_dropped_v5 = safe_read_parquet(DROPPED_V5)

    # All dropped rows across stages (for drop_reason stats)
    df_dropped_all = pd.concat(
        [df_dropped_v2, df_dropped_v4, df_dropped_v5],
        axis=0,
        ignore_index=True,
    )

    # Choose lang source (v4 if available, else v2)
    lang_source = df_clean_v4 if "lang_score" in df_clean_v4.columns else df_clean_v2

    # -----------------------------------------------------------------
    # 1) Length & core histograms (plots/)
    # -----------------------------------------------------------------

    # Token length (Stage 7)
    if "n_tokens" in df_tok_v7.columns:
        save_histogram(
            df_tok_v7["n_tokens"],
            title="Token Length Distribution (n_tokens)",
            xlabel="n_tokens",
            out_path=PLOTS_DIR / "hist_n_tokens.png",
            bins=80,
        )

    # Character length (Stage 2)
    if "char_len" in df_clean_v2.columns:
        save_histogram(
            df_clean_v2["char_len"],
            title="Character Length Distribution (char_len)",
            xlabel="char_len",
            out_path=PLOTS_DIR / "hist_char_len.png",
            bins=80,
        )

    # Word count (Stage 2)
    if "word_count" in df_clean_v2.columns:
        save_histogram(
            df_clean_v2["word_count"],
            title="Word Count Distribution",
            xlabel="word_count",
            out_path=PLOTS_DIR / "hist_word_count.png",
            bins=80,
        )

    # Language score distribution
    if "lang_score" in lang_source.columns:
        save_histogram(
            lang_source["lang_score"],
            title="Language Confidence Distribution (lang_score)",
            xlabel="lang_score",
            out_path=PLOTS_DIR / "hist_lang_score.png",
            bins=50,
        )

    # Quality score distribution (Stage 6)
    if "quality_score" in df_scored_v6.columns:
        save_histogram(
            df_scored_v6["quality_score"],
            title="Quality Score Distribution",
            xlabel="quality_score",
            out_path=PLOTS_DIR / "hist_quality_score.png",
            bins=50,
        )

    # -----------------------------------------------------------------
    # 2) Noise / integrity metrics histograms
    # -----------------------------------------------------------------
    if "alpha_ratio" in df_clean_v2.columns:
        save_histogram(
            df_clean_v2["alpha_ratio"],
            title="Alpha Ratio Distribution",
            xlabel="alpha_ratio",
            out_path=PLOTS_DIR / "hist_alpha_ratio.png",
            bins=50,
        )

    if "repetition_ratio" in df_clean_v2.columns:
        save_histogram(
            df_clean_v2["repetition_ratio"],
            title="Repetition Ratio Distribution",
            xlabel="repetition_ratio",
            out_path=PLOTS_DIR / "hist_repetition_ratio.png",
            bins=50,
        )

    if "unique_token_ratio" in df_clean_v4.columns:
        save_histogram(
            df_clean_v4["unique_token_ratio"],
            title="Unique Token Ratio Distribution",
            xlabel="unique_token_ratio",
            out_path=PLOTS_DIR / "hist_unique_token_ratio.png",
            bins=50,
        )

    if "stopword_ratio" in df_clean_v4.columns:
        save_histogram(
            df_clean_v4["stopword_ratio"],
            title="Stopword Ratio Distribution",
            xlabel="stopword_ratio",
            out_path=PLOTS_DIR / "hist_stopword_ratio.png",
            bins=50,
        )

    # -----------------------------------------------------------------
    # 3) Duplication markers (from Stage 5 dropped parquet)
    # -----------------------------------------------------------------
    # (Plots for bar_exact_duplicates and bar_near_duplicates removed)

    # -----------------------------------------------------------------
    # 4) PII hit-rate plots (per type + summary)
    # -----------------------------------------------------------------
    pii_columns = ["pii_email_hits", "pii_phone_hits", "pii_cc_hits", "pii_iban_hits"]

    for col in pii_columns:
        if col in df_clean_v4.columns:
            save_bar_counts(
                df_clean_v4[col],
                title=f"{col} Distribution",
                xlabel=col,
                out_path=PLOTS_DIR / f"bar_{col}.png",
            )

    if all(c in df_clean_v4.columns for c in pii_columns):
        pii_sum = df_clean_v4[pii_columns].sum()
        plt.figure(figsize=(8, 5))
        pii_sum.plot(kind="bar")
        plt.title("PII Hit Summary by Type")
        plt.ylabel("Total Hits")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "bar_pii_summary.png")
        plt.close()
        logging.info("Saved PII summary plot")

    # -----------------------------------------------------------------
    # 5) Metrics export for dashboards (reports/)
    # -----------------------------------------------------------------
    metrics = {}

    # Token length stats
    if "n_tokens" in df_tok_v7.columns:
        metrics["n_tokens"] = series_describe_for_json(df_tok_v7["n_tokens"])

    # Lang score stats
    if "lang_score" in lang_source.columns:
        metrics["lang_score"] = series_describe_for_json(lang_source["lang_score"])

    # Quality scores
    if "quality_score" in df_scored_v6.columns:
        metrics["quality_score"] = series_describe_for_json(df_scored_v6["quality_score"])

    # Subset distribution
    if "subset" in df_scored_v6.columns:
        subset_counts = df_scored_v6["subset"].value_counts().to_dict()
        metrics["subset_counts"] = {str(k): int(v) for k, v in subset_counts.items()}

    # Count per stage
    metrics["counts"] = {
        "clean_v2_rows": int(len(df_clean_v2)),
        "clean_v4_rows": int(len(df_clean_v4)),
        "scored_v6_rows": int(len(df_scored_v6)),
        "tok_v7_rows": int(len(df_tok_v7)),
        "dropped_v2_rows": int(len(df_dropped_v2)),
        "dropped_v4_rows": int(len(df_dropped_v4)),
        "dropped_v5_rows": int(len(df_dropped_v5)),
    }

    # PII stats: combine v4 + dropped v4 + dropped v5
    pii_cols_full = [
        "pii_email_hits",
        "pii_phone_hits",
        "pii_cc_hits",
        "pii_iban_hits",
        "has_pii",
    ]

    df_pii = pd.concat(
        [
            df_clean_v4[[c for c in pii_cols_full if c in df_clean_v4.columns]],
            df_dropped_v4[[c for c in pii_cols_full if c in df_dropped_v4.columns]],
            df_dropped_v5[[c for c in pii_cols_full if c in df_dropped_v5.columns]],
        ],
        axis=0,
        ignore_index=True,
    )

    if not df_pii.empty:
        pii_agg = {}
        for col in pii_cols_full:
            if col in df_pii.columns:
                if df_pii[col].dtype == bool:
                    pii_agg[col] = int(df_pii[col].sum())
                else:
                    pii_agg[col] = int(df_pii[col].fillna(0).sum())
        metrics["pii_aggregate"] = pii_agg

        df_pii.to_csv(PII_STATS_CSV, index=False)
        logging.info("Saved PII stats CSV: %s", PII_STATS_CSV)

    # Drop reason counts CSV
    if not df_dropped_all.empty and "drop_reason" in df_dropped_all.columns:
        drop_counts = (
            df_dropped_all["drop_reason"].value_counts()
            .rename_axis("drop_reason")
            .reset_index(name="count")
        )
        drop_counts.to_csv(DROP_COUNTS_CSV, index=False)
        logging.info("Saved drop_reason counts CSV: %s", DROP_COUNTS_CSV)

    # Write metrics JSON
    with METRICS_JSON.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logging.info("Saved metrics JSON: %s", METRICS_JSON)


if __name__ == "__main__":
    main()
