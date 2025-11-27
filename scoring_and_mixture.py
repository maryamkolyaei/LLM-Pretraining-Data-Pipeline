# ============================================
#  Document scoring & mixture creation
"""
Inputs

- mainpipe_cleaned_v5.parquet

Outputs
-------
- mainpipe_scored_v6.parquet
    Parquet with previous Stage columns plus:
      - quality_score : float ∈ [0, 1]
      - mixture_name  : str
      - subset        : {"high_quality", "rest"}

- mainpipe_scored_v6.jsonl
    JSONL format suitable for training:
      {
        "doc_id":        ...,
        "text":          ...,
        "source":        ...,
        "quality_score": ...,
        "subset":        ...,
        "mixture_name":  ...
      }

"""


import logging
import json
from pathlib import Path
from typing import Union
import pandas as pd

PathLike = Union[str, Path]

logger = logging.getLogger("stage6_scoring")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)



# A — Scoring function

def compute_quality_score(
    row: pd.Series,
    min_tokens_pref: int = 20,
    max_tokens_pref: int = 1000,
) -> float:
    """
    Compute a simple quality_score in [0, 1] from:
        - lang_score
        - token_count
        - unique_token_ratio
        - has_pii flag
    """

    # ---- Language confidence ----
    lang_conf = float(row.get("lang_score", 1.0) or 0.0)
    lang_score = max(0.0, min(1.0, lang_conf))

    # ---- Length preference ----
    token_count = int(row.get("token_count", 0) or 0)
    if token_count <= 0:
        length_score = 0.0
    elif token_count < min_tokens_pref:
        length_score = token_count / float(min_tokens_pref)
    elif token_count > max_tokens_pref:
        # Smooth penalty for overly long documents
        max_cap = max_tokens_pref * 4
        capped = min(token_count, max_cap)
        length_score = 1.0 - (capped - max_tokens_pref) / float(max_cap - max_tokens_pref)
        length_score = max(0.0, length_score)
    else:
        length_score = 1.0

    # ---- Uniqueness score ----
    uniq_ratio = float(row.get("unique_token_ratio", 0.0) or 0.0)
    uniq_ratio = max(0.0, min(1.0, uniq_ratio))
    uniqueness_score = uniq_ratio

    # ---- PII penalty ----
    has_pii = bool(row.get("has_pii", False))
    pii_score = 0.0 if has_pii else 1.0

    # ---- Weighted combination ----
    w_lang = 0.4
    w_len  = 0.3
    w_uniq = 0.2
    w_pii  = 0.1

    score = (
        w_lang * lang_score +
        w_len  * length_score +
        w_uniq * uniqueness_score +
        w_pii  * pii_score
    )

    return float(max(0.0, min(1.0, score)))



# B — Scoring + mixture assignment
def scoring_and_mixture_stage(
    df: pd.DataFrame,
    mixture_name: str = "web_sample",
    high_quality_threshold: float = 0.8,
    min_tokens_pref: int = 20,
    max_tokens_pref: int = 1000,
) -> pd.DataFrame:
    """
    Adds:
        - quality_score
        - mixture_name
        - subset = {high_quality, rest}
    """

    df_scored = df.copy()

    # Quality score
    df_scored["quality_score"] = df_scored.apply(
        lambda row: compute_quality_score(
            row,
            min_tokens_pref=min_tokens_pref,
            max_tokens_pref=max_tokens_pref,
        ),
        axis=1,
    )

    # Mixture name (single dataset mixture)
    df_scored["mixture_name"] = mixture_name

    # Subset assignment
    df_scored["subset"] = df_scored["quality_score"].apply(
        lambda q: "high_quality" if q >= high_quality_threshold else "rest"
    )

    # Logs
    logger.info("=== Stage 6: Scoring & mixture summary ===")
    logger.info("Rows scored: %d", len(df_scored))
    logger.info("Quality score stats:\n%s", df_scored["quality_score"].describe())
    logger.info("Subset distribution:\n%s", df_scored["subset"].value_counts())

    return df_scored



# C Entrypoint

if __name__ == "__main__":

    INPUT_PARQUET_V5 = "mainpipe_cleaned_v5.parquet"
    SCORED_PARQUET_V6 = "mainpipe_scored_v6.parquet"
    SCORED_JSONL_V6 = "mainpipe_scored_v6.jsonl"

    logger.info("Loading deduplicated parquet from: %s", INPUT_PARQUET_V5)
    df_stage5 = pd.read_parquet(INPUT_PARQUET_V5)
    logger.info("✔ Loaded Stage 5 parquet successfully")

    # Run Stage 6
    df_stage6 = scoring_and_mixture_stage(
        df_stage5,
        mixture_name="web_sample",
        high_quality_threshold=0.8,
        min_tokens_pref=20,
        max_tokens_pref=1000,
    )
    logger.info("✔ Scoring & mixture assignment completed")

    # Save parquet
    logger.info("Writing scored parquet to: %s", SCORED_PARQUET_V6)
    df_stage6.to_parquet(SCORED_PARQUET_V6, index=False)

    # Save JSONL
    logger.info("Writing scored JSONL to: %s", SCORED_JSONL_V6)
    with open(SCORED_JSONL_V6, "w", encoding="utf-8") as f:
        for _, row in df_stage6.iterrows():
            text_out = (
                row.get("text_pii_masked")
                or row.get("text_deep_clean")
                or row.get("text_norm")
                or row.get("text")
            )
            rec = {
                "doc_id": row.get("doc_id"),
                "text": text_out,
                "source": row.get("source"),
                "quality_score": row.get("quality_score"),
                "subset": row.get("subset"),
                "mixture_name": row.get("mixture_name"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("✔ Saved scored parquet")
    logger.info("Stage Scoring & mixture complete.")
