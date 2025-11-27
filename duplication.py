
#  Deduplication (exact + near-dup)
import logging
import json
import hashlib
import re
from pathlib import Path
from typing import Tuple, Union

import pandas as pd

PathLike = Union[str, Path]

logger = logging.getLogger("dedup_stage")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    


# A — Canonicalisation helpers

WHITESPACE_RE = re.compile(r"\s+")


def canonicalize_for_exact(text: str) -> str:
    """
    Canonical form for exact dedup:
      - handle NaN safely
      - lowercase
      - strip leading/trailing
      - collapse internal whitespace
    """
    if pd.isna(text):
        return ""
    s = str(text)
    s = s.lower()
    s = s.strip()
    s = WHITESPACE_RE.sub(" ", s)
    return s


def canonicalize_for_near(text: str, max_chars: int = 500) -> str:
    """
    Idea: if the first ~500 characters are identical after normalization,
    it's very likely a near-duplicate, especially for long docs.
    """
    base = canonicalize_for_exact(text)
    if not base:
        return ""
    return base[:max_chars]


def sha256_hash(s: str) -> str:
    """
    Compute a strong SHA256 hash of a string.
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()



# b Deduplication stage
def dedup_stage(
    df: pd.DataFrame,
    text_col: str = "text_pii_masked",  # final text we care about (from last stage)
    near_dup_min_len: int = 200,        # only consider near-dup for longer docs
    near_key_chars: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
 
    Steps:
      1) Exact dedup:
         - canonicalize text (lowercase + whitespace normalization)
         - hash with SHA256 -> exact_hash
         - keep the first row per exact_hash
         - mark others:
             is_dup_exact = True
             dup_of       = doc_id (or index) of first
             drop_reason  = 'exact_duplicate'

      2) Near-duplicate heuristic (simple):
         - for texts with len(canonical_text) >= near_dup_min_len:
             * build near_key by canonicalizing and truncating to `near_key_chars`
             * group by near_key
             * within each group, keep the first, drop others as 'near_duplicate'
               (only if they haven't been dropped already)

    Adds columns:
      - exact_canon_text
      - exact_hash
      - is_dup_exact (bool)
      - dup_of (doc_id or index of canonical doc)
      - near_key
      - is_dup_near (bool)

    Returns:
      kept_df    : rows kept after dedup
      dropped_df : rows dropped due to 'exact_duplicate' / 'near_duplicate'
                   (plus any previous drop_reason carried through)
    """

    df_work = df.copy()

    #  drop_reason 
    if "drop_reason" not in df_work.columns:
        df_work["drop_reason"] = pd.NA

    # Ensure stable ID
    if "doc_id" in df_work.columns:
        df_work["dedup_id"] = df_work["doc_id"].astype(str)
    else:
        df_work["dedup_id"] = df_work.index.astype(str)

    # Exact dedup ----------

    df_work["exact_canon_text"] = df_work[text_col].map(canonicalize_for_exact)
    df_work["exact_hash"] = df_work["exact_canon_text"].map(sha256_hash)

    # Mark exact duplicates: keep first, drop rest
    df_work["is_dup_exact"] = False
    df_work["dup_of"] = pd.NA  # will store dedup_id of canonical doc

    # For each hash, identify duplicates
    first_ids = {} 

    for idx, row in df_work.iterrows():
        h = row["exact_hash"]
        if h not in first_ids:
            # first time we see this hash
            first_ids[h] = row["dedup_id"]
        else:
            # duplicate of an earlier row
            df_work.at[idx, "is_dup_exact"] = True
            df_work.at[idx, "dup_of"] = first_ids[h]

    # Set drop_reason for exact duplicates if not already dropped
    mask_exact_dup = df_work["is_dup_exact"] & df_work["drop_reason"].isna()
    df_work.loc[mask_exact_dup, "drop_reason"] = "exact_duplicate"

    #  2) Simple near-duplicate heuristic ----------
    df_work["near_key"] = df_work[text_col].map(
        lambda s: canonicalize_for_near(s, max_chars=near_key_chars)
    )
    df_work["is_dup_near"] = False

    # Filter to candidates: long enough, non-empty near_key
    candidate_mask = (
        df_work["exact_canon_text"].str.len() >= near_dup_min_len
    ) & df_work["near_key"].ne("")

    candidates = df_work[candidate_mask].copy()

    # Group by near_key and mark duplicates (within group)
    for near_key, group in candidates.groupby("near_key"):
        if len(group) <= 1:
            continue  # no duplicates here

        # Sort by something stable (e.g., index) and keep the first
        group_sorted = group.sort_index()
        first_id = group_sorted.iloc[0]["dedup_id"]
        dup_indices = group_sorted.index[1:]

        # Mark near duplicates only if not already exact dup
        for idx in dup_indices:
            if not df_work.at[idx, "is_dup_exact"]:
                df_work.at[idx, "is_dup_near"] = True
                # only assign drop_reason if still NA
                if pd.isna(df_work.at[idx, "drop_reason"]):
                    df_work.at[idx, "drop_reason"] = "near_duplicate"
                # if dup_of is still NA, set it to canonical doc
                if pd.isna(df_work.at[idx, "dup_of"]):
                    df_work.at[idx, "dup_of"] = first_id

    # ---------- Split kept vs dropped ----------
    mask_dropped = df_work["drop_reason"].notna()
    kept_df = df_work[~mask_dropped].copy()
    dropped_df = df_work[mask_dropped].copy()

    # ---------- Summary ----------
    logger.info("=== Stage 5: Deduplication summary ===")
    logger.info("Input rows           : %d", len(df))
    logger.info("Kept rows            : %d", len(kept_df))
    logger.info("Dropped rows         : %d", len(dropped_df))

    if len(dropped_df) > 0:
        logger.info("Drop reasons (including dedup):")
        logger.info("\n%s", dropped_df["drop_reason"].value_counts())

    logger.info("Exact duplicate rows : %d", int(df_work["is_dup_exact"].sum()))
    logger.info("Near-duplicate rows  : %d", int(df_work["is_dup_near"].sum()))

    return kept_df, dropped_df



if __name__ == "__main__":

    INPUT_PARQUET_V4 = "mainpipe_cleaned_v4.parquet"

    CLEANED_PARQUET_V5 = "mainpipe_cleaned_v5.parquet"
    DROPPED_PARQUET_V5 = "mainpipe_dropped_v5.parquet"
    CLEANED_JSONL_V5 = "mainpipe_cleaned_v5.jsonl"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    logger.info("Loading Stage 4 cleaned parquet from: %s", INPUT_PARQUET_V4)
    df_stage4 = pd.read_parquet(INPUT_PARQUET_V4)
    logger.info("✔ Loaded Stage 4 cleaned parquet successfully")

    # Run Stage 5 deduplication
    stage5_clean_df, stage5_dropped_df = dedup_stage(
        df_stage4,
        text_col="text_pii_masked",  # or "text_deep_clean"/"text_norm" if you prefer
        near_dup_min_len=200,
        near_key_chars=500,
    )
    logger.info("✔ Deduplication step finished")


    # Save deduped cleaned parquet
    logger.info("Writing deduplicated cleaned parquet to: %s", CLEANED_PARQUET_V5)
    stage5_clean_df.to_parquet(CLEANED_PARQUET_V5, index=False)
    logger.info("✔ Saved deduplicated cleaned parquet")

    # Save deduped dropped parquet
    logger.info("Writing deduplicated dropped parquet to: %s", DROPPED_PARQUET_V5)
    stage5_dropped_df.to_parquet(DROPPED_PARQUET_V5, index=False)

    # JSONL: use the same convention as Stage 4 — prefer masked text
    logger.info("Writing deduplicated cleaned JSONL to: %s", CLEANED_JSONL_V5)
    with open(CLEANED_JSONL_V5, "w", encoding="utf-8") as f:
        for _, row in stage5_clean_df.iterrows():
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
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Stage 5 complete.")
    logger.info("Final docs after dedup: %d", len(stage5_clean_df))
    logger.info(
        "Total dropped due to dedup (or earlier reasons in this stage): %d",
        len(stage5_dropped_df),
    )
