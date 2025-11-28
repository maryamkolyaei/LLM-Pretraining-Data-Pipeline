#!/usr/bin/env python
"""
Final export step: JSONL with a mandatory 'text' field (non-tokenised).

- Input  : a cleaned parquet from your pipeline (e.g. mainpipe_scored_v6.parquet)
- Output : JSONL where each line has at least:
    {
      "text": "<non-tokenised cleaned text>",
      ...optional metadata...
    }

"""

import json
import logging
from pathlib import Path
from typing import Union, Sequence

import pandas as pd


INPUT_PARQUET: str = "mainpipe_scored_v6.parquet"
OUTPUT_JSONL: str = "mainpipe_scored_v6_text.jsonl"

TEXT_COLS_PRIORITY: Sequence[str] = (
    "text_pii_masked",
    "text_deep_clean",
    "text_norm",
    "text",             
)


EXTRA_FIELDS: Sequence[str] = (
    "doc_id",
    "source",
    "url",
    "subset",
    "mixture_name",
    "quality_score",
)

# --------------------------------------------------------------------
# LOGGING
# --------------------------------------------------------------------
logger = logging.getLogger("export_clean_jsonl")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)



# CORE 
PathLike = Union[str, Path]


def export_clean_jsonl(
    parquet_in: PathLike,
    jsonl_out: PathLike,
    text_cols_priority: Sequence[str] = TEXT_COLS_PRIORITY,
    extra_fields: Sequence[str] = EXTRA_FIELDS,
    drop_empty_text: bool = False,
) -> None:
    """
    Export a JSONL with a REQUIRED 'text' field containing the
    non-tokenised cleaned text.

    Parameters
    ----------
    parquet_in : str or Path
        Input Parquet path (e.g. mainpipe_scored_v6.parquet).
    jsonl_out : str or Path
        Output JSONL path (e.g. mainpipe_scored_v6_text.jsonl).
    text_cols_priority : list/tuple of str
        Ordered list of candidate text columns. First non-empty wins.
    extra_fields : list/tuple of str
        Additional columns to carry over if they exist in the dataframe.
    drop_empty_text : bool
        If True, rows with empty final 'text' are dropped from the JSONL.
    """
    parquet_in = Path(parquet_in)
    jsonl_out = Path(jsonl_out)

    if not parquet_in.exists():
        raise FileNotFoundError(f"Input parquet not found: {parquet_in}")

    logger.info("Loading cleaned parquet from: %s", parquet_in)
    df = pd.read_parquet(parquet_in)
    logger.info("Loaded %d rows.", len(df))

    # Pick the best available text column PER ROW
    logger.info("Selecting canonical 'text' using priority: %s", text_cols_priority)

    def pick_text(row) -> str:
        for col in text_cols_priority:
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    s = str(val).strip()
                    if s:
                        return s
        return ""  # fallback if everything is empty/NA

    df["text"] = df.apply(pick_text, axis=1)

    # Optionally drop rows with empty text
    empty_mask = df["text"].str.strip() == ""
    empty_count = int(empty_mask.sum())
    if empty_count > 0:
        logger.warning(
            "There are %d rows with empty 'text' after selection.", empty_count
        )
        if drop_empty_text:
            logger.info("Dropping rows with empty 'text'.")
            df = df[~empty_mask].copy()
            logger.info("Remaining rows after drop_empty_text: %d", len(df))

    logger.info("Writing JSONL to: %s", jsonl_out)
    with jsonl_out.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {"text": row["text"]}  # << REQUIRED FIELD

            # Attach extra metadata fields if present
            for col in extra_fields:
                if col in df.columns:
                    rec[col] = row[col]

            # ensure_ascii=False so UTF-8 is preserved
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Export complete. Wrote %d JSONL lines.", len(df))



if __name__ == "__main__":
    export_clean_jsonl(
        parquet_in=INPUT_PARQUET,
        jsonl_out=OUTPUT_JSONL,
        text_cols_priority=TEXT_COLS_PRIORITY,
        extra_fields=EXTRA_FIELDS,
        drop_empty_text=False,  
    )
