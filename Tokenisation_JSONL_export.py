"""
Tokenisation & training-ready formatting.


1. Tokenise each document using a HuggingFace tokenizer.
2. Compute `n_tokens` per document.
3. Filter out documents that are:
     - too short (< MIN_TOKENS)
     - too long  (> MAX_TOKENS)
4. Export a training-ready JSONL file with tokenised fields and key metadata.

Inputs
------
- mainpipe_scored_v6.parquet

Outputs
-------
- mainpipe_tokenised_v7.parquet
    Parquet with all columns from previous step plus:
      - input_ids      : List[int]
      - attention_mask : List[int]
      - n_tokens       : int

- train_web_sample_tokenised.jsonl
    JSONL for training. Each line roughly:
      {
        "input_ids":      [...],
        "attention_mask": [...],
        "doc_id":         "...",
        "url":            "...",
        "subset":         "...",
        "mixture_name":   "...",
        "quality_score":  0.93
      }
"""


#  Tokenisation & training-ready formatting
import json
from typing import Tuple

import pandas as pd
from transformers import AutoTokenizer

import logging
logger = logging.getLogger("stage6_scoring")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Config


TEXT_COL = "text_pii_masked"          # primary text column from previous stages
MIN_TOKENS = 10                       # drop very short docs
MAX_TOKENS = 2048                     # drop very long docs (full length)
INPUT_PARQUET_V6 = "mainpipe_scored_v6.parquet"
TOKENISED_PARQUET_V7 = "mainpipe_tokenised_v7.parquet"
TRAIN_JSONL_PATH = "train_web_sample_tokenised.jsonl"



# B — Tokenisation helpers


def tokenize_texts(
    df: pd.DataFrame,
    tokenizer,
    text_col: str = TEXT_COL,
    add_special_tokens: bool = True,
) -> pd.DataFrame:
    """
    Tokenise texts in `text_col` using a HuggingFace-style tokenizer.

    Adds:
      - input_ids      : List[int]
      - attention_mask : List[int]
      - n_tokens       : int (len(input_ids))

    Notes
    -----
    - `truncation=False` so n_tokens reflects the FULL document length.
      Very long docs are filtered later by `filter_by_token_length`.
    """

    df_tok = df.copy()

    input_ids_list = []
    attn_masks_list = []
    n_tokens_list = []

    for text in df_tok[text_col].tolist():
        if pd.isna(text):
            text = ""

        encoded = tokenizer(
            str(text),
            add_special_tokens=add_special_tokens,
            truncation=False,              # keep full length, drop long later
            return_attention_mask=True,
        )

        ids = encoded["input_ids"]
        mask = encoded["attention_mask"]

        input_ids_list.append(ids)
        attn_masks_list.append(mask)
        n_tokens_list.append(len(ids))

    df_tok["input_ids"] = input_ids_list
    df_tok["attention_mask"] = attn_masks_list
    df_tok["n_tokens"] = n_tokens_list

    return df_tok


def filter_by_token_length(
    df: pd.DataFrame,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter documents based on token length.

    """

    df_work = df.copy()

    if "drop_reason" not in df_work.columns:
        df_work["drop_reason"] = pd.NA

    too_short_mask = df_work["n_tokens"] < min_tokens
    too_long_mask  = df_work["n_tokens"] > max_tokens

    # Only overwrite drop_reason if it was NA
    df_work.loc[too_short_mask & df_work["drop_reason"].isna(), "drop_reason"] = "too_few_tokens"
    df_work.loc[too_long_mask  & df_work["drop_reason"].isna(), "drop_reason"] = "too_many_tokens"

    mask_dropped = df_work["drop_reason"].notna()
    kept_df    = df_work[~mask_dropped].copy()
    dropped_df = df_work[mask_dropped].copy()

    print("=== Stage 7: Token length filter summary ===")
    print(f"Input rows           : {len(df)}")
    print(f"Kept rows            : {len(kept_df)}")
    print(f"Dropped rows         : {len(dropped_df)}")

    print("\nDrop reasons (incl. previous stages if any):")
    print(dropped_df["drop_reason"].value_counts())

    return kept_df, dropped_df


def export_tokenised_jsonl(
    df: pd.DataFrame,
    path: str,
    include_attention_mask: bool = True,
) -> None:
    """
    Export a tokenised dataset to JSONL, one document per line.

    Each line roughly looks like:
      {
        "input_ids":      [...],
        "attention_mask": [...],   # optional
        "doc_id":         "...",
        "url":            "...",
        "subset":         "...",
        "mixture_name":   "...",
        "quality_score":  0.93
      }

    Adjust the `fields` list as needed to match your training loop.
    """

    # Order of fields in JSON (if present)
    fields = [
        "input_ids",
        "doc_id",
        "url",
        "subset",
        "mixture_name",
        "quality_score",
    ]

    if include_attention_mask:
        fields.insert(1, "attention_mask")  # right after input_ids

    print(f"Writing JSONL to {path} ...")
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {}
            for field in fields:
                if field in row.index:
                    val = row[field]
                    # ensure lists etc. are JSON-serialisable
                    rec[field] = val
            f.write(json.dumps(rec) + "\n")
    print("Done.")


# ---------------------------
if __name__ == "__main__":

    # 1) Load previous step dataset
    print(f"Loading Stage 6 scored parquet from: {INPUT_PARQUET_V6}")
    df_stage6 = pd.read_parquet(INPUT_PARQUET_V6)
    logger.info("✔ Loaded Stage 6 parquet successfully")

    # Safety check: make sure we have the text column we expect
    if TEXT_COL not in df_stage6.columns:
        raise ValueError(
            f"TEXT_COL='{TEXT_COL}' not found in Stage 6 dataframe. "
            f"Available columns: {list(df_stage6.columns)[:20]} ..."
        )

    # 2) Load tokenizer 
    print("Loading tokenizer: gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    logger.info("✔ Tokenizer loaded")

    # If your model needs a padding token and it's not set (GPT2 case)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # common trick for GPT-style models


    stage7_tok_df = tokenize_texts(
        df_stage6,
        tokenizer=tokenizer,
        text_col=TEXT_COL,
        add_special_tokens=True,
    )
    logger.info("✔ Tokenisation completed")

    print("\nToken count stats (n_tokens):")
    print(stage7_tok_df[["n_tokens"]].describe())

    # 4) Filter by token length (e.g. keep 10–2048 tokens)
    stage7_kept_df, stage7_dropped_df = filter_by_token_length(
        stage7_tok_df,
        min_tokens=MIN_TOKENS,
        max_tokens=MAX_TOKENS,
    )

    print("\n>>> Final training docs:", len(stage7_kept_df))
    print(">>> Dropped (too short/long or earlier reasons):", len(stage7_dropped_df))

    # 5) Save tokenised parquet (optional but useful for inspection)
    print(f"\nWriting tokenised parquet to: {TOKENISED_PARQUET_V7}")
    stage7_kept_df.to_parquet(TOKENISED_PARQUET_V7, index=False)

    # 6) Export to JSONL for training
    export_tokenised_jsonl(
        stage7_kept_df,
        path=TRAIN_JSONL_PATH,
        include_attention_mask=True,
    )

    print("\nSample lines from tokenised JSONL:")
    with open(TRAIN_JSONL_PATH, "r", encoding="utf-8") as f:
        for _ in range(3):
            line = f.readline()
            if not line:
                break
            print(line.strip())
