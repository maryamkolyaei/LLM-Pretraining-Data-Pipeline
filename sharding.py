
import os
import math
import json
from datetime import datetime
from pathlib import Path
from typing import Union

import logging
import numpy as np
import pandas as pd

PathLike = Union[str, Path]

logger = logging.getLogger("stage8_sharding")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ---- Config ----

TOKENISED_PARQUET_V7 = "mainpipe_tokenised_v7.parquet"   # from previous step
SHARD_DIR = "shards"
SHARD_BASE_NAME = "train_shard"                          # train_shard_00001.jsonl, ...
DOCS_PER_SHARD = 50_000                                  # adjust as needed
TOY_SHARD_PATH = "tiny_train.jsonl"                      # small sample for quick experiments
MANIFEST_PATH = "manifest.json"
TOKENIZER_NAME = "gpt2"                           



# A — Helpers

def ensure_dir(path: str) -> None:
    """
    Create directory if it doesn't exist.
    """
    os.makedirs(path, exist_ok=True)


def _json_safe(val):
    """
    Make values JSON-serialisable:
      - numpy arrays -> lists
      - numpy scalars -> Python scalars
      - pandas Timestamps -> ISO strings
    """
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.generic,)):  # numpy scalar types
        return val.item()
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    return val


def export_shard_jsonl(
    df: pd.DataFrame,
    path: str,
    include_attention_mask: bool = True,
) -> None:
    """
    Export a single shard to JSONL.

    Field layout is consistent with Stage 7's training JSONL:
      - input_ids
      - attention_mask (optional)
      - doc_id
      - url
      - subset
      - mixture_name
      - quality_score
    """

    fields = [
        "input_ids",
        "doc_id",
        "url",
        "subset",
        "mixture_name",
        "quality_score",
    ]

    if include_attention_mask:
        fields.insert(1, "attention_mask")

    logger.info("  Writing shard: %s", path)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {}
            for field in fields:
                if field in row.index:
                    val = row[field]
                    rec[field] = _json_safe(val)
            f.write(json.dumps(rec) + "\n")


def make_shards_from_df(
    df: pd.DataFrame,
    out_dir: str = SHARD_DIR,
    base_name: str = SHARD_BASE_NAME,
    docs_per_shard: int = DOCS_PER_SHARD,
    include_attention_mask: bool = True,
    tokenizer_name: str = TOKENIZER_NAME,
    manifest_path: str = MANIFEST_PATH,
) -> dict:
    """
    Split a tokenised DataFrame into JSONL shards and write a manifest.

    Strategy:
      - Shard by document count (docs_per_shard).
      - Each shard is a JSONL file with the same schema as Stage 7 export.
      - Manifest contains per-shard doc counts and token counts.
    """

    ensure_dir(out_dir)

    n_docs = len(df)
    n_shards = math.ceil(n_docs / docs_per_shard) if n_docs > 0 else 0

    logger.info("=== Stage 8: Sharding & exports ===")
    logger.info("Total docs       : %d", n_docs)
    logger.info("Docs per shard   : %d", docs_per_shard)
    logger.info("Number of shards : %d", n_shards)

    shards_info = []

    for shard_idx in range(n_shards):
        start = shard_idx * docs_per_shard
        end = min((shard_idx + 1) * docs_per_shard, n_docs)

        shard_df = df.iloc[start:end]
        shard_id = shard_idx + 1

        shard_filename = f"{base_name}_{shard_id:05d}.jsonl"
        shard_path = os.path.join(out_dir, shard_filename)

        export_shard_jsonl(
            shard_df,
            path=shard_path,
            include_attention_mask=include_attention_mask,
        )

        num_docs = len(shard_df)
        total_tokens = int(shard_df["n_tokens"].sum()) if "n_tokens" in shard_df.columns else None

        shards_info.append(
            {
                "shard_id": shard_id,
                "filename": shard_filename,
                "path": shard_path,
                "num_docs": num_docs,
                "total_tokens": total_tokens,
            }
        )

    manifest = {
        "created_at_utc": datetime.utcnow().isoformat(),
        "tokenizer_name": tokenizer_name,
        "docs_per_shard": docs_per_shard,
        "num_shards": n_shards,
        "total_docs": n_docs,
        "total_tokens": int(df["n_tokens"].sum()) if "n_tokens" in df.columns else None,
        "shards": shards_info,
    }

    logger.info("Writing manifest to: %s", manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest written. num_shards=%d total_docs=%d", n_shards, n_docs)
    return manifest


def make_toy_shard(
    df: pd.DataFrame,
    path: str = TOY_SHARD_PATH,
    num_docs: int = 1_000,
    include_attention_mask: bool = True,
) -> None:
    """
    Create a small 'toy' shard for quick experiments.

    - Random sample of num_docs (or fewer if df smaller).
    - JSONL format, same as training shards.
    """

    sample_n = min(num_docs, len(df))
    toy_df = df.sample(sample_n, random_state=42)

    logger.info("Creating toy shard with %d docs at: %s", sample_n, path)
    export_shard_jsonl(
        toy_df,
        path=path,
        include_attention_mask=include_attention_mask,
    )



# Entrypoint 
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # 1) Load tokenised 
    logger.info("Loading tokenised parquet from: %s", TOKENISED_PARQUET_V7)
    df_tok = pd.read_parquet(TOKENISED_PARQUET_V7)

    if "input_ids" not in df_tok.columns:
        raise ValueError(
            f"'input_ids' column not found in {TOKENISED_PARQUET_V7}. "
            "Make sure Stage 7 (tokenisation) ran successfully."
        )

    # 2) Create JSONL shards + manifest
    manifest = make_shards_from_df(
        df_tok,
        out_dir=SHARD_DIR,
        base_name=SHARD_BASE_NAME,
        docs_per_shard=DOCS_PER_SHARD,
        include_attention_mask=True,
        tokenizer_name=TOKENIZER_NAME,
        manifest_path=MANIFEST_PATH,
    )

    logger.info(
        "Manifest summary: num_shards=%d total_docs=%d total_tokens=%s",
        manifest["num_shards"],
        manifest["total_docs"],
        str(manifest["total_tokens"]),
    )

    # 3) Small shard for quick experiments
    make_toy_shard(
        df_tok,
        path=TOY_SHARD_PATH,
        num_docs=1_000,
        include_attention_mask=True,
    )

    logger.info("Sharding complete.")
