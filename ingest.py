import logging
import datetime
import hashlib
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np 

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
    
PathLike = Union[str, Path]


# Data loading
def load_raw(path: PathLike) -> pd.DataFrame:
    """
    Parameters
    ----------
    path : str or Path
        Path to the input JSONL file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the raw records.
    """
    path = Path(path)
    logging.info(f"Loading raw data from {path} ...")

    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_json(path, lines=True)

    logging.info(f"Loaded {len(df):,} raw records.")
    return df



# document ID creation
def make_doc_id(row: pd.Series) -> str:
    """
Parameters
    ----------
    row : pd.Series 

    Returns
    -------
    str
        A hexadecimal SHA1 hash string.
    """
    # Convert to string explicitly to avoid issues with NaN / None
    url = str(row.get("url", ""))
    text = str(row.get("text", ""))

    key = f"{url}||{text}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def add_doc_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Input df expected to contain 'url' and 'text' columns.

    Returns
    -------
    pd.DataFrame
        df with an additional 'doc_id' column.
    """
    required_cols = {"url", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for doc_id creation: {missing}")

    logging.info("Adding stable document IDs (doc_id) ...")


    df["doc_id"] = df.apply(make_doc_id, axis=1)

    logging.info("doc_id column added.")
    return df



# Ingestion pipeline
def ingest_mainpipe_v1(raw_path: PathLike, out_path: PathLike) -> pd.DataFrame:
    """
    Ingest the raw dataset and produce a canonical ingested dataset.

    Steps:
      1. Load raw JSONL data from `raw_path`.
      2. Add stable document IDs (`doc_id`) based on URL + text.
      3. Add basic provenance metadata (e.g., 'source').
      4. Add ingestion timestamp.
      5. Save the ingested dataset to Parquet at `out_path`.

    Parameters
    ----------
    raw_path and out_path : str or Path

    Returns
    -------
    pd.DataFrame
        The ingested DataFrame (also written to disk).
    """
    raw_path = Path(raw_path)
    out_path = Path(out_path)

    # Load raw data
    df_raw = load_raw(raw_path)

    # Create ingested copy to avoid mutating the original
    df_ing = df_raw.copy()

    # Add stable IDs
    df_ing = add_doc_ids(df_ing)

    # Add source metadata
    df_ing["source"] = "mainpipe_v1"

    # Add ingestion timestamp
    ingest_ts = datetime.datetime.utcnow().isoformat()
    df_ing["ingest_ts"] = ingest_ts

    # 6. Save as Parquet
    logging.info(f"Saving ingested dataset to: {out_path}")
    df_ing.to_parquet(out_path, index=False, engine="fastparquet")
    logging.info("Ingested dataset successfully written to disk.")

    return df_ing



# Convenience loader for the ingested dataset
def load_ingested_parquet(path: PathLike) -> pd.DataFrame:
    """
    Load a previously ingested parquet dataset.

    Parameters
    ----------
    path : str or Path
        Path to the Parquet file produced by `ingest_mainpipe_v1`.

    Returns
    -------
    pd.DataFrame
        The ingested DataFrame.
    """
    path = Path(path)
    logging.info(f"Loading ingested dataset from {path} ...")

    if not path.exists():
        raise FileNotFoundError(f"Ingested parquet file not found: {path}")

    df = pd.read_parquet(path, engine="fastparquet")
    logging.info(f"Loaded {len(df):,} ingested records.")
    return df
RAW_PATH = "mainpipe_data_v1.jsonl"
INGESTED_PATH = "mainpipe_ingested_v1.parquet"

# Run ingestion
df_ing = ingest_mainpipe_v1(RAW_PATH, INGESTED_PATH)

# Load back for next steps
df_loaded = load_ingested_parquet(INGESTED_PATH)

