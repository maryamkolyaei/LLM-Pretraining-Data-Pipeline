"""
- Load ingested Parquet from the ingestion step.
- a: text normalisation & early prefiltering.
- b: compute quality metrics (length, alpha ratio, repetition).
- c: language heuristic, PII heuristics, toxicity heuristic, model-based quality score hook.
- d: quality filters (English-only, length, alpha ratio, repetition, URL, PII, toxicity, model_q)
  with clear drop reasons.
- Export:
    * Cleaned Parquet (with metrics, lang, pii, tox, model_q, drop_reason).
    * Cleaned JSONL with 'text' column (required).
    * Optional Parquet with dropped rows for analysis.
- Provide inspectability via drop stats & metrics.
"""

import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
# from detoxify import Detoxify
import pandas as pd
import math
from dataclasses import dataclass, field
# DETOX_MODEL = Detoxify("original")
# PathLike = Union[str, Path]
from typing import Dict, List, Optional, Set, Tuple, Union
from langdetect import detect_langs, DetectorFactory


DetectorFactory.seed = 2025
PathLike = Union[str, Path]
USE_LANG_FILTER = False 

logger = logging.getLogger("stage6_scoring")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# a — Text normalisation

# Precompute control characters
CONTROL_CHARS = "".join(
    map(chr, list(range(0, 32)) + list(range(127, 160)))
)
CONTROL_CHAR_RE = re.compile("[%s]" % re.escape(CONTROL_CHARS))


def normalize_text(text: str) -> Optional[str]:
    """
    Normalize and lightly clean a text string.

    Operations:
      - Return None for missing / NaN values
      - Convert input to string (defensive)
      - Unicode normalize to NFKC
      - Remove control characters
      - Collapse all whitespace to single spaces
      - Strip leading/trailing whitespace
      - Return None if result is empty after cleaning
    """
    if pd.isna(text):
        return None

    text = str(text)

    # Unicode normalization (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters that can break tooling
    text = CONTROL_CHAR_RE.sub(" ", text)

    # Collapse all whitespace (spaces, tabs, newlines) to a single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing spaces
    text = text.strip()

    return text if text else None


def prefilter_and_normalise(
    df: pd.DataFrame,
    text_col: str = "text", #column that contains text
    min_chars: int = 20, #minimum number of characters after normalisation to keep
    min_words: int = 2, #minimum number of words after normalisation to keep
    max_chars: int = 100_000, #maximum number of characters allowed
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
     Pre-filter & normalisation (early junk filtering).

    For each row:
      - Normalize raw text into `text_norm`
      - Compute `char_len` and `word_count` on the normalized text
      - Assign a `drop_reason` if the row should be filtered out, with reasons:
            * 'missing'
            * 'non_string'
            * 'empty_after_clean'
            * 'null_like'
            * 'numeric_like'
            * 'too_short_chars'
            * 'too_short_words'
            * 'too_long'

    Returns:
      clean_df    : rows with drop_reason is NA     (kept)
      dropped_df  : rows with drop_reason not NA    (filtered out)
      mask_problem: boolean Series over original df
                    (True = dropped, False = kept)
    """
    df_work = df.copy()

    # Normalise into text column
    df_work["text_norm"] = df_work[text_col].map(normalize_text)

    # Character length of normalized text
    df_work["char_len"] = df_work["text_norm"].str.len()

    # Word count of normalized text
    df_work["word_count"] = df_work["text_norm"].fillna("").str.split().str.len()

    # Initialise drop_reason as missing
    df_work["drop_reason"] = pd.NA

    #I adopted below rules from AWS
    #Rule 1: missing values
    mask_missing = df[text_col].isna()
    df_work.loc[mask_missing, "drop_reason"] = "missing"

    # Rule 2: non-string values
    mask_nonstring = ~df[text_col].apply(lambda x: isinstance(x, str) or pd.isna(x))
    df_work.loc[
        mask_nonstring & df_work["drop_reason"].isna(), "drop_reason"
    ] = "non_string"

    # Rule 3: empty after cleaning
    empty_after_clean = df_work["text_norm"].isna()
    df_work.loc[
        empty_after_clean & df_work["drop_reason"].isna(), "drop_reason"
    ] = "empty_after_clean"

    # Rule 4: null-like strings after cleaning
    null_strings = {"nan", "none", "null", "n/a", "null value"}
    mask_null_like = (
        df_work["text_norm"]
        .fillna("")
        .str.strip()
        .str.lower()
        .isin(null_strings)
    )
    df_work.loc[
        mask_null_like & df_work["drop_reason"].isna(), "drop_reason"
    ] = "null_like"

    # Rule 5: numeric-like strings
    mask_numeric_like = df_work["text_norm"].fillna("").str.match(r"^[\d\.\-]+$")
    df_work.loc[
        mask_numeric_like & df_work["drop_reason"].isna(), "drop_reason"
    ] = "numeric_like"

    # Rule 6: too short by character length
    mask_too_short_chars = (df_work["char_len"] < min_chars) & ~empty_after_clean
    df_work.loc[
        mask_too_short_chars & df_work["drop_reason"].isna(), "drop_reason"
    ] = "too_short_chars"

    # Rule 7: too few words
    mask_too_short_words = (df_work["word_count"] < min_words) & ~empty_after_clean
    df_work.loc[
        mask_too_short_words & df_work["drop_reason"].isna(), "drop_reason"
    ] = "too_short_words"

    # Rule 8: too long by character length
    mask_too_long = (df_work["char_len"] > max_chars) & ~empty_after_clean
    df_work.loc[
        mask_too_long & df_work["drop_reason"].isna(), "drop_reason"
    ] = "too_long"

    mask_problem = df_work["drop_reason"].notna()
    clean_df = df_work[~mask_problem].copy()
    dropped_df = df_work[mask_problem].copy()

    logging.info("=== summary of rre-filter & normalisation (a) ===")
    logging.info(f"Input rows           : {len(df)}")
    logging.info(f"Kept rows            : {len(clean_df)}")
    logging.info(f"Dropped rows         : {len(dropped_df)}")
    if len(dropped_df) > 0:
        logging.info("Drop reasons (early):")
        logging.info("\n%s", dropped_df["drop_reason"].value_counts())
    else:
        logging.info("No rows dropped in a.")

    return clean_df, dropped_df, mask_problem



# b — Quality metrics, lang, PII, toxicity, model-based hook

def compute_quality_metrics(text: str) -> Dict[str, float]:
    """
    Compute quality metrics.

    Metrics:
      - n_chars
      - n_words
      - alpha_ratio: fraction of characters that are alphabetic
      - repetition_ratio: 1 - (unique_words / n_words)
    """
    if text is None:
        text = ""
    text = str(text)

    n_chars = len(text)
    tokens = text.split()
    n_words = len(tokens)

    alpha_chars = sum(ch.isalpha() for ch in text)
    alpha_ratio = alpha_chars / n_chars if n_chars > 0 else 0.0 #What fraction of characters are letters?

    unique_words = len(set(tokens)) if n_words > 0 else 0
    repetition_ratio = 1.0 - (unique_words / n_words) if n_words > 0 else 0.0
    
    
    return {
        "n_chars": n_chars,
        "n_words": n_words,
        "alpha_ratio": alpha_ratio,
        "repetition_ratio": repetition_ratio,
    }


def simple_pii_hits(text: str) -> Dict[str, int]:
    """
    Extremely rough PII hit counters using regexes.

    Returns:
      - email_hits
      - phone_hits
    """
    if text is None:
        text = ""
    text = str(text)

    email_hits = len(
        re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    )
    phone_hits = len(
        re.findall(r"\+?\d[\d\- ]{7,}\d", text)
    )

    return {
        "email_hits": email_hits,
        "phone_hits": phone_hits,
    }




#For this stage ML can be used, or use a more comprehensive words in huristicc, for take home I decided to make it faster and lighter
# def toxicity_ml(text: str) -> Dict[str, float]:
#     """
#     ML-based toxicity scoring using Detoxify.

#     Returns Detoxify scores such as:
#       - toxicity
#       - severe_toxicity
#       - insult
#       - threat
#       - identity_attack
#       - sexual_explicit

#     All scores are in [0, 1].
#     """
#     if text is None:
#         text = ""
#     text = str(text).strip()
#     if not text:
#         # empty text => no toxicity
#         return {
#             "toxicity": 0.0,
#             "severe_toxicity": 0.0,
#             "insult": 0.0,
#             "threat": 0.0,
#             "identity_attack": 0.0,
#             "sexual_explicit": 0.0,
#         }

#     scores = DETOX_MODEL.predict(text)
#     # Ensure plain Python floats
#     return {k: float(v) for k, v in scores.items()}



# --- Simple keyword-based toxicity heuristic ---

BAD_WORDS = {
    "fuck", "fucking", "shit", "bitch", "bastard", "asshole", "crap",
    "damn", "dick", "piss",
}

INSULT_WORDS = {
    "idiot", "moron", "stupid", "loser", "dumb",
}

THREAT_WORDS = {
    "kill", "murder", "hurt", "shoot", "stab",
}

SEXUAL_WORDS = {
    "sex", "porn", "nude", "naked",
}

SLUR_WORDS = {
    "nigger", "nigga", "faggot", "retard",
}


def toxicity_heuristic(text: str) -> Dict[str, float]:
    """
    Very simple toxicity heuristic.

    - Token-level keyword match.
    - Scores are just scaled fractions of hits in [0, 1].
    """
    if text is None:
        text = ""
    text = str(text)
    text_lower = text.lower()

    tokens = re.findall(r"\w+", text_lower)
    n_tokens = max(1, len(tokens))

    bad_count = sum(t in BAD_WORDS for t in tokens)
    insult_count = sum(t in INSULT_WORDS for t in tokens)
    threat_count = sum(t in THREAT_WORDS for t in tokens)
    sexual_count = sum(t in SEXUAL_WORDS for t in tokens)
    slur_count = sum(t in SLUR_WORDS for t in tokens)

    # Fraction of tokens that are any “bad” term
    toxic_frac = (bad_count + insult_count + threat_count +
                  sexual_count + slur_count) / n_tokens

    # Simple scaled scores, clipped to [0, 1]
    toxicity = min(1.0, toxic_frac * 5.0)
    insult_score = min(1.0, insult_count / n_tokens * 5.0)
    threat_score = min(1.0, threat_count / n_tokens * 5.0)
    sexual_score = min(1.0, sexual_count / n_tokens * 5.0)
    identity_attack_score = min(1.0, slur_count / n_tokens * 5.0)

    severe_toxicity = 1.0 if toxicity >= 0.9 else 0.0

    return {
        "toxicity": float(toxicity),
        "severe_toxicity": float(severe_toxicity),
        "insult": float(insult_score),
        "threat": float(threat_score),
        "identity_attack": float(identity_attack_score),
        "sexual_explicit": float(sexual_score),
    }



def model_quality_score(text: str) -> float:
    """
    Placeholder for model-based quality scoring.

    Returns:
      score in [0, 1] (higher = better quality).
    """
    # TODO later: integrate a real classifier at scale
    return 0.5






def detect_lang_with_score(text: str) -> Tuple[str, float]:
    """
    Safe wrapper around langdetect.detect_langs.
    Returns:
      (lang_code, score) like ('en', 0.98)
      or ('unk', 0.0) on failure / empty input.
    """
    if pd.isna(text):
        return "unk", 0.0

    text = str(text).strip()
    if not text:
        return "unk", 0.0

    try:
        # detect_langs returns a list like [en:0.99, de:0.01]
        langs = detect_langs(text)
        if not langs:
            return "unk", 0.0

        best = max(langs, key=lambda l: l.prob)  # Select the language candidate with highest probability
        return best.lang, float(best.prob)
    except Exception:
        return "unk", 0.0


def language_filter_stage(
    df: pd.DataFrame,
    text_col: str = "text",      # use the cleaned text
    allowed_langs = ("en",),     # languages we will *keep*
    min_conf: float = 0.80,      # minimum confidence for English
    drop_non_latin_heavy: bool = True,
    non_latin_threshold: float = 0.50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
     Language ID & quality filtering.

    Adds:
      - lang_pred  : predicted language code
      - lang_score : confidence (0.0–1.0)
      - non_latin_ratio (optional; fraction of alphabetic chars that are non-ASCII)

    Updates / creates:
      - drop_reason for:
          * 'lang_unknown'
          * 'non_english'
          * 'low_lang_confidence'
          * 'non_latin_heavy' (if enabled)

    Returns:
      kept_df    : rows with drop_reason is NA  (kept for downstream)
      dropped_df : rows with drop_reason not NA (filtered out at this stage)
    """

    df_lang = df.copy()

    # drop_reason
    if "drop_reason" not in df_lang.columns:
        df_lang["drop_reason"] = pd.NA

    # --- Run language detection ---
    lang_results = df_lang[text_col].map(detect_lang_with_score)
    df_lang["lang_pred"], df_lang["lang_score"] = zip(*lang_results)

    # --- non-Latin character heuristic ---
    if drop_non_latin_heavy:
        def non_latin_ratio(s: str) -> float:
            if pd.isna(s):
                return 0.0
            s = str(s)
            if not s:
                return 0.0
            total = 0
            non_latin = 0
            for ch in s:
                if ch.isalpha():
                    total += 1
                    # crude: non-ASCII ≈ non-Latin
                    if not ch.encode("utf-8").isascii():
                        non_latin += 1
            if total == 0:
                return 0.0
            return non_latin / total

        df_lang["non_latin_ratio"] = df_lang[text_col].map(non_latin_ratio)
    else:
        df_lang["non_latin_ratio"] = 0.0

    # --- unknown language ---
    mask_unknown = (df_lang["lang_pred"] == "unk") & df_lang["drop_reason"].isna()
    df_lang.loc[mask_unknown, "drop_reason"] = "lang_unknown"

    # --- non-English languages ---
    mask_non_english = (
        ~df_lang["lang_pred"].isin(allowed_langs)
        & (df_lang["lang_pred"] != "unk")
        & df_lang["drop_reason"].isna()
    )
    df_lang.loc[mask_non_english, "drop_reason"] = "non_english"

    # --- low-confidence English ---
    mask_low_conf_en = (
        df_lang["lang_pred"].isin(allowed_langs)
        & (df_lang["lang_score"] < min_conf)
        & df_lang["drop_reason"].isna()
    )
    df_lang.loc[mask_low_conf_en, "drop_reason"] = "low_lang_confidence"

    # --- heavy non-Latin content (optional) ---
    if drop_non_latin_heavy:
        mask_non_latin_heavy = (
            (df_lang["non_latin_ratio"] > non_latin_threshold)
            & df_lang["drop_reason"].isna()
        )
        df_lang.loc[mask_non_latin_heavy, "drop_reason"] = "non_latin_heavy"

    mask_dropped = df_lang["drop_reason"].notna()
    kept_df    = df_lang[~mask_dropped].copy()
    dropped_df = df_lang[mask_dropped].copy()

    # --- Logging: English kept vs non-English dropped ---
    total_rows     = len(df_lang)
    en_kept        = (kept_df["lang_pred"] == "en").sum()
    non_en_dropped = (dropped_df["drop_reason"] == "non_english").sum()

    logging.info("=== Language filter summary (English-only) ===")
    logging.info(f"Input rows          : {total_rows}")
    logging.info(f"Kept rows (EN only) : {en_kept}")
    logging.info(f"Dropped rows (lang) : {len(dropped_df)}")

    # Explicit English vs non-English breakdown
    logging.info(
        "English kept        : %d (%.3f%%)",
        en_kept,
        (en_kept / total_rows * 100.0) if total_rows > 0 else 0.0,
    )
    logging.info(
        "Non-English dropped : %d (%.3f%%)",
        non_en_dropped,
        (non_en_dropped / total_rows * 100.0) if total_rows > 0 else 0.0,
    )

    if len(dropped_df) > 0:
        logging.info("Language drop reasons:")
        logging.info("\n%s", dropped_df["drop_reason"].value_counts())

    return kept_df, dropped_df





def quality_filter_row(row: pd.Series) -> Tuple[bool, str]:
    """
    Row-wise quality decision (b/c).

    Returns
    -------
    keep : bool
        Whether this row is kept.
    reason : str
        Empty if kept, otherwise a short 'drop_reason'.
    """
    text = row.get("text", "")

    # 0) Empty text
    if text is None or str(text).strip() == "":
        return False, "empty_text"

    # # 1) Language filter (English only for this assignment)
    # if row.get("lang", "unknown") != "en":
    #     return False, "non_english"

    # 2) Basic metrics
    n_words = row.get("n_words", 0)
    alpha_ratio = row.get("alpha_ratio", 0.0)
    repetition_ratio = row.get("repetition_ratio", 0.0)

    if n_words < 5:
        return False, "too_short"
    if n_words > 5000:
        return False, "too_long"
    if alpha_ratio < 0.5:
        return False, "low_alpha_ratio"
    if repetition_ratio > 0.8:
        return False, "high_repetition"

    # 3) Metadata-based filters (example: blocked URLs)
    url = str(row.get("url", ""))
    blocked_patterns = [
        r"/ads/",
        r"example-spam-site\.com",
    ]
    for pat in blocked_patterns:
        if re.search(pat, url):
            return False, "blocked_url"

    # 4) PII heuristic: drop extremely PII-heavy docs
    email_hits = row.get("email_hits", 0)
    phone_hits = row.get("phone_hits", 0)
    if email_hits + phone_hits > 20:
        return False, "pii_heavy"


    # 5)ML Toxicity filtering (Detoxify)
    toxicity = row.get("toxicity", 0.0)
    if toxicity >= 0.8:  # adjust threshold as you like
        return False, "high_toxicity"

    # If we reach here, we keep the row
    return True, ""








# C — Combined clean+filter pipeline
def clean_and_filter(
    ingested_parquet: PathLike,
    cleaned_parquet: PathLike,
    cleaned_jsonl: PathLike,
    dropped_parquet: Optional[PathLike] = None,
) -> pd.DataFrame:
    """
    1. Load ingested parquet (output of ingest_mainpipe_v1).
    2. a: robust normalisation + early prefiltering.
    3. b: compute quality metrics (n_chars, n_words, alpha_ratio, repetition).
    4. c: compute lang + PII + toxicity + model_q.
    5. 2d: apply quality filters and record drop reasons.
    6. Save:
        - cleaned parquet (with metrics + lang + pii + tox + model_q + drop_reason)
        - cleaned JSONL with required 'text' column.

    Returns
    -------
    pd.DataFrame
        The final cleaned (kept) DataFrame.
    """
    ingested_parquet = Path(ingested_parquet)
    cleaned_parquet = Path(cleaned_parquet)
    cleaned_jsonl = Path(cleaned_jsonl)
    dropped_parquet_path = Path(dropped_parquet) if dropped_parquet else None

    if not ingested_parquet.exists():
        raise FileNotFoundError(f"Ingested parquet not found: {ingested_parquet}")

    logging.info(f"Loading ingested data from { ingested_parquet } ...")
    df_in = pd.read_parquet(ingested_parquet)
    logging.info(f"Loaded {len(df_in):,} rows.")
    logging.info("✔ .... Loaded ingested parquet successfully")

    
    # Ensure we have a 'text' column
    if "text" not in df_in.columns:
        raise ValueError("Expected a 'text' column in ingested dataset.")

    # a: prefilter + normalise
    df_clean_stage2a, df_dropped_stage2a, mask_problem = prefilter_and_normalise(
        df_in,
        text_col="text",
        min_chars=20,
        min_words=2,
        max_chars=100_000,
    )
    logging.info("✔ Step a: Prefilter & normalisation completed successfully")

    # Use normalised text as canonical 'text' for downstream stages
    df_clean = df_clean_stage2a.copy()
    df_clean["text"] = df_clean["text_norm"]

    # b: quality metrics
    logging.info("Computing quality metrics (b) ...")
    metrics = df_clean["text"].apply(compute_quality_metrics).apply(pd.Series)
    df_clean = pd.concat([df_clean, metrics], axis=1)
    summary_metrics = pd.DataFrame({
        "n_chars_min": [df_clean["n_chars"].min()],
        "n_chars_max": [df_clean["n_chars"].max()],
        "n_chars_mean": [df_clean["n_chars"].mean()],
        "n_chars_median": [df_clean["n_chars"].median()],
        
        "n_words_min": [df_clean["n_words"].min()],
        "n_words_max": [df_clean["n_words"].max()],
        "n_words_mean": [df_clean["n_words"].mean()],
        "n_words_median": [df_clean["n_words"].median()],
        
        "alpha_ratio_min": [df_clean["alpha_ratio"].min()],
        "alpha_ratio_max": [df_clean["alpha_ratio"].max()],
        "alpha_ratio_mean": [df_clean["alpha_ratio"].mean()],
        "alpha_ratio_median": [df_clean["alpha_ratio"].median()],
        
        "repetition_ratio_min": [df_clean["repetition_ratio"].min()],
        "repetition_ratio_max": [df_clean["repetition_ratio"].max()],
        "repetition_ratio_mean": [df_clean["repetition_ratio"].mean()],
        "repetition_ratio_median": [df_clean["repetition_ratio"].median()],
    })
    logging.info("")
    logging.info("=== Summary of Quality Metrics (b) ===")
    logging.info("\n%s", summary_metrics.T)
    logging.info("✔ Step b: Quality metrics computed successfully")



    # c: language filter (keep English only)
    logging.info("Applying language filter (English-only) ...")
    df_lang_kept, df_lang_dropped = language_filter_stage(
        df_clean,
        text_col="text",
        allowed_langs=("en",),
        min_conf=0.80,
        drop_non_latin_heavy=True,
        non_latin_threshold=0.50,
    )

    df_clean = df_lang_kept.copy()
    df_clean["lang"] = df_clean["lang_pred"]  





    pii = df_clean["text"].apply(simple_pii_hits).apply(pd.Series)
    df_clean = pd.concat([df_clean, pii], axis=1)

    # tox_scores = df_clean["text"].apply(toxicity_ml).apply(pd.Series)
    # df_clean = pd.concat([df_clean, tox_scores], axis=1)

    # Heuristic toxicity (fast)
    tox_scores = df_clean["text"].apply(toxicity_heuristic).apply(pd.Series)
    df_clean = pd.concat([df_clean, tox_scores], axis=1)
    


    df_clean["model_q"] = df_clean["text"].apply(model_quality_score)
    logging.info("✔ Step c: Language, PII, toxicity, and model_q computed successfully")

    # -----------------------------
    # d: quality filter
    logging.info("Applying quality filters (2d) ...")
    decisions = df_clean.apply(quality_filter_row, axis=1, result_type="expand")
    decisions.columns = ["keep_final", "drop_reason_stage2b"]
    df_clean = pd.concat([df_clean, decisions], axis=1)

    # Merge Stage b drop reasons 
    df_clean.loc[~df_clean["keep_final"], "drop_reason"] = df_clean.loc[
        ~df_clean["keep_final"], "drop_reason_stage2b"
    ]

    df_kept = df_clean[df_clean["keep_final"]].copy()
    df_dropped_stage2b = df_clean[~df_clean["keep_final"]].copy()
    logging.info("✔ Step d: Quality filters applied successfully")


    
    # Toxicity statistics: dropped vs kept
    total_rows = len(df_in)

    # Rows dropped due to toxicity
    tox_dropped = (df_dropped_stage2b["drop_reason"] == "high_toxicity").sum()
    tox_dropped_pct = (tox_dropped / total_rows * 100) if total_rows > 0 else 0.0

    # Rows that were toxic but NOT dropped (toxicity > 0 but passed filters)
    tox_present = (df_clean["toxicity"] > 0).sum()  # all rows with toxic words
    tox_kept = (df_kept["toxicity"] > 0).sum()      # toxic rows that survived
    tox_kept_pct = (tox_kept / total_rows * 100) if total_rows > 0 else 0.0

    logging.info("=== Toxicity Filtering Report ===")
    logging.info("Total rows: %d", total_rows)
    logging.info("Toxic rows dropped     : %d (%.3f%%)", tox_dropped, tox_dropped_pct)
    logging.info("Toxic rows remaining   : %d (%.3f%%)", tox_kept, tox_kept_pct)
    logging.info("Total toxic rows found : %d (%.3f%%)", tox_present,
                 (tox_present / total_rows * 100 if total_rows > 0 else 0.0))




    

    # Aggregate dropped rows from both stages for inspectability
    df_dropped_all = pd.concat(
        [df_dropped_stage2a, df_dropped_stage2b], axis=0, ignore_index=True
    )

    logging.info("=== Stage 2b/2c/2d: Additional filtering summary ===")
    logging.info(f"Kept after Stage 2d : {len(df_kept):,}")
    logging.info(f"Dropped in Stage 2d : {len(df_dropped_stage2b):,}")

    # Drop-reason counts across both stages
    if len(df_dropped_all) > 0:
        logging.info("Combined drop reason counts (Stage 2a + 2d):")
        logging.info("\n%s", df_dropped_all["drop_reason"].value_counts())
    else:
        logging.info("No rows dropped in combined stages (unexpected).")


    
    # Save outputs
    # -----------------------------
    # Cleaned parquet (drop helper columns if you want)
    if "keep_final" in df_kept.columns:
        df_kept = df_kept.drop(
            columns=["keep_final", "drop_reason_stage2b"], errors="ignore"
        )
    logging.info(f"Writing cleaned parquet to {cleaned_parquet} ...")
    df_kept.to_parquet(cleaned_parquet, index=False)

    # Optional: save all dropped rows
    if dropped_parquet_path is not None:
        logging.info(f"Writing dropped rows parquet to {dropped_parquet_path} ...")
        df_dropped_all.to_parquet(dropped_parquet_path, index=False)

    # Cleaned JSONL with required 'text' column (+ doc_id, source if present)
    logging.info(f"Writing cleaned JSONL to {cleaned_jsonl} ...")
    with cleaned_jsonl.open("w", encoding="utf-8") as f:
        for _, row in df_kept.iterrows():
            rec = {
                "doc_id": row.get("doc_id"),
                "text": row["text"],
                "source": row.get("source"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logging.info("Combined clean+filter step completed.")
    logging.info(f"Final kept rows: {len(df_kept):,} / {len(df_in):,} original.")


    return df_kept


if __name__ == "__main__":
    INGESTED_PATH = "mainpipe_ingested_v1.parquet"
    CLEANED_PARQUET = "mainpipe_cleaned_v2.parquet"
    CLEANED_JSONL = "mainpipe_cleaned_v2.jsonl"
    DROPPED_PARQUET = "mainpipe_dropped_v2.parquet"

    df_kept = clean_and_filter(
        ingested_parquet=INGESTED_PATH,
        cleaned_parquet=CLEANED_PARQUET,
        cleaned_jsonl=CLEANED_JSONL,
        dropped_parquet=DROPPED_PARQUET,
    );

    logging.info("All done.")

clean_and_filter(
    ingested_parquet="mainpipe_ingested_v1.parquet",
    cleaned_parquet="mainpipe_cleaned_v2.parquet",
    cleaned_jsonl="mainpipe_cleaned_v2.jsonl",
    dropped_parquet="mainpipe_dropped_v2.parquet"
)

