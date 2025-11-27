"""
Deep cleaning and for LLM pre-training data.

- Structural cleanup:
    * HTML stripping
    * Boilerplate removal (cookie banners, footers, etc.)
    * Normalisation of repeated characters

- Content heuristics:
    * Token statistics (token count, unique-token ratio, stopword ratio)
    * Low-information filtering

- PII detection & masking:
    * Email, phone, credit card, IBAN-like patterns
    * PII masking with placeholder tokens
    * Dropping of PII-containing rows

Typical usage (script mode):
    1) Load text_clean_and_filter (e.g. with a `text_norm` column).
    2) Run `deep_clean_and_pii_stage`.
    3) Persist:
         - cleaned parquet
         - dropped parquet
         - JSONL with PII-masked text for training.
"""

import json
import logging
import re
from typing import Tuple, Optional
from collections import Counter
import pandas as pd
# --- GLOBAL LOGGER ---
logger = logging.getLogger("deep_clean_and_pii")
logger.setLevel(logging.INFO)

# If no handler exists, add one (prevents duplicate logging)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# HTML parser
try:
    from bs4 import BeautifulSoup
    HAVE_BS4 = True
except ImportError:
    HAVE_BS4 = False


# Debug counters for strip_html stats
STRIP_HTML_TOTAL = 0
STRIP_HTML_HAD_TAGS = 0
STRIP_HTML_CHANGED = 0


# Debug counters for boilerplate removal
BOILER_TOTAL_LINES = 0
BOILER_REMOVED_LINES = 0
BOILER_DOCS_WITH_REMOVALS = 0

# Stats for repeated char normalization
REPEAT_TOTAL = 0
REPEAT_CHANGED = 0

# Stats for structural cleanup
STRUCT_TOTAL = 0
STRUCT_CHANGED = 0




#Texts with no meaningful natural language content
BOILERPLATE_PATTERNS = [
    r"cookie(s)? policy",
    r"accept( all)? cookies",
    r"privacy policy",
    r"terms of service",
    r"all rights reserved",
    r"sign up for our newsletter",
    r"subscribe to our newsletter",
    r"contact us",
]


# A — Structural cleanup helpers

def strip_html(text: Optional[str]) -> Optional[str]:
    """
    Strip HTML tags using regex and record debug stats.
    """
    # Use global counters
    global STRIP_HTML_TOTAL, STRIP_HTML_HAD_TAGS, STRIP_HTML_CHANGED

    STRIP_HTML_TOTAL += 1

    if pd.isna(text):
        return text

    s = str(text)

    # Detect if HTML tags exist
    if "<" in s and ">" in s:
        STRIP_HTML_HAD_TAGS += 1

    cleaned = re.sub(r"<[^>]+>", " ", s)

    # Check if cleaning made a difference
    if cleaned != s:
        STRIP_HTML_CHANGED += 1

    return cleaned



def remove_boilerplate_lines(text: Optional[str]) -> Optional[str]:
    """
    Remove generic website boilerplate (cookie banners, footers, legal text, etc.)
    """
    global BOILER_TOTAL_LINES, BOILER_REMOVED_LINES, BOILER_DOCS_WITH_REMOVALS

    if pd.isna(text):
        return text

    s = str(text)
    lines = s.splitlines()

    BOILER_TOTAL_LINES += len(lines)

    keep_lines = []
    removed_in_this_doc = 0

    for line in lines:
        lnorm = line.lower()
        if any(re.search(pat, lnorm) for pat in BOILERPLATE_PATTERNS):
            removed_in_this_doc += 1
            continue
        keep_lines.append(line)

    # Track per-document removal
    if removed_in_this_doc > 0:
        BOILER_DOCS_WITH_REMOVALS += 1
        BOILER_REMOVED_LINES += removed_in_this_doc

    return "\n".join(keep_lines).strip()


def normalize_repeated_chars(text: Optional[str]) -> Optional[str]:
    global REPEAT_TOTAL, REPEAT_CHANGED

    if pd.isna(text):
        return text

    s = str(text)
    REPEAT_TOTAL += 1

    before = s
    # Collapse repeated punctuation to max 2
    s = re.sub(r"([!?.,])\1{2,}", r"\1\1", s)
    # Collapse any character repeated >= 4 times down to 3
    s = re.sub(r"(.)\1{3,}", r"\1\1\1", s)

    if s != before:
        REPEAT_CHANGED += 1

    return s



def structural_cleanup(text: Optional[str]) -> Optional[str]:
    """
    Apply all structural cleanup steps in order:

      1) Strip HTML.
      2) Remove boilerplate lines (cookie banners, legal footers, etc.).
      3) Normalise repeated characters.

    Parameters
    ----------
    text : str or None
        Input text (raw or lightly cleaned). May be NaN/None.

    Returns
    -------
    str or None
        Cleaned text. If the input is NaN/None, it is returned unchanged.
    """
    global STRUCT_TOTAL, STRUCT_CHANGED
    if pd.isna(text):
        return text

    STRUCT_TOTAL += 1
    before = text

    s = strip_html(text) #Remove anything that looks like HTML tags
    s = remove_boilerplate_lines(s) #Remove lines containing common “website garbage”
    s = normalize_repeated_chars(s) #Prevent texts with very long character repeats from appearing in training

    if s != before:
        STRUCT_CHANGED += 1
    return s



# B — Simple content heuristics, in a real data set we can optimise all of these steps to make them more efficient and faster
# Small built-in stopword set (for a rough ratio)
EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "at",
    "is", "it", "this", "that", "with", "as", "by", "from", "be", "are",
    "was", "were", "will", "would", "can", "could", "has", "have", "had",
    "about", "into", "over", "after", "before", "between", "up", "down",
}


def compute_token_stats(text: Optional[str]) -> Tuple[int, int, float, float]:
    """

    The function tokenises on whitespace (`str.split()`), then computes:
      - total_tokens
      - unique_tokens
      - unique_token_ratio = unique_tokens / total_tokens
      - stopword_ratio    = (#stopwords in EN_STOPWORDS) / total_tokens

    Empty / whitespace-only / NaN inputs return zeros.

    Parameters
    ----------
    text : str or None
        Input text. May be NaN/None.

    Returns
    -------
    total_tokens : int
    unique_tokens : int
    unique_token_ratio : float
    stopword_ratio : float
    """
    if pd.isna(text):
        return 0, 0, 0.0, 0.0

    s = str(text).strip()
    if not s:
        return 0, 0, 0.0, 0.0

    tokens = s.split()
    total = len(tokens)
    if total == 0:
        return 0, 0, 0.0, 0.0

    unique_tokens = len(set(tokens))
    unique_ratio = unique_tokens / total

    stopwords = sum(1 for t in tokens if t.lower() in EN_STOPWORDS)
    stopword_ratio = stopwords / total

    return total, unique_tokens, unique_ratio, stopword_ratio



# C — PII detection & masking

EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)

PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{2,4}\)?[\s\-\.]?)?\d{3,4}[\s\-\.]?\d{3,4}\b"
)

CREDIT_CARD_RE = re.compile(
    r"\b(?:\d[ -]*?){13,16}\b"
)

IBAN_RE = re.compile(
    r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"
)


def detect_and_mask_pii(text: Optional[str]) -> Tuple[Optional[str], int, int, int, int, bool]:
    """
    Detect basic PII (emails, phones, credit cards, IBAN-like patterns)
    and mask them in the returned text. I dont remove them here becasue 
    sometime they contain useful info tat can be used later

    PII types and placeholders
    --------------------------
    - Email addresses     -> ``<EMAIL>``
    - Phone numbers       -> ``<PHONE>``
    - Credit card numbers -> ``<CREDIT_CARD>``
    - IBAN-like strings   -> ``<IBAN>``

    Parameters
    ----------
    text : str or None
        Input text on which to run regex-based PII detection. May be NaN/None.

    Returns
    -------
    masked_text : str or None
        Text with PII patterns replaced by placeholder tokens. If input is
        NaN/None, it is returned unchanged.
    email_hits : int
        Number of email matches found.
    phone_hits : int
        Number of phone number matches found.
    cc_hits : int
        Number of credit card-like matches found.
    iban_hits : int
        Number of IBAN-like matches found.
    has_pii : bool
        True if any PII pattern was detected, False otherwise.
    """
    if pd.isna(text):
        return text, 0, 0, 0, 0, False

    s = str(text)

    email_hits = len(EMAIL_RE.findall(s))
    phone_hits = len(PHONE_RE.findall(s))
    cc_hits = len(CREDIT_CARD_RE.findall(s))
    iban_hits = len(IBAN_RE.findall(s))

    has_pii = any([email_hits, phone_hits, cc_hits, iban_hits])

    s_masked = EMAIL_RE.sub("<EMAIL>", s)
    s_masked = PHONE_RE.sub("<PHONE>", s_masked)
    s_masked = CREDIT_CARD_RE.sub("<CREDIT_CARD>", s_masked)
    s_masked = IBAN_RE.sub("<IBAN>", s_masked)

    return s_masked, email_hits, phone_hits, cc_hits, iban_hits, has_pii



# D —  main function
def deep_clean_and_pii_stage(
    df: pd.DataFrame,
    text_col: str = "text_norm",      # input text (from Stage 2/3 parquet)
    drop_pii: bool = False,           # if True, drop any row with has_pii
    low_unique_ratio_thresh: float = 0.20,   # below this -> low-info
    high_stopword_ratio_thresh: float = 0.95,  # above this -> low-info
    min_tokens_for_stats: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function takes a DataFrame produced by earlier and:

      1) Structural cleanup:
         - Strip HTML
         - Remove boilerplate lines
         - Normalise repeated characters
         -> new column: ``text_deep_clean``

      2) Content heuristics:
         - Compute token-level stats on ``text_deep_clean``:
              * token_count
              * unique_tokens
              * unique_token_ratio
              * stopword_ratio
         - Apply optional low-information filters:
              * ``low_unique_token_ratio`` (if ``unique_token_ratio`` is
                below ``low_unique_ratio_thresh``)
              * ``high_stopword_ratio`` (if ``stopword_ratio`` is above
                ``high_stopword_ratio_thresh``)

      3) PII detection:
         - Regex-based detection of:
              * emails
              * phone numbers
              * credit cards
              * IBAN-like patterns
         - Add columns:
              * ``text_pii_masked``
              * ``pii_email_hits``
              * ``pii_phone_hits``
              * ``pii_cc_hits``
              * ``pii_iban_hits``
              * ``has_pii``
         - If ``drop_pii=True``, rows with ``has_pii`` and no existing
           ``drop_reason`` are marked with ``drop_reason='pii'``.

    Returns
    -------
    clean_df : pd.DataFrame
        Rows kept after Stage 4 (no ``drop_reason`` assigned in this
        or previous stages).
    dropped_df : pd.DataFrame
        Rows dropped due to Stage 4 filters or previous stages
        (i.e., where ``drop_reason`` is not null).
    """
    df_work = df.copy()

    if "drop_reason" not in df_work.columns:
        df_work["drop_reason"] = pd.NA

    if text_col not in df_work.columns:
        raise ValueError(f"Expected input column '{text_col}' not found in DataFrame.")

    # ---- 1) Structural cleanup ----
    df_work["text_deep_clean"] = df_work[text_col].map(structural_cleanup)

    # ---- 2) Content heuristics (token stats) ----
    stats = df_work["text_deep_clean"].map(compute_token_stats)
    (
        df_work["token_count"],
        df_work["unique_tokens"],
        df_work["unique_token_ratio"],
        df_work["stopword_ratio"],
    ) = zip(*stats)



    #Repetitive token spam filter ----
    

    def is_repetitive_token_spam(text: str,
                                 threshold: float = 0.70,
                                 min_tokens: int = 3) -> bool:
        """
        Detect 'spammy' texts where one token dominates the document.
        Example drops:
          - 'yes yes yes yes yes'
          - 'ok ok ok ok'
          - 'nooooo nooooo nooooo' (after char normalisation)
        Logic:
          max_token_frequency / token_count > `threshold`
        """
        if not isinstance(text, str):
            return False

        tokens = text.split()
        if len(tokens) < min_tokens:
            return False

        counts = Counter(tokens)
        most_common_freq = counts.most_common(1)[0][1]
        frac = most_common_freq / len(tokens)
        return frac > threshold

    spam_mask = df_work["text_deep_clean"].map(is_repetitive_token_spam)
    spam_mask = spam_mask & df_work["drop_reason"].isna()
    df_work.loc[spam_mask, "drop_reason"] = "repetitive_token_spam"


    # Drop for low unique-token ratio
    mask_low_unique = (
        (df_work["token_count"] >= min_tokens_for_stats)
        & (df_work["unique_token_ratio"] < low_unique_ratio_thresh)
        & df_work["drop_reason"].isna()
    )
    df_work.loc[mask_low_unique, "drop_reason"] = "low_unique_token_ratio"

    # Drop for extremely high stopword ratio (very low-content)
    mask_high_stopword = (
        (df_work["token_count"] >= min_tokens_for_stats)
        & (df_work["stopword_ratio"] > high_stopword_ratio_thresh)
        & df_work["drop_reason"].isna()
    )
    df_work.loc[mask_high_stopword, "drop_reason"] = "high_stopword_ratio"

    # 3) PII detection & dropping ----
    pii_results = df_work["text_deep_clean"].map(detect_and_mask_pii)

    

    # Drop for low unique-token ratio (if enough tokens to be meaningful)
    mask_low_unique = (
        (df_work["token_count"] >= min_tokens_for_stats)
        & (df_work["unique_token_ratio"] < low_unique_ratio_thresh)
        & df_work["drop_reason"].isna()
    )
    df_work.loc[mask_low_unique, "drop_reason"] = "low_unique_token_ratio"

    # Drop for extremely high stopword ratio (very low-content)
    mask_high_stopword = (
        (df_work["token_count"] >= min_tokens_for_stats)
        & (df_work["stopword_ratio"] > high_stopword_ratio_thresh)
        & df_work["drop_reason"].isna()
    )
    df_work.loc[mask_high_stopword, "drop_reason"] = "high_stopword_ratio"

    # ---- 3) PII detection & dropping ----
    pii_results = df_work["text_deep_clean"].map(detect_and_mask_pii)
    (
        df_work["text_pii_masked"],
        df_work["pii_email_hits"],
        df_work["pii_phone_hits"],
        df_work["pii_cc_hits"],
        df_work["pii_iban_hits"],
        df_work["has_pii"],
    ) = zip(*pii_results)

    if drop_pii:
        mask_pii_drop = df_work["has_pii"] & df_work["drop_reason"].isna()
        df_work.loc[mask_pii_drop, "drop_reason"] = "pii"

    # ---- Split clean vs dropped ----
    mask_dropped = df_work["drop_reason"].notna()
    clean_df = df_work[~mask_dropped].copy()
    dropped_df = df_work[mask_dropped].copy()

        # ---- Summary ----
    logger.info("=== Stage 4: Deep cleaning & PII summary ===")
    logger.info("Input rows           : %d", len(df))
    logger.info("Kept rows            : %d", len(clean_df))
    logger.info("Dropped rows         : %d", len(dropped_df))

    # Drop reasons
    if len(dropped_df) > 0:
        logger.info("Drop reasons (Stage 4 + previous):")
        reason_counts = dropped_df["drop_reason"].value_counts()
        for reason, count in reason_counts.items():
            logger.info("  %-25s %7d", reason, count)
    else:
        logger.info("Drop reasons (Stage 4 + previous): (no rows dropped)")

    # PII stats
    logger.info("PII stats (all rows):")
    logger.info("  Rows with has_pii = True: %d", int(df_work["has_pii"].sum()))

    # strip_html stats
    logger.info("=== strip_html statistics ===")
    logger.info("  Total texts processed      : %d", STRIP_HTML_TOTAL)
    logger.info("  Texts that contained <...> : %d", STRIP_HTML_HAD_TAGS)
    logger.info("  Texts actually changed     : %d", STRIP_HTML_CHANGED)

    # Boilerplate stats
    logger.info("=== Boilerplate removal statistics ===")
    logger.info("  Documents processed        : %d", len(df))
    logger.info("  Documents with removal     : %d", BOILER_DOCS_WITH_REMOVALS)
    logger.info("  Total lines processed      : %d", BOILER_TOTAL_LINES)
    logger.info("  Total lines removed        : %d", BOILER_REMOVED_LINES)

    if BOILER_TOTAL_LINES > 0:
        pct = (BOILER_REMOVED_LINES / BOILER_TOTAL_LINES) * 100.0
        logger.info("  Percent of lines removed   : %.2f%%", pct)


     # --- Token statistics summary ---
    logger.info("=== Token statistics summary ===")

       # Basic stats
    logger.info(
        "token_count:     min=%d  median=%d  max=%d",
        df_work["token_count"].min(),
        df_work["token_count"].median(),
        df_work["token_count"].max()
    )
    logger.info(
        "unique_ratio:    min=%.3f  median=%.3f  max=%.3f",
        df_work["unique_token_ratio"].min(),
        df_work["unique_token_ratio"].median(),
        df_work["unique_token_ratio"].max()
    )
    
    logger.info(
        "stopword_ratio:  min=%.3f  median=%.3f  max=%.3f",
        df_work["stopword_ratio"].min(),
        df_work["stopword_ratio"].median(),
        df_work["stopword_ratio"].max()
    )
    
    # Low-information counts
    low_unique_count = (
        (df_work["token_count"] >= min_tokens_for_stats)
        & (df_work["unique_token_ratio"] < low_unique_ratio_thresh)
    ).sum()
    
    high_stopword_count = (
        (df_work["token_count"] >= min_tokens_for_stats)
        & (df_work["stopword_ratio"] > high_stopword_ratio_thresh)
    ).sum()
    
    logger.info(
        "Documents with low unique-token ratio (< %.2f): %d",
        low_unique_ratio_thresh,
        low_unique_count
    )
    
    logger.info(
        "Documents with high stopword ratio (> %.2f): %d",
        high_stopword_ratio_thresh,
        high_stopword_count
    )
    
    logger.info(
        "Interpretation: token_count measures text length, "
        "unique_token_ratio measures vocabulary diversity, "
        "stopword_ratio measures proportion of 'function words'."
    )

     #Documents that WOULD be dropped
    mask_low_unique = (
        (df_work["token_count"] >= min_tokens_for_stats)
        & (df_work["unique_token_ratio"] < low_unique_ratio_thresh)
    )
    
    logger.info(
        "Documents with low unique-token ratio (< %.2f): %d",
        low_unique_ratio_thresh,
        mask_low_unique.sum(),
    )
    
    
    if mask_low_unique.sum() > 0:
        logger.info("Example low-unique documents:")
        logger.info(df_work.loc[mask_low_unique, text_col].head().to_string())

    #How many documents were dropped for token spam?
    spam_dropped_count = (dropped_df["drop_reason"] == "repetitive_token_spam").sum()
    logger.info(
        "Documents dropped for repetitive_token_spam: %d",
        spam_dropped_count,
    )


    return clean_df, dropped_df



# Script entrypoint — use previous step's output directly

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # Input: output parquet from previous Stage
    INPUT_PARQUET = "mainpipe_cleaned_v2.parquet"  


    CLEANED_PARQUET_V4 = "mainpipe_cleaned_v4.parquet"
    DROPPED_PARQUET_V4 = "mainpipe_dropped_v4.parquet"
    CLEANED_JSONL_V4 = "mainpipe_cleaned_v4.jsonl"

    logger.info("Loading output from: %s", INPUT_PARQUET)
    df_stage2 = pd.read_parquet(INPUT_PARQUET)
    logger.info("✔ Loaded Stage 2 cleaned parquet successfully")


    df_clean_v4, df_dropped_v4 = deep_clean_and_pii_stage(
        df_stage2,
        text_col="text_norm",   # or "text" depending on your previous step
        drop_pii=False,         # set True to DROP any row with PII
        low_unique_ratio_thresh=0.20,
        high_stopword_ratio_thresh=0.95,
        min_tokens_for_stats=10,
    )
    logger.info("✔ Deep-clean + PII stage finished")
    

    logger.info("Writing cleaned parquet to: %s", CLEANED_PARQUET_V4)
    df_clean_v4.to_parquet(CLEANED_PARQUET_V4, index=False)
    logger.info("✔ Saved cleaned parquet (Stage 4 output)")

    logger.info("Writing dropped parquet to: %s", DROPPED_PARQUET_V4)
    df_dropped_v4.to_parquet(DROPPED_PARQUET_V4, index=False)
    logger.info("✔ Saved dropped-rows parquet for inspection")

    # JSONL: use the PII-masked text as the output text
    logger.info("Writing cleaned JSONL to: %s", CLEANED_JSONL_V4)
    with open(CLEANED_JSONL_V4, "w", encoding="utf-8") as f:
        for _, row in df_clean_v4.iterrows():
            # Prefer masked text, fall back to deep_clean, then to original text
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

    logger.info("All done.")
