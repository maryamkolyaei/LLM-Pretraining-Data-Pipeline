# LLM-Pretraining-Data-Pipeline

**Title:** Mainpipe Data Preparation Pipeline<br>
**Author:** Mary Kolyaei<br>
**Email:** maryamkolyaie@gmail.com<br>
**Sources:** <br>
AWS: https://aws.amazon.com/blogs/machine-learning/an-introduction-to-preparing-your-own-dataset-for-llm-training/<br>
Hugging Face: https://huggingface.co/docs/datasets/en/quickstart<br>
https://huggingface.co/docs/datasets/v1.4.0/loading_datasets.html
**Language:** Python 3.10.10<br>

**The pipeline is orchestrated via:**
- 1. Install dependencies: pip install -r requirements.txt, pereferably in a clean environment and 2. python run_pipeline.py
- 2. Run inside Docker


!python run_pipeline.py #executes every stage of the end-to-end data pipeline

🌱 Stage 1 — Ingestion

**Script:** `ingest.py`  
**Input:** `mainpipe_data_v1.jsonl`  
**Output:** `mainpipe_ingested_v1.parquet`

- Loads raw `.jsonl` dataset  
- Creates stable `doc_id` using SHA1 hash  
- Adds ingestion timestamp (`ingest_ts`)  
- Adds metadata fields (`source`)  
- Saves an ingested, structured Parquet file


🧹 Stage 2 — Text Cleaning & Filtering

**Script:** text_clean_and_filter.py
**Input:** mainpipe_ingested_v1.parquet
**Outputs:** mainpipe_cleaned_v2.parquet  &  mainpipe_cleaned_v2.jsonl  &  mainpipe_dropped_v2.parquet

- Removes empty, whitespace & extremely short texts
- Normalises whitespace & punctuation
- Computes basic content stats
- Basic filtering (empty, too short, etc.)


🧽 Stage 3 — Deep Cleaning + PII Masking

**Script:** deep_clean_and_pii.py
**Input:** mainpipe_cleaned_v2.parquet
**Outputs:** mainpipe_cleaned_v4.parquet  &  mainpipe_dropped_v4.parquet  &  mainpipe_cleaned_v4.jsonl

- Removes HTML tags
- Removes boilerplate (cookie banners, footers, disclaimers)
- Normalises repeated characters
- Detects and masks: <EMAIL>, <PHONE>, <CREDIT_CARD>, <IBAN>
- Applies token-based heuristics:
- stopword ratio
- unique-token ratio
- repetitive-token spam
- Drops low-information documents


🔍 Stage 4 — Deduplication (Exact + Near-Dup)

**Script:** duplication.py
**Input:** mainpipe_cleaned_v4.parquet
**Outputs:** mainpipe_cleaned_v5.parquet  &  mainpipe_dropped_v5.parquet  &  mainpipe_cleaned_v5.jsonl

- Builds canonical version of each text
- Exact dedup: SHA256 hash
- Near-dup: match first 500 chars of canonical text
- Drops duplicate documents, drop_reason, etc


⭐ Stage 5 — Scoring & Mixture Assignment

**Script:** scoring_and_mixture.py
**Input:** mainpipe_cleaned_v5.parquet
**Outputs:**  mainpipe_scored_v6.parquet  &  mainpipe_scored_v6.jsonl

- Computes quality_score ∈ [0, 1] using:
- language confidence
- token count
- unique-token ratio
- PII penalty


🧩 Stage 6 — Tokenisation + Training JSONL Export

**Script:** Tokenisation_JSONL_export.py
**Input:** mainpipe_scored_v6.parquet
**Outputs:** mainpipe_tokenised_v7.parquet  &  train_web_sample_tokenised.jsonl

- Loads HuggingFace tokenizer (GPT-2)
- input_ids
- attention_mask
- n_tokens
- Drops extremely short or extremely long docs
- Saves training-ready JSONL for model training


📦 Stage 7 — Sharding

**Script:** sharding.py
**Input:** mainpipe_tokenised_v7.parquet
**Outputs:** Shard files (e.g., train_shard_00001.jsonl, train_shard_00002.jsonl, …)  &  manifest.json  &  tiny_train.jsonl

- Splits tokenised dataset into smaller .jsonl chunks
- Writes a manifest describing:
- number of shards
- number of docs
- total tokens
- tokenizer name
- file paths


📊 Metrics & Inspectability:

🧨 Scaling Plan (Distributed) and Improvements for future : 
- The current approach was selected for a take-home task, the current dataset 
-Scalable approach would be:
1. Hash via pyspark
2. Join back to mark duplicates
3. Faster tokeniser implementations (Rust-based tokenisers)
4. In this project, I performed language detection using the Lingua library. However, Lingua is computationally heavy because it loads multiple statistical language models into memory. To scale this stage, I would consider replacing or optimising Lingua using, for example a lightweight supervised ML model.


For large datasets we use spark, Batch Tokenisation, Shards: (use 50k–100k docs per shard, Write shards in parallel,
- hashed partitions for dedup
- LM-based perplexity filtering
- distributed tokenisation
