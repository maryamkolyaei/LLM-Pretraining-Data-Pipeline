# Data-Pipeline task

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
- 2. A Dockerfile is also provided for containerised execution.
- 3. An **example** end-to-end run is demonstrated in the notebook:Maincode project.ipynb



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


🧾 Stage 8 — Final Clean JSONL Export (Non-Tokenised text)

Script: export_text_jsonl.py
Input: mainpipe_scored_v6.parquet
Output: mainpipe_scored_v6_text.jsonl




📊 **Metrics & Inspectability:** The pipeline logs key performance metrics and intermediate outputs, enabling quick inspection, reproducibility, and debugging, 
- plots and tables are provided.
- **How to run:** after running the full pipeline: python run_pipeline.py,  generate all plots and metrics summaries by using plots_charts.py

🧨 **Scaling Plan (Distributed) and Improvements for future :** 
- The current implementation is lightweight for a take-home exercise, it processes the provided dataset on a single machine in a research compuiting portal equipped with 32 virtual CPUs and 128GB of RAM. I used Python/Pandas for clarity and reproducibility.
  
-An scalable approach would be:
1. At real scale, however, the pipeline would run on a distributed compute cluster, and the entire workflow changes to accommodate multi-billion-document datasets. For large datasets, all heavy lifting would be done using PySpark, where Spark reads thousands of files in parallel from S3/GCS rather than loading data locally. Data cleaning, normalisation, and filtering run as distributed map operations, and memory pressure is handled automatically by Spark executors rather than the local machine. Deduplication also changes: instead of grouping in Pandas, we compute a strong hash (e.g., SHA256) for each cleaned document using a Spark UDF, repartition the dataset by a hash prefix so that all potential duplicates land on the same worker, and then drop duplicates inside each hash bucket. This pattern keeps deduplication scalable and avoids expensive shuffles

2. Near-duplicate detection would rely on Spark-compatible approaches such as MinHash/LSH or SimHash to cluster similar documents across the corpus. Tokenisation also needs to scale dramatically. Rather than relying on Python tokenisers, which are slow and single-threaded. This can be done directly inside Spark.

3. Language detection would also be redesigned. In this project I used Lingua because it is accurate, but Lingua loads multiple statistical language models into memory and becomes too heavy when scaled up. In a production setting, I would consider replacing or optimising Lingua with a lightweight supervised ML model that runs efficiently in batches. Additional large-scale improvements include hashed partitions that make incremental deduplication cheap


