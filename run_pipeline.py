"""
run_pipeline.py

"""
import logging       # logging framework
logger = logging.getLogger("stage8_sharding")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)




import subprocess
import sys
from pathlib import Path

# Update script names here if yours differ
STAGES = [
    ("Stage 1: ingest:Raw JSONL → Ingested Parquet pipeline",
     ["python", "ingest.py"]),

    ("Stage 2: text_clean_and_filter",
     ["python", "text_clean_and_filter.py"]),

    ("Stage 3: deep_clean_and_pii",
     ["python", "deep_clean_and_pii.py"]),

    ("Stage 4: duplication",
     ["python", "duplication.py"]),

    ("Stage 5: scoring_and_mixture",
     ["python", "scoring_and_mixture.py"]),
    
    ("Stage 6: Tokenisation_JSONL_export",
     ["python", "Tokenisation_JSONL_export.py"]),

     ("Stage 7: sharding",
     ["python", "sharding.py"]),

     ("Stage 8: Export_to_jsonl",
     ["python", " Export_to_jsonl.py"]),

]


def run_stage(name: str, cmd: list) -> None:
    """
    Run a single stage as a subprocess.

    - Logs start/end.
    - If the command fails (non-zero exit code), exits the whole pipeline.
    """
    logging.info("=" * 80)
    logging.info("Starting %s", name)
    logging.info("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd)

    if result.returncode != 0:
        logging.error("❌ %s FAILED with return code %d", name, result.returncode)
        sys.exit(result.returncode)

    logging.info("✅ %s completed successfully", name)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("==== Running full data pipeline ====")

    try:
        project_root = Path(__file__).resolve().parent
    except NameError:
        # __file__ is not defined in Jupyter / interactive environments
        project_root = Path.cwd()

    logging.info("Project root: %s", project_root)

    for name, cmd in STAGES:
        run_stage(name, cmd)

    logging.info("🎉 Pipeline completed successfully. All stages finished.")

