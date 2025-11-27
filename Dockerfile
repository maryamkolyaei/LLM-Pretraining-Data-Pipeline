# ============================================
# Base image
# ============================================
FROM python:3.10-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ============================================
# System dependencies
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Working directory
# ============================================
WORKDIR /app

# ============================================
# Python dependencies
# ============================================
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Copy all pipeline code
# ============================================
COPY . /app

# ============================================
# Default command — run the pipeline
# ============================================
CMD ["python", "run_pipeline.py"]
