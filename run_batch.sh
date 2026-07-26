#!/bin/bash


PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"


CSV_FILE="$PROJECT_DIR/data/jds.csv"


OUT_FILE="$PROJECT_DIR/outputs/batch_results.json"

# Python environment
PYTHON_BIN="python"  

echo "[RUN] Batch processing CSV..."
$PYTHON_BIN "$PROJECT_DIR/src/run_batch_recommend.py" \
    --csv "$CSV_FILE" \
    --n 10 \
    --top-k 20 \
    --out "$OUT_FILE"

echo "[DONE] Saved results to: $OUT_FILE"
