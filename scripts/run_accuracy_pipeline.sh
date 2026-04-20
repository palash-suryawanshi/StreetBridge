#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
YOLO_BIN="${YOLO_BIN:-${PROJECT_ROOT}/venv/bin/yolo}"

BASE_DATASET="${PROJECT_ROOT}/datasets/yolo_homeless_4class_clean"
BALANCED_DATASET="${PROJECT_ROOT}/datasets/yolo_homeless_4class_clean_balanced"
RUNS_DIR="${PROJECT_ROOT}/runs/detect"
RUN_NAME="${RUN_NAME:-homeless4_accuracy_v3}"

MODEL="${MODEL:-yolov8s.pt}"
EPOCHS="${EPOCHS:-60}"
IMGSZ="${IMGSZ:-832}"
BATCH="${BATCH:-8}"
DEVICE="${DEVICE:-cpu}"
WORKERS="${WORKERS:-0}"
PATIENCE="${PATIENCE:-20}"
SEED="${SEED:-42}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -x "${YOLO_BIN}" ]]; then
  echo "YOLO executable not found: ${YOLO_BIN}" >&2
  exit 1
fi

echo "==> Step 1/4: Build clean grouped-split dataset"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/prepare_yolo_dataset.py" \
  --src "${PROJECT_ROOT}/annotated_images" \
  --out "${BASE_DATASET}" \
  --seed "${SEED}" \
  --train 0.7 \
  --val 0.2 \
  --test 0.1 \
  --ambiguous-person-policy exclude

echo "==> Step 2/4: Build balanced training dataset"
STREETBRIDGE_BALANCE_SRC="${BASE_DATASET}" \
STREETBRIDGE_BALANCE_OUT="${BALANCED_DATASET}" \
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_balanced_dataset.py"

echo "==> Step 3/4: Train model"
"${YOLO_BIN}" detect train \
  model="${MODEL}" \
  data="${BALANCED_DATASET}/data.yaml" \
  epochs="${EPOCHS}" \
  imgsz="${IMGSZ}" \
  batch="${BATCH}" \
  device="${DEVICE}" \
  optimizer=AdamW \
  lr0=0.001 \
  lrf=0.01 \
  weight_decay=0.0005 \
  close_mosaic=10 \
  patience="${PATIENCE}" \
  workers="${WORKERS}" \
  seed="${SEED}" \
  deterministic=True \
  project="${RUNS_DIR}" \
  name="${RUN_NAME}" \
  exist_ok=True

echo "==> Step 4/4: Evaluate on held-out test split"
"${YOLO_BIN}" detect val \
  model="${RUNS_DIR}/${RUN_NAME}/weights/best.pt" \
  data="${BALANCED_DATASET}/data.yaml" \
  split=test \
  device="${DEVICE}" \
  workers="${WORKERS}"

echo "Pipeline complete."
echo "Training run: ${RUNS_DIR}/${RUN_NAME}"
