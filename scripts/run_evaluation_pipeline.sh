#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
YOLO_BIN="${YOLO_BIN:-${PROJECT_ROOT}/venv/bin/yolo}"

DATASET="${DATASET:-${PROJECT_ROOT}/datasets/yolo_homeless_4class_clean_balanced}"
RUN_NAME="${RUN_NAME:-homeless4_accuracy_v3}"
RUNS_DIR="${RUNS_DIR:-${PROJECT_ROOT}/runs/detect}"
RUN_DIR="${RUN_DIR:-${RUNS_DIR}/${RUN_NAME}}"
MODEL_PATH="${MODEL_PATH:-${RUN_DIR}/weights/best.pt}"
DEVICE="${DEVICE:-cpu}"
WORKERS="${WORKERS:-0}"
SPLIT="${SPLIT:-test}"

EVAL_PROJECT="${EVAL_PROJECT:-${PROJECT_ROOT}/runs/eval}"
EVAL_NAME="${EVAL_NAME:-${RUN_NAME}_${SPLIT}}"
EVAL_DIR="${EVAL_PROJECT}/${EVAL_NAME}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -x "${YOLO_BIN}" ]]; then
  echo "YOLO executable not found: ${YOLO_BIN}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model weights not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${RUN_DIR}/results.csv" ]]; then
  echo "Training results not found: ${RUN_DIR}/results.csv" >&2
  exit 1
fi

mkdir -p "${EVAL_PROJECT}"

echo "==> Step 1/2: Run YOLO evaluation on ${SPLIT}"
"${YOLO_BIN}" detect val \
  model="${MODEL_PATH}" \
  data="${DATASET}/data.yaml" \
  split="${SPLIT}" \
  device="${DEVICE}" \
  workers="${WORKERS}" \
  project="${EVAL_PROJECT}" \
  name="${EVAL_NAME}" \
  exist_ok=True

echo "==> Step 2/2: Build evaluation report"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/summarize_evaluation.py" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --run-dir "${RUN_DIR}" \
  --eval-dir "${EVAL_DIR}" \
  --out-json "${EVAL_DIR}/evaluation_report.json" \
  --out-md "${EVAL_DIR}/evaluation_report.md"

echo "Evaluation complete."
echo "Artifacts: ${EVAL_DIR}"
