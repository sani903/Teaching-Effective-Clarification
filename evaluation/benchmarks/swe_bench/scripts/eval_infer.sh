#!/usr/bin/env bash
set -e

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <input_file> <eval_note> <dataset> <split> [local|remote] [extra_args...]"
  exit 1
fi

INPUT_FILE="$1"
EVAL_NOTE="$2"
DATASET="$3"
SPLIT="$4"
ENVIRONMENT="${5:-local}"  # default to local if not provided
shift 5 || true

if [ "$ENVIRONMENT" != "local" ] && [ "$ENVIRONMENT" != "remote" ]; then
  echo "Error: ENVIRONMENT must be either 'local' or 'remote'"
  exit 1
fi

MODEL_NAME_OR_PATH=$(basename "$(dirname "$INPUT_FILE")")
RESULT_OUTPUT_DIR="evaluation/evaluation_outputs/outputs/${DATASET//\//__}-${SPLIT}/${MODEL_NAME_OR_PATH}"

mkdir -p "$RESULT_OUTPUT_DIR"

echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH"
echo "RESULT_OUTPUT_DIR: $RESULT_OUTPUT_DIR"

if [ "$ENVIRONMENT" = "remote" ]; then
  echo "Running via AllHands remote runtime..."
  RUNTIME=remote \
  ALLHANDS_API_KEY="${ALLHANDS_API_KEY}" \
  SANDBOX_REMOTE_RUNTIME_API_URL="${SANDBOX_REMOTE_RUNTIME_API_URL:-https://runtime.eval.all-hands.dev}" \
  EVAL_DOCKER_IMAGE_PREFIX="${EVAL_DOCKER_IMAGE_PREFIX:-us-central1-docker.pkg.dev/evaluation-092424/swe-bench-images}" \
  poetry run python evaluation/benchmarks/swe_bench/eval_infer.py \
    --input-file "$INPUT_FILE" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --eval-output-dir "$RESULT_OUTPUT_DIR" \
    --eval-note "$EVAL_NOTE" \
    "$@"

else
  echo "Running locally..."
  poetry run python evaluation/benchmarks/swe_bench/eval_infer.py \
    --input-file "$INPUT_FILE" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --num_workers 16 \
    --eval-output-dir "$RESULT_OUTPUT_DIR" \
    --eval-note "$EVAL_NOTE" \
    "$@"
fi



