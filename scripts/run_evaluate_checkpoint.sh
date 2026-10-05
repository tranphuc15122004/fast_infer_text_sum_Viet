#!/usr/bin/env bash
# ==============================================================================
# Script Đánh giá Checkpoint DFlash sau Huấn luyện bằng vLLM trên VietBench
# ==============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. Load Master Config (nếu có)
MASTER_CONFIG_PATH_FILE="$ROOT/config/master.path"
if [[ -f "$MASTER_CONFIG_PATH_FILE" && -z "${FAST_INFER_MASTER_CONFIG:-}" ]]; then
  FAST_INFER_MASTER_CONFIG="$(cat "$MASTER_CONFIG_PATH_FILE" | tr -d '[:space:]')"
fi

if [[ -n "${FAST_INFER_MASTER_CONFIG:-}" && -f "$FAST_INFER_MASTER_CONFIG" ]]; then
  echo ">>> Loading master configuration from: $FAST_INFER_MASTER_CONFIG"
  set -a
  # shellcheck source=/dev/null
  source "$FAST_INFER_MASTER_CONFIG"
  set +a
fi

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON:-${FAST_INFER_PYTHON:-python3}}"
TARGET_MODEL="${TARGET_MODEL_PATH:-${MODEL_QWEN3_4B:-${MODEL_TARGET:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B}}}"
DEFAULT_CHECKPOINT_DIR="/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/checkpoints"

CHECKPOINT=""
ALL_BASELINES=0
DATASETS="vietnews,wikilingua,vims,vlsp"
MAX_SAMPLES=100
BATCH_SIZE=1
MAX_NEW_TOKENS=512
OUTPUT_DIR=""
SKIP_VANILLA=0
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
  cat <<EOF
Sử dụng: $(basename "$0") [TÙY CHỌN]

Đánh giá mô hình DFlash Draft Model vừa train bằng vLLM trên 4 bộ dữ liệu VietBench:
  - vietnews (100 mẫu)
  - wikilingua (100 mẫu)
  - vims (100 mẫu)
  - vlsp (100 mẫu)

Tùy chọn:
  --checkpoint PATH        Đường dẫn thư mục checkpoint (hoặc file draft_state_dict.pt)
  --all-baselines          Tự động đánh giá cả 3 baselines (scratch, finetuned, growmtp)
  --batch-size INT         Inference batch size (mặc định: 1 cho Latency Benchmark chuẩn)
  --target-model PATH      Đường dẫn Target Model (mặc định: $TARGET_MODEL)
  --gpu ID                 GPU ID để chạy vLLM (mặc định: $GPU)
  --datasets LIST          Danh sách dataset phân tách bằng dấu phẩy (mặc định: $DATASETS)
  --max-samples INT        Số mẫu mỗi dataset (mặc định: $MAX_SAMPLES)
  --max-new-tokens INT     Độ dài sinh tối đa (mặc định: $MAX_NEW_TOKENS)
  --output-dir PATH        Thư mục lưu kết quả benchmark (mặc định: auto theo batch size)
  --skip-vanilla           Bỏ qua chạy lại baseline Vanilla (nếu chỉ muốn đo draft)
  --no-enforce-eager       Tắt eager mode, bật capture CUDA graphs (mất thêm ~7 phút)
  -h, --help               Hiển thị hướng dẫn này
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --all-baselines)
      ALL_BASELINES=1
      shift 1
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --target-model)
      TARGET_MODEL="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --skip-vanilla)
      SKIP_VANILLA=1
      shift 1
      ;;
    --enforce-eager)
      ENFORCE_EAGER=1
      shift 1
      ;;
    --no-enforce-eager)
      ENFORCE_EAGER=0
      shift 1
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Tùy chọn không hợp lệ: $1" >&2
      usage
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU"

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$ROOT/outputs/vllm_evaluation_bs${BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUTPUT_DIR"

# Tự động tái sử dụng vanilla_vllm_records_bs${BATCH_SIZE}.jsonl từ lần chạy trước nếu có đúng batch size
PREV_VANILLA="/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/benchmark_eval_bs${BATCH_SIZE}/vanilla_vllm_records_bs${BATCH_SIZE}.jsonl"
if [[ ! -f "$OUTPUT_DIR/vanilla_vllm_records_bs${BATCH_SIZE}.jsonl" && -f "$PREV_VANILLA" ]]; then
  echo ">>> Tự động liên kết kết quả Vanilla (BS=${BATCH_SIZE}) đã đo trước đó từ: $PREV_VANILLA"
  cp "$PREV_VANILLA" "$OUTPUT_DIR/vanilla_vllm_records_bs${BATCH_SIZE}.jsonl"
fi

if [[ "$ALL_BASELINES" -eq 1 ]]; then
  echo "================================================================================"
  echo "🎯 BƯỚC 1: EXPORT TẤT CẢ 3 BASELINES SANG ĐỊNH DẠNG vLLM"
  echo "================================================================================"
  
  BASE_DIR="/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs"
  
  # 1. Scratch
  CP_SCRATCH="$(find "$BASE_DIR/qwen3_4b_phase1_Viet/checkpoints" -maxdepth 1 -name "*-step*" -type d 2>/dev/null | sort -V | tail -n 1 || true)"
  EXP_SCRATCH="$OUTPUT_DIR/exported_scratch"
  if [[ -n "$CP_SCRATCH" ]]; then
    echo ">>> Exporting Scratch: $CP_SCRATCH -> $EXP_SCRATCH"
    "$PYTHON_BIN" "$ROOT/scripts/export_trained_checkpoint.py" \
      --checkpoint "$CP_SCRATCH" \
      --target-model "$TARGET_MODEL" \
      --output-dir "$EXP_SCRATCH" \
      --block-size 16 --num-draft-layers 5
  fi

  # 2. Finetuned
  CP_FT="$(find "$BASE_DIR/qwen3_4b_dflash_finetuned/checkpoints" -maxdepth 1 -name "*-step*" -type d 2>/dev/null | sort -V | tail -n 1 || true)"
  EXP_FT="$OUTPUT_DIR/exported_finetuned"
  if [[ -n "$CP_FT" ]]; then
    echo ">>> Exporting Finetuned: $CP_FT -> $EXP_FT"
    "$PYTHON_BIN" "$ROOT/scripts/export_trained_checkpoint.py" \
      --checkpoint "$CP_FT" \
      --target-model "$TARGET_MODEL" \
      --output-dir "$EXP_FT" \
      --block-size 16 --num-draft-layers 5
  fi

  # 3. GrowMTP
  CP_MTP="$(find "$BASE_DIR/qwen3_4b_dflash_growmtp/checkpoints" -maxdepth 1 -name "*-step*" -type d 2>/dev/null | sort -V | tail -n 1 || true)"
  EXP_MTP="$OUTPUT_DIR/exported_growmtp"
  if [[ -n "$CP_MTP" ]]; then
    echo ">>> Exporting GrowMTP: $CP_MTP -> $EXP_MTP"
    "$PYTHON_BIN" "$ROOT/scripts/export_trained_checkpoint.py" \
      --checkpoint "$CP_MTP" \
      --target-model "$TARGET_MODEL" \
      --output-dir "$EXP_MTP" \
      --block-size 16 --num-draft-layers 5
  fi

  DRAFT_MODELS_ARG="scratch:$EXP_SCRATCH,finetuned:$EXP_FT,growmtp:$EXP_MTP"

  echo ""
  echo "================================================================================"
  echo "🚀 BƯỚC 2: TIẾN HÀNH ĐÁNH GIÁ ĐỒNG LOẠT 3 BASELINES TRÊN VIETBENCH (BS=${BATCH_SIZE})"
  echo "================================================================================"

  EVAL_CMD=(
    "$PYTHON_BIN" "$ROOT/scripts/evaluate_vllm_vietbench.py"
    --model "$TARGET_MODEL"
    --draft-models "$DRAFT_MODELS_ARG"
    --datasets "$DATASETS"
    --max-samples "$MAX_SAMPLES"
    --batch-size "$BATCH_SIZE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --output-dir "$OUTPUT_DIR"
  )

  if [[ "$SKIP_VANILLA" -eq 1 ]]; then
    EVAL_CMD+=(--skip-vanilla)
  fi

  if [[ "$ENFORCE_EAGER" -eq 0 ]]; then
    EVAL_CMD+=(--no-enforce-eager)
  fi

  "${EVAL_CMD[@]}"

else
  # Đánh giá 1 Checkpoint cụ thể
  if [[ -z "$CHECKPOINT" ]]; then
    CHECKPOINT="$DEFAULT_CHECKPOINT_DIR"
  fi

  if [[ -d "$CHECKPOINT" ]]; then
    LATEST_CP="$(find "$CHECKPOINT" -maxdepth 1 -name "*-step*" -type d 2>/dev/null | sort -V | tail -n 1 || true)"
    if [[ -n "$LATEST_CP" ]]; then
      echo ">>> Tự động phát hiện Checkpoint mới nhất: $LATEST_CP"
      CHECKPOINT="$LATEST_CP"
    fi
  fi

  if [[ -z "$CHECKPOINT" || ! -e "$CHECKPOINT" ]]; then
    echo "❌ LỖI: Không tìm thấy checkpoint để đánh giá. Vui lòng truyền --checkpoint /path/to/checkpoint hoặc dùng --all-baselines" >&2
    exit 1
  fi

  EXPORTED_DIR="$OUTPUT_DIR/exported_draft_model"

  echo "================================================================================"
  echo "🎯 BƯỚC 1: EXPORT CHECKPOINT SANG ĐỊNH DẠNG vLLM / HUGGINGFACE"
  echo "================================================================================"
  "$PYTHON_BIN" "$ROOT/scripts/export_trained_checkpoint.py" \
    --checkpoint "$CHECKPOINT" \
    --target-model "$TARGET_MODEL" \
    --output-dir "$EXPORTED_DIR" \
    --block-size 16 \
    --num-draft-layers 5

  echo ""
  echo "================================================================================"
  echo "🚀 BƯỚC 2: TIẾN HÀNH ĐÁNH GIÁ TRÊN VIETBENCH BẰNG vLLM (BS=${BATCH_SIZE})"
  echo "================================================================================"

  EVAL_CMD=(
    "$PYTHON_BIN" "$ROOT/scripts/evaluate_vllm_vietbench.py"
    --model "$TARGET_MODEL"
    --draft-model "$EXPORTED_DIR"
    --datasets "$DATASETS"
    --max-samples "$MAX_SAMPLES"
    --batch-size "$BATCH_SIZE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --output-dir "$OUTPUT_DIR"
  )

  if [[ "$SKIP_VANILLA" -eq 1 ]]; then
    EVAL_CMD+=(--skip-vanilla)
  fi

  if [[ "$ENFORCE_EAGER" -eq 0 ]]; then
    EVAL_CMD+=(--no-enforce-eager)
  fi

  "${EVAL_CMD[@]}"
fi

echo ""
echo "================================================================================"
echo "🎉 HOÀN TẤT ĐÁNH GIÁ (BS=${BATCH_SIZE})! Xem kết quả chi tiết tại:"
echo "   - Markdown Report: $OUTPUT_DIR/evaluation_summary.md"
echo "   - JSON Report:     $OUTPUT_DIR/evaluation_summary.json"
echo "   - CSV Report:      $OUTPUT_DIR/evaluation_summary.csv"
echo "================================================================================"
