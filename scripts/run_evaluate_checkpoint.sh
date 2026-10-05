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

PYTHON_BIN="${PYTHON:-${FAST_INFER_PYTHON:-python3}}"
TARGET_MODEL="${TARGET_MODEL_PATH:-${MODEL_QWEN3_4B:-${MODEL_TARGET:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B}}}"
DEFAULT_CHECKPOINT_DIR="/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/checkpoints"

CHECKPOINT=""
DATASETS="vietnews,wikilingua,vims,vlsp"
MAX_SAMPLES=100
MAX_NEW_TOKENS=512
OUTPUT_DIR="$ROOT/outputs/vllm_evaluation_$(date +%Y%m%d_%H%M%S)"
SKIP_VANILLA=0
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
  --target-model PATH      Đường dẫn Target Model (mặc định: $TARGET_MODEL)
  --gpu ID                 GPU ID để chạy vLLM (mặc định: $GPU)
  --datasets LIST          Danh sách dataset phân tách bằng dấu phẩy (mặc định: $DATASETS)
  --max-samples INT        Số mẫu mỗi dataset (mặc định: $MAX_SAMPLES)
  --max-new-tokens INT     Độ dài sinh tối đa (mặc định: $MAX_NEW_TOKENS)
  --output-dir PATH        Thư mục lưu kết quả benchmark (mặc định: $OUTPUT_DIR)
  --skip-vanilla           Bỏ qua chạy lại baseline Vanilla (nếu chỉ muốn đo draft)
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

# Tự động tìm checkpoint mới nhất nếu người dùng không truyền
if [[ -z "$CHECKPOINT" ]]; then
  if [[ -d "$DEFAULT_CHECKPOINT_DIR" ]]; then
    LATEST_CP="$(find "$DEFAULT_CHECKPOINT_DIR" -maxdepth 1 -name "*-step*" -type d | sort -V | tail -n 1)"
    if [[ -n "$LATEST_CP" ]]; then
      CHECKPOINT="$LATEST_CP"
      echo ">>> Tự động phát hiện Checkpoint mới nhất: $CHECKPOINT"
    fi
  fi
fi

if [[ -z "$CHECKPOINT" || ! -e "$CHECKPOINT" ]]; then
  echo "❌ LỖI: Không tìm thấy checkpoint để đánh giá. Vui lòng truyền --checkpoint /path/to/checkpoint" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
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
echo "🚀 BƯỚC 2: TIẾN HÀNH ĐÁNH GIÁ TRÊN VIETBENCH BẰNG vLLM"
echo "================================================================================"

EVAL_CMD=(
  "$PYTHON_BIN" "$ROOT/scripts/evaluate_vllm_vietbench.py"
  --model "$TARGET_MODEL"
  --draft-model "$EXPORTED_DIR"
  --datasets "$DATASETS"
  --max-samples "$MAX_SAMPLES"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --output-dir "$OUTPUT_DIR"
)

if [[ "$SKIP_VANILLA" -eq 1 ]]; then
  EVAL_CMD+=(--skip-vanilla)
fi

"${EVAL_CMD[@]}"

echo ""
echo "================================================================================"
echo "🎉 HOÀN TẤT ĐÁNH GIÁ! Xem kết quả chi tiết tại: $OUTPUT_DIR/evaluation_summary.md"
echo "================================================================================"
