#!/usr/bin/env bash
# ==============================================================================
# Script Đánh giá Hợp nhất (1-Click) Toàn bộ 3 Baseline DFlash trên VietBench
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

DIR_SCRATCH="/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/checkpoints"
DIR_FINETUNED="/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_finetuned/checkpoints"
DIR_GROWMTP="/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_growmtp/checkpoints"

DATASETS="vietnews,wikilingua,vims,vlsp"
MAX_SAMPLES=100
MAX_NEW_TOKENS=512
OUTPUT_DIR="$ROOT/outputs/all_baselines_eval"
SKIP_VANILLA=0
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
  cat <<EOF
Sử dụng: $(basename "$0") [TÙY CHỌN]

Đánh giá gộp toàn bộ 3 Baseline DFlash (Scratch, Finetuned, GrowMTP)
trên 4 bộ dữ liệu VietBench (vietnews, wikilingua, vims, vlsp) trong 1 lần chạy:
  - Khởi chạy Vanilla vLLM một lần duy nhất (hoặc bỏ qua nếu đã đo)
  - Tự động export cả 3 checkpoint sang định dạng vLLM
  - Lần lượt đánh giá cả 3 baseline và sinh bảng so sánh đối đầu toàn diện

Tùy chọn:
  --gpu ID                 GPU ID để chạy vLLM (mặc định: $GPU)
  --output-dir PATH        Thư mục lưu kết quả benchmark (mặc định: $OUTPUT_DIR)
  --skip-vanilla           Bỏ qua chạy lại baseline Vanilla (tái sử dụng kết quả cũ)
  --enforce-eager          Bật eager mode (chạy nhanh không cần compile/capture CUDA graph 7 phút)
  --max-samples INT        Số mẫu mỗi dataset (mặc định: $MAX_SAMPLES)
  --max-new-tokens INT     Độ dài sinh tối đa (mặc định: $MAX_NEW_TOKENS)
  -h, --help               Hiển thị hướng dẫn này
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
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
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
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
mkdir -p "$OUTPUT_DIR"

find_latest_checkpoint() {
  local cp_dir="$1"
  if [[ -d "$cp_dir" ]]; then
    local latest
    latest="$(find "$cp_dir" -maxdepth 1 -name "*-step*" -type d | sort -V | tail -n 1)"
    if [[ -n "$latest" ]]; then
      echo "$latest"
      return 0
    fi
  fi
  echo "$cp_dir"
}

CP_SCRATCH="$(find_latest_checkpoint "$DIR_SCRATCH")"
CP_FINETUNED="$(find_latest_checkpoint "$DIR_FINETUNED")"
CP_GROWMTP="$(find_latest_checkpoint "$DIR_GROWMTP")"

echo "================================================================================"
echo "🎯 ĐÁNH GIÁ HỢP NHẤT TOÀN BỘ 3 BASELINE DFLASH TRÊN VIETBENCH"
echo "================================================================================"
echo " - GPU Sử dụng              : $GPU"
echo " - Output Directory         : $OUTPUT_DIR"
echo " - Baseline 1 (Scratch)     : $CP_SCRATCH"
echo " - Baseline 2 (Finetuned)   : $CP_FINETUNED"
echo " - Baseline 3 (GrowMTP)     : $CP_GROWMTP"
echo " - Chế độ Eager             : $(( ENFORCE_EAGER == 1 ? 1 : 0 ))"
echo " - Bỏ qua Vanilla Baseline  : $(( SKIP_VANILLA == 1 ? 1 : 0 ))"
echo "================================================================================"

# BƯỚC 1: EXPORT CẢ 3 CHECKPOINT
EXPORT_SCRATCH="$OUTPUT_DIR/exported_draft_scratch"
EXPORT_FINETUNED="$OUTPUT_DIR/exported_draft_finetuned"
EXPORT_GROWMTP="$OUTPUT_DIR/exported_draft_growmtp"

export_one() {
  local name="$1"
  local cp="$2"
  local out="$3"
  echo ""
  echo ">>> [Export $name] Checkpoint: $cp -> $out"
  "$PYTHON_BIN" "$ROOT/scripts/export_trained_checkpoint.py" \
    --checkpoint "$cp" \
    --target-model "$TARGET_MODEL" \
    --output-dir "$out" \
    --block-size 16 \
    --num-draft-layers 5
}

export_one "Baseline 1 (Scratch)" "$CP_SCRATCH" "$EXPORT_SCRATCH"
export_one "Baseline 2 (Finetuned)" "$CP_FINETUNED" "$EXPORT_FINETUNED"
export_one "Baseline 3 (GrowMTP)" "$CP_GROWMTP" "$EXPORT_GROWMTP"

# BƯỚC 2: TIẾN HÀNH ĐÁNH GIÁ TRÊN VIETBENCH
echo ""
echo "================================================================================"
echo "🚀 BƯỚC 2: TIẾN HÀNH ĐÁNH GIÁ LIÊN HOÀN TRÊN VIETBENCH BẰNG vLLM"
echo "================================================================================"

DRAFT_MODELS_ARG="dflash_scratch:$EXPORT_SCRATCH,dflash_finetuned:$EXPORT_FINETUNED,dflash_growmtp:$EXPORT_GROWMTP"

EVAL_CMD=(
  "$PYTHON_BIN" "$ROOT/scripts/evaluate_vllm_vietbench.py"
  --model "$TARGET_MODEL"
  --draft-models "$DRAFT_MODELS_ARG"
  --datasets "$DATASETS"
  --max-samples "$MAX_SAMPLES"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --output-dir "$OUTPUT_DIR"
)

if [[ "$SKIP_VANILLA" -eq 1 ]]; then
  EVAL_CMD+=(--skip-vanilla)
fi

if [[ "$ENFORCE_EAGER" -eq 1 ]]; then
  EVAL_CMD+=(--enforce-eager)
fi

"${EVAL_CMD[@]}"

echo ""
echo "================================================================================"
echo "🎉 HOÀN TẤT ĐÁNH GIÁ TẤT CẢ BASELINES!"
echo "📄 Xem bảng so sánh đầy đủ tại: $OUTPUT_DIR/evaluation_summary.md"
echo "================================================================================"
