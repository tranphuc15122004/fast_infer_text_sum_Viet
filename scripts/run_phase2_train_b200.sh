#!/usr/bin/env bash
# ==============================================================================
# Phase 2 Runner for B200 — Fine-tuning DFlash on Cached Features
#
# Huấn luyện DFlash Draft Model (5 layers) trên tập feature cache đã tạo từ Phase 1.
# - Không cần chạy lại target model trong lúc train (features đã nằm sẵn trên ổ cứng).
# - Hỗ trợ phân tán Multi-GPU qua PyTorch DDP.
# - Tự động dò tìm batch size tối ưu trên B200 qua Adaptive Batch.
# - Tự động lưu checkpoint và hỗ trợ Resume từ checkpoint gần nhất.
# ==============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load master config nếu có
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

# Default Paths & Training Hyperparameters
CONFIG="${CONFIG:-$ROOT/src/Finetuning/configs/qwen3_4b.yaml}"
DEFAULT_SERVER_QWEN3_4B="${MODEL_QWEN3_4B:-${MODEL_TARGET:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B}}"
TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-$DEFAULT_SERVER_QWEN3_4B}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet}"
TRAIN_INPUT="${TRAIN_INPUT:-/workspace/storage-shared/nlp/dungdx4/bien_projects/LLM2Seq/src/eviseq_new/datasets/50k/train_clean.jsonl}"
EVAL_INPUT="${EVAL_INPUT:-$ROOT/datasets/eval_100/vietnews_100.jsonl}"

# GPU Selection
GPU_IDS="${GPUS:-${CUDA_VISIBLE_DEVICES:-}}"

MAX_STEPS="${MAX_STEPS:-}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-}"
DRAFT_INIT_PATH="${DRAFT_INIT_PATH:-}"
LOSS_TYPE="${LOSS_TYPE:-}"
EXTRA_ARGS=()

usage() {
  cat <<EOF
Sử dụng: $(basename "$0") [TÙY CHỌN]

Huấn luyện (Phase 2) DFlash Draft Model cho Qwen3-4B trên server B200 từ Feature Cache (Chuẩn theo paper DFlash 6 epochs).

Tùy chọn:
  --output-root PATH             Thư mục run root đã chạy Phase 1 (chứa features/ và checkpoints/)
  --draft-init-path PATH         Checkpoint DFlash gốc để finetune (safetensors hoặc draft_export)
  --loss-type TYPE               Hàm loss: "dflash" hoặc "growmtp" (DCA + VGM)
  --gpus IDS                     Danh sách GPU huấn luyện, vd: "0,1" hoặc "0,1,2,3" (mặc định: tất cả GPU)
  --num-gpus N                   Số GPU DDP workers
  --epochs INT                   Số epoch huấn luyện (mặc định: 6 theo paper gốc)
  --max-steps INT                Số bước huấn luyện tối đa (nếu muốn giới hạn steps thay vì epochs)
  --batch-size INT               Batch size cố định (nếu không dùng adaptive batch)
  --target-model-path PATH       Đường dẫn model Qwen3-4B (mặc định: $TARGET_MODEL_PATH)
  --config PATH                  File config YAML (mặc định: configs/qwen3_4b.yaml)
  -h, --help                     Hiển thị hướng dẫn này

Ví dụ:
  1. Train chuẩn 6 epochs trên 2 GPU (GPU 0, 1):
     bash $(basename "$0") --gpus 0,1 --output-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet

  2. Train 6 epochs trên 4 GPU:
     bash $(basename "$0") --gpus 0,1,2,3 --epochs 6 --output-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --gpus)
      GPU_IDS="$2"
      shift 2
      ;;
    --num-gpus)
      NPROC_OVERRIDE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --target-model-path)
      TARGET_MODEL_PATH="$2"
      shift 2
      ;;
    --draft-init-path)
      DRAFT_INIT_PATH="$2"
      shift 2
      ;;
    --loss-type)
      LOSS_TYPE="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift 1
      ;;
  esac
done

if [[ -n "$GPU_IDS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
  IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
  NUM_GPUS="${NPROC_OVERRIDE:-${#GPU_ARRAY[@]}}"
else
  if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS="${NPROC_OVERRIDE:-$(nvidia-smi -L | wc -l)}"
  else
    NUM_GPUS="${NPROC_OVERRIDE:-1}"
  fi
fi

echo "================================================================================"
echo "🎯 KHỞI ĐỘNG PHASE 2: HUẤN LUYỆN DFLASH (Qwen3-4B) TRÊN B200"
echo "================================================================================"
echo " - Output Root (chứa Cache) : $OUTPUT_ROOT"
echo " - Model Target             : $TARGET_MODEL_PATH"
if [[ -n "$DRAFT_INIT_PATH" ]]; then
  echo " - Init từ DFlash Model     : $DRAFT_INIT_PATH"
fi
if [[ -n "$LOSS_TYPE" ]]; then
  echo " - Hàm Loss Mục tiêu        : $LOSS_TYPE (DCA + VGM)"
fi
echo " - Config YAML              : $CONFIG"
echo " - GPUs tham gia Train      : ${CUDA_VISIBLE_DEVICES:-'Tất cả GPU'} (Tổng: $NUM_GPUS workers)"
echo " - Checkpoint Lưu tại       : $OUTPUT_ROOT/checkpoints/"
echo "================================================================================"

CMD=(
  "$PYTHON_BIN" "$ROOT/scripts/run_finetuning_b200.py"
  --config "$CONFIG"
  --target-model-path "$TARGET_MODEL_PATH"
  --train-input "$TRAIN_INPUT"
  --eval-input "$EVAL_INPUT"
  --output-root "$OUTPUT_ROOT"
  --nproc-per-node "$NUM_GPUS"
  --stages train
  --epochs "$EPOCHS"
)

if [[ -n "$DRAFT_INIT_PATH" ]]; then
  CMD+=(--draft-init-path "$DRAFT_INIT_PATH")
fi

if [[ -n "$LOSS_TYPE" ]]; then
  CMD+=(--loss-type "$LOSS_TYPE")
fi

if [[ -n "$MAX_STEPS" ]]; then
  CMD+=(--max-steps "$MAX_STEPS")
fi

if [[ -n "$BATCH_SIZE" ]]; then
  CMD+=(--batch-size "$BATCH_SIZE")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo ">>> Thực thi lệnh:"
printf '%q ' "${CMD[@]}"
echo -e "\n"

exec "${CMD[@]}"
