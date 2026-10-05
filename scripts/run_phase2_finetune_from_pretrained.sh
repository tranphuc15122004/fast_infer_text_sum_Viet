#!/usr/bin/env bash
# ==============================================================================
# 1-Click Runner: Finetune DFlash cho Tiếng Việt từ Checkpoint Gốc Pretrained
#
# Tự động:
# 1. Tạo thư mục output riêng: outputs/qwen3_4b_dflash_finetuned
# 2. Symlink features/ và teacher/ từ Phase 1 (không tốn dung lượng ổ đĩa)
# 3. Khởi tạo trọng số từ checkpoint pretrained: Qwen3-4B-DFlash-b16
# 4. Huấn luyện 6 epochs trên 2 GPU B200 (GPU 0, 1)
# ==============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load master config nếu có
MASTER_CONFIG_PATH_FILE="$ROOT/config/master.path"
if [[ -f "$MASTER_CONFIG_PATH_FILE" && -z "${FAST_INFER_MASTER_CONFIG:-}" ]]; then
  FAST_INFER_MASTER_CONFIG="$(cat "$MASTER_CONFIG_PATH_FILE" | tr -d '[:space:]')"
fi

if [[ -n "${FAST_INFER_MASTER_CONFIG:-}" && -f "$FAST_INFER_MASTER_CONFIG" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$FAST_INFER_MASTER_CONFIG"
  set +a
fi

PHASE1_OUTPUT="${PHASE1_OUTPUT:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet}"
FINETUNE_OUTPUT="${FINETUNE_OUTPUT:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_finetuned}"
DFLASH_PRETRAINED="${DFLASH_PRETRAINED:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B-DFlash-b16}"
GPU_IDS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
EPOCHS="${EPOCHS:-6}"

echo "================================================================================"
echo "🎯 KHỞI ĐỘNG FINETUNE DFLASH TỪ CHECKPOINT GỐC PRETRAINED"
echo "================================================================================"
echo " - Checkpoint Gốc Init       : $DFLASH_PRETRAINED"
echo " - Output Root Mới           : $FINETUNE_OUTPUT"
echo " - Nguồn Cache Features      : $PHASE1_OUTPUT/features"
echo " - GPUs tham gia Train       : $GPU_IDS"
echo " - Epochs                    : $EPOCHS"
echo "================================================================================"

# 1. Chuẩn bị thư mục và symlink cache
mkdir -p "$FINETUNE_OUTPUT"

if [[ -d "$PHASE1_OUTPUT/features" ]]; then
  echo ">>> Liên kết (symlink) features cache từ Phase 1..."
  ln -sfn "$PHASE1_OUTPUT/features" "$FINETUNE_OUTPUT/features"
else
  echo "❌ LỖI: Không tìm thấy $PHASE1_OUTPUT/features" >&2
  exit 1
fi

if [[ -d "$PHASE1_OUTPUT/teacher" ]]; then
  echo ">>> Liên kết (symlink) teacher trajectories từ Phase 1..."
  ln -sfn "$PHASE1_OUTPUT/teacher" "$FINETUNE_OUTPUT/teacher"
fi

# 2. Khởi chạy huấn luyện
exec bash "$ROOT/scripts/run_phase2_train_b200.sh" \
  --draft-init-path "$DFLASH_PRETRAINED" \
  --output-root "$FINETUNE_OUTPUT" \
  --run-id "qwen3-4b-finetuned" \
  --gpus "$GPU_IDS" \
  --epochs "$EPOCHS"
