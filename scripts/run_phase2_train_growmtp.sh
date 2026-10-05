#!/usr/bin/env bash
# ==============================================================================
# 1-Click Runner: Huấn luyện DFlash với Loss Mới GrowMTP (DCA + VGM) trên B200
#
# Tự động:
# 1. Tạo thư mục output riêng: outputs/qwen3_4b_dflash_growmtp
# 2. Symlink features/ và teacher/ từ Phase 1 (tái sử dụng 100% cache)
# 3. Kích hoạt hàm loss: --loss-type growmtp (Dynamic Chain Acceptance + Verify-Gated Masking)
# 4. Hỗ trợ chạy trên 1 GPU (GPU 0) hoặc multi-GPU tùy chọn
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
GROW_OUTPUT="${GROW_OUTPUT:-/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_growmtp}"
DRAFT_INIT_PATH="${DRAFT_INIT_PATH:-}"
GPU_IDS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
EPOCHS="${EPOCHS:-6}"

usage() {
  cat <<EOF
Sử dụng: $(basename "$0") [TÙY CHỌN]

Huấn luyện DFlash Draft Model cho Qwen3-4B với hàm loss GrowMTP (DCA + VGM):
  - Tự động symlink feature cache từ Phase 1
  - Tự động lưu checkpoint tại: $GROW_OUTPUT/checkpoints/

Tùy chọn:
  --gpus IDS               Danh sách GPU huấn luyện (mặc định: "$GPU_IDS")
  --epochs INT             Số epoch (mặc định: $EPOCHS)
  --output-root PATH       Thư mục lưu outputs/checkpoints (mặc định: $GROW_OUTPUT)
  --draft-init-path PATH   Đường dẫn pretrained DFlash nếu muốn finetune (mặc định: train from scratch)
  -h, --help               Hiển thị hướng dẫn này
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPU_IDS="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --output-root)
      GROW_OUTPUT="$2"
      shift 2
      ;;
    --draft-init-path)
      DRAFT_INIT_PATH="$2"
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

echo "================================================================================"
echo "🎯 KHỞI ĐỘNG HUẤN LUYỆN DFLASH VỚI LOSS GROWMTP (DCA + VGM)"
echo "================================================================================"
echo " - Hàm Loss Mục tiêu        : growmtp (Dynamic Chain Acceptance + Verify-Gated Masking)"
echo " - Output Root              : $GROW_OUTPUT"
echo " - Nguồn Cache Features     : $PHASE1_OUTPUT/features"
if [[ -n "$DRAFT_INIT_PATH" ]]; then
  echo " - Khởi tạo từ Trọng số     : $DRAFT_INIT_PATH"
else
  echo " - Chế độ Huấn luyện        : From Scratch (Ngẫu nhiên -> Học tiếng Việt)"
fi
echo " - GPUs tham gia Train      : GPU $GPU_IDS"
echo " - Epochs                   : $EPOCHS"
echo "================================================================================"

# 1. Chuẩn bị thư mục và dọn dẹp symlink nếu có (tránh ValueError từ OfflineFeatureDataset)
mkdir -p "$GROW_OUTPUT"
if [[ -L "$GROW_OUTPUT/features" ]]; then
  rm -f "$GROW_OUTPUT/features"
fi
if [[ -L "$GROW_OUTPUT/teacher" ]]; then
  rm -f "$GROW_OUTPUT/teacher"
fi

if [[ ! -d "$PHASE1_OUTPUT/features" ]]; then
  echo "❌ LỖI: Không tìm thấy $PHASE1_OUTPUT/features" >&2
  exit 1
fi

# 2. Xây dựng lệnh chạy
TRAIN_CMD=(
  bash "$ROOT/scripts/run_phase2_train_b200.sh"
  --loss-type "growmtp"
  --output-root "$GROW_OUTPUT"
  --feature-cache-dir "$PHASE1_OUTPUT"
  --run-id "qwen3-4b-growmtp"
  --gpus "$GPU_IDS"
  --epochs "$EPOCHS"
)

if [[ -n "$DRAFT_INIT_PATH" ]]; then
  TRAIN_CMD+=(--draft-init-path "$DRAFT_INIT_PATH")
fi

exec "${TRAIN_CMD[@]}"
