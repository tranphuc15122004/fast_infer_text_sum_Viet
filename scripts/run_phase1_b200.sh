#!/usr/bin/env bash
# ==============================================================================
# Phase 1 Runner for B200 (180GB VRAM) — DFlash on Qwen3-4B
#
# Chạy toàn bộ quy trình Phase 1:
#   1. Generate Teacher Targets (train + eval) qua SGLang / vLLM server pool
#   2. Validate Teacher Targets (ROUGE-1 + Anomaly filter)
#   3. Capture Hidden Features (train + eval) qua SGLang / FlashInfer
#
# Tối ưu hóa cho B200 180GB VRAM và hỗ trợ Resume toàn diện ở mọi stage.
# ==============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ------------------------------------------------------------------------------
# 1. Load Master Config (nếu có) & Python Runtime
# ------------------------------------------------------------------------------
MASTER_CONFIG_PATH_FILE="$ROOT/config/master.path"
if [[ -f "$MASTER_CONFIG_PATH_FILE" && -z "${FAST_INFER_MASTER_CONFIG:-}" ]]; then
  FAST_INFER_MASTER_CONFIG="$(cat "$MASTER_CONFIG_PATH_FILE" | tr -d '[:space:]')"
fi

if [[ -n "${FAST_INFER_MASTER_CONFIG:-}" && -f "$FAST_INFER_MASTER_CONFIG" ]]; then
  echo ">>> Loading master configuration from: $FAST_INFER_MASTER_CONFIG"
  # Export variables from .env ignoring comments
  set -a
  # shellcheck source=/dev/null
  source "$FAST_INFER_MASTER_CONFIG"
  set +a
fi

PYTHON_BIN="${PYTHON:-${FAST_INFER_PYTHON:-python3}}"

# ------------------------------------------------------------------------------
# 2. Default Parameters & B200 VRAM Optimization Defaults
CONFIG="${CONFIG:-$ROOT/src/Finetuning/configs/qwen3_4b.yaml}"

# Resolve TARGET_MODEL_PATH across master config variables and server paths
resolve_target_model() {
  local candidates=(
    "${TARGET_MODEL_PATH:-}"
    "${MODEL_QWEN3_4B:-}"
    "${MODEL_TARGET:-}"
    "${B200_TARGET_MODEL:-}"
    "${MODEL_ROOT:-/workspace/storage-shared/nlp/dungdx4/BERT}/Qwen3-4B"
    "/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B"
    "/workspace/storage-shared/nlp/dungdx4/models/Qwen3-4B"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -d "$candidate" && -f "$candidate/config.json" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

TARGET_MODEL_PATH="$(resolve_target_model || true)"

DEFAULT_SERVER_TRAIN_50K="/workspace/storage-shared/nlp/dungdx4/bien_projects/LLM2Seq/src/eviseq_new/datasets/50k/train_clean.jsonl"

if [[ -z "${TRAIN_INPUT:-}" ]]; then
  if [[ -f "$DEFAULT_SERVER_TRAIN_50K" ]]; then
    TRAIN_INPUT="$DEFAULT_SERVER_TRAIN_50K"
  else
    TRAIN_INPUT="$ROOT/datasets/eval_100/vietnews_100.jsonl"
  fi
fi

EVAL_INPUT="${EVAL_INPUT:-$ROOT/datasets/eval_100/vietnews_100.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/qwen3_4b_phase1}"
RUN_ID="${RUN_ID:-qwen3-4b-phase1}"

# GPU Selection (mặc định lấy tất cả GPU hoặc theo biến GPUS / CUDA_VISIBLE_DEVICES)
GPU_IDS="${GPUS:-${CUDA_VISIBLE_DEVICES:-}}"

# Generation & Caching Backends
GEN_BACKEND="${GEN_BACKEND:-sglang}"       # sglang | vllm | hf
CACHE_BACKEND="${CACHE_BACKEND:-sglang}"   # sglang | hf
CAPTURE_METHOD="${CAPTURE_METHOD:-dflash}" # dflash | eagle3 | dspark

# B200 180GB VRAM Tuning
# - mem_fraction 0.90: Tận dụng ~162GB VRAM cho KV Cache và hidden feature forward
# - concurrency: 16-32 request đồng thời trên mỗi GPU
# - max_tokens_per_batch: 131072 (128k tokens per batch forward trên B200)
# - adaptive_max_batch_size: 128
GEN_MEM_FRACTION="${GEN_MEM_FRACTION:-0.90}"
GEN_CONCURRENCY="${GEN_CONCURRENCY:-16}"
CACHE_MEM_FRACTION="${CACHE_MEM_FRACTION:-0.90}"
MAX_TOKENS_PER_BATCH="${MAX_TOKENS_PER_BATCH:-131072}"
ADAPTIVE_MAX_BATCH_SIZE="${ADAPTIVE_MAX_BATCH_SIZE:-128}"
ADAPTIVE_MIN_BATCH_SIZE="${ADAPTIVE_MIN_BATCH_SIZE:-1}"
TARGET_MEMORY_FRACTION="${TARGET_MEMORY_FRACTION:-0.90}"

# Trajectory Validation Gating
MAX_ANOMALY_RATE="${MAX_ANOMALY_RATE:-0.20}"
MIN_TEACHER_ROUGE1="${MIN_TEACHER_ROUGE1:-0.0}"
FILTER_ANOMALIES="${FILTER_ANOMALIES:-1}"
VALIDATE_WARN_ONLY="${VALIDATE_WARN_ONLY:-0}"

DRY_RUN=0
EXTRA_ARGS=()

# ------------------------------------------------------------------------------
# 3. CLI Argument Parsing & Help
# ------------------------------------------------------------------------------
usage() {
  cat <<EOF
Sử dụng: $(basename "$0") [TÙY CHỌN]

Chạy Phase 1 (Regenerate + Validation + Caching) cho DFlash Qwen3-4B trên server B200.

Tùy chọn GPU & Server:
  --gpus IDS                     Danh sách GPU ID cần dùng, vd: "0,1,2,3" hoặc "0" (mặc định: tất cả GPU)
  --num-gpus N                   Số lượng GPU tiến trình (mặc định: tự động đếm từ --gpus hoặc nvidia-smi)
  --target-model-path PATH       Đường dẫn local snapshot model Qwen3-4B
  --config PATH                  Đường dẫn file config YAML (mặc định: configs/qwen3_4b.yaml)
  --train-input PATH             Đường dẫn file input train JSONL
  --eval-input PATH              Đường dẫn file input eval JSONL
  --output-root PATH             Thư mục run root lưu checkpoints, features, logs (mặc định: outputs/qwen3_4b_phase1)
  --run-id NAME                  Tên định danh run (mặc định: qwen3-4b-phase1)

Tùy chọn Backend & Tối ưu hóa B200 (180GB VRAM):
  --gen-backend BACKEND          Backend sinh target: sglang | vllm | hf (mặc định: sglang)
  --cache-backend BACKEND        Backend trích xuất hidden states: sglang | hf (mặc định: sglang)
  --gen-mem-fraction FLOAT       Tỷ lệ VRAM cấp cho server sinh (mặc định: 0.90 ~ 162GB/GPU)
  --gen-concurrency INT          Số request đồng thời mỗi GPU server (mặc định: 16)
  --cache-mem-fraction FLOAT     Tỷ lệ VRAM cấp cho SGLang capture (mặc định: 0.90)
  --max-tokens-per-batch INT     Trần số tokens xử lý trong 1 batch cache (mặc định: 131072)
  --max-batch-size INT           Batch size tối đa khi trích xuất features (mặc định: 128)

Tùy chọn Kiểm soát & Chịu lỗi:
  --max-anomaly-rate FLOAT       Tỷ lệ lỗi dị thường tối đa cho phép (mặc định: 0.20)
  --validate-warn-only           Chỉ cảnh báo khi vượt ngưỡng lỗi, không dừng tiến trình
  --no-filter-anomalies          Không tự động lọc bỏ mẫu lỗi khỏi tập dữ liệu
  --dry-run                      Chỉ in lệnh và kiểm tra preflight, không chạy thực tế
  -h, --help                     Hiển thị hướng dẫn này

Ví dụ thực thi:
  1. Chạy trên 4 GPU (GPU 0, 1, 2, 3) với SGLang:
     bash $(basename "$0") --gpus 0,1,2,3 --target-model-path /data/models/Qwen3-4B

  2. Chạy trên 1 GPU riêng biệt (GPU 4):
     bash $(basename "$0") --gpus 4 --target-model-path /data/models/Qwen3-4B

  3. Dry-run kiểm tra cấu hình:
     bash $(basename "$0") --gpus 0,1 --target-model-path /data/models/Qwen3-4B --dry-run
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPU_IDS="$2"
      shift 2
      ;;
    --num-gpus)
      NPROC_OVERRIDE="$2"
      shift 2
      ;;
    --target-model-path)
      TARGET_MODEL_PATH="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --train-input)
      TRAIN_INPUT="$2"
      shift 2
      ;;
    --eval-input)
      EVAL_INPUT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --gen-backend)
      GEN_BACKEND="$2"
      shift 2
      ;;
    --cache-backend)
      CACHE_BACKEND="$2"
      shift 2
      ;;
    --gen-mem-fraction)
      GEN_MEM_FRACTION="$2"
      shift 2
      ;;
    --gen-concurrency)
      GEN_CONCURRENCY="$2"
      shift 2
      ;;
    --cache-mem-fraction)
      CACHE_MEM_FRACTION="$2"
      shift 2
      ;;
    --max-tokens-per-batch)
      MAX_TOKENS_PER_BATCH="$2"
      shift 2
      ;;
    --max-batch-size)
      ADAPTIVE_MAX_BATCH_SIZE="$2"
      shift 2
      ;;
    --max-anomaly-rate)
      MAX_ANOMALY_RATE="$2"
      shift 2
      ;;
    --validate-warn-only)
      VALIDATE_WARN_ONLY=1
      shift 1
      ;;
    --no-filter-anomalies)
      FILTER_ANOMALIES=0
      shift 1
      ;;
    --dry-run)
      DRY_RUN=1
      shift 1
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

# ------------------------------------------------------------------------------
# 4. Resolve Target Model & GPU Topologies
# ------------------------------------------------------------------------------
if [[ -z "$TARGET_MODEL_PATH" ]]; then
  echo "ERROR: Chưa chỉ định model target. Truyền qua --target-model-path hoặc biến MODEL_QWEN3_4B." >&2
  exit 1
fi

if [[ -n "$GPU_IDS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
  # Đếm số lượng GPU từ chuỗi phân tách bởi dấu phẩy
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
echo "🚀 KHỞI ĐỘNG PHASE 1: REGENERATE & CACHING (DFlash Qwen3-4B) TRÊN B200"
echo "================================================================================"
echo " - Model Target          : $TARGET_MODEL_PATH"
echo " - Config YAML           : $CONFIG"
echo " - GPUs sử dụng          : ${CUDA_VISIBLE_DEVICES:-'Tất cả GPU'} (Tổng: $NUM_GPUS workers)"
echo " - Train Input           : $TRAIN_INPUT"
echo " - Eval Input            : $EVAL_INPUT"
echo " - Output Root           : $OUTPUT_ROOT"
echo " - Generation Backend    : $GEN_BACKEND (mem_fraction: $GEN_MEM_FRACTION, concurrency: $GEN_CONCURRENCY/server)"
echo " - Cache Backend         : $CACHE_BACKEND/$CAPTURE_METHOD (mem_fraction: $CACHE_MEM_FRACTION, max_tokens: $MAX_TOKENS_PER_BATCH)"
echo " - Trajectory Validation : max_anomaly_rate=$MAX_ANOMALY_RATE, filter_anomalies=$FILTER_ANOMALIES, warn_only=$VALIDATE_WARN_ONLY"
echo " - Resume & Chịu lỗi     : Kích hoạt tự động (theo dõi tại $OUTPUT_ROOT/.state/)"
echo "================================================================================"

# ------------------------------------------------------------------------------
# 5. Build Final Launcher Command
# ------------------------------------------------------------------------------
CMD=(
  "$PYTHON_BIN" "$ROOT/scripts/run_finetuning_b200.py"
  --config "$CONFIG"
  --target-model-path "$TARGET_MODEL_PATH"
  --train-input "$TRAIN_INPUT"
  --eval-input "$EVAL_INPUT"
  --output-root "$OUTPUT_ROOT"
  --run-id "$RUN_ID"
  --nproc-per-node "$NUM_GPUS"
  --phase1-only
  --generation-backend "$GEN_BACKEND"
  --capture-backend "$CACHE_BACKEND"
  --capture-method "$CAPTURE_METHOD"
  --target-memory-fraction "$TARGET_MEMORY_FRACTION"
  --adaptive-min-batch-size "$ADAPTIVE_MIN_BATCH_SIZE"
  --adaptive-max-batch-size "$ADAPTIVE_MAX_BATCH_SIZE"
  --max-tokens-per-batch "$MAX_TOKENS_PER_BATCH"
  --max-anomaly-rate "$MAX_ANOMALY_RATE"
  --min-teacher-rouge1 "$MIN_TEACHER_ROUGE1"
)

# Nếu dùng backend server độc lập (SGLang / vLLM), tự động quản lý vòng đời target server
if [[ "$GEN_BACKEND" != "hf" ]]; then
  CMD+=(
    --generation-launch-servers
    --generation-server-mem-fraction "$GEN_MEM_FRACTION"
    --generation-concurrency-per-server "$GEN_CONCURRENCY"
    --generation-server-tp-size 1
  )
fi

# Thiết lập cho SGLang cache
if [[ "$CACHE_BACKEND" == "sglang" ]]; then
  CMD+=(
    --sglang-mem-fraction-static "$CACHE_MEM_FRACTION"
    --sglang-attention-backend "flashinfer"
  )
fi

if [[ "$FILTER_ANOMALIES" -eq 1 ]]; then
  CMD+=(--filter-anomalies)
fi

if [[ "$VALIDATE_WARN_ONLY" -eq 1 ]]; then
  CMD+=(--validate-warn-only)
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  CMD+=(--dry-run)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo ">>> Thực thi lệnh:"
printf '%q ' "${CMD[@]}"
echo -e "\n"

exec "${CMD[@]}"
