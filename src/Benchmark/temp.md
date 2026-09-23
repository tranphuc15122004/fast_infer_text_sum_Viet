Trên server B200, làm theo quy trình dưới đây. Không dùng `/home/tuantb/fast_infer_text_sum/.venv`; venv đó chỉ để mô phỏng local. Server dùng `python3` hệ thống Python 3.12.

## 1. Đặt biến môi trường

Điều chỉnh `REPO` nếu repo nằm ở vị trí khác:

```bash
export REPO=/workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum_Viet
export SHARED=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data
export MASTER=$SHARED/fast_infer_master_Viet.env

cd "$REPO"

export FAST_INFER_MASTER_CONFIG="$MASTER"
export FI_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub

export FAST_INFER_CACHE_ROOT=/tmp/fast_infer_cache
export TRITON_CACHE_DIR=$FAST_INFER_CACHE_ROOT/triton
export FLASHINFER_WORKSPACE_BASE=$FAST_INFER_CACHE_ROOT/flashinfer
export TORCH_EXTENSIONS_DIR=$FAST_INFER_CACHE_ROOT/torch_extensions

mkdir -p "$TRITON_CACHE_DIR" \
         "$FLASHINFER_WORKSPACE_BASE" \
         "$TORCH_EXTENSIONS_DIR"
```

Kiểm tra GPU và Python:

```bash
python3 --version
nvidia-smi

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
PY
```

Kỳ vọng:

```text
Python 3.12.x
CUDA 13.0
NVIDIA B200
cuda_available: True
```

## 2. Kiểm tra source và checkpoint

Server phải có các source đã sửa:

```bash
test -d externals/EAGLE
test -d externals/dflash
test -d externals/Domino
test -d externals/SpecForge
test -f datasets/eval_100/vietnews_100.jsonl
```

Trong master config, đặt đúng đường dẫn checkpoint local:

```bash
MODEL_ROOT=/workspace/storage-shared/models

MODEL_TARGET=$MODEL_ROOT/Qwen3-4B
MODEL_EAGLE_DRAFT=$MODEL_ROOT/Qwen3-4B_eagle3
MODEL_DFLASH_DRAFT=$MODEL_ROOT/Qwen3-4B-DFlash-b16
MODEL_DOMINO_DRAFT=$MODEL_ROOT/Qwen3-4B-Domino-b16
MODEL_DSPARK_DRAFT=$MODEL_ROOT/Qwen3-4B-DSpark
```

Kiểm tra từng model:

```bash
for model in \
  "$MODEL_TARGET" \
  "$MODEL_EAGLE_DRAFT" \
  "$MODEL_DFLASH_DRAFT" \
  "$MODEL_DOMINO_DRAFT" \
  "$MODEL_DSPARK_DRAFT"
do
  echo "=== $model ==="
  test -f "$model/config.json" || echo "THIEU config.json"
  find "$model" -maxdepth 1 -type f \
    \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \) \
    | head
done
```

Các checkpoint tương ứng:

```text
Qwen/Qwen3-4B
AngelSlim/Qwen3-4B_eagle3
z-lab/Qwen3-4B-DFlash-b16
Huang2020/Qwen3-4B-Domino-b16
deepseek-ai/dspark_qwen3_4b_block7
```

Server không có Internet. Nếu thiếu checkpoint, phải copy snapshot từ máy có Internet hoặc mirror nội bộ trước; không chạy benchmark khi thiếu model.

## 3. Cài dependency

Trước hết chỉ chạy preflight:

```bash
python3 scripts/setup_server_env.py \
  --check \
  --master-config "$MASTER" \
  --data-dir "$REPO/datasets/eval_100" \
  --profile server

python3 scripts/check_shared_env.py --profile server
```

Stack đã được debug thực tế trên B200/Modal là:

```text
torch: 2.13.0+cu130
transformers: 5.12.1
sglang: 0.5.19
flashinfer: 0.6.18
sglang-kernel: 0.4.6.post1
```

Nếu server thiếu SGLang/FlashInfer, chỉ cài từ wheelhouse nội bộ:

```bash
export B200_WHEELHOUSE=/workspace/storage-shared/nlp/dungdx4/phuc_projects/offline_wheelhouse

python3 -m pip install \
  --no-index \
  --find-links "$B200_WHEELHOUSE" \
  'sglang==0.5.19' \
  'sglang-kernel==0.4.6.post1' \
  'flashinfer-python[cu13]==0.6.18'
```

Không trộn wheel CUDA khác với Torch hiện tại. `libnuma1` và `libnuma-dev` cần được administrator cài bằng system package manager nếu server chưa có.

## 4. Chạy smoke toàn bộ 6 baseline

Đầu tiên chạy một mẫu VietNews:

```bash
export SGLANG_ENABLE_JIT_DEEPGEMM=false
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=false

RUN_ID="b200-qwen3-smoke-$(date -u +%Y%m%dT%H%M%SZ)"

bash scripts/run_longbench_200.sh \
  --mode smoke \
  --baselines "vanilla_hf vanilla_fa eagle3 dflash domino dspark" \
  --datasets vietnews \
  --data-dir datasets/eval_100 \
  --output-dir "outputs/longbench_viet_100/$RUN_ID" \
  --max-samples 1 \
  --max-new-tokens 8 \
  --max-input-tokens 4096 \
  --warmup-runs 1 \
  --gpu-ids 0 \
  --no-data-parallel \
  --strict \
  --collect \
  --run-id "$RUN_ID"
```

Nếu server có CUDA toolkit đầy đủ, giữ:

```bash
export LONG_BENCH_SGLANG_ATTENTION_BACKEND=flashinfer
```

Nếu FlashInfer lỗi JIT/SM100, thử:

```bash
export LONG_BENCH_SGLANG_ATTENTION_BACKEND=triton
```

Không dùng `returncode=0` làm tiêu chí duy nhất. Kiểm tra manifest:

```bash
RUN_DIR="outputs/longbench_viet_100/$RUN_ID"

jq '.cells[] | {
  baseline,
  dataset,
  status,
  returncode,
  unresolved_sample_count,
  successful_samples,
  metric_audit_summary
}' "$RUN_DIR/run_manifest.json"
```

Điều kiện đạt:

```text
status = success
successful_samples = 1
unresolved_sample_count = 0
metric_contract.status = complete
issue_counts = {}
```

Audit lại:

```bash
python3 scripts/audit_benchmark_metrics.py \
  --run-dir "$RUN_DIR" \
  --expected-samples 1 \
  --expected-output-tokens 8
```

## 5. Chạy representative benchmark

Sau khi smoke pass:

```bash
RUN_ID="b200-qwen3-representative-$(date -u +%Y%m%dT%H%M%SZ)"

bash scripts/run_longbench_200.sh \
  --mode representative \
  --baselines "vanilla_hf vanilla_fa eagle3 dflash domino dspark" \
  --datasets "vietnews wikilingua vims vlsp" \
  --data-dir datasets/eval_100 \
  --output-dir "outputs/longbench_viet_100/$RUN_ID" \
  --max-samples 20 \
  --max-new-tokens 2048 \
  --warmup-runs 3 \
  --sample-retries 2 \
  --retry-backoff-seconds 5 \
  --gpu-ids 0 \
  --no-data-parallel \
  --strict \
  --collect \
  --run-id "$RUN_ID"
```

Nếu có nhiều B200, dùng data parallel:

```bash
bash scripts/run_longbench_200.sh \
  --mode representative \
  --data-parallel \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --dp-gpus-per-shard 1 \
  --dp-processes-per-gpu 1 \
  --baselines "vanilla_hf vanilla_fa eagle3 dflash domino dspark" \
  --datasets "vietnews wikilingua vims vlsp" \
  --sample-retries 2 \
  --strict \
  --collect \
  --run-id "$RUN_ID"
```

## 6. Chạy full 100 mẫu

Chỉ chạy sau khi representative pass:

```bash
RUN_ID="b200-qwen3-full-$(date -u +%Y%m%dT%H%M%SZ)"

bash scripts/run_longbench_200.sh \
  --mode full \
  --data-parallel \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --dp-gpus-per-shard 1 \
  --dp-processes-per-gpu 1 \
  --baselines "vanilla_hf vanilla_fa eagle3 dflash domino dspark" \
  --datasets "vietnews wikilingua vims vlsp" \
  --max-samples 100 \
  --max-new-tokens 2048 \
  --sample-retries 2 \
  --oom-retries 1 \
  --strict \
  --collect \
  --run-id "$RUN_ID"
```

Kết quả nằm tại:

```text
outputs/longbench_viet_100/<run-id>/
├── run_manifest.json
├── metrics_summary.json
├── metrics_summary.csv
├── metrics_summary.md
├── logs/
├── vanilla_hf/
├── vanilla_fa/
├── eagle3/
├── dflash/
├── domino/
└── dspark/
```

Các hướng dẫn này tương ứng với [docs/vietbench_qwen3.md](/home/tuantb/fast_infer_text_sum_Viet/docs/vietbench_qwen3.md) và tài liệu server canonical [server_environment.md](/home/tuantb/fast_infer_text_sum/docs/server_environment.md).