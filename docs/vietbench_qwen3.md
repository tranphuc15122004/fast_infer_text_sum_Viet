# Benchmark Qwen3-4B trên dữ liệu tiếng Việt

Runner dùng cùng cơ chế orchestration với project tham chiếu nhưng đọc bốn
file chuẩn trong `datasets/eval_100/`:

```text
vietnews_100.jsonl
wikilingua_100.jsonl
vims_100.jsonl
vlsp_100.jsonl
```

Target cố định là Qwen3-4B. Ma trận baseline theo phương pháp gồm
`vanilla_hf`, `vanilla_fa`, `eagle3`, `dflash`, `domino` và `dspark`. Code
speculative vẫn do implementation upstream trong `externals/` thực hiện; các
script của repo chỉ chuyển input, khởi động runtime và chuẩn hóa output.

## Preflight bằng venv mô phỏng B200

```bash
export FAST_INFER_VENV=/home/tuantb/fast_infer_text_sum/.venv
export FAST_INFER_MASTER_CONFIG=/path/to/fast_infer_master_Viet.env

bash scripts/run_longbench_200.sh \
  --mode smoke \
  --preflight-only \
  --allow-unsupported \
  --data-dir datasets/eval_100 \
  --output-dir /tmp/longbench_viet_preflight
```

Master production canonical là file `_Viet` trên server; không dùng master
config của project tham chiếu.

Master cần cung cấp target và checkpoint draft tương ứng (đường dẫn snapshot
local hoặc repo ID đã có trong cache):

```bash
MODEL_TARGET=/models/Qwen3-4B
MODEL_EAGLE_DRAFT=/models/Qwen3-4B_eagle3
MODEL_DFLASH_DRAFT=/models/Qwen3-4B-DFlash
MODEL_DOMINO_DRAFT=/models/Qwen3-4B-Domino-b16
MODEL_DSPARK_DRAFT=/models/Qwen3-4B-DSpark
```

Có thể override riêng bằng `LONG_BENCH_*_MODEL`. `MODEL_DOMINO_DRAFT` và
`MODEL_DSPARK_DRAFT` được launcher ánh xạ sang baseline tương ứng.

## Chạy smoke/full

```bash
bash scripts/run_longbench_200.sh \
  --mode smoke \
  --baselines "vanilla_hf vanilla_fa eagle3 dflash domino dspark" \
  --datasets "vietnews wikilingua vims vlsp"

bash scripts/run_longbench_200.sh \
  --mode full \
  --data-parallel \
  --gpu-ids 0,1,2,3,4,5,6,7
```

SGLang speculative baseline mặc định dùng `LONG_BENCH_BATCH_SIZE=auto`.
Trên B200 180 GiB, `auto` bắt đầu ở batch 8 và có thể ghi đè bằng:

```bash
LONG_BENCH_AUTO_BATCH_SIZE=4 bash scripts/run_longbench_200.sh --mode representative
```

`vanilla_fa` được chọn làm dense reference trước; nếu không có output thì
runner fallback sang `vanilla_hf`. Baseline speculative không chạy lại vanilla
batch size 1. Speedup chỉ được tính khi join được cùng `sample_id` và cùng
output token budget.

Mỗi run lưu raw JSONL, timing phase, acceptance metadata, VRAM, batch/TP/DP
assignment, `run_manifest.json`, live logs và `metrics_summary.{json,csv,md}`.

## Chạy full an toàn và tự khôi phục

Full run mặc định không dừng khi một child process, cell hoặc shard bị lỗi.
Sau mỗi cell, runner kiểm tra coverage theo `sample_id`, retry riêng từng
sample còn thiếu/lỗi ở batch 1, chờ slot VRAM khi chạy data-parallel, rồi ghi
status `failed` với timing `null` nếu retry vẫn thất bại. Vì vậy sample lỗi
không biến mất khỏi JSONL và các cell còn lại vẫn tiếp tục chạy.

```bash
bash scripts/run.sh longbench_200 \
  --mode full \
  --data-parallel \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --sample-retries 2 \
  --retry-backoff-seconds 5
```

Các tham số tương đương trong master là `LONG_BENCH_SAMPLE_RETRIES`,
`LONG_BENCH_RETRY_BACKOFF_SECONDS`, `LONG_BENCH_RETRY_FAILED_SAMPLES` và
`LONG_BENCH_CONTINUE_ON_ERROR`. Mặc định là retry 2 lần/sample, backoff tăng
theo cấp số nhân và tiếp tục toàn bộ ma trận. `run_manifest.json` ghi
`retry_history`, `unresolved_sample_ids`, `safe_eval_complete` và trạng thái
từng cell; các failure/OOM/timeout không được dùng như timing hợp lệ.

Các input/output của từng lần retry được giữ trong `attempts/` để chẩn đoán;
runner không sửa raw JSONL hoặc ghi đè artifact của run cũ.
