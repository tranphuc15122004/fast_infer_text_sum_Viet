# Benchmark native Transformers với FlashAttention-4

Runner so sánh năm method `vanilla_hf`, `eagle3`, `dflash`, `domino`, `dspark`
trực tiếp trên GPU B200 của server. Cả target và draft đều phải dispatch qua
FlashAttention-4 (FA4); runtime gate dừng run nếu phát hiện attention fallback.
Inference chạy bằng Transformers native, batch size 1, greedy decoding, không
import hay gọi vLLM. Runner dùng Python 3.12 và package FA4 đã cài trên server;
nó không tạo virtualenv, cài package, hoặc tải checkpoint từ internet.

Target và draft checkpoint được lấy từ master config mà `config/master.path`
trỏ tới. Bốn JSONL dưới `datasets/eval_100/` là dữ liệu chung, mỗi file có 100
mẫu. Mọi method dùng cùng tokenizer, prompt tiếng Việt, seed và ngân sách token.
Representative chọn 20 mẫu/dataset phủ dải độ dài; full chạy đủ 100 mẫu/dataset.
Input dài hơn giới hạn sẽ được truncate bằng helper dùng chung và ghi lại số
token nguồn cùng cờ truncation.

## Chạy trên server B200

Chạy từ thư mục checkout repo Việt trên server. Mặc định launcher đọc
`config/master.path`; để chỉ rõ master config `_Viet`, truyền path làm đối số
đầu tiên hoặc đặt `FAST_INFER_MASTER_CONFIG`:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects
export FAST_INFER_MASTER_CONFIG=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master_Viet.env
```

Preflight kiểm tra Python 3.12, FA4 import qua compatibility shim của repo,
B200 (SM100+), kernel tree-mask của EAGLE và tính toàn vẹn của `eval_100/`; nó
không nạp model checkpoint:

```bash
FI_GPU_IDS=0 bash scripts/run_fa4_benchmark.sh --preflight-only
```

Sau khi preflight pass, smoke chạy một mẫu mỗi dataset qua đủ năm baseline:

```bash
FI_GPU_IDS=0 bash scripts/run_fa4_benchmark.sh \
  --mode smoke --datasets all --samples-per-dataset 1 \
  --max-new-tokens 64
```

Representative chạy 20 mẫu cho mỗi dataset được chọn. Có thể bắt đầu với một
dataset để kiểm tra thời gian và mức dùng VRAM; lệnh dưới đây chạy cả bốn:

```bash
FI_GPU_IDS=0 bash scripts/run_fa4_benchmark.sh \
  --mode representative --datasets all --max-new-tokens 512
```

Full dùng đủ 100 mẫu/dataset:

```bash
FI_GPU_IDS=0 bash scripts/run_fa4_benchmark.sh \
  --mode full --datasets all --max-new-tokens 512 --repetitions 1
```

`FI_GPU_IDS=0` dùng GPU vật lý số 0. Có thể bỏ nếu master config đã chọn đúng
GPU hoặc scheduler đã gán `CUDA_VISIBLE_DEVICES`. Mỗi method được nạp và chạy
tuần tự trên cùng một GPU, vì vậy có thể chạy khi chỉ GPU 0 đang rảnh; runner
không chia một run sang nhiều card.

Các option chính: `--datasets`, `--methods`, `--samples-per-dataset`,
`--max-input-tokens`, `--max-new-tokens`, `--warmup-tokens`, `--repetitions`,
`--seed`, `--sample-retries`, `--checkpoint-interval`, `--run-id`, `--resume`,
`--direct-target-audit`, `--verifier-audit`, `--preflight-only`, `--output-dir`.
Mặc định output nằm dưới `outputs/fa4_native_benchmark/<run-id>/`; có thể nối
log ra file bằng `2>&1 | tee <log-file>`. Khi resume phải truyền cùng run ID
và cấu hình như lần chạy trước.

Nếu vLLM nằm trong shared Python environment vì các job khác, điều đó không làm
benchmark này thành vLLM: runtime guard ghi nhận package có cài nhưng yêu cầu
không module vLLM nào được import trong process benchmark. Không gỡ hoặc cài lại
package toàn cục.

## Metric và artifact

Mỗi sample/repeat lưu output IDs/text, model, input/output tokens, peak memory,
E2E, prefill/TTFT, decode, TPOT, throughput, QPS, draft/verification time,
acceptance counters, ROUGE-1/2/L, ROUGE-Lsum, BLEU-1..4, length ratio,
repetition/quality guard, exact greedy match và token LCS với Vanilla. Summary
báo mean/median/p90/std, DSR, ESR và metric theo từng dataset. DSR/ESR ghép cùng
sample và repeat; ESR tính trên số token output nhỏ hơn của cặp. Speedup báo theo
số đo thực tế, có thể nhỏ hơn 1.

`prefill_ms` dùng native TTFT nếu baseline cung cấp; Vanilla và DSpark dùng
CUDA-event duration của target forward đầu tiên. `tpot_ms` là
`(e2e_ms - prefill_ms) / (output_tokens - 1)`. Queue wait, server startup, batch
wait và server-reported E2E để `null` vì đây không phải server request API.

Artifact trong mỗi run directory gồm `results.jsonl` (sample records và summary
cuối), `run_report.json`, `report_vi.md`, `metrics_summary.csv`, `warmup.jsonl`,
`events.jsonl`, `samples.jsonl`, `excluded_samples.jsonl`, `progress.json`,
`state.json` và `results.partial.jsonl`.

## Chạy qua Modal

Modal là đường chạy riêng, không sử dụng GPU server. Nếu cần gọi lại Modal, dùng
launcher tương thích này:

```bash
bash scripts/run_fa4_modal_benchmark.sh \
  --mode smoke --datasets vietnews --samples-per-dataset 1 --max-new-tokens 32
```
