# Hướng dẫn và kiến trúc Benchmark

`src/Benchmark` chứa adapter inference, runtime guard, schema kết quả và metric
cho benchmark tóm tắt văn bản tiếng Việt. Pipeline so sánh năm baseline trên
GPU Modal Blackwell là **native Transformers + FlashAttention-4 (FA4)**; không
dùng vLLM:

| Method | Cơ chế | Model draft |
|---|---|---|
| `vanilla_hf` | Autoregressive reference | — |
| `eagle3` | Tree speculative decoding | EAGLE-3 |
| `dflash` | Block-diffusion drafting | DFlash block 16 |
| `domino` | Parallel block + causal correction | Domino block 16 |
| `dspark` | Learned parallel speculative decoding | DSpark block 7 |

Runner chính: [`scripts/modal_flashattn_pilot.py`](../../scripts/modal_flashattn_pilot.py);
runbook đầy đủ: [`docs/flashattn4_benchmark.md`](../../docs/flashattn4_benchmark.md).
Tên file pilot được giữ để tương thích các lệnh Modal đã dùng trước đó; CLI hỗ
trợ smoke, representative và full benchmark.

## Contract chung

- Qwen3-4B target, cùng tokenizer/chat prompt, dữ liệu từ `datasets/eval_100/`,
  greedy decoding, cùng giới hạn input/output và batch size 1.
- Target và draft đều cấu hình FA4. Runner xác minh dispatch khi chạy và báo
  lỗi nếu method fallback sang attention backend khác. EAGLE tree mask được
  kiểm tra trên GPU trước benchmark.
- Mỗi method nạp tuần tự trên một GPU; benchmark ghi riêng model-load time,
  warmup, latency và peak memory. Mặc định warmup một lần cho từng prompt.
- Model repository, package pins, GPU, dữ liệu, mã mẫu, method config, seed và
  thời gian chạy được lưu trong metadata.
- Output ghi theo schema `Benchmark.common.benchmark_runtime` và kết thúc bằng
  summary record qua `JsonlWriter`.

## Lệnh Modal

```bash
# Kiểm tra FA4/GPU/package pins/tree mask, không sinh output
bash scripts/run_fa4_benchmark.sh --preflight-only

# Smoke tất cả baseline trên một dataset
bash scripts/run_fa4_benchmark.sh \
  --mode smoke --datasets vietnews --samples-per-dataset 1 \
  --max-new-tokens 32

# 20 mẫu/dataset, chọn quantile độ dài có tính xác định
bash scripts/run_fa4_benchmark.sh \
  --mode representative --datasets all --max-new-tokens 512

# 100 mẫu/dataset trên bốn dataset
bash scripts/run_fa4_benchmark.sh \
  --mode full --datasets all --max-new-tokens 512 --repetitions 1
```

Tùy chọn CLI gồm dataset/method selection, sample cap, input/output token cap,
warmup tokens, repetitions, seed, retry, run ID/resume, CUDA launch debugging,
target-only greedy audit, verifier audit và thư mục tải artifact. `--resume`
cần `--run-id` và cùng cấu hình với run đã bắt đầu.

## Metric

Mỗi sample/repeat giữ output text và token IDs; E2E, prefill/TTFT, decode, TPOT,
throughput, decode throughput, QPS, peak memory, acceptance length/rate, số draft
token được đề xuất/nhận, verification steps, ROUGE-1/2/L, ROUGE-Lsum, BLEU-1..4,
length ratio, repetition guard, exact greedy parity và token LCS với Vanilla.
Summary có mean/median/p90/std và breakdown từng dataset.

DSR là mean Vanilla TPOT chia mean method TPOT. ESR chuẩn hóa theo độ dài output
nhỏ hơn trong từng cặp sample. Paired metrics chỉ tính khi có cả hai kết quả và
timing dương; speedup được báo đúng theo số đo và có thể nhỏ hơn 1.

Native HF không cung cấp cùng server request timeline như vLLM. Vì vậy queue
wait, batch wait, server startup và server-reported E2E để `null`. Prefill của
baseline có native TTFT dùng timer đó; với Vanilla/DSpark runner dùng CUDA-event
duration của target forward đầu tiên. TPOT được tính từ E2E trừ prefill, chia số
token sau token đầu. Báo cáo đánh dấu measurement boundary này để không nhầm
proxy timing với server-side ITL.

`quality_valid` yêu cầu output không rỗng, ít nhất 4 token và không có repetition
collapse flag. Quality, greedy parity, FA4 dispatch và tốc độ là các gate riêng;
không thay đổi output model để khiến gate pass.

## Artifacts và các runner khác

Modal lưu checkpoint trên Volume `fast-infer-viet-fa4-results` và tải kết quả
sang `outputs/modal_flashattn_benchmark/<run-id>/`:

- `results.jsonl`, `run_report.json`, `report_vi.md`, `metrics_summary.csv`;
- `warmup.jsonl`, `events.jsonl`, `samples.jsonl`, `excluded_samples.jsonl`;
- `progress.json`, `state.json`, `results.partial.jsonl`.

`vllm_all_baselines.py`, `run_vllm_all.sh` và kết quả vLLM cũ vẫn nằm trong repo
để tái hiện lịch sử; chúng không phải pipeline FA4 hiện hành. `run_longbench_200.py`
là orchestration cho các baseline LongBench khác và không thay cho runner Modal
này.
