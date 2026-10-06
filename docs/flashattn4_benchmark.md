# Benchmark native Transformers với FlashAttention-4

Runner `scripts/modal_flashattn_pilot.py` triển khai một benchmark batch-1 trên
GPU Blackwell của Modal cho năm method trong `src/Benchmark`:
`vanilla_hf`, `eagle3`, `dflash`, `domino`, `dspark`. Target là Qwen3-4B; mọi
target và draft attention đều phải dispatch qua FA4, nếu có fallback thì runtime
gate báo lỗi. Runner không nạp vLLM.

Các file JSONL trong `datasets/eval_100/` là nguồn dữ liệu. Tất cả method dùng
cùng prompt, tokenized input, seed, greedy decoding và ngân sách token. Mẫu được
chọn theo quantile độ dài có tính xác định; full mode dùng đủ 100 mẫu của mỗi
dataset được chọn. Nếu input dài hơn giới hạn, runner giữ head và suffix của
prompt bằng cùng helper truncation dùng trong benchmark chung, đồng thời ghi số
token nguồn và cờ truncation.

## Chạy

Trước tiên kiểm tra GPU, package pins, khả năng import FA4 và kernel tree mask:

```bash
bash scripts/run_fa4_benchmark.sh --preflight-only
```

Smoke nhanh một dataset qua đủ năm method:

```bash
bash scripts/run_fa4_benchmark.sh \
  --mode smoke --datasets vietnews --samples-per-dataset 1 \
  --max-new-tokens 32
```

Benchmark representative mặc định 20 mẫu/dataset; full mặc định đủ 100:

```bash
bash scripts/run_fa4_benchmark.sh \
  --mode representative --datasets all --max-new-tokens 512

bash scripts/run_fa4_benchmark.sh \
  --mode full --datasets all --max-new-tokens 512 --repetitions 1
```

Chọn tập/method hoặc đổi model repository qua các biến `MODAL_QWEN3_MODEL`,
`MODAL_EAGLE3_MODEL_REPO`, `MODAL_DFLASH_MODEL_REPO`,
`MODAL_DOMINO_MODEL_REPO`, `MODAL_DSPARK_MODEL_REPO`. `--methods` phải có cả
`vanilla_hf` và `dflash` vì đây là cặp reference tối thiểu của runner.

Các tham số chính: `--samples-per-dataset 0` chọn toàn bộ eligible rows,
`--max-input-tokens`, `--warmup-tokens`, `--repetitions`, `--seed`,
`--sample-retries`, `--checkpoint-interval`, `--run-id`, `--resume`,
`--direct-target-audit`, `--verifier-audit`, `--output-dir`. Mặc định mỗi mẫu
được warmup riêng bằng 8 token; kết quả đo lặp được giữ qua `repeat_index` và
tổng hợp trên từng sample/repeat.

## Metric và artifact

Mỗi sample/repeat có output IDs/text, model, input/output tokens, memory peak,
E2E, prefill/TTFT, decode, TPOT, throughput, QPS, draft/verification time,
acceptance counters, ROUGE-1/2/L, ROUGE-Lsum, BLEU-1..4, length ratio,
repetition/quality guard, exact greedy match và token LCS overlap với Vanilla.
Summary báo mean/median/p90/std, DSR, ESR và metric riêng theo dataset cùng
metric gộp. DSR/ESR chỉ ghép cùng sample và repeat; ESR dùng số token output
chung là độ dài nhỏ hơn của cặp. Speedup được đo trung thực, có thể dưới 1.

`prefill_ms` lấy native TTFT của baseline nếu có; với Vanilla và DSpark dùng
CUDA-event duration của target forward đầu tiên. `tpot_ms` là
`(e2e_ms - prefill_ms) / (output_tokens - 1)`. Queue wait, server startup,
batch wait và server-reported E2E là `null` vì runner không chạy request server.
Không so sánh các metric proxy này với server-side timing như thể chúng cùng
measurement boundary.

Modal ghi tiến độ/checkpoint lên Volume `fast-infer-viet-fa4-results`; sau khi
hoàn tất, launcher tải artifact về `outputs/modal_flashattn_benchmark/<run-id>/`:

- `results.jsonl`: mọi sample/repeat và summary cuối.
- `run_report.json`, `report_vi.md`, `metrics_summary.csv`.
- `warmup.jsonl`, `events.jsonl`, `samples.jsonl`, `excluded_samples.jsonl`.
- `progress.json`, `state.json`, `results.partial.jsonl` để resume/kiểm toán.

Nếu Modal ngắt một run, dùng lại `--run-id <id> --resume` với đúng cấu hình.
Chỉ skip cell đã có kết quả thành công; cell lỗi được chạy lại theo số lần
`--sample-retries`. Trước khi dùng full run cho paper, chạy smoke trên đúng
checkpoint, xem runtime gate, sample coverage, quality guard, greedy parity và
đối chiếu report/JSONL.

Nếu inference đã hoàn tất nhưng máy local ngắt trước khi tải kết quả, khôi phục
artifact mà không chạy lại model bằng:

```bash
bash scripts/run_fa4_benchmark.sh --download-only --run-id <run-id>
```
