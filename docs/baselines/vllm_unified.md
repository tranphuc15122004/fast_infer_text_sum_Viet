# Đánh giá đồng bộ các baseline bằng vLLM

`scripts/run.sh vllm_all` chạy vanilla, EAGLE3, DFlash, Domino và DSpark lần
lượt trên cùng target model, cùng prompt token IDs, cùng tham số greedy và cùng
phiên bản vLLM. Mỗi lần chỉ nạp một draft model để tránh chiếm VRAM đồng thời.
Runner dùng checkpoint local trong master config và bật chế độ offline; nó không
tải model từ Hub.

## Chạy trên server

```bash
# Preflight: kiểm tra package, GPU, target/tokenizer và config các draft.
bash scripts/run.sh vllm_all --preflight-only

# Smoke: 2 mẫu chung để kiểm tra runtime và output schema.
bash scripts/run.sh vllm_all --smoke

# Full: toàn bộ mẫu đủ điều kiện trong VLLM_DATA_FILE.
bash scripts/run.sh vllm_all
```

Mặc định config được resolve qua `config/master.path`. Có thể override một lần:

```bash
bash scripts/run.sh vllm_all \
  --config /path/to/fast_infer_master_Viet.env \
  --max-new-tokens 512
```

Các biến `VLLM_*` có thể override model, data, output, phương pháp, dtype,
context limit và mức dùng VRAM. Model/data mặc định lấy từ master config:

| Biến | Master config mặc định |
|---|---|
| `VLLM_TARGET_MODEL` | `MODEL_TARGET` |
| `VLLM_EAGLE3_MODEL` | `MODEL_EAGLE_DRAFT` |
| `VLLM_DFLASH_MODEL` | `MODEL_DFLASH_DRAFT` |
| `VLLM_DOMINO_MODEL` | `MODEL_DOMINO_DRAFT` |
| `VLLM_DSPARK_MODEL` | `MODEL_DSPARK_DRAFT` |
| `VLLM_DATA_FILE` | `LONG_BENCH_DATA_FILE`, rồi `DATA_INPUT` |

Mặc định runner dùng tối đa 8.192 input tokens, 12.288 tổng context tokens,
512 output tokens, BF16 và greedy decode. Mỗi prompt được warmup riêng trước khi
đo để không tính chi phí JIT vào request đầu tiên của từng shape. Các mẫu vượt giới hạn input chung bị
bỏ khỏi toàn bộ method và được ghi vào manifest; mọi method còn lại nhận cùng
sample IDs và token IDs. Nên chạy khi GPU không có workload khác để tránh nhiễu timing.
`--max-samples N` giới hạn số mẫu đầu tiên đủ điều
kiện; mặc định `0` nghĩa là chạy hết. `--preflight-only` xác nhận model path,
Python package, CUDA và GPU trước khi bắt đầu full run.

## Artifact và tiêu chí đọc kết quả

Mỗi lần chạy tạo thư mục riêng dưới `outputs/vllm_unified/<run-id>/`:

- `console.log`: toàn bộ stdout/stderr của launcher và evaluator, bao gồm stack
  trace nếu lỗi.
- `results.jsonl`: record từng method/sample, token IDs và text, timing, metric
  chất lượng, parity, bộ nhớ GPU, RequestOutput/metrics thô của vLLM và speculative
  metrics thô.
- `samples.jsonl`: hàng dữ liệu đầu vào gốc, prompt đã render, prompt token IDs,
  reference và số từ; có thể dùng để chấm lại hoặc dựng metric mới offline.
- `excluded_samples.jsonl`: hàng gốc và lý do mẫu bị loại.
- `warmup.jsonl`: output/token IDs, RequestOutput thô, thời gian và bộ nhớ của
  từng request warmup.
- `events.jsonl`: timeline JSONL có UTC timestamp và elapsed time; ghi config nạp
  model/generation, warmup, request, GPU snapshot, lỗi, tổng hợp và hoàn tất.
- `run_report.json`: config đầy đủ, config target/draft, SamplingParams, phiên
  bản thư viện, hash dữ liệu, tokenizer, thông tin môi trường và metric tổng hợp.
- `report_vi.md`: bảng TPOT, DSR, ESR, ROUGE-L, repetition và token overlap.
- `progress.json`: method/sample đang chạy và số record đã lưu, cập nhật khi request bắt đầu/kết thúc.
- `results.partial.jsonl`: checkpoint theo từng request; được xóa sau khi
  `results.jsonl` hoàn tất, nhưng giữ lại nếu tiến trình bị ngắt.

Mỗi record giữ cả trường đã chuẩn hóa lẫn payload thô để có thể sửa cách tính
metric mà không chạy lại model. GPU được ghi bằng bộ đếm PyTorch của tiến trình
điều khiển và snapshot `nvidia-smi` theo từng giai đoạn (trước/sau load, sau
method), để thấy cả worker vLLM. Các trường latency draft/verify riêng chỉ có nếu
vLLM cung cấp trong payload thô; runner không suy diễn chúng từ tổng latency.
Runner không yêu cầu lưu logits/logprobs theo token vì điều đó làm thay đổi overhead
của phép đo tốc độ; metric cần xác suất token phải được tính trong lượt chẩn đoán
riêng.

DSR/ESR chỉ được tính khi cả method và vanilla có timing hợp lệ trên cùng tập
sample IDs. `correctness_pass=false` báo có method không khớp greedy vanilla;
khi đó không diễn giải speedup của method ấy như hiệu năng đã xác nhận. Speedup
dưới 1 vẫn được báo cáo nếu đầu ra hợp lệ và phép đo hoàn tất. Runner trả mã lỗi
nếu request/schema hoặc quality guard thất bại; artifact vẫn được giữ.
