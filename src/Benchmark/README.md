# Hướng dẫn & Kiến trúc Benchmark (`src/Benchmark`)

Module `src/Benchmark` là package cốt lõi chịu trách nhiệm thực thi các phép đo lường, đánh giá tốc độ inference và chất lượng tóm tắt văn bản dài tiếng Việt cho mô hình **Qwen3-4B** trên phần cứng **NVIDIA B200**.

> [!IMPORTANT]
> **Quyết định kiến trúc chính thức:**
> Pipeline **vLLM Đồng Bộ** (`src/Benchmark/vllm_all_baselines.py` qua launcher `scripts/run_vllm_all.sh`) hiện tại được chọn làm **Pipeline Benchmark Chính (Primary Benchmark Pipeline)** của dự án, thay thế cho việc chạy các engine phân tán độc lập của từng baseline trong `run_longbench_200.py`.

---

## 1. Tại sao chuyển sang Pipeline vLLM Đồng Bộ làm Benchmark Chính?

Trước đây, mỗi phương pháp suy đoán (Speculative Decoding) sử dụng runtime riêng biệt của repo gốc (`externals/`):
- **EAGLE3**: Chạy qua codebase native của EAGLE (PyTorch CLI loop).
- **DFlash**: Chạy qua codebase DFlash Transformers native.
- **Domino & DSpark**: Chạy thông qua máy chủ SGLang Server (HTTP loopback API).
- **Vanilla**: Chạy qua HuggingFace Transformers hoặc FlashAttention.

### Hạn chế của phương pháp cũ:
1. **Runtime Confounders (Nhiễu do sai khác engine)**: Rất khó phân biệt được tốc độ tăng lên là nhờ *thuật toán suy đoán (speculative algorithm)* hay do *sự tối ưu của engine (SGLang C++ runtime vs Python Transformers loop)*.
2. **Khó kiểm soát bộ nhớ**: Nhiều baseline chạy đồng thời server daemon dễ gây tranh chấp VRAM hoặc rò rỉ bộ nhớ.
3. **Độ lệch đo lường**: Độ trễ HTTP, IPC, cơ chế batching ngầm khác nhau khiến phép đo E2E và Decode Latency bị lệch.

### Ưu điểm vượt trội của Pipeline vLLM Đồng Bộ:
- **100% Đồng nhất Runtime**: Cả 5 baseline (`vanilla_vllm`, `eagle3`, `dflash`, `domino`, `dspark`) đều chạy trên cùng một engine **`vLLM 0.30.0` (V2 Model Runner)**.
- **Cùng cấu hình tải nghiêm ngặt**:
  - Batch size cố định = 1, `max_num_seqs = 1`.
  - Greedy decoding tuyệt đối (`temperature = 0.0`, `seed = 42`).
  - Vô hiệu hóa prefix caching (`enable_prefix_caching = False`), không dùng chung cache giữa các request.
  - Cùng tập tokenized prompt IDs (cùng tokenizer, chat template và giới hạn context).
- **Warmup độc lập theo từng mẫu (Per-sample Warmup)**: Mỗi prompt đều được thực hiện warmup ngắn (`max_tokens = min(warmup_tokens, max_new_tokens)`) nhằm loại trừ chi phí CUDA Graph compilation / JIT overhead ra khỏi thời gian đo đạc chính thức.
- **Cách ly tài nguyên an toàn**: Nạp tuần tự đúng 1 target model + 1 draft model tương ứng trên 1 GPU. Khi chạy xong một baseline, engine được shutdown hoàn toàn, thu hồi VRAM (`gc.collect()`, `torch.cuda.empty_cache()`) trước khi khởi tạo baseline tiếp theo.

---

## 2. Các Baseline Được So Sánh trong Pipeline vLLM

| Baseline | Kiến trúc / Cơ chế Speculative | Speculative Tokens ($K$) | Ghi chú tích hợp trong vLLM |
|---|---|:---:|---|
| **`vanilla_vllm`** | Autoregressive Baseline | — | **Dense Reference** chuẩn cho toàn bộ benchmark |
| **`eagle3`** | Tree-based Speculative Drafting | 16 | vLLM 0.30 native với aux hidden state layer resolution |
| **`dflash`** | Block-Diffusion Parallel Drafting | 16 | vLLM 0.30 native DFlash backbone |
| **`domino`** | Block-parallel + Causal Correction (GRU) | 16 | Tích hợp qua monkey-patching adapter trên đường dẫn DFlash của vLLM |
| **`dspark`** | SpecForge Parallel Draft Model | 7 | vLLM 0.30 native DSpark backend |

### Cơ chế Adapter & Monkey-patching cho Domino (`vllm_pilot.py`)
Do vLLM 0.30 gốc chưa có native runner cho Domino (sử dụng GRU prefix và causal projection head), repo cung cấp một giải pháp tích hợp trực tiếp:
- [`src/Benchmark/common/vllm_pilot.py`](file:///home/tuantb/fast_infer_text_sum_Viet/src/Benchmark/common/vllm_pilot.py) bổ sung `prefix_gru` và `embed_proj` vào `DFlashQwen3Model` và chèn hàm `domino_greedy_sample` vào `DFlashProposer._sample_draft_tokens` và `DFlashSpeculator.load_draft_model`.
- Đăng ký tự động qua **vLLM General Plugin** (`src/Benchmark/common/vllm_pilot_plugin.py` và `src/fast_infer_viet_vllm_plugin-0.1.0.dist-info/`).

---

## 3. Hệ thống Chỉ số & Công thức Đo lường (Metric Formulation)

Toàn bộ thời gian được trích xuất từ server metrics chính thức của vLLM (`request_output.metrics`):

1. **TPOT (Time Per Output Token)**:
   $$\text{TPOT} = \text{mean\_itl\_ms} = \frac{\text{last\_token\_ts} - \text{first\_token\_ts}}{\text{output\_tokens} - 1}$$
   *(Đo riêng thời gian inter-token latency của decode, loại trừ prefill và queue wait).*

2. **DSR (Decode Speedup Ratio)**:
   $$\text{DSR} = \frac{\text{Mean TPOT of } \texttt{vanilla\_vllm}}{\text{Mean TPOT of method}}$$

3. **ESR (End-to-end Speedup Ratio)**:
   $$\text{ESR} = \frac{\bar{T}_{\text{prefill, ref}} + \bar{T}_{\text{TPOT, ref}} \times \bar{L}_{\text{paired\_min}}}{\bar{T}_{\text{prefill, ref}} + \bar{T}_{\text{TPOT, method}} \times \bar{L}_{\text{paired\_min}}}$$
   *(Trong đó $\bar{L}_{\text{paired\_min}}$ là độ dài đầu ra tối thiểu của cặp mẫu để chuẩn hóa công bằng khối lượng sinh).*

4. **Token LCS Overlap with Vanilla**:
   $$\text{LCS Overlap} = \frac{\sum \text{LCS}(\text{output\_ids}_{\text{method}}, \text{output\_ids}_{\text{vanilla}})}{\sum \text{len}(\text{output\_ids}_{\text{vanilla}})}$$

5. **Cổng kiểm tra Greedy Parity (`correctness_pass`)**:
   Kiểm tra xem 100% token IDs sinh ra bởi baseline suy đoán có trùng khớp hoàn toàn với greedy vanilla decoding không.

---

## 4. Kết quả Thực nghiệm Thực tế trên B200 (`full-all4-20260930T225348-3093798`)

Kết quả chạy trên **NVIDIA B200** (vLLM 0.30.0, PyTorch 2.13.0+cu130, target Qwen3-4B, 399 mẫu tiếng Việt từ 4 bộ dữ liệu VietNews, WikiLingua, ViMs, VLSP):

| Phương pháp | Prefill TB (ms) | TPOT (ms/token) | Throughput (tok/s) | DSR $\uparrow$ | ESR $\uparrow$ | ROUGE-L | Khớp Token Tuyệt đối | LCS Overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`vanilla_vllm`** *(Ref)* | **21.36** | 2.914 | 343.17 | 1.000 | 1.000 | **0.1956** | 399/399 (100.0%) | 100.00% |
| **`dspark`** | 22.34 | **1.825** | **548.09** | **1.597×** | **1.557×** | 0.1953 | 182/399 (45.6%) | 91.19% |
| **`dflash`** | 21.98 | **2.028** | **493.18** | **1.437×** | **1.410×** | 0.1937 | 200/399 (50.1%) | 91.97% |
| **`eagle3`** | 24.51 | 4.392 | 227.70 | **0.664×** | **0.673×** | 0.1937 | 200/399 (50.1%) | 91.97% |
| **`domino`** | 23.84 | 4.590 | 217.86 | **0.635×** | **0.645×** | 0.1937 | 200/399 (50.1%) | 91.97% |

### Phân tích Khoa học từ Kết quả:
- **`dspark` và `dflash`** đạt hiệu quả tăng tốc ấn tượng (tăng tốc lần lượt **1.597×** và **1.437×** về tốc độ decode, đẩy throughput lên **493 - 548 tok/s**).
- **`eagle3` và `domino`** bị suy giảm tốc độ trên B200 (DSR < 1) do overhead xử lý cấu trúc cây hoặc GRU tuần tự lớn hơn chi phí tính toán decode nhanh của mô hình gốc Qwen3-4B trên B200.
- **Parity Gate**: Các phương pháp đạt độ trùng khớp ngữ nghĩa rất cao (LCS Overlap > 91%, ROUGE-L tương đương) nhưng chỉ đạt 45.6% - 50.1% exact greedy match. Báo cáo ghi nhận trung thực `correctness_pass=false` để thể hiện speedup mang tính quan sát thực nghiệm.

---

## 5. Hướng dẫn Chạy Benchmark trên Server B200

Mọi lệnh chạy đều được bọc trong launcher shell [`scripts/run_vllm_all.sh`](file:///home/tuantb/fast_infer_text_sum_Viet/scripts/run_vllm_all.sh):

```bash
# 1. Kiểm tra môi trường, GPU, model paths & tokenizer (không chạy inference)
bash scripts/run_vllm_all.sh --preflight-only

# 2. Chạy smoke test nhanh (2 mẫu) để kiểm tra luồng end-to-end
bash scripts/run_vllm_all.sh --smoke

# 3. Chạy full benchmark toàn bộ bộ dữ liệu 400 mẫu tiếng Việt
bash scripts/run_vllm_all.sh --full

# 4. Tùy biến tham số hoặc master config
bash scripts/run_vllm_all.sh \
  --config /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master_Viet.env \
  --max-new-tokens 512 \
  --gpu-memory-utilization 0.88
```

### Cấu trúc Artifacts Xuất ra:
Mỗi run tạo một thư mục độc lập dưới `outputs/vllm_unified/<run-id>/`:
- `report_vi.md`: Báo cáo markdown tổng hợp TPOT, DSR, ESR, ROUGE, Parity.
- `run_report.json`: JSON đầy đủ metadata môi trường, tham số model, thống kê chi tiết.
- `results.jsonl`: Dữ liệu chi tiết từng sample, token IDs, latency breakdown, speculative metrics.
- `warmup.jsonl`: Ghi nhận dữ liệu các bước warmup từng prompt.
- `events.jsonl`: Timeline sự kiện theo thời gian thực (nạp model, warmup, request, snapshot GPU).
- `samples.jsonl` & `excluded_samples.jsonl`: Danh sách mẫu được chọn và mẫu bị loại trừ.
- `console.log`: Log đầy đủ của phiên chạy.

---

## 6. Quan hệ với Hệ thống Cũ (`run_longbench_200.py`)

Runner cũ `src/Benchmark/run_longbench_200.py` vẫn được lưu giữ trong repo nhằm mục đích:
1. Đối chiếu (cross-validation) với kết quả triển khai native của các tác giả upstream.
2. Kiểm tra tính tương thích ngược với format dữ liệu và cấu hình LongBench truyền thống.

Tuy nhiên, đối với các kết quả báo cáo chính thức, so sánh công bằng và xuất bảng paper, **Pipeline vLLM Đồng Bộ là chuẩn mực duy nhất được sử dụng**.
