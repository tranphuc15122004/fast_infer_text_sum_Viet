# Tiến độ Huấn luyện & Sổ tay Thực nghiệm DFlash Tiếng Việt (Server B200)

> **Mục tiêu dự án:** Huấn luyện, tối ưu hoá và đánh giá công bằng các mô hình suy luận đầu cơ (Speculative Decoding) họ DFlash cho bài toán **Tóm tắt văn bản tiếng Việt** trên nền mô hình gốc **Qwen3-4B**, so sánh trực tiếp với baseline autoregressive tiêu chuẩn (Vanilla vLLM).

Tài liệu này ghi nhận đầy đủ hiện trạng, các lỗi kỹ thuật đã xử lý, đường dẫn dữ liệu/mô hình trên cụm B200, và toàn bộ lệnh chạy huấn luyện cũng như đánh giá để các phiên làm việc tiếp theo dễ dàng theo dõi và tái lập.

---

## 1. Bảng Tổng quan Tiến độ 3 Baseline

| Baseline | Kiến trúc / Phương pháp | Trọng số Init | Hàm Loss | Trạng thái Train | Trạng thái Eval VietBench |
|---|---|---|---|---|---|
| **Baseline 1** | DFlash (5 layers, b16) | From Scratch (ngẫu nhiên) | DFlash chuẩn ($\gamma=7.0$) | ✅ **Hoàn thành 6 epochs** (6,162 steps) | ⏳ Sẵn sàng chạy Eval |
| **Baseline 2** | DFlash (5 layers, b16) | `Qwen3-4B-DFlash-b16` gốc | DFlash chuẩn ($\gamma=7.0$) | ✅ **Hoàn thành Finetune** | ⏳ Sẵn sàng chạy Eval |
| **Baseline 3** | DFlash (5 layers, b16) | From Scratch (ngẫu nhiên) | **GrowMTP** (DCA + VGM) | ✅ **Hoàn thành Train** | ⏳ Sẵn sàng chạy Eval |

---

## 2. Hệ thống Đường dẫn Chuẩn (Canonical Paths) trên Cụm Server B200

| Thành phần | Đường dẫn Tuyệt đối trên B200 | Ghi chú |
|---|---|---|
| **Repo Root (Mã nguồn)** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum_Viet-main` | Nhánh `main` đồng bộ với GitHub |
| **Master Config** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master_Viet.env` | Chứa cấu hình môi trường offline |
| **Target Model (Teacher)** | `/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B` | Mô hình mục tiêu Qwen3-4B |
| **Pretrained DFlash gốc** | `/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B-DFlash-b16` | Checkpoint DFlash đa ngôn ngữ gốc |
| **Tập Train sạch (36,973 mẫu)** | `/workspace/storage-shared/nlp/dungdx4/bien_projects/LLM2Seq/src/eviseq_new/datasets/50k/train_clean.jsonl` | Dữ liệu văn bản tiếng Việt làm sạch |
| **Bộ Eval Chuẩn (VietBench 100)** | `datasets/eval_100/` (`vietnews`, `wikilingua`, `vims`, `vlsp`) | 4 bộ test chuẩn hoá, 100 mẫu/bộ |
| **Phase 1 Cache (Features)** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/features/` | Tensors 5 target layers `[1, 9, 17, 25, 33]` |
| **Phase 1 Cache (Teacher)** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/teacher/` | Trajectories sinh bởi Qwen3-4B |
| **Output Baseline 1 (Scratch)** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/` | Checkpoint lưu tại `checkpoints/` |
| **Output Baseline 2 (Finetuned)** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_finetuned/` | Checkpoint lưu tại `checkpoints/` |
| **Output Baseline 3 (GrowMTP)** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_growmtp/` | Checkpoint lưu tại `checkpoints/` |

---

## 3. Nhật ký Huấn luyện Chi tiết 3 Baseline

### 3.1. Baseline 1: DFlash From Scratch (Chuẩn $\gamma=7.0$)
- **Kiến trúc & Cấu hình:** 5 Draft Layers, Block Size 16, Mask Token ID 151669, 2x B200 GPUs.
- **Tổng số bước:** 6,162 steps (chính xác 6 epochs trọn vẹn trên 36,973 mẫu tiếng Việt).
- **Thời gian thực thi:** 2 giờ 29 phút 17 giây (Tốc độ trung bình: 1.45s / step, thông lượng ~25,924 tokens/s).
- **Hội tụ Loss & Accuracy:**
  - `Step 1`: `loss = 12.5981`, `acc = 0.0%`
  - `Step 500`: `loss = 3.6521`, `acc = 22.4%`
  - `Step 3000`: `loss = 2.4510`, `acc = 35.8%`
  - `Step 6000`: `loss = 2.0366`, `acc = 42.4%` (lr = 9.34e-08)
  - `Step 6162 (End)`: `loss = 2.0608`, `acc = 40.4%` (lr = 0.0)
- **Checkpoint lưu tại:** `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/checkpoints/`

### 3.2. Baseline 2: Finetune từ Pretrained DFlash Gốc (`Qwen3-4B-DFlash-b16`)
- **Trọng số khởi tạo:** Nạp toàn bộ trọng số pre-trained đa ngôn ngữ từ `/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B-DFlash-b16`.
- **Dữ liệu chuyển tiếp:** Huấn luyện thích nghi domain tóm tắt tiếng Việt trên 36,973 mẫu văn bản chất lượng cao, tái sử dụng offline feature cache từ Phase 1.
- **Mục tiêu:** Đánh giá lợi thế hội tụ và tốc độ chấp nhận token (acceptance rate) khi khởi động từ mô hình đã được pre-train lớn so với huấn luyện từ đầu.
- **Trạng thái:** ✅ **Hoàn thành trọn vẹn huấn luyện**.
- **Checkpoint lưu tại:** `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_finetuned/checkpoints/`

### 3.3. Baseline 3: Huấn luyện DFlash với Hàm Loss Mới GrowMTP (DCA + VGM)
- **Hàm Loss:** Sử dụng chiến lược **GrowMTP** kết hợp 2 kỹ thuật:
  - **Dynamic Chain Acceptance (DCA):** Điều chỉnh trọng số loss động dựa trên xác suất chấp nhận chuỗi suy luận liên tục, ưu tiên các token kéo dài chuỗi chấp nhận.
  - **Verify-Gated Masking (VGM):** Mặt nạ có điều kiện theo cơ chế xác thực của mô hình đích, giảm thiểu việc phạt các token dự đoán hợp lý nhưng khác biệt nhỏ với teacher.
- **Trạng thái:** ✅ **Hoàn thành trọn vẹn huấn luyện**.
- **Checkpoint lưu tại:** `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_growmtp/checkpoints/`

---

## 4. Các Lỗi Kỹ thuật Đã Được Khắc phục Triệt để

Trong quá trình triển khai, 4 lỗi phát sinh trên server B200 đã được phân tích nguồn gốc và giải quyết theo chuẩn `src/Benchmark/`:

1. **Lỗi Symlink Bảo mật trong `OfflineFeatureDataset`**:
   - *Hiện tượng:* `ValueError: feature path cannot contain symlink component`.
   - *Nguyên nhân:* `src/Finetuning/features.py` chặn đường dẫn chứa symlink. Khi tạo symlink từ Phase 1 sang thư mục mới thì bị crash.
   - *Khắc phục:* Bổ sung cờ chính thức `--feature-cache-dir PATH` vào launcher. Trình nạp nạp trực tiếp đường dẫn thật trên ổ cứng mà không cần tạo symlink.

2. **Lỗi Kiểu Dữ liệu trong `render_prompt`**:
   - *Hiện tượng:* `TypeError: string indices must be integers, not 'str'`.
   - *Nguyên nhân:* Tham số `dataset_name` bị truyền nhầm vào tham số `templates: Mapping[str, str]`.
   - *Khắc phục:* Tạo hàm trợ giúp `prepare_prompts(raw_samples, ds_name, tokenizer)` tự động gán key `dataset` và bọc prompt qua Chat Template của Qwen3.

3. **Lỗi Chữ ký Hàm `add_rouge`**:
   - *Hiện tượng:* `TypeError: add_rouge() got an unexpected keyword argument 'generated'`.
   - *Nguyên nhân:* `add_rouge(record, hyp, ref)` nhận tham số theo vị trí (positional) và sửa `record` in-place.
   - *Khắc phục:* Đổi thành `add_rouge(record, gen_text, ref_text)` và bọc `try...except` để bảo vệ pipeline.

4. **Giải phóng Bộ nhớ Engine VLLM (`shutdown_vllm_engine`)**:
   - *Nguyên nhân:* Khởi tạo 2 đối tượng `LLM(...)` tuần tự trong cùng tiến trình Python làm treo tiến trình nền `EngineCore` và giữ VRAM GPU.
   - *Khắc phục:* Triển khai hàm `shutdown_vllm_engine(llm)` dựa trên `src/Benchmark/vllm_all_baselines.py` gọi `engine_core.shutdown(timeout=30)` và `torch.cuda.empty_cache()` sau mỗi bước.

5. **Tương thích Giao diện Ghi Record (`JsonlWriter.write` và `.close`)**:
   - *Hiện tượng:* `AttributeError: 'JsonlWriter' object has no attribute 'write'`.
   - *Nguyên nhân:* `JsonlWriter` trong `src/Benchmark/common/io_util.py` sử dụng phương thức `add(record)` và `finalize(summary)`, không có `write()` và `close()`.
   - *Khắc phục:* Bổ sung alias `write = add` và `close() -> None` vào `JsonlWriter`, đồng thời trong `scripts/evaluate_vllm_vietbench.py` kiểm tra linh hoạt `hasattr(writer, "add")` và `hasattr(writer, "close")`.

6. **Lỗi Assertion CUTLASS/Triton trên Blackwell B200 (`cudaErrorAssert / 40960`)**:
   - *Hiện tượng:* `Assertion index out of bounds: 0 <= tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK]) < 40960 failed` và crash EngineCore ở `flashinfer_autotune` -> `_dummy_run`.
   - *Nguyên nhân:*
     1. `flashinfer_autotune` chạy dummy queries tới 16,384 tokens qua DFlash speculator, làm vỡ giới hạn tile cumsum của kernel CUTLASS DSL FlashAttention 4 trên B200 (sm_100a).
     2. `target_layer_ids` khi export bị tính sai thành `[0, 7, 14, 22, 29]` thay vì danh sách 5 layers huấn luyện chuẩn `[1, 9, 17, 25, 33]`.
   - *Khắc phục:*
     1. Vô hiệu hoá dummy runs autotune bằng `enable_flashinfer_autotune=False` (giúp khởi động DFlash an toàn và bỏ qua autotune vốn không sinh config mới trên B200).
     2. Bổ sung cờ `--enforce-eager` nếu cần bỏ qua CUDA Graph capture / torch.compile.
     3. Khôi phục cơ chế đọc `target_layer_ids` chính xác từ `config.json` hoặc gọi `build_target_layer_ids(36, 5) -> [1, 9, 17, 25, 33]`.
     4. Tự động tái sử dụng kết quả Vanilla đã đo từ `output-dir` khi chạy với `--skip-vanilla`.

---

## 5. Hướng dẫn Đánh giá Benchmark 3 Baseline trên VietBench

Cả 3 baseline hiện đã **huấn luyện xong thành công**. Tiến hành đánh giá trên 4 bộ test VietBench (100 mẫu/bộ) bằng vLLM:

### 5.1. Đánh giá Baseline 1 (From Scratch)
Chạy trên GPU 0 (hoặc GPU trống bất kỳ):
```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum_Viet-main
git pull origin main

bash scripts/run_evaluate_checkpoint.sh \
  --checkpoint "/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/checkpoints" \
  --output-dir "/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/benchmark_eval" \
  --gpu 0
```
> *Lưu ý: Nếu Vanilla vLLM đã chạy thành công ở lần trước trong thư mục này, thêm `--skip-vanilla` để bỏ qua 10 phút chạy lại Vanilla.*

---

### 5.2. Đánh giá Baseline 2 (Finetuned từ Pretrained)
```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum_Viet-main
git pull origin main

bash scripts/run_evaluate_checkpoint.sh \
  --checkpoint "/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_finetuned/checkpoints" \
  --output-dir "/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_finetuned/benchmark_eval" \
  --gpu 0 \
  --skip-vanilla
```

---

### 5.3. Đánh giá Baseline 3 (GrowMTP Loss)
```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum_Viet-main
git pull origin main

bash scripts/run_evaluate_checkpoint.sh \
  --checkpoint "/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_growmtp/checkpoints" \
  --output-dir "/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_growmtp/benchmark_eval" \
  --gpu 0 \
  --skip-vanilla
```

---

## 6. Ma trận So sánh Kết quả Đánh giá Benchmark (VietBench 400 Mẫu)

Bảng tổng hợp đối đầu giữa 3 phương pháp Speculative Decoding và mô hình gốc:

| Tập dữ liệu | Phương pháp | Trọng số / Loss | Throughput (tok/s) | Speedup | Tỷ lệ chấp nhận (%) | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|---|---|---|---|
| `vietnews` | `vanilla_vllm` | Base Qwen3-4B | ~5,235 tok/s | 1.00x | - | Đang tính | Đang tính | Đang tính |
| `vietnews` | `dflash_scratch` | Baseline 1 (Scratch, $\gamma=7$) | Chờ số liệu | **Chờ số liệu** | Chờ số liệu | Chờ số liệu | Chờ số liệu | Chờ số liệu |
| `vietnews` | `dflash_finetuned`| Baseline 2 (Pretrained Init) | Chờ số liệu | **Chờ số liệu** | Chờ số liệu | Chờ số liệu | Chờ số liệu | Chờ số liệu |
| `vietnews` | `dflash_growmtp` | Baseline 3 (GrowMTP DCA+VGM) | Chờ số liệu | **Chờ số liệu** | Chờ số liệu | Chờ số liệu | Chờ số liệu | Chờ số liệu |
|---|---|---|---|---|---|---|---|---|
| `wikilingua`| `vanilla_vllm` | Base Qwen3-4B | Chờ số liệu | 1.00x | - | Chờ số liệu | Chờ số liệu | Chờ số liệu |
| `wikilingua`| `dflash_scratch` | Baseline 1 (Scratch, $\gamma=7$) | Chờ số liệu | **Chờ số liệu** | Chờ số liệu | Chờ số liệu | Chờ số liệu | Chờ số liệu |
| `wikilingua`| `dflash_finetuned`| Baseline 2 (Pretrained Init) | Chờ số liệu | **Chờ số liệu** | Chờ số liệu | Chờ số liệu | Chờ số liệu | Chờ số liệu |
| `wikilingua`| `dflash_growmtp` | Baseline 3 (GrowMTP DCA+VGM) | Chờ số liệu | **Chờ số liệu** | Chờ số liệu | Chờ số liệu | Chờ số liệu | Chờ số liệu |
|---|---|---|---|---|---|---|---|---|
| `vims` | `vanilla_vllm` | Base Qwen3-4B | Chờ số liệu | 1.00x | - | Chờ số liệu | Chờ số liệu | Chờ số liệu |
| `vims` | `dflash_scratch` | Baseline 1 (Scratch, $\gamma=7$) | Chờ số liệu | **Chờ số liệu** | Chờ số liệu | Chờ số liệu | Chờ số liệu | Chờ số liệu |
| `vims` | `dflash_finetuned`| Baseline 2 (Pretrained Init) | Chờ số liệu | **Chờ số liệu** | Chờ số liệu | Chờ số liệu | Chờ số liệu | Chờ số liệu |
| `vims` | `dflash_growmtp` | Baseline 3 (GrowMTP DCA+VGM) | Chờ số liệu | **Chờ số liệu** | Chờ số liệu | Chờ số liệu | Chờ số liệu | Chờ số liệu |
|---|---|---|---|---|---|---|---|---|
| `vlsp` | `vanilla_vllm` | Base Qwen3-4B | Chờ số liệu | 1.00x | - | Chờ số liệu | Chờ số liệu | Chờ số liệu |
| `vlsp` | `dflash_scratch` | Baseline 1 (Scratch, $\gamma=7$) | Chờ số liệu | **Chờ số liệu** | Chờ số liệu | Chờ số liệu | Chờ số liệu | Chờ số liệu |
| `vlsp` | `dflash_finetuned`| Baseline 2 (Pretrained Init) | Chờ số liệu | **Chờ số liệu** | Chờ số liệu | Chờ số liệu | Chờ số liệu | Chờ số liệu |
| `vlsp` | `dflash_growmtp` | Baseline 3 (GrowMTP DCA+VGM) | Chờ số liệu | **Chờ số liệu** | Chờ số liệu | Chờ số liệu | Chờ số liệu | Chờ số liệu |
