# Tiến độ Huấn luyện & Sổ tay Thực nghiệm DFlash Tiếng Việt (Server B200)

> **Mục tiêu dự án:** Huấn luyện, tối ưu hoá và đánh giá công bằng các mô hình suy luận đầu cơ (Speculative Decoding) họ DFlash cho bài toán **Tóm tắt văn bản tiếng Việt** trên nền mô hình gốc **Qwen3-4B**, so sánh trực tiếp với baseline autoregressive tiêu chuẩn (Vanilla vLLM).

Tài liệu này ghi nhận đầy đủ hiện trạng, các lỗi kỹ thuật đã xử lý, đường dẫn dữ liệu/mô hình trên cụm B200, và toàn bộ lệnh chạy huấn luyện cũng như đánh giá để các phiên làm việc tiếp theo dễ dàng theo dõi và tái lập.

---

## 1. Bảng Tổng quan Tiến độ 3 Baseline

| Baseline | Kiến trúc / Phương pháp | Trọng số Init | Hàm Loss | Trạng thái Train | Trạng thái Eval |
|---|---|---|---|---|---|
| **Baseline 1** | DFlash (5 layers, b16) | From Scratch (ngẫu nhiên) | DFlash chuẩn ($\gamma=7.0$) | ✅ **Hoàn thành 6 epochs** (6162 steps) | ⏳ Sẵn sàng chạy Eval trên GPU 0 |
| **Baseline 2** | DFlash (5 layers, b16) | `Qwen3-4B-DFlash-b16` gốc | DFlash chuẩn ($\gamma=7.0$) | 🚀 Sẵn sàng chạy (1-click script) | ⏳ Sẵn sàng chạy sau khi có checkpoint |
| **Baseline 3** | DFlash (5 layers, b16) | From Scratch (ngẫu nhiên) | **GrowMTP** (DCA + VGM) | 🚀 Sẵn sàng chạy (1-click script) | ⏳ Sẵn sàng chạy sau khi có checkpoint |

---

## 2. Hệ thống Đường dẫn Chuẩn (Canonical Paths) trên Cụm Server B200

| Thành phần | Đường dẫn Tuyệt đối trên B200 | Ghi chú |
|---|---|---|
| **Repo Root (Mã nguồn)** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum_Viet-main` | Nhánh `main` đồng bộ với GitHub |
| **Master Config** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master_Viet.env` | Chứa cấu hình môi trường offline |
| **Target Model (Teacher)** | `/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B` | Mô hình mục tiêu Qwen3-4B |
| **Pretrained DFlash gốc** | `/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B-DFlash-b16` | Checkpoint DFlash đa ngôn ngữ có sẵn |
| **Tập Train sạch (36,973 mẫu)** | `/workspace/storage-shared/nlp/dungdx4/bien_projects/LLM2Seq/src/eviseq_new/datasets/50k/train_clean.jsonl` | Đã sinh teacher & trích xuất features |
| **Bộ Eval Chuẩn (VietBench 100)** | `datasets/eval_100/` (`vietnews`, `wikilingua`, `vims`, `vlsp`) | 4 bộ test chuẩn hoá, 100 mẫu/bộ |
| **Phase 1 Cache (Features)** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/features/` | Tensors sharded 5 target layers |
| **Phase 1 Cache (Teacher)** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/teacher/` | Trajectories sinh bởi Qwen3-4B |
| **Output Baseline 1 (Scratch)** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/` | Checkpoint lưu tại `checkpoints/` |
| **Output Baseline 2 (Finetuned)** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_finetuned/` | Checkpoint lưu tại `checkpoints/` |
| **Output Baseline 3 (GrowMTP)** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_growmtp/` | Checkpoint lưu tại `checkpoints/` |

---

## 3. Nhật ký Huấn luyện Chi tiết Baseline 1 (From Scratch)

- **Cấu hình Huấn luyện:** 5 Draft Layers, Block Size 16, Mask Token ID 151669, 2x B200 GPUs qua PyTorch DDP.
- **Tổng số bước:** 6162 steps (chính xác 6 epochs trọn vẹn trên 36,973 mẫu tiếng Việt).
- **Thời gian thực thi:** 2 giờ 29 phút 17 giây (Tốc độ trung bình: 1.45s / step, thông lượng ~25,924 tokens/s).
- **Hội tụ Loss & Accuracy:**
  - `Step 1`: `loss = 12.5981`, `acc = 0.0%`
  - `Step 500`: `loss = 3.6521`, `acc = 22.4%`
  - `Step 3000`: `loss = 2.4510`, `acc = 35.8%`
  - `Step 6000`: `loss = 2.0366`, `acc = 42.4%` (lr = 9.34e-08)
  - `Step 6162 (End)`: `loss = 2.0608`, `acc = 40.4%` (lr = 0.0)
- **Checkpoint tốt nhất:**
  - `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/checkpoints/qwen3-4b-phase1-step6000/`

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

## 5. Hướng dẫn Lệnh Thực thi Chuẩn (Standard Operating Procedures)

### 5.1. Đánh giá Benchmark Baseline 1 (Checkpoint step 6000)

Chạy trên GPU 0 (hoặc bất kỳ GPU nào còn trống):

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum_Viet-main
git pull

bash scripts/run_evaluate_checkpoint.sh \
  --checkpoint "/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/checkpoints/qwen3-4b-phase1-step6000" \
  --output-dir "/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/benchmark_eval" \
  --gpu 0
```

> **Kết quả đầu ra:** File markdown tổng hợp `evaluation_summary.md` và file log chi tiết từng mẫu `vanilla_vllm_records.jsonl`, `dflash_spec_records.jsonl`.

---

### 5.2. Huấn luyện & Đánh giá Baseline 2 (Finetune từ Pretrained)

#### A. Khởi chạy Huấn luyện (6 epochs):
```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum_Viet-main
git pull

# Chạy trên 2 GPU (GPU 0, 1):
bash scripts/run_phase2_finetune_from_pretrained.sh --gpus 0,1

# Hoặc chạy trên 1 GPU (GPU 0):
bash scripts/run_phase2_finetune_from_pretrained.sh --gpus 0
```

#### B. Đánh giá sau khi Huấn luyện xong:
```bash
# Thêm --skip-vanilla để không chạy lại Vanilla (tiết kiệm 10 phút)
bash scripts/run_evaluate_checkpoint.sh \
  --checkpoint "/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_finetuned/checkpoints/qwen3-4b-finetuned-step6000" \
  --output-dir "/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_finetuned/benchmark_eval" \
  --gpu 0 \
  --skip-vanilla
```

---

### 5.3. Huấn luyện & Đánh giá Baseline 3 (Hàm Loss Mới GrowMTP)

#### A. Khởi chạy Huấn luyện:
```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum_Viet-main
git pull

# Chạy trên 1 GPU (GPU 0):
bash scripts/run_phase2_train_growmtp.sh --gpus 0

# Hoặc chạy trên 2 GPU (GPU 0, 1):
bash scripts/run_phase2_train_growmtp.sh --gpus 0,1
```

#### B. Đánh giá sau khi Huấn luyện xong:
```bash
bash scripts/run_evaluate_checkpoint.sh \
  --checkpoint "/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_growmtp/checkpoints/qwen3-4b-growmtp-step6000" \
  --output-dir "/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_growmtp/benchmark_eval" \
  --gpu 0 \
  --skip-vanilla
```

---

## 6. Định dạng Bảng Kết quả Đánh giá Benchmark

Sau khi chạy xong lệnh `run_evaluate_checkpoint.sh`, báo cáo sẽ xuất hiện dưới dạng bảng so sánh chuẩn:

| Tập dữ liệu | Phương pháp | Số mẫu | Throughput (tok/s) | Speedup | Tỷ lệ chấp nhận (%) | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|---|---|---|---|
| `vietnews` | `vanilla_vllm` | 100 | ~3,689 tok/s | 1.00x | - | Đang tính | Đang tính | Đang tính |
| `vietnews` | `dflash_spec` | 100 | Chờ số liệu | **Chờ số liệu** | Chờ số liệu | Chờ số liệu | Chờ số liệu | Chờ số liệu |
| `wikilingua`| `vanilla_vllm` | 100 | Chờ số liệu | 1.00x | - | Chờ số liệu | Chờ số liệu | Chờ số liệu |
| `wikilingua`| `dflash_spec` | 100 | Chờ số liệu | **Chờ số liệu** | Chờ số liệu | Chờ số liệu | Chờ số liệu | Chờ số liệu |
| `vims` | `vanilla_vllm` | 100 | Chờ số liệu | 1.00x | - | Chờ số liệu | Chờ số liệu | Chờ số liệu |
| `vims` | `dflash_spec` | 100 | Chờ số liệu | **Chờ số liệu** | Chờ số liệu | Chờ số liệu | Chờ số liệu | Chờ số liệu |
| `vlsp` | `vanilla_vllm` | 100 | Chờ số liệu | 1.00x | - | Chờ số liệu | Chờ số liệu | Chờ số liệu |
| `vlsp` | `dflash_spec` | 100 | Chờ số liệu | **Chờ số liệu** | Chờ số liệu | Chờ số liệu | Chờ số liệu | Chờ số liệu |
