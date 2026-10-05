# Tiến độ Huấn luyện & Nhật ký Thực nghiệm DFlash Tiếng Việt (Server B200)

Tài liệu này tổng hợp toàn bộ tiến độ, các mốc thực nghiệm, cấu trúc đường dẫn canonical trên server B200, và hướng dẫn tái lập/đánh giá các baseline cho bài toán **Tăng tốc Tóm tắt Văn bản dài Tiếng Việt bằng DFlash**.

---

## 1. Bảng Trạng thái Tiến độ Tổng thể

| Hạng mục / Giai đoạn | Phương pháp | Trạng thái | Kết quả chính / Chỉ số |
|---|---|---|---|
| **Phase 1: Teacher Generation** | Qwen3-4B (Greedy offline) | ✅ **100% Hoàn thành** | 36,973 mẫu train clean & 100 mẫu eval |
| **Phase 1: Feature Caching** | Trích xuất 5 Target Layers | ✅ **100% Hoàn thành** | Đã lưu tensor sharded trong `features/` |
| **Baseline 1: Train From Scratch** | DFlash Loss chuẩn (Decay $\gamma=7.0$) | ✅ **100% Hoàn thành** | **6162/6162 steps (6 epochs)**, Loss: $12.59 \to 2.06$, Acc: **40.4%** |
| **Đánh giá Benchmark Baseline 1** | vLLM Speculative Decoding | ⏳ **Sẵn sàng chạy** | Đo trên VietBench (`vietnews`, `wikilingua`, `vims`, `vlsp`) |
| **Baseline 2: Finetune từ DFlash gốc** | Init từ `Qwen3-4B-DFlash-b16` | 🔄 Đã triển khai code | Sẵn sàng chạy với `--draft-init-path` |
| **Baseline 3: Nghiên cứu Loss GrowMTP** | DCA (Chain Acceptance) + VGM | 🔄 Đã triển khai & Test 100% | Sẵn sàng chạy với `--loss-type growmtp` |

---

## 2. Hệ thống Đường dẫn Canonical trên Server B200

| Mục đích | Đường dẫn Tuyệt đối trên B200 |
|---|---|
| **Mã nguồn Dự án (Repo Root)** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum_Viet-main` |
| **Master Config chuẩn** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master_Viet.env` |
| **Target Model (Teacher)** | `/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B` |
| **Pretrained DFlash Model gốc** | `/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B-DFlash-b16` |
| **Tập dữ liệu Huấn luyện gốc** | `/workspace/storage-shared/nlp/dungdx4/bien_projects/LLM2Seq/src/eviseq_new/datasets/50k/train_clean.jsonl` |
| **Tập dữ liệu Eval Chuẩn (VietBench 100)** | `datasets/eval_100/` (`vietnews_100.jsonl`, `wikilingua_100.jsonl`, `vims_100.jsonl`, `vlsp_100.jsonl`) |
| **Output Baseline 1 (From Scratch)** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/` |
| **Output Baseline 2 (Finetune từ gốc)** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_finetuned/` |
| **Output Baseline 3 (GrowMTP Loss)** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_growmtp/` |

---

## 3. Nhật ký Huấn luyện Baseline 1 (DFlash From Scratch)

- **Cấu hình:** 5 Draft Layers, Block Size 16, Mask Token ID 151669, 2x B200 GPUs (DDP).
- **Tổng số bước:** 6162 steps (tương ứng trọn vẹn 6 epochs trên 36,973 mẫu tiếng Việt).
- **Thời gian chạy:** 2 giờ 29 phút 17 giây (Trung bình: 1.45s / step, tốc độ xử lý ~25,924 tokens/s).
- **Hội tụ:**
  - Step 1: `loss = 12.5981`, `accuracy = 0.0%`
  - Step 94: `loss = 6.1919`, `accuracy = 8.8%`
  - Step 6000: `loss = 2.0366`, `accuracy = 42.4%`
  - Step 6162: `loss = 2.0608`, `accuracy = 40.4%`
- **Checkpoint khả dụng:**
  - `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/checkpoints/qwen3-4b-phase1-step6000/`

---

## 4. Các Công cụ & Tính năng Đã Tích hợp vào Repo

### 4.1. Bộ Đánh giá Benchmark Tự động với vLLM (`scripts/`)
- [`scripts/export_trained_checkpoint.py`](file:///home/tuantb/fast_infer_text_sum_Viet/scripts/export_trained_checkpoint.py): Chuyển đổi trọng số `draft_state_dict.pt` sang định dạng `model.safetensors` và `config.json` tiêu chuẩn của HuggingFace/vLLM.
- [`scripts/evaluate_vllm_vietbench.py`](file:///home/tuantb/fast_infer_text_sum_Viet/scripts/evaluate_vllm_vietbench.py): Chạy vLLM Speculative Decoding trên 4 bộ test VietBench, tính throughput, speedup ratio, acceptance rate, và gọi trực tiếp `Benchmark.common.rouge` để chấm điểm ROUGE-1/2/L.
- [`scripts/run_evaluate_checkpoint.sh`](file:///home/tuantb/fast_infer_text_sum_Viet/scripts/run_evaluate_checkpoint.sh): Wrapper launcher 1-click tự động tìm checkpoint mới nhất, export và chạy benchmark.

### 4.2. Hỗ trợ Finetune từ Trọng số DFlash gốc (`--draft-init-path`)
- Cho phép nạp trực tiếp trọng số từ checkpoint pretrained `Qwen3-4B-DFlash-b16` (hỗ trợ cả `.safetensors` và `.pt`).
- Tự động bỏ qua `--resume-from` khi finetune từ model mới.

### 4.3. Nghiên cứu Hàm Loss GrowMTP (DCA + VGM) (`--loss-type growmtp`)
- **DCA (Dynamic Chain Acceptance):** Tối ưu hóa trực tiếp kỳ vọng độ dài chuỗi được chấp nhận thông qua logsumexp chuỗi xác suất tích lũy:
  $$L_{\mathrm{DCA}} = -\mathrm{logsumexp}\left(-\mathrm{cumsum}(\text{neg\_log\_q})\right)$$
- **VGM (Verify-Gated Masking):** Cắt loss tại vị trí từ chối đầu tiên $j$. Giữ lại gradient tại $j$ để sửa sai, triệt tiêu gradient ($0.0$) ở tất cả các vị trí sau $j$.
- Đã được verify qua unit tests trong `src/Finetuning/tests/test_objective.py`.

---

## 5. Hướng dẫn Lệnh Thực thi trên Server B200

### 5.1. Chạy Đánh giá Benchmark Baseline 1 (From Scratch)
```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum_Viet-main
git pull

# Chạy đánh giá trên GPU 0
bash scripts/run_evaluate_checkpoint.sh \
  --checkpoint "/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/checkpoints/qwen3-4b-phase1-step6000" \
  --output-dir "/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/benchmark_eval" \
  --gpu 0
```

### 5.2. Chạy Huấn luyện Baseline 2 (Finetune từ DFlash gốc)
Cách 1-click (khuyên dùng):
```bash
bash scripts/run_phase2_finetune_from_pretrained.sh --gpus 0,1
```

### 5.3. Chạy Huấn luyện Baseline 3 (Nghiên cứu Loss GrowMTP)
Chạy trên 1 GPU (GPU 0 trống):
```bash
bash scripts/run_phase2_train_growmtp.sh --gpus 0
```
Hoặc nếu có cả 2 GPU:
```bash
bash scripts/run_phase2_train_growmtp.sh --gpus 0,1
```
Script sẽ tự động symlink feature cache, cấu hình loss GrowMTP (DCA + VGM), và lưu checkpoint vào `/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_dflash_growmtp/checkpoints/`.

