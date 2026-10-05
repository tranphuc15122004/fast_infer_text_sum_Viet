# Tài liệu Cấu hình Đường dẫn & Artifacts trên Server B200

Tài liệu này lưu trữ thông tin canonical về cấu trúc thư mục làm việc, dữ liệu Teacher Regeneration, Feature Cache và Training Checkpoint trên **Server B200** để tra cứu và sử dụng lâu dài.

---

## 1. Đường dẫn Canonical trên Server B200

| Mục đích | Đường dẫn Canonical trên B200 |
|---|---|
| **Repository Root** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum_Viet` |
| **Shared Data & Config** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/data` |
| **Master Config** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master_Viet.env` |
| **Target Model (Qwen3-4B)** | `/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B` |
| **Train Dataset Gốc (50k Clean)** | `/workspace/storage-shared/nlp/dungdx4/bien_projects/LLM2Seq/src/eviseq_new/datasets/50k/train_clean.jsonl` |
| **Offline Wheelhouse** | `/workspace/storage-shared/nlp/dungdx4/phuc_projects/offline_wheelhouse` |

---

## 2. Thư mục Output Root, Regen và Cache

Mọi artifact sinh ra từ quy trình Fine-tuning Phase 1 (Regeneration + Feature Caching) và Phase 2 (Train DFlash) đều nằm trong thư mục output mục tiêu:

```text
TARGET_ROOT="/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet"
```

### Cấu trúc chi tiết:

```text
/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/
│
├── 📁 teacher/                               # [REGEN] Dữ liệu sinh từ Teacher Model
│   ├── train.jsonl                          # Trajectory sinh cho tập train (36,973+ mẫu)
│   ├── eval.jsonl                           # Trajectory sinh cho tập eval
│   ├── train_validation_report.json         # Báo cáo lọc anomaly & ROUGE-1 tập train
│   └── eval_validation_report.json          # Báo cáo lọc anomaly & ROUGE-1 tập eval
│
├── 📁 features/                              # [CACHE] Hidden-state Tensors đã trích xuất
│   ├── train/
│   │   ├── manifest.json                    # Manifest xác thực cache tập train
│   │   └── feature_*.pt                     # Các tensor hidden states (sharded)
│   └── eval/
│       ├── manifest.json                    # Manifest xác thực cache tập eval
│       └── feature_*.pt
│
├── 🎯 checkpoints/                           # [TRAIN OUTPUT] Checkpoint DFlash qua các epoch
│   ├── qwen3-4b-phase1-stepXXXX/            # Checkpoint các step trung gian
│   │   ├── model.pt (hoặc weights)
│   │   └── extra.json                       # Metadata batch adaptive và VRAM peak
│   └── qwen3-4b-phase1-stepXXXX/COMPLETE    # Checkpoint hoàn tất 6 epochs
│
├── 📋 logs/                                  # Log chi tiết từng công đoạn
│   ├── generate_train.log / generate_eval.log
│   ├── validate_teacher_train.log / validate_teacher_eval.log
│   ├── cache_train.log / cache_eval.log
│   ├── train.log                            # Log tiến độ huấn luyện Phase 2
│   └── console.log                          # Tổng hợp console output
│
├── ⚙️ .state/                                # Marker trạng thái hỗ trợ tự động Resume
│   ├── generate_train.json
│   ├── generate_eval.json
│   ├── validate_teacher_train.json
│   ├── validate_teacher_eval.json
│   ├── cache_train.json
│   ├── cache_eval.json
│   ├── train.json                           # Đánh dấu Phase 2 hoàn thành
│   └── .run.lock                            # File lock chống ghi đè tiến trình
│
└── 📄 run_config.yaml                        # Configuration YAML đã materialize cho run
```

---

## 3. Các Lệnh Theo Dõi & Quản Lý

### Theo dõi tiến độ Train (Phase 2):
```bash
# Xem log huấn luyện thời gian thực
tail -n 100 -f /workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/logs/train.log

# Kiểm tra danh sách checkpoint đã lưu
ls -la /workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/checkpoints/

# Kiểm tra trạng thái hoàn thành Phase 2
cat /workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet/.state/train.json 2>/dev/null || echo "Đang chạy..."
```

### Resume lại tiến trình nếu bị gián đoạn:
Launcher hỗ trợ resume thông minh dựa trên `.state/` và `manifest.json`. Chỉ cần chạy lại đúng lệnh ban đầu, các bước đã hợp lệ sẽ tự động được `SKIP`:

```bash
bash scripts/run_phase2_train_b200.sh \
  --gpus 0,1 \
  --epochs 6 \
  --output-root "/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/qwen3_4b_phase1_Viet"
```
