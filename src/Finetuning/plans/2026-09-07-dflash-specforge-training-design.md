# Thiết kế port quy trình training DFlash của SpecForge

**Ngày:** 2026-09-07  
**Trạng thái:** Đã được duyệt để lập implementation plan  
**Experiment directory:** `src/Finetuning/`

## Mục tiêu

Port độc lập quy trình training DFlash được triển khai trong
`externals/SpecForge` vào `src/Finetuning/`, để sau này huấn luyện draft model
cho task tóm tắt tiếng Việt. Giai đoạn đầu chỉ reproduce faithful DFlash cho
Qwen3-4B và Qwen3-8B trên một GPU, không đưa các thay đổi MR-DFlash vào.

Đây là training draft model, không phải fine-tune target model. Target Qwen3
được giữ frozen; draft học cách dự đoán các token tiếp theo từ target hidden
states theo objective block-diffusion của DFlash.

## Nguồn sự thật và nguyên tắc fidelity

Các file sau trong SpecForge là nguồn tham chiếu trực tiếp:

- `externals/SpecForge/specforge/modeling/draft/dflash.py`
- `externals/SpecForge/specforge/modeling/draft/dflash_kernels.py`
- `externals/SpecForge/specforge/algorithms/common/dflash_family_model.py`
- `externals/SpecForge/specforge/algorithms/common/dflash_family_data.py`
- `externals/SpecForge/specforge/algorithms/dflash/providers.py`
- `externals/SpecForge/specforge/algorithms/model_providers.py`
- `externals/SpecForge/specforge/training/strategies/base.py`
- `externals/SpecForge/specforge/training/trainer.py`
- `externals/SpecForge/specforge/training/assembly.py`
- `externals/SpecForge/specforge/training/model_loading.py`
- `externals/SpecForge/specforge/training/checkpoint.py`
- `externals/SpecForge/specforge/training/schedule.py`
- `externals/SpecForge/specforge/config/schema.py`
- `externals/SpecForge/specforge/data/preprocessing.py`
- `externals/SpecForge/specforge/data/loss_mask.py`
- `externals/SpecForge/specforge/runtime/data_plane/offline_reader.py`

`src/Finetuning` sẽ là bản copy self-contained của phần cần cho DFlash offline,
không import ngược từ `src/MR_DFlash` và không phụ thuộc runtime vào
`externals/SpecForge`. Các khác biệt vì cấu trúc repo hiện tại phải được ghi
trong implementation plan và kiểm tra bằng contract tests.

## Phạm vi phase 1

### Bao gồm

- DFlash draft backbone tương thích `Qwen3Config`, gồm attention, RoPE, RMSNorm,
  MLP, projector và layer layout của SpecForge.
- DFlash-family model wrapper và objective `dflash`; giữ các tùy chọn loss
  D-PACE mà upstream expose qua cùng wrapper nếu không làm thay đổi semantics
  mặc định.
- Data normalization, loss mask assistant/summary, feature manifest, offline
  reader và collator cho contract `input_ids`, `loss_mask`, `hidden_states`.
- Target embedding và LM head frozen; draft initialization, model assembly,
  optimizer, gradient accumulation, warmup/cosine schedule, clipping, logging,
  checkpoint/resume và final evaluation.
- Attention backends upstream hỗ trợ cho DFlash (`eager`, `sdpa`,
  `flex_attention`), với backend an toàn cho CPU smoke và backend GPU tương
  thích khi chạy thật.
- CLI/config cho hai target đầu tiên:
  - `Qwen/Qwen3-4B`
  - `Qwen/Qwen3-8B`
- Offline single-GPU launch trước. Các abstraction cần thiết cho DDP sẽ được
  giữ ở ranh giới trainer nhưng DDP chưa là acceptance criterion của phase 1.
- Synthetic fixture để kiểm thử end-to-end trong điều kiện hiện chưa có
  dataset tiếng Việt.

### Không bao gồm trong phase 1

- Dataset tiếng Việt thật hoặc download dataset qua internet.
- Fine-tune target model.
- MR-DFlash, Domino, DSpark, EAGLE3 hoặc các biến thể ngoài DFlash.
- Online/disaggregated capture bằng SGLang/Mooncake.
- DDP/multi-GPU execution, FSDP, USP và model parallel.
- Llama 3.1. Llama sẽ là phase 2 với một DFlash backbone dùng Llama config,
  vì SpecForge hiện chỉ có Llama implementation cho Eagle3.
- Claim về ROUGE hoặc chất lượng tóm tắt tiếng Việt khi chưa có dữ liệu thật.

## Kiến trúc và luồng dữ liệu

```text
local JSONL document/summary
        │
        ▼
text preparation + assistant loss_mask
        │
        ▼
offline feature capture/normalization
  input_ids + loss_mask + hidden_states
        │
        ▼
manifest reader + padding collator
        │
        ▼
frozen target embed/lm_head + DFlash draft
        │
        ▼
anchor sampling → masked blocks → block attention
        │
        ▼
DFlash CE objective → backward → AdamW step
        │
        ▼
metrics + checkpoint/resume + evaluation
```

Code trong `src/Finetuning` sẽ được chia theo boundary của upstream thay vì
gộp thành một script lớn:

| Boundary | Trách nhiệm |
|---|---|
| `modeling/` | DFlash draft model và kernel factory; giữ checkpoint key/layout tương thích upstream |
| `algorithms/` | anchor sampling, mask, block forward, DFlash/D-PACE objective và feature contract |
| `data.py`/`prepare_data.py` | JSONL summary adapter, chat rendering, assistant loss mask và feature preparation |
| `training/` | strategy, trainer lifecycle, schedule, checkpoint, metrics và evaluation |
| `config.py` | typed config, validation, Qwen3 defaults và CLI/YAML mapping |
| `run_train.py` | single-GPU offline entry point |
| `configs/` | config reproducible cho Qwen3-4B và Qwen3-8B |
| `tests/` | deterministic unit, contract và end-to-end synthetic tests |

Core code không import test/validation code. Validation script chỉ quan sát
training entry point và artifact bên ngoài.

## DFlash semantics phải giữ nguyên

Với một batch sequence độ dài `S`:

1. Chỉ chọn anchor tại vị trí `t` khi `loss_mask[t]` và `loss_mask[t+1]` đều
   được supervise; số anchor tối đa là `num_anchors`.
2. Mỗi anchor tạo một block dài `block_size`: offset 0 dùng embedding token
   anchor, các offset còn lại dùng embedding `mask_token` của target.
3. Draft query được attend context thật trước anchor và draft token trong cùng
   block; không attend sang block khác. Ở layer `full_attention`, mọi draft
   offset trong cùng block đều visible; ở layer `sliding_attention`, draft
   visibility vẫn causal theo offset (`kv_offset <= q_offset`).
4. Offset `k` dự đoán token thật tại `anchor + k`; offset 0 bị loại khỏi loss.
   Weight còn chịu bounds, block validity, `loss_mask` và positional decay
   `exp(-(k-1)/loss_decay_gamma)` nếu bật.
5. Logits dùng frozen target LM head, loss mặc định là hard-label cross
   entropy. Accuracy và simulated acceptance được ghi riêng, không dùng để
   thay đổi gradient.
6. Target hidden states, target embedding và target LM head không nhận gradient;
   chỉ draft parameters được optimizer cập nhật.

Các tensor shape, checkpoint keys, config fields và metric names cần được test
để bảo đảm không vô tình chuyển sang semantics của `src/MR_DFlash`.

## Dữ liệu và contract tóm tắt

Hiện chưa có dataset tiếng Việt. Adapter phase 1 sẽ nhận JSONL local dạng tối
thiểu:

```json
{"id": "sample-001", "document": "...", "summary": "..."}
```

Adapter render document và summary thành một causal-LM sequence theo tokenizer
đã chọn. `loss_mask=0` cho instruction/document/prompt; `loss_mask=1` cho
summary assistant span. Nếu sequence sau truncate không còn ít nhất hai token
summary liên tiếp, sample bị loại với lý do rõ ràng.

Offline feature artifact phải chứa:

- `input_ids`: token ids của toàn sequence;
- `loss_mask`: mask cùng chiều sequence;
- `hidden_states`: concat hidden states tại các target layer đã resolve;
- manifest ghi target model/revision, tokenizer, max length, layer ids và số
  chiều feature.

Feature reader phải từ chối artifact sai schema, sai sequence length, sai
feature width hoặc không có cặp token supervise liên tiếp. Không tự động tải
model/dataset từ internet.

## Evaluation và validation

Phase 1 không đánh giá ROUGE tiếng Việt thật. Evaluation core sẽ dùng chung cho
hai entry mode:

- in-memory model trong lúc training;
- checkpoint đã lưu bằng CLI riêng.

Metric tối thiểu là `loss`, supervised token accuracy, số token hợp lệ và
simulated acceptance/acceptance length nếu input fixture hỗ trợ. Trainer quyết
định thời điểm evaluation theo step; mặc định smoke có evaluation cuối run,
config production có thể đặt `eval_interval` theo optimizer step.

Mỗi evaluation phải có phase-start message, progress indicator, phase-end
message, result summary và efficiency summary. Các lỗi checkpoint thiếu/hỏng,
restore thất bại, validation loader rỗng/sai, metric không aggregate được,
metric non-finite hoặc evaluation bị stall phải fail rõ ràng.

Acceptance phase 1:

- unit/contract tests chạy được trên CPU với synthetic tensors;
- Qwen3-4B one-step smoke chạy được nếu snapshot local có sẵn;
- tiny overfit run trên synthetic feature data có loss hữu hạn và giảm theo
  step, không NaN/Inf;
- checkpoint final load lại được và evaluation checkpoint cho kết quả hợp lệ;
- log có step, loss, grad norm, learning rate, throughput/progress và đường dẫn
  checkpoint;
- không claim quality tiếng Việt trước khi có dataset thật.

## Phase sau

Sau khi phase 1 pass, các thay đổi được tách thành các phase riêng:

1. DDP multi-GPU cho cùng offline feature contract và deterministic resume.
2. Online/disaggregated capture tương thích SGLang/Mooncake của SpecForge.
3. Llama 3.1 DFlash backbone cho base và instruct, gồm Llama RoPE/MLP/norm,
   draft config, weight mapping và parity tests với Qwen3 implementation.
4. Dataset tiếng Việt thật, ROUGE evaluation và benchmark generation.

Mỗi phase phải có design/plan riêng hoặc impact update trước khi sửa code của
phase trước.

## Rủi ro và cách kiểm soát

- **Transformers API lệch upstream:** khóa contract tests cho Qwen3 config,
  attention output, checkpoint keys và target layer extraction.
- **Mask sai nhưng loss vẫn chạy:** test từng mask row/block và test gradient
  chỉ đi qua draft parameters.
- **Loss mask summary sai:** test prompt/document không contribute gradient và
  summary span giữ được hai token liên tiếp sau truncate.
- **Offline artifact trộn model/layer:** manifest validation bắt buộc trước khi
  load model lớn.
- **Thiếu dataset thật:** synthetic fixture chỉ dùng để kiểm tra pipeline, không
  được báo cáo như kết quả tóm tắt tiếng Việt.
- **CPU dev không có CUDA:** dùng eager/sdpa và tensor nhỏ cho smoke; không cố
  sửa môi trường GPU T4 theo quy định repo.
