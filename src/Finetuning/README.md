# Fine-tune DFlash cho tóm tắt tiếng Việt

`src/Finetuning` là pipeline DFlash độc lập cho Qwen3. Nó không import từ
`src/MR_DFlash`; target Qwen luôn frozen, còn optimizer chỉ cập nhật
`DFlashDraftModel`.

Điều cần phân biệt: DFlash học để tăng tốc chính target, không làm target có
chất lượng tóm tắt cao hơn. Vì vậy pipeline train trên **summary do chính
target sinh**, nhưng giữ gold summary của người viết để đo ROUGE sau cùng.

## Quy trình thật

Input gốc UTF-8 JSONL:

```json
{"id":"vi-001","document":"...","summary":"gold summary"}
```

Chạy ba stage riêng biệt. Mọi model đều phải là snapshot local khi
`offline: true`.

```bash
# 1. Sinh trajectory greedy của frozen target. summary trong file output là
#    target trajectory; reference_summary vẫn là gold của dữ liệu gốc.
PYTHONPATH=src python3 -m Finetuning.generate_targets \
  --input /data/vietnamese_train.jsonl \
  --output /work/teacher_train.jsonl \
  --target-model-path /models/Qwen3-4B \
  --max-length 2048 --max-source-tokens 1536 --max-summary-tokens 384 \
  --torch-dtype bfloat16 --device cuda \
  --adaptive-batch --target-memory-fraction 0.90 \
  --adaptive-max-batch-size 256 --bucket-window 512

# 2. Capture hidden state một lần. --num-draft-layers=5 sẽ chọn cùng rule
#    target layer với config khi target_layer_ids: null.
PYTHONPATH=src python3 -m Finetuning.capture_features \
  --input /work/teacher_train.jsonl \
  --output /work/features_train \
  --target-model-path /models/Qwen3-4B \
  --num-draft-layers 5 \
  --max-length 2048 --max-source-tokens 1536 --max-summary-tokens 384 \
  --torch-dtype bfloat16 --device cuda \
  --adaptive-batch --target-memory-fraction 0.90 \
  --adaptive-max-batch-size 256 --bucket-window 512

# Lặp lại stage 1–2 cho validation và thay các path trong config.
PYTHONPATH=src python3 -m Finetuning.run_train \
  --config src/Finetuning/configs/qwen3_4b.yaml --device cuda
```

`run_train` chỉ nhận `data.hidden_states_path` và
`data.eval_hidden_states_path`; nó chủ động từ chối raw JSONL. Nhờ vậy capture
không đồng thời giữ hai bản target trên GPU. Manifest cache kiểm tra nghiêm
ngặt target/tokenizer, target layers, dtype, prompt template và các budget
trước khi train. Cache được publish atomic; có thể chạy lại capture an toàn
sau lỗi, nhưng phải dùng thư mục output chỉ dành cho feature store.

## Chạy một job dài end-to-end trên server B200

Để chạy tự động cả regenerate train/eval, capture feature train/eval và DFlash
training, dùng launcher ở repo root:

```bash
bash scripts/run_finetuning_b200.sh \
  --config src/Finetuning/configs/qwen3_4b.yaml \
  --train-input /server/data/vietnamese_train.jsonl \
  --eval-input /server/data/vietnamese_eval.jsonl \
  --target-model-path /server/models/Qwen3-4B \
  --output-root /server/work/dflash-qwen3-4b \
  --nproc-per-node 8
```

Nếu bỏ `--nproc-per-node`, launcher tự lấy số GPU từ `nvidia-smi -L`. Có thể
chọn interpreter khác bằng `FINETUNING_PYTHON=/path/to/python`.

Launcher chạy theo thứ tự cố định:

```text
generate_train → generate_eval → cache_train → cache_eval → train
```

Mỗi stage có log riêng trong `output-root/logs/` và marker trong
`output-root/.state/`. Nếu job bị ngắt hoặc máy mất kết nối, chạy lại đúng
cùng lệnh; các stage đã có artifact hợp lệ sẽ được bỏ qua, stage chưa hoàn tất
sẽ chạy tiếp. `output-root/.run.lock` ngăn hai job cùng ghi một run. Không đổi
model, input, số GPU hoặc output root khi resume; launcher sẽ từ chối nếu
`run_manifest.json` không còn khớp.

Artifact sau khi chạy:

```text
output-root/
├── teacher/{train,eval}.jsonl
├── features/{train,eval}/manifest.json
├── checkpoints/<run-id>-stepN/COMPLETE
├── logs/{generate_*,cache_*,train}.log
└── run_config.yaml
```

Batch preparation và training đều bật adaptive batching; mặc định target là
90% VRAM, batch tối đa 256 mẫu/GPU. Có thể điều chỉnh khi khởi chạy:

```bash
  --target-memory-fraction 0.90 \
  --adaptive-max-batch-size 256 \
  --bucket-window 512 \
  --probe-batches 2
```

Metadata batch đã chọn và peak VRAM nằm trong `extra.json` của checkpoint.
Launcher không đọc `datasets/`; toàn bộ dữ liệu đi qua hai đường dẫn JSONL
được truyền bằng `--train-input` và `--eval-input`.

## Adaptive batching cho regeneration và caching

Trên CUDA, hai phase chuẩn bị tự bật adaptive batching. Mỗi rank chỉ giữ một
window giới hạn (`bucket-window`), sắp xếp sample theo độ dài, rồi tạo các batch
được padding tối thiểu. Batch khởi đầu được probe bằng chính
`generate`/forward thật và giữ dưới `target-memory-fraction` của VRAM. Nếu một
bucket dài hơn gây CUDA OOM, batch được giảm một nửa và retry; lỗi không phải
OOM sẽ không bị che giấu.

Các tham số chính:

```text
--adaptive-max-batch-size 256   # cap số sample/GPU
--max-tokens-per-batch 0        # 0 = chỉ giới hạn bởi VRAM probe
--bucket-window 512             # giới hạn RAM CPU dùng để bucketing
--probe-batches 2               # số lần warm-up khi probe
--no-adaptive-batch             # baseline một sample/lần
```

Kết quả cuối phase in JSON gồm số sample/token, throughput, số lần OOM retry,
batch đã chọn, peak reserved VRAM và VRAM target. `max-tokens-per-batch` nên
được đặt nếu RAM host hạn chế; generation ước lượng cả prompt và số token tóm
tắt tối đa, còn caching dùng độ dài sequence thực tế.

## OfflineSGLangCapture cho hidden-state cache

`capture_features` có hai backend:

```text
--capture-backend hf       # mặc định, tương thích Transformers
--capture-backend sglang   # SpecForge OfflineSGLangCapture nội bộ
```

Backend SGLang không gọi HTTP/OpenAI endpoint. Nó tái sử dụng
`externals/SpecForge/specforge/offline_capture`, khởi tạo `ModelRunner` trong
process, bật `enable_return_hidden_states`, lấy các layer theo hook của
EAGLE3/DFlash/DSpark, rồi tách lại packed prefill theo từng sequence. Cách này
khác hoàn toàn việc lấy text từ SGLang server và không tự động đảm bảo parity
với mọi version SGLang.

Mỗi lần chạy SGLang bắt buộc chạy parity trên các sample đầu tiên bằng cách so
sánh với forward Transformers. Cache chỉ được publish nếu đạt các ngưỡng:

```bash
PYTHONPATH=src python3 -m Finetuning.capture_features \
  --input /work/teacher_train.jsonl \
  --output /work/features_train \
  --target-model-path /models/Qwen3-4B \
  --target-layer-ids 1,9,17,25,33 \
  --max-length 2048 --max-source-tokens 1536 --max-summary-tokens 384 \
  --torch-dtype bfloat16 --device cuda \
  --capture-backend sglang --capture-method dflash \
  --sglang-attention-backend flashinfer \
  --sglang-max-running-requests 8 \
  --parity-samples 2 \
  --adaptive-batch --adaptive-max-batch-size 8
```

`manifest.json` ghi `capture_backend`, `capture_method` và các tham số
SGLang; JSON summary ghi thêm `parity`, batch đã chọn và số lần OOM retry.
Nếu dependency SGLang/SpecForge không import được hoặc parity thất bại, lệnh
trả lỗi và không publish generation mới. Khi chạy nhiều GPU bằng `torchrun`,
backend hiện hỗ trợ data parallel (`--sglang-tp-size 1`); TP lớn hơn một cần
launcher DP-aware riêng để mọi TP rank nhận cùng input batch.

## Chạy nhiều GPU trên một server

Pipeline hỗ trợ single-node DDP qua `torchrun`. Mỗi process dùng một GPU; chỉ
draft DFlash được đồng bộ gradient, còn target embedding/lm_head vẫn frozen và
được replicate trên từng GPU. `training.batch_size` là batch size **mỗi GPU**;
global batch size bằng:

```text
batch_size * số GPU * accumulation_steps
```

Hai config Qwen mẫu bật `training.adaptive_batch_size: true`. Trước optimizer
và trước khi bọc DDP, mỗi rank sẽ chạy một preflight forward/backward trên
feature thật, tìm batch lớn nhất còn dưới `target_memory_fraction` (mặc định
90% VRAM), rồi đồng bộ lấy giá trị nhỏ nhất giữa các GPU. Batch được chọn giữ
cố định trong toàn run; `accumulation_steps` vẫn là cách điều chỉnh global
batch mà không làm tăng peak VRAM. Giới hạn tìm kiếm nằm trong
`adaptive_min_batch_size` và `adaptive_max_batch_size`, còn
`adaptive_probe_batches` kiểm soát số batch dùng để đo.

Kết quả được ghi vào `extra.json` của checkpoint dưới khóa
`adaptive_batch`. Khi resume từ checkpoint này, pipeline dùng lại batch đã
chọn và không probe lại; vì vậy phải dùng cùng số GPU và cùng global batch.
Nếu batch 1 không vừa, hãy giảm `objective_chunk_blocks`, `num_anchors` hoặc
độ dài sequence trước khi tăng `adaptive_max_batch_size`. Nếu metadata báo
`hit_maximum: true` mà peak vẫn dưới mục tiêu, có thể tăng cap này (sau khi
kiểm tra RAM host vì feature batch cũng được đọc vào bộ nhớ CPU).

Các stage chuẩn bị dữ liệu cũng có thể chạy song song. Mỗi rank ghi artifact
tạm với hậu tố `.rank00000`, sau đó rank 0 merge theo thứ tự sample và publish
atomic:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m Finetuning.generate_targets \
  --input /server/data/vietnamese_train.jsonl \
  --output /server/work/teacher_train.jsonl \
  --target-model-path /server/models/Qwen3-4B \
  --max-length 2048 --max-source-tokens 1536 --max-summary-tokens 384 \
  --torch-dtype bfloat16 --device cuda \
  --adaptive-batch --target-memory-fraction 0.90 \
  --adaptive-max-batch-size 256 --bucket-window 512

torchrun --standalone --nproc_per_node=8 \
  -m Finetuning.capture_features \
  --input /server/work/teacher_train.jsonl \
  --output /server/features/train \
  --target-model-path /server/models/Qwen3-4B \
  --num-draft-layers 5 \
  --max-length 2048 --max-source-tokens 1536 --max-summary-tokens 384 \
  --torch-dtype bfloat16 --device cuda \
  --adaptive-batch --target-memory-fraction 0.90 \
  --adaptive-max-batch-size 256 --bucket-window 512

torchrun --standalone --nproc_per_node=8 \
  -m Finetuning.run_train \
  --config src/Finetuning/configs/qwen3_4b.yaml \
  --device cuda
```

Các rank phải nhìn thấy cùng filesystem cho input, feature store và
`output_dir`. Checkpoint, `metrics.jsonl`, `train.log` và `draft_export/` chỉ
được ghi bởi rank 0. Khi resume phải dùng cùng số GPU với checkpoint đã tạo.
Nếu chỉ chạy một GPU, các lệnh Python cũ vẫn giữ nguyên.

Prompt mặc định là chỉ dẫn tóm tắt tiếng Việt có một placeholder
`{document}`. Nếu thay `data.prompt_template`, phải dùng đúng chuỗi đó cho
**cả generate, capture và evaluate**; manifest sẽ từ chối cache khác contract.

Mỗi checkpoint có thư mục `draft_export/` gồm `draft_state_dict.pt`,
`draft_metadata.json` và `COMPLETE`. Có thể tiếp tục domain adaptation bằng:

```yaml
model:
  draft_init_path: /work/previous-run/qwen3-4b-step1000/draft_export
```

Không kết hợp `draft_init_path` với `--resume-from`. Resume kiểm tra metadata
export (target, layer IDs, mask token, block size và draft config) trước khi
phục hồi optimizer/scheduler/RNG.

## Đánh giá sau train

Đánh giá generation độc lập dùng cùng prompt contract và file validation gốc
hoặc file trajectory (nếu là trajectory nó tự dùng `reference_summary`).

```bash
PYTHONPATH=src python3 -m Finetuning.generation_evaluation \
  --input /work/teacher_eval.jsonl \
  --output /work/dflash_eval.jsonl \
  --target-model-path /models/Qwen3-4B \
  --draft-export /work/output/qwen3-4b-step1000/draft_export \
  --max-length 2048 --max-source-tokens 1536 --max-summary-tokens 384 \
  --torch-dtype bfloat16 --device cuda
```

Output ghi ROUGE-1/2/L F1 với gold summary, `target_token_match` cho từng mẫu,
và summary có `target_exact_rate` cùng paired `speedup`. Với greedy decoding,
`target_exact_rate` phải bằng 1.0 trước khi dùng số speedup để kết luận. Nếu
không đạt, checkpoint hoặc contract target/draft không phù hợp.

Đánh giá cũng có thể chia theo nhiều GPU bằng cách thêm
`torchrun --standalone --nproc_per_node=N` trước module; rank 0 sẽ merge các
bản ghi và ghi summary cuối cùng vào `--output`.

## Quy mô và smoke

Five target layers của Qwen3-8B ở BF16, sequence 2048, xấp xỉ 80 MiB hidden
state/mẫu. Bắt đầu với 500 → 2K mẫu, quan sát validation loss/accuracy và
`target_exact_rate`, rồi mới capture tập lớn. Dataset chỉ được mở lazy qua
`DataLoader`; không materialize toàn bộ tensor cache vào RAM.

Smoke CPU không cần snapshot:

```bash
PYTHONPATH=src .venv/bin/python -m Finetuning.run_train \
  --config src/Finetuning/configs/synthetic_smoke.yaml --device cpu
```

Smoke chỉ kiểm tra lifecycle, không đại diện cho ROUGE tiếng Việt, VRAM hay
speedup thực tế.
