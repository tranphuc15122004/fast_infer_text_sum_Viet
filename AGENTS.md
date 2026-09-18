# Project Guidelines — fast_infer_text_sum_Viet

Biến thể **tiếng Việt** của benchmark `fast_infer_text_sum`: so sánh công bằng các
baseline tăng tốc inference (speculative decoding, semantic reduction, KV
optimization) cho **tóm tắt văn bản dài tiếng Việt** trên cùng dữ liệu và cùng
output schema. Toàn bộ docs viết bằng tiếng Việt.

Repo này **mirror cấu trúc** của project tham chiếu `fast_infer_text_sum`
(`src/`, `scripts/`, `config/`, `docs/`, `tests/`). Convention chi tiết
(baseline pattern, runtime contract, output schema, lệnh LongBench) **không lặp
lại ở đây** — đọc project tham chiếu, xem [Tài liệu canonical](#tài-liệu-canonical-ngoài-repo).

## Ba môi trường — ĐỌC TRƯỚC KHI CHẠY BẤT CỨ THỨ GÌ

| Môi trường | Vai trò | Đặc điểm |
|---|---|---|
| `tuantb@teslaT4` (máy local) | Dev/debug **CPU-only** | Tesla T4 15 GB, **không sudo** |
| `/home/tuantb/fast_infer_text_sum/.venv` | Venv **mô phỏng** môi trường B200 | **Nằm NGOÀI repo này** |
| Server B200 | Chạy thực nghiệm, sinh số liệu thật | `python3` hệ thống (3.12), **không internet** |

### Máy local `tuantb@teslaT4` — không chạy được GPU

- Driver 550.163 (max CUDA 12.4) **không** chạy được stack cu130 của server ⇒
  `torch.cuda.is_available()` **luôn False**. Đừng cố sửa GPU trên máy này.
- Máy **không có sudo** ⇒ không cài system package, không build `flash-attn`.
- T4 chỉ để dev/debug logic trên CPU. **Mọi số liệu benchmark phải lấy từ B200.**

### Venv mô phỏng B200 — nằm ngoài repo

Venv dùng để chạy thử thực nghiệm **không nằm trong workspace này**; nó ở thư mục
project tham chiếu. Trỏ tới nó bằng biến chuẩn của runtime:

```bash
export FAST_INFER_VENV=/home/tuantb/fast_infer_text_sum/.venv
```

Không tạo `.venv` riêng trong repo này. `.venv` chỉ để mô phỏng trước khi đưa code
lên server; production dùng `python3` hệ thống trên B200.

### Server B200 — đường dẫn canonical

```text
Repo root (repo này):
/workspace/storage-shared/nlp/dungdx4/phuc_projects/

Master config của repo này:
/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master_Viet.env
```

- Master config là **nguồn chân lý duy nhất** cho path/model/cache. Repo chỉ giữ
  `config/master.path` trỏ tới file đó — không nhúng path server vào code.
- Hậu tố `_Viet` phân biệt với `fast_infer_master.env` của project tham chiếu.
  **Đừng** trỏ nhầm sang file của project kia.
- Override khi cần: `FAST_INFER_MASTER_CONFIG=/path/khác.env`.
- Server **không có internet**: model/dataset/wheel phải được mirror sẵn trong
  offline wheelhouse.

## Trạng thái hiện tại của repo này

| Thành phần | Trạng thái |
|---|---|
| `datasets/` | ✅ Dữ liệu tiếng Việt (1.2 GB raw, 158k file) + bộ eval chuẩn hoá |
| `externals/` | ✅ 5 baseline repo vendored (~42 MB) |
| `src/Finetuning/` | ✅ Đã port (kèm `tests/`, `configs/`, `plans/`) |
| `scripts/data/` | ✅ Pipeline dữ liệu (phân tích phân phối + build bộ eval) |
| `docs/` | ⚠️ Mới có `docs/superpowers/plans/` |
| `config/` | ❌ **Chưa có** — cần `config/master.path` trỏ tới master config `_Viet` |
| `tests/` (cấp repo) | ❌ Chưa có (chỉ có `src/Finetuning/tests/`) |
| `checkpoints/` | ❌ Chưa có |

Khi port phần còn lại: sao chép convention nguyên trạng từ project tham chiếu, chỉ
Việt hoá phần prompt/dữ liệu. Không tự phát minh layout mới.

⚠️ **Bẫy khi port `config/`**: `config/master.path` của project tham chiếu trỏ tới
`fast_infer_master.env` (không có hậu tố). Copy nguyên trạng sẽ khiến repo này
**âm thầm dùng sai master config**. Phải sửa thành `fast_infer_master_Viet.env`.

## Dữ liệu tiếng Việt

Nằm trong `datasets/`:

- `datasets/raw/` — VietNews, ViMs, VLSP, WikiLingua (+ file `.zip` gốc). **Gitignored.**
- `datasets/infer_test/` — hiện đang rỗng.

`datasets/raw/` bị `.gitignore` chặn vì rất nặng; dữ liệu gốc tái tạo được từ `.zip`.
Schema JSONL plug-and-play và quy tắc reference/ROUGE xem
[`data/README.md`](</home/tuantb/fast_infer_text_sum/data/README.md>) của project tham chiếu.

### Bộ eval chuẩn hoá — `datasets/eval_100/`

4 bộ test **cùng một format**, mỗi bộ 100 mẫu, dùng để chạy mọi baseline:

| File | Document lấy từ | Reference |
|---|---|---|
| `vietnews_100.jsonl` | block 3 trở đi của `.seg` (de-segment `_`→space) | block 2 (summary) |
| `wikilingua_100.jsonl` | `src` nối lại (file `.json` **thực chất là JSONL**) | `tgt` nối lại |
| `vims_100.jsonl` | ghép 4–10 bài/cluster, marker `### Tài liệu N:` | `0.gold.txt` (`1.gold.txt` trong `answers`) |
| `vlsp_100.jsonl` | ghép 2–5 bài từ `single_documents` | `summary` |

Vào pool: VietNews/WikiLingua dùng split `test`; ViMs dùng cả 300 cluster; **VLSP dùng
`vlsp_2022_abmusu.jsonl` (có gold) — KHÔNG dùng `vlsp_abmusu_test_data.jsonl` vì test
chính thức không có `summary`, không chấm được ROUGE.**

Chạy lại:

```bash
python3 scripts/data/analyze_distribution.py --include-train-counts
python3 scripts/data/build_eval_100.py --samples 100
```

Lấy mẫu có chủ đích chống OOD: loại mẫu ngoài `[p1, p99]`, loại mẫu suy biến
(`document < 100` từ, hoặc `reference >= document`), rồi chia 5 bin độ dài và phân
bổ theo phân phối. Thuật toán **deterministic** (điểm cách đều + seed trong manifest)
nên chạy lại cho kết quả y hệt. `manifest.json` ghi minh bạch số OOD và số mẫu bị loại.

⚠️ **`datasets/normalized/` (109 MB) là gitignored** — nó tái tạo được từ raw. Chỉ
`datasets/eval_100/` (4,4 MB) được commit.

## Externals — baseline vendored

5 repo, **untracked, không phải git submodule**, không có `.git` riêng:

| Repo | Mục đích | Import / entry | Tier GPU |
|---|---|---|---|
| `DeepSpec` | Train draft model (DSpark, DFlash, Eagle3) | `deepspec` · `train.py`, `eval.py` | 8 GPU; target cache ~38 TB |
| `dflash` | Block-diffusion draft model (inference/serve) | `dflash` · `dflash/cli.py` | Lớn (MoE 30B+) |
| `Domino` | Block-parallel drafting + causal correction head | *(không có `__init__.py`)* · `run_*_benchmark.sh` | 8 GPU, SGLang tp2 |
| `EAGLE` | EAGLE-1/2/3 spec decoding + training | `eagle` · `eagle/evaluation/gen_ea_answer_*.py` | **Nhỏ nhất: 8x RTX 3090** |
| `SpecForge` | Framework train draft model → serve trên SGLang | `specforge` · `specforge train --config ...` | **B200/B300/H200, multi-node** |

Chỉ **EAGLE** quảng bá chạy được trên GPU nhỏ; **SpecForge** hướng B200-class.
Không repo nào hỗ trợ T4. Khi cần hiểu quan hệ giữa các method, đọc
`externals/SpecForge/docs/sections/concepts/`.

## Convention phải theo

- **1 baseline = 1 bộ file**: `scripts/infer_<b>.py` + `scripts/run_<b>.sh` +
  `docs/baselines/<b>.md`, nối vào `scripts/run.sh` và thêm loader
  `fast_infer__load_<b>()` trong `scripts/common/config.sh`.
- Mọi launcher resolve master config qua `config/master.path` (hoặc
  `FAST_INFER_MASTER_CONFIG`), rồi source `scripts/common/runtime.sh`.
- **Smoke vs full**: mặc định `--smoke`; full cần GPU lớn + model/cache thật.
- **Output schema**: mọi record ghi qua `io_util.JsonlWriter`
  (`scripts/common/io_util.py`), kết thúc bằng summary record. Key chuẩn xem
  `BASE_SCHEMA_KEYS` / `SPEC_SCHEMA_KEYS`.
- **ROUGE**: khi có reference, gọi `rouge.add_rouge()`; summary gọi
  `rouge.aggregate_rouge()` (`scripts/common/rouge.py`).
- **Output/checkpoint không commit** — `outputs/` và `checkpoints/` gitignored.

## Lệnh (sau khi đã port scripts/)

```bash
# Preflight interpreter/import, không tải model
FAST_INFER_VENV=/home/tuantb/fast_infer_text_sum/.venv \
  /home/tuantb/fast_infer_text_sum/.venv/bin/python scripts/check_shared_env.py

# Dev CPU local
CUDA_VISIBLE_DEVICES="" DEVICE=cpu SMOKE=1 bash scripts/run_<baseline>.sh

# Chạy 1 baseline
bash scripts/run.sh <baseline>
```

Lệnh setup server, LongBench matrix (data-parallel / nhiều process mỗi GPU) và
MR-DFlash smoke: chép nguyên văn từ
[`docs/server_environment.md`](</home/tuantb/fast_infer_text_sum/docs/server_environment.md>)
của project tham chiếu.

## Gotchas

- **Không internet trên server**: không dùng installer online; mọi wheel/direct URL
  phải có sẵn trong mirror/wheelhouse.
- **`requirements.txt` hygiene**: chỉ requirement theo tên/version hoặc artifact đã
  mirror. Không `file://`, editable path, host URL, hay package OS (Ubuntu).
- **transformers 5.x**: `apply_chat_template(..., return_tensors="pt")` trả
  `BatchEncoding` ⇒ **phải thêm `return_dict=False`** trước khi dùng
  `.shape[1]` / `.to(device)` / `model.generate()`.
- **Bẫy cú pháp shell/Python**: dùng `sys.exit(1) if cond else None`, **KHÔNG**
  `raise SystemExit(...) if cond else None` (parse thành `raise None` → TypeError).
- **CUDA mismatch**: `flash-attn`, `flashinfer`, Triton, vLLM, `sglang-kernel` phải
  khớp torch/CUDA/GPU. `sglang-kernel` phải là wheel cu130 trong wheelhouse.
- **Model gated (Llama)** cần `HF_TOKEN`; ưu tiên snapshot local.
- **OOM với input dài**: `vanilla_hf` từng OOM ở `--max-input-tokens 20480`. Trần VRAM
  vận hành ~170 GiB/card; dùng `--cache-auto-batch` (target 160 / hard 163 GiB) và
  backoff tự động thay vì tăng batch cố định.
- **Không tạo venv riêng cho từng baseline.**
- **Không commit** `datasets/raw/`, `outputs/`, `checkpoints/`, hay `HF_TOKEN`.
- Đừng diễn giải `unsupported_cpu`, `missing_checkpoint`, `missing_dependency`,
  `unsupported_dataset` như số liệu tốc độ — timing của chúng là `null`.

## Git & artifact hygiene

Chủ đề này lặp lại nhiều lần trong lịch sử làm việc của project ("dữ liệu có nặng
quá không", "commit hiện tại nặng bao nhiêu", "giảm tải kết quả thực nghiệm") —
nên xử lý theo quy tắc cố định sau.

- Repo chỉ track **source + docs**. Không commit: `datasets/raw/` (đã gitignored),
  `externals/` (vendored, untracked), `outputs/`, `checkpoints/`, cache runtime.
- Kiểm tra dung lượng **trước khi** commit:
  ```bash
  git status --short                      # file sẽ vào commit
  { git ls-files -z; git ls-files -o --exclude-standard -z; } | xargs -0 du -ch | tail -1
  ```
- Asset nặng đã bị ignore nhưng thật sự cần versioned: `git add -f <path>` (cân nhắc
  trước — file đó sẽ nằm trong lịch sử vĩnh viễn).
- **`externals/` là untracked, không phải submodule** ⇒ xoá file trong đó **không có
  git để khôi phục**. Phải kiểm tra reference (`grep`) trước khi xoá bất cứ thứ gì.
- Khi giảm tải kết quả thực nghiệm: giữ file tổng hợp/báo cáo, bỏ JSONL thô.

## Tài liệu canonical (ngoài repo)

Repo này chưa có `docs/`; các tài liệu dưới đây thuộc project tham chiếu
`/home/tuantb/fast_infer_text_sum/` và là **nguồn chân lý** cần đọc:

| Tài liệu | Nội dung |
|---|---|
| [`docs/server_environment.md`](</home/tuantb/fast_infer_text_sum/docs/server_environment.md>) | Path canonical, runtime server, venv B200, MR-DFlash smoke |
| [`docs/cpu_dev_workflow.md`](</home/tuantb/fast_infer_text_sum/docs/cpu_dev_workflow.md>) | Dev CPU trên T4, checklist debug, bug đã sửa |
| [`docs/README.md`](</home/tuantb/fast_infer_text_sum/docs/README.md>) | Baseline Inference Guide + bảng baseline |
| [`docs/fast_infer_master.example.env`](</home/tuantb/fast_infer_text_sum/docs/fast_infer_master.example.env>) | Template master config (mọi key `FI_*`, `MODEL_*`, `LONG_BENCH_*`) |
| [`data/README.md`](</home/tuantb/fast_infer_text_sum/data/README.md>) | Schema dữ liệu, ROUGE, plug-and-play |
| [`externals/baseline_repo_guide.md`](</home/tuantb/fast_infer_text_sum/externals/baseline_repo_guide.md>) | Taxonomy baseline + **§13 unified result schema** |
| [`AGENTS.md`](</home/tuantb/fast_infer_text_sum/AGENTS.md>) | Convention đầy đủ của project tham chiếu |
| [`src/Finetuning/README.md`](</home/tuantb/fast_infer_text_sum/src/Finetuning/README.md>) | Pipeline fine-tune DFlash cho tóm tắt tiếng Việt |

Cấu trúc thư mục, runtime contract và lệnh đầy đủ: xem
[`AGENTS.md`](</home/tuantb/fast_infer_text_sum/AGENTS.md>) của project tham chiếu
thay vì tự suy đoán.
