# Triển khai metric speedup thống nhất cho sáu baseline

**Mục tiêu:** Sau một lượt benchmark B200 trên VietNews, WikiLingua, ViMs và VLSP (100 mẫu/bộ), xuất bảng paper có end-to-end speedup chung, speedup nội bộ, decode speedup được định nghĩa đúng, chất lượng và độ phủ; mọi bảng tính lại offline từ JSONL mà không chạy GPU thêm.

**Đặc tả đo lường:** [`docs/baselines/speedup_metric_contract.md`](../../baselines/speedup_metric_contract.md). Kế hoạch này là hướng dẫn triển khai cho đặc tả đó. Đây là benchmark inference, không có training loop.

**Trạng thái triển khai (2026-09-28):** các hạng mục 1–6 đã được đưa vào source, gồm runner paper profile, v2 estimator/report, strict timing adapters và tracked vendor timing patches. CPU fixture tests đã qua: 91 tests pass; B200 smoke/full chưa chạy. Chưa có B200 smoke/full output trong workspace; các kết quả paper vẫn cần validation trên B200.

**Phạm vi:** `vanilla_hf`, `vanilla_fa`, `eagle3`, `dflash`, `domino`, `dspark`; bốn file `datasets/eval_100/*_100.jsonl`. Chỉ dùng model/cache từ master config `_Viet` trên B200. Không chạy benchmark GPU trên T4 local. Giữ nguyên JSONL legacy và các thay đổi đang có của người dùng; schema mới mang version riêng.

**Ngân sách GPU tối thiểu:** 400 mẫu × 9 generation/mẫu = 3.600 generation: HF, FA, DFlash + block-1, EAGLE + naive, Domino, DSpark, và một SGLang target-only dùng chung. Cộng warmup ngắn, ba anchor đo lại mỗi session, và smoke một mẫu trước full. Nếu target-only artifact không tương thích giữa Domino và DSpark, cần 400 generation nữa; báo chi phí đó trước full run.

## Quyết định khóa trước khi code

- **Hai câu hỏi khác nhau:** `common_*` dùng một Vanilla reference cố định cho toàn run để so sánh giữa baseline; `native_*` dùng reference cùng runtime của method để đo lợi ích thuật toán. Không cộng/trộn hai nhóm trong key `speedup` hay `dense_*` cũ.
- **Reference chung:** mặc định `vanilla_fa`. Có thể chọn `vanilla_hf` *trước khi đo bất kỳ mẫu nào* nếu FA không khả dụng. Không fallback theo dataset/sample. Reference baseline của run nhận 1,0× identity trên các mẫu thành công.
- **Native reference:** DFlash `dflash_generate(block_size=1)`; EAGLE `naivegenerate`; Domino/DSpark SGLang `target_only` cùng fingerprint. HF/FA không có native reference.
- **Load:** batch 1, một request/process trên mỗi GPU, không data parallel chia sẻ GPU; warmup ngắn ở mỗi mode; cùng target checkpoint/tokenizer/prompt đã cắt/max-token/greedy EOS. Ghi GPU count/TP; nếu khác tài nguyên thì nhãn `unequal_resource`, không đưa vào ranking cùng tài nguyên.
- **Natural EOS:** không ép fixed-K. Output mismatch không loại timing; báo text/token agreement, output length, ROUGE và output degeneracy riêng. Tỷ số latency là tốc độ *hoàn thành tác vụ* với output thực tế, không là speedup trên cùng một chuỗi token.
- **Strict decode:** đã instrument/vận dụng phase boundary riêng cho Vanilla, DFlash, EAGLE và SGLang 0.5.20; smoke gate kiểm tra common decode cho cả sáu baseline và native decode cho bốn speculative baseline. Không nâng `completion_latency` hay `eagle_time` native thành strict decode nếu patch không xác minh được. Thiếu phase thì metric đó `null` và full run không qua gate. `common_output_rate_ratio` từ request wall là output rate, không phải decode speedup.
- **Một lượt B200:** toàn bộ join, kiểm toán, bootstrap và bảng paper làm offline. Raw timing lưu precision gốc; chỉ làm tròn khi render báo cáo.

## Schema và phép tính v2

Mỗi observation có `contract_version=2`, `run_id`, `dataset`, `sample_id`, `method`, `status`, `sample_order`, `prompt_token_sha256`, `target_checkpoint_sha256` hoặc revision, `tokenizer_revision`, `generation_config_sha256`, `hardware_fingerprint`, `gpu_count`, `tp_size`, `batch_size`, `concurrency`, `cache_policy`, `actual_input_tokens`, `timed_generated_tokens`, `visible_output_tokens`, `stop_reason`, `text`, `request_wall_ms`, `native_elapsed_ms`, `native_timing_scope`, `decode_active_ms`, `decode_token_count`, `decode_phase_definition`, `prefill_ms`, và `timing_source`. Chỉ lưu token IDs khi backend có; không dùng text/token IDs như timing validity gate. `timed_generated_tokens` tính throughput; `visible_output_tokens` và text tính chất lượng (quan trọng khi EAGLE trim EOS sau timer).

Reference chung là JSONL Vanilla đã chạy; Domino/DSpark có một sidecar `references/sglang_target_only/<dataset>.jsonl` chứa cùng schema và fingerprint. Native DFlash/EAGLE có thể giữ timing/reference output nhúng cùng sample record, nhưng normalizer đọc nó như một `ReferenceObservation`. Không ghi đè raw JSONL để attach reference. `JsonlWriter` tiếp tục append record ngay sau mỗi sample; `summary` chỉ chứa metadata/coverage, không làm nguồn chân lý cho metric v2.

Trên tập ID hợp lệ *riêng của từng metric* `I`:

```text
common_esr = Σ common_ref.request_wall_ms / Σ method.request_wall_ms
native_esr = Σ native_ref.native_elapsed_ms / Σ method.native_elapsed_ms
common_decode_time_ratio = Σ common_ref.decode_active_ms / Σ method.decode_active_ms
native_decode_time_ratio = Σ native_ref.decode_active_ms / Σ method.decode_active_ms
common_decode_rate_ratio =
  (Σ method.decode_token_count / Σ method.decode_active_ms) /
  (Σ common_ref.decode_token_count / Σ common_ref.decode_active_ms)
native_decode_rate_ratio: cùng công thức, thay common_ref bằng native_ref
common_output_rate_ratio =
  (Σ method.timed_generated_tokens / Σ method.request_wall_ms) /
  (Σ common_ref.timed_generated_tokens / Σ common_ref.request_wall_ms)
```

Mọi tổng chỉ lấy đúng cùng tập ID ở tử và mẫu; timing phải hữu hạn, dương; decode rate cần cả hai `decode_token_count > 0`. Prefill/TTFT ratio dùng cùng phép tổng thời gian và namespace common/native **chỉ khi** hai bên có cùng boundary đã xác minh; `ttft_ms=prefill_ms` hiện tại không đủ bằng chứng cho TTFT xuyên engine. Nếu thiếu, ghi `null`/reason. Không lấy mean tỷ số từng sample. Lưu số cặp, sample IDs, reason loại trừ và reference ID cho từng metric. Dataset table dùng bốn tỷ số riêng; cột gộp dùng geometric mean bốn dataset trên shared-set, mỗi dataset trọng số bằng nhau. Bảng ranking dùng giao sample thành công của cả sáu baseline cho cùng metric; bảng pairwise riêng không làm mất mẫu chỉ vì baseline khác lỗi. Paired bootstrap (seed cố định, 10.000 resamples theo sample ID trong từng dataset) tạo 95% percentile CI; với cột gộp bootstrap phân tầng theo dataset. CI phản ánh sample variability, không phản ánh run-to-run GPU noise.

## Shared scaffold

- Có sẵn: `src/Benchmark/common/{io_util,benchmark_runtime,metrics,metric_audit,longbench_adapter}.py`, `src/Benchmark/{run_longbench_200,collect_metrics,infer_dflash,infer_sglang_spec,eagle3_infer_qwen3}.py`, `src/Benchmark/common/vanilla_inference.py`.
- Output dưới `outputs/longbench_viet_100/<run_id>/`; tạo `references/`, `audit_v2.json`, `paper_speedup.csv`, `paper_speedup.md` trong run directory; `outputs/` gitignored. Không tạo venv mới, không sửa repo external không có bản patch versioned.
- Những test hiện có cần điều chỉnh theo contract v2: `tests/test_longbench_metric_contract.py`, `tests/test_sglang_adapter.py`, `tests/test_eagle_compat.py`, `tests/test_vanilla_inference.py`. Thêm test nhỏ chỉ cho semantics mới; không duplicate test implementation.

## Subtask 1: Manifest v2 và prompt/config identity — implemented

**Files:** `src/Benchmark/common/benchmark_runtime.py`, `src/Benchmark/common/longbench_adapter.py`, `src/Benchmark/run_longbench_200.py`; tests trong `tests/test_longbench_metric_contract.py`.

**Implement:** Hàm thuần Python tạo SHA-256 từ token IDs sau chat template + truncate (serialize length và từng ID có endian/version ổn định), config generation canonical và hardware fingerprint. Manifest khóa bốn SHA dataset, thứ tự sample, pinned common reference, checkpoint/tokenizer revision, effective attention backend, dtype, GPU count/TP, speculative algorithm, seed/stop/cache/warmup, code/runtime versions. Preflight tạo hash cho cả sáu adapter từ chính prompt path; đối chiếu `input_tokens` phía SGLang server với số token local. Nếu mismatch, không chạy full; nếu cần sửa SGLang text tokenization, làm trước production. Dùng config thực tế, không chỉ flag yêu cầu. `domino` hiện mặc định SGLang `DFLASH`, `dspark` là `DSPARK`; xác nhận checkpoint Domino có chủ ý dùng backend này và ghi đúng nhãn trong manifest.

**CPU cases:** cùng token IDs cho cùng hash; đổi một token/stop/budget/backend phải đổi fingerprint; duplicate sample ID hoặc thiếu dataset bị reject; khác GPU footprint được gắn nhãn; manifest reference không đổi giữa dataset.

**Done:** một `run_manifest.json` có đủ mọi giá trị dùng cho join; preflight trả lý do cụ thể trước GPU khi input/config không tương thích.

## Subtask 2: Outer wall time cho Vanilla HF/FA và DFlash — implemented

**Files:** `src/Benchmark/common/vanilla_inference.py`, `src/Benchmark/infer_dflash.py`, `src/Benchmark/common/benchmark_runtime.py`; tests `tests/test_vanilla_inference.py`, `tests/test_dflash_compat.py`.

**Implement:** Timer `request_wall_ms` từ trước chat format/tokenize đến khi decoded text sẵn sàng; đồng bộ CUDA trước/sau GPU generation. Giữ `e2e_ms`/`prefill_ms` native cũ và `native_elapsed_ms` rõ boundary. HF/FA giữ real backend/kv cache metadata. DFlash đo `block_size=1` và speculative như hiện có, kể cả khi Vanilla reference tồn tại; lưu token counts/text và raw native timing của cả hai. DFlash upstream `time_per_output_token` lấy `total_decode_time / num_output_tokens` dù decode bắt đầu sau token đầu; v2 lưu phase time và token count đúng, không dùng upstream TPOT làm strict DSR. Record cả `timed_generated_tokens` và text/tokens sau EOS.

**CPU cases:** fake tokenizer/model clock xác nhận outer timer bao format và decode; block-1 vẫn chạy khi có common reference; output text lệch vẫn giữ pair; `n=1` không tạo decode rate từ 0 token.

**Done:** HF/FA/DFlash phát sample record v2 có outer wall và prompt hash; DFlash có native reference đúng mỗi sample.

## Subtask 3: EAGLE native pair và strict decode boundary — implemented

**Files:** `src/Benchmark/eagle3_infer_qwen3.py`, `src/Benchmark/eagle_compat.py`, bản patch có version dưới `scripts/patches/` nếu phải đổi `externals/EAGLE/eagle/model/{ea_model,utils}.py`; tests `tests/test_eagle_compat.py`.

**Implement:** Thêm outer wall từ raw prompt đến decoded text (hiện prompt IDs chuẩn bị ngoài timer). Giữ `eagenerate`/`naivegenerate` trên cùng input, seed, stop và target; `eagle_time` từ vendor là generation **gồm prefill**, không lấy nó làm `decode_ms`. Vendor `initialize_tree()` append token đầu rồi còn tạo draft tree trước khi trả về; timestamp decode chuẩn phải đặt ngay sau token đầu đã commit, không lấy mốc cuối `initialize_tree()` làm first-token time. Nếu phải sửa vendor untracked, giữ tracked patch + SHA nguồn và apply idempotent ở offline setup; không dựa vào sửa tay không version. Ghi `timed_generated_tokens`, `visible_output_tokens`, số EOS/stop token bị trim và raw phase timings cho spec/naive.

**CPU cases:** stub generation kiểm tra outer/native timing; first-token timestamp và n-1 count; mismatch output không vô hiệu timing; trim EOS không làm sai timed count; patch không apply hai lần.

**Done:** EAGLE có common outer timing, native pair và phase decode có nguồn gốc đo lường rõ.

## Subtask 4: SGLang target-only dùng chung cho Domino/DSpark — implemented

**Files:** `src/Benchmark/infer_sglang_spec.py`, `src/Benchmark/common/longbench_adapter.py`, tracked patch/inventory cho SGLang nếu cần; tests `tests/test_sglang_adapter.py`, `tests/test_sglang_algorithm_preflight.py`.

**Implement:** `--target-only-reference-file` đọc sidecar theo sample ID; standalone không có file vẫn tự chạy reference. Orchestrator chuẩn bị một target-only artifact/dataset, chỉ reuse khi fingerprint target, tokenizer, prompt IDs, sampling/stop, GPU/TP, backend, cache, batch/concurrency khớp. Không đưa draft checkpoint vào fingerprint của target-only. SGLang warmup một request ngắn ở mỗi mode, không tính timing; batch_size=1, max_running_requests=1, fixed order; không coi server startup là sample latency. `_request_one` chốt outer timer sau text đã sẵn sàng. Paper latency profile **tắt prefix cache giữa các request** cho SGLang để giống HF/DFlash/EAGLE, hoặc flush cache trước từng sample ngoài timer nếu pinned SGLang không có flag tắt; kiểm tra effective setting/cache hits trong smoke. Target-only, Domino và DSpark dùng cùng chính sách. Nếu không vô hiệu được reuse, gắn nhãn warm-cache system comparison và không đưa vào ranking cùng no-cache baselines. Kiểm tra source của đúng SGLang wheel trên B200 để xác định `completion_latency`/`prompt_latency`; nếu không trùng strict decode contract, làm tracked version-pinned instrumentation rồi xác minh bằng request 1 và 2 token trước full. Không lấy field bằng tên giống nhau làm bằng chứng đủ.

**CPU cases:** sidecar join out-of-order đúng ID, thiếu/duplicate/mismatch fingerprint reject rõ; reuse một reference khi hai method tương thích; nếu khác config tạo hai artifact có nhãn, không silent fallback; fake HTTP clock bao prompt prep+parse+text; batch >1 không được vào latency profile.

**Done:** Domino/DSpark có native pair qua cùng SGLang target-only artifact, common outer timing, strict decode hợp lệ hoặc smoke gate dừng.

## Subtask 5: Một estimator, audit và paper report — implemented

**Files:** tạo `src/Benchmark/common/paired_reference.py`; sửa `src/Benchmark/common/{metrics,metric_audit}.py`, `src/Benchmark/collect_metrics.py`; tests `tests/test_longbench_metric_contract.py` và một fixture matrix nhỏ trong `tests/`.

**Implement:** Normalizer chuyển Vanilla JSONL, DFlash/EAGLE embedded native refs, SGLang sidecar thành `Observation`; reject duplicate ID và `NaN/Inf`; join theo dataset/sample/prompt hash/config/clock. Hàm aggregate chung nhận `reference_scope=common|native`, trả mỗi metric value, valid IDs/count, exclusions theo reason. Phân biệt `phase_unverified`, `missing_time`, `nonpositive_time`, `prompt_mismatch`, `config_mismatch`, `unequal_resource`, `failed_status`. DSR strict chỉ khi definition/count tương thích. Tính paired bootstrap offline, shared-set và pairwise tables; quality/length/exact match luôn độc lập với timing. Collector tạo `audit_v2.json`, `paper_speedup.csv`, `paper_speedup.md`, giữ `metrics_summary.*` legacy nhưng không trộn `esr/dsr` cũ vào bảng v2. Tái chạy collector phải byte-stable trừ timestamp được tách khỏi bảng.

**CPU fixture có số biết trước:** ref wall `[100,200]`, method `[50,100]` ⇒ ESR `2.0`; dù output text khác vẫn `n=2`. Với ref decode tokens/time `30/120`, method `24/60` ⇒ DSR rate `1.6`. Sample thứ ba fail ⇒ exclusion count tăng, không chèn 0. Chạy thêm trường hợp `output_tokens=1`, thiếu phase, mixed resources, shared-set nhỏ, bootstrap seed ổn định.

**Done:** một lệnh collector từ JSONL gốc tái tạo cùng báo cáo không cần model/GPU.

## Subtask 6: Runner paper-profile và one-shot validation [INTEGRATION] — implemented; B200 validation pending

**Files:** `src/Benchmark/run_longbench_200.py`, `src/Benchmark/common/longbench_adapter.py`, `src/Benchmark/common/io_util.py` nếu cần, `src/Benchmark/collect_metrics.py`, `tests/test_safe_longbench.py`, `tests/test_longbench_metric_contract.py`; cập nhật `docs/baselines/speedup_metric_contract.md` nếu contract phải làm rõ.

**Assemble:** Thêm profile/flag paper-speedup vào runner. Pin reference trước mọi cell; chạy reference sớm để fail fast nhưng giữ thứ tự method ghi manifest; ghép sau run bằng estimator v2 thay vì `_attach_external_reference_metrics()` sửa raw file. **Sửa cả đường chính và retry:** hiện `cfg["skip_reference"] = external_reference_path is not None` và `retry_cfg` tương tự; bỏ skip cho DFlash paper-profile. Dù vậy `--retry-failed-samples` phải tắt trong bảng paper chính vì retry trong session/load khác; nếu người vận hành bật, ghi attempt riêng và không trộn vào main estimator. Tắt data parallel dùng chung GPU, giữ batch/concurrency 1. Sau mỗi cell, kiểm tra 100 status/sample, output không rỗng, prompt/config hash, timing hữu hạn, reference availability; lưu partial JSONL khi lỗi. Ba anchor ngắn/vừa/dài đặt đầu mỗi cell và chạy lại cuối session; repeats chỉ vào calibration file. Ngưỡng trước run: ≥95/100 success/cell, ≥90/100 shared-set/dataset, ≥90/100 native e2e và decode-rate cho mỗi speculative baseline/dataset, anchor drift ≤10%; vượt ngưỡng thì output vẫn có nhưng bảng ranking đánh dấu không đủ điều kiện, không chạy lại âm thầm.

**CPU integration check:** matrix giả 2–3 sample × 6 baseline, sidecar SGLang, một record failure và một output mismatch; chạy runner/collector không model, kiểm tra cả file audit, paper CSV/MD, reference pin, shared-set, native refs và không có raw mutation. Chỉ những check này mới cần thiết trước GPU vì lỗi join sau GPU rất đắt.

**B200 validation gate:** dùng master config `_Viet`, preflight checkpoint/wheel/GPU, rồi chạy smoke **một sample/dataset/method** với cùng paper-profile. Kiểm tra `contract_version=2`, count/hash/boundary/phase/fingerprint, Domino actual SGLang algorithm, DFlash block-1 tồn tại và target-only SGLang reuse. Nếu strict common DSR không có ở cả sáu, dừng trước full run để instrument; không sinh bảng giả. Smoke có thể có `max_new_tokens=8`; full dùng budget đã khóa trong manifest.

**Full B200 run:** một production run gồm bốn dataset × 100 mẫu với pinned config; không đổi model, backend, batch, sampling hoặc reference giữa cell. Mỗi cell flush JSONL; nếu fail, giữ partial output và status. Sau run, collector offline sinh hai bảng: per-dataset pairwise và six-way shared-set, 95% paired bootstrap CI, quality/length/acceptance/coverage và calibration drift. Không claim run-to-run variance đã được đo.

**Expected conclusion:** Nếu gate và ngưỡng pass, bảng paper có common ESR/rate cho cả sáu, native ESR/DSR cho bốn speculative methods, strict common DSR cho cả sáu, cùng chất lượng và valid-pair counts. Nếu một metric thiếu, report `null`/reason; không đổi định nghĩa sau khi nhìn số và không gọi bảng đó là complete.

## Thứ tự thực hiện và giới hạn công bố

Thứ tự 1 → 2/3/4 → 5 → 6. Hoàn thành tất cả code và CPU checks trước khi tốn B200; smoke gate là điểm cuối trước full run. Một full run và paired bootstrap đủ để báo **kết quả đo trong một phiên** khi workload/config/coverage rõ; không đủ để khẳng định độ ổn định giữa nhiều phiên GPU. Bài báo phải ghi giới hạn đó, cùng GPU-hours, checkpoint revisions, điều kiện cache, output length và chất lượng. Không biến output mismatch thành timing failure, nhưng cũng không dùng mismatch để tuyên bố lossless decoding.

## Lệnh kiểm tra profile và chạy B200

Chạy CPU semantics checks trong repo (không tải model):

```bash
PYTHONPATH=src python3 -m pytest \
  tests/test_longbench_metric_contract.py \
  tests/test_sglang_adapter.py \
  tests/test_sglang_algorithm_preflight.py \
  tests/test_eagle_compat.py \
  tests/test_vanilla_inference.py \
  tests/test_dflash_compat.py \
  tests/test_safe_longbench.py -q
```

Kết quả đã chạy trên CPU: `91 passed`, không gọi CUDA; synthetic ESR `2.0`, decode-rate ratio `1.6`, sidecar/config mismatch, output mismatch và anchor drift đều được kiểm tra. Trên B200, `scripts/run_longbench_200.sh` tự đọc `config/master.path` trỏ tới master `_Viet`; chỉ dùng `--config` nếu chủ ý override. `--paper-speedup` đã được triển khai. Chỉ chạy inference trên B200:

```bash
bash scripts/run_longbench_200.sh --paper-speedup --preflight-only --mode full \
  --baselines 'vanilla_fa vanilla_hf dflash eagle3 domino dspark' \
  --datasets 'vietnews wikilingua vims vlsp' \
  --no-data-parallel --dp-processes-per-gpu 1 \
  --no-retry-failed-samples --sample-retries 0 --strict
```

Expected: 24/24 cell ready, đúng master `_Viet`, pinned common reference và checkpoint/algorithm fingerprints hiện trong manifest; không phát generation. Sau đó smoke 1 mẫu/cell và kiểm tra `audit_v2.json` trước full:

```bash
bash scripts/run_longbench_200.sh --paper-speedup --mode smoke --max-samples 1 \
  --baselines 'vanilla_fa vanilla_hf dflash eagle3 domino dspark' \
  --datasets 'vietnews wikilingua vims vlsp' \
  --no-data-parallel --dp-processes-per-gpu 1 \
  --no-retry-failed-samples --sample-retries 0 --strict
```

Kỳ vọng: 24/24 ô thành công; hash prompt và số token client/server khớp; gate decode common/native đạt; DFlash có block-1; Domino/DSpark dùng chung sidecar target-only. Full run chỉ khi smoke audit đạt gate:

```bash
bash scripts/run_longbench_200.sh --paper-speedup --mode full --max-samples 100 \
  --smoke-audit-run outputs/longbench_viet_100/<smoke_run_id> \
  --baselines 'vanilla_fa vanilla_hf eagle3 dflash domino dspark' \
  --datasets 'vietnews wikilingua vims vlsp' \
  --no-data-parallel --dp-processes-per-gpu 1 \
  --no-retry-failed-samples --sample-retries 0 --strict
```

Expected: một `run_id` với 24 cell, mỗi cell 100 sample/status và một summary; `paper_speedup.csv/.md`, `audit_v2.json`, `run_manifest.json`, sidecars và raw JSONL trong `outputs/longbench_viet_100/<run_id>/`. Kiểm tra audit/coverage trước khi dùng số liệu cho bài báo; không diễn giải `null` hoặc unsupported status như speedup.
