# VietBench Evaluation Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended) to implement this plan task-by-task.

**Goal:** Hoàn thiện quy trình đánh giá VietBench để mọi baseline trong ma trận 6 phương pháp có cùng sample coverage, metric contract, reference timing và báo cáo strict như runner canonical.

**Architecture:** Giữ orchestrator hiện tại làm nơi điều phối profile, retry và data-parallel; bổ sung một metric contract dùng chung để chuẩn hóa measurement scope và chặn cell không đủ dữ liệu. Các baseline inference ghi thêm telemetry/semantic metrics cần thiết; collector chỉ tổng hợp output canonical, không tính retry artifact.

**Tech Stack:** Python 3, pytest, JSONL, `scripts/common/metric_audit.py`, `scripts/common/metrics.py`, `scripts/common/rouge.py`, child-process runner hiện có.

**Spec:** Yêu cầu đối chiếu runner `/home/tuantb/fast_infer_text_sum/scripts/run_longbench_200.py` với runner hiện tại và hoàn thiện benchmark 6 baseline trên `datasets/eval_100/`.

## Global Constraints

- Dữ liệu benchmark chỉ lấy từ `datasets/eval_100/<dataset>_100.jsonl`, 4 dataset, 100 mẫu/dataset; manifest và checksum là nguồn xác nhận profile.
- Không dùng `unsupported_cpu`, `missing_checkpoint`, `missing_dependency`, `unsupported_dataset` như số liệu tốc độ; các timing của status record phải là `null`.
- Mọi record inference dùng `io_util.JsonlWriter`/schema hiện có; summary record vẫn là dòng cuối JSONL.
- Reference speedup chỉ join theo `sample_id` và chỉ hợp lệ khi output token budget của hai phía bằng nhau.
- Giữ các cải tiến Việt hiện có: `domino`/`dspark`, isolated sample retry, DP shard, port SGLang riêng theo shard, và không sửa dữ liệu raw/eval trong lúc chạy.
- Không chạy GPU trên máy T4; preflight/smoke logic phải chạy được CPU và full/representative phải yêu cầu CUDA trừ khi `--allow-unsupported`.

---

### Task 1: Viết test contract và reference selection trước khi sửa

**Files:**
- Create: `tests/test_longbench_metric_contract.py`
- Modify: `tests/test_baseline_registry.py`

**Interfaces:**
- Consumes: `common.metric_audit.audit_output_file`, `common.metric_audit.validate_cell_metric_contract`, `run_longbench_200._normalize_child_output`, `run_longbench_200._select_external_reference`.
- Produces: regression tests cho scope backfill, metric contract, quality guard và strict reference selection.

- [ ] **Step 1: Viết test failing cho scope và contract.**

```python
def test_viet_baseline_contract_requires_full_e2e_and_persists_summary(tmp_path):
    from common.metric_audit import audit_output_file

    record = {
        "method": "vanilla_hf", "dataset": "vietnews", "sample_id": "a",
        "status": "success", "measurement_scope": "full_e2e",
        "input_tokens": 10, "output_tokens": 4, "retained_tokens": 10,
        "batch_size": 1, "model_load_ms": 1.0, "peak_memory_gb": 2.0,
        "device": "cuda", "e2e_ms": 20.0, "prefill_ms": 3.0,
        "ttft_ms": 3.0, "decode_ms": 17.0, "throughput_tok_s": 200.0,
        "text": "tóm tắt", "reference_output": "tóm tắt",
        "rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0,
        "rouge1_p": 1.0, "rouge1_r": 1.0, "rouge1_f": 1.0,
        "rouge2_p": 1.0, "rouge2_r": 1.0, "rouge2_f": 1.0,
        "rougeL_p": 1.0, "rougeL_r": 1.0, "rougeL_f": 1.0,
        "rougeLsum_p": 1.0, "rougeLsum_r": 1.0, "rougeLsum_f": 1.0,
        "bleu1": 1.0, "bleu2": 1.0, "bleu3": 1.0, "bleu4": 1.0,
        "length_ratio": 1.0,
    }
    output = tmp_path / "vietnews.jsonl"
    output.write_text(json.dumps(record) + "\n", encoding="utf-8")

    summary = audit_output_file(
        output, baseline="vanilla_hf", dataset="vietnews",
        audit_path=tmp_path / "audit.json", expected_output_tokens=8,
        expected_samples=1,
    )

    assert summary["metric_contract"]["status"] == "complete"
    persisted = json.loads(output.read_text(encoding="utf-8").splitlines()[-1])
    assert persisted["metric_contract"]["status"] == "complete"
```

- [ ] **Step 2: Viết test failing cho `_normalize_child_output` backfill scope và quality guard.**

```python
def test_normalize_dflash_backfills_full_e2e_scope(tmp_path):
    from run_longbench_200 import _normalize_child_output

    output = tmp_path / "dflash.jsonl"
    output.write_text(json.dumps({
        "sample_id": "a", "status": "success", "input_tokens": 10,
        "output_tokens": 4, "e2e_ms": 20.0, "prefill_ms": 3.0,
        "ttft_ms": 3.0, "decode_ms": 17.0, "text": "tóm tắt",
    }) + "\n", encoding="utf-8")
    _normalize_child_output(
        output, baseline="dflash", dataset="vietnews",
        source_records=[{"id": "a", "reference_output": "tóm tắt", "task_type": "summarization"}],
        config={"model": "m", "device": "cuda", "max_new_tokens": 8},
        run_id="r1",
    )
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["measurement_scope"] == "full_e2e"
```

- [ ] **Step 3: Bổ sung test reference degenerate bị loại và chạy riêng test mới để xác nhận RED.**

Run: `PYTHONPATH=src pytest -q tests/test_longbench_metric_contract.py tests/test_baseline_registry.py`

Expected: FAIL vì contract API/scope map/quality guard chưa đầy đủ trong code hiện tại.

### Task 2: Port metric audit contract cho 6 baseline Việt

**Files:**
- Modify: `scripts/common/metric_audit.py`
- Test: `tests/test_longbench_metric_contract.py`

**Interfaces:**
- Consumes: normalized sample rows do runner tạo ra.
- Produces: `BASELINE_MEASUREMENT_SCOPE`, `required_direct_metrics()`, `validate_cell_metric_contract()`, và `audit_output_file(..., expected_samples=...)` với `metric_contract` trong summary/sidecar.

- [ ] **Step 1: Thêm baseline scope và raw metric policy theo tên hiện tại.**

```python
BASELINE_MEASUREMENT_SCOPE = {
    "vanilla_hf": "full_e2e", "vanilla_fa": "full_e2e",
    "eagle3": "full_e2e", "dflash": "full_e2e",
    "domino": "full_e2e", "dspark": "full_e2e",
}
_SPECULATIVE_BASELINES = {"eagle3", "dflash", "domino", "dspark"}
```

Giữ derived metrics (`throughput`, speedup, acceptance ratio) ngoài hard contract; yêu cầu coverage, scope, output budget, timing full-e2e, raw speculative telemetry và semantic quality fields.

- [ ] **Step 2: Port `required_direct_metrics`, `_missing_quality_fields` và `validate_cell_metric_contract` từ canonical rồi điều chỉnh baseline names.**

- [ ] **Step 3: Mở rộng `audit_output_file` nhận `expected_samples`, ghi `metric_contract` vào summary và payload audit.**

- [ ] **Step 4: Chạy test contract và test benchmark hiện có.**

Run: `PYTHONPATH=src pytest -q tests/test_longbench_metric_contract.py tests/test_safe_longbench.py tests/test_baseline_registry.py tests/test_vietbench_contract.py`

Expected: PASS.

### Task 3: Nối metric contract vào orchestrator, giữ retry/DP Việt

**Files:**
- Modify: `scripts/run_longbench_200.py`
- Modify: `tests/test_longbench_metric_contract.py`

**Interfaces:**
- Consumes: `validate_cell_metric_contract` từ Task 2 và output safe-recovery hiện có.
- Produces: mỗi cell có audit/contract persisted; cell success nhưng metric contract incomplete chuyển thành `metric_incomplete` khi strict; reference vanilla degenerate không được dùng làm dense timing.

- [ ] **Step 1: Import `BASELINE_MEASUREMENT_SCOPE` và `validate_cell_metric_contract`; backfill scope khi output không khai báo.**

Chỉ dùng `decode_only` khi EAGLE thực sự thiếu một trong các phase `prefill_ms`, `ttft_ms`, `decode_ms`, `e2e_ms`; các baseline còn lại lấy scope map hiện tại.

- [ ] **Step 2: Port `_reference_quality_error` và để `_select_external_reference` loại file malformed/degenerate.**

- [ ] **Step 3: Truyền `expected_samples=len(normalized)` vào mọi `_audit_cell_output` call; không gate status-only preflight rows thành dữ liệu tốc độ.**

- [ ] **Step 4: Sau audit, strict-gate `metric_contract.status`; tăng `failures`, ghi `metric_incomplete` vào cell/manifest, nhưng vẫn giữ raw output và cho collector chạy để báo cáo.**

- [ ] **Step 5: Chạy test RED→GREEN và smoke preflight CPU với đúng quoting.**

Run: `DEVICE=cpu CUDA_VISIBLE_DEVICES='' python3 scripts/run_longbench_200.py --mode smoke --baselines 'vanilla_hf vanilla_fa eagle3 dflash domino dspark' --datasets 'vietnews' --data-dir datasets/eval_100 --output-dir /tmp/fast_infer_viet_preflight --preflight-only --allow-unsupported --no-collect --no-retry-failed-samples`

Expected: tạo manifest/status cho 6 baseline, không launch model, không có timing hợp lệ.

### Task 4: Hoàn thiện telemetry và semantic metrics của baseline

**Files:**
- Modify: `scripts/eagle3_infer_qwen3.py`
- Modify: `scripts/infer_dflash.py`
- Modify: `scripts/infer_sglang_spec.py`
- Modify: `scripts/common/vanilla_inference.py`
- Test: `tests/test_longbench_metric_contract.py`

**Interfaces:**
- Consumes: cùng prompt/reference và output schema hiện tại.
- Produces: raw timing, scope, model/device/batch metadata, speculative acceptance/phase fields và full semantic metrics trên từng success record.

- [ ] **Step 1: Test output helper/record contract cho EAGLE và DFlash trước khi sửa.**

- [ ] **Step 2: EAGLE ghi `model_load_ms`, `retained_tokens`, `device`, `batch_size`, `draft_latency_ms`, `verification_latency_ms`, draft token counts, acceptance fields và giữ `measurement_scope` do phase timing quyết định.**

- [ ] **Step 3: DFlash ghi `model_load_ms`, `retained_tokens`, `device`, `peak_memory_gb`, `measurement_scope`, phase/acceptance telemetry và throughput đúng theo E2E/decode field tương ứng.**

- [ ] **Step 4: Gọi `metrics.add_semantic` sau `rouge.add_rouge` ở vanilla, DFlash, EAGLE và SGLang; giữ ROUGE summary tương thích.**

- [ ] **Step 5: Chạy static compile và unit tests.**

Run: `python3 -m py_compile scripts/run_longbench_200.py scripts/common/metric_audit.py scripts/eagle3_infer_qwen3.py scripts/infer_dflash.py scripts/infer_sglang_spec.py scripts/common/vanilla_inference.py && PYTHONPATH=src pytest -q tests`

### Task 5: Cập nhật tài liệu và kiểm chứng end-to-end offline

**Files:**
- Modify: `docs/vietbench_qwen3.md`
- Create: `docs/vietbench_evaluation.md`

**Interfaces:**
- Consumes: CLI/defaults, manifest/output schema và strict contract sau các task trước.
- Produces: hướng dẫn tiếng Việt tái lập preflight, smoke, representative/full, collector, audit và cách đọc status/metric-incomplete.

- [ ] **Step 1: Ghi bảng đối chiếu ngắn giữa canonical LongBench và VietBench 4×100.**

- [ ] **Step 2: Ghi lệnh chính xác với `--baselines '...'`, `--datasets '...'`, master `_Viet`, `FAST_INFER_VENV`, DP/VRAM/retry.**

- [ ] **Step 3: Ghi rõ dense reference selection, token-budget parity, metric contract và status không phải tốc độ.**

- [ ] **Step 4: Chạy preflight, audit/collector trên artifact status và xác nhận manifest không chứa raw data ngoài repo.**

Plan complete when: targeted tests, static compile và CPU preflight pass; GPU full benchmark chỉ được kết luận “ready to run” khi preflight server B200 xác nhận model/checkpoint/dependency.
