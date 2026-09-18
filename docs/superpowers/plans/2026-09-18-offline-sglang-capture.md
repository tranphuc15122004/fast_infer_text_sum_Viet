# OfflineSGLangCapture Implementation Plan

> **For Codex:** triển khai theo TDD, reuse code chính thức của SpecForge và
> chỉ chuyển sang SGLang sau khi parity gate đạt.

**Goal:** Tăng throughput tạo hidden-state cache bằng OfflineSGLangCapture mà
không thay đổi feature contract hoặc chấp nhận hidden state chưa được kiểm chứng.

**Experiment directory:** `src/Finetuning/` và `docs/superpowers/plans/`

**Validation scope:** TDD/unit, shared-venv static/preflight và GPU Modal smoke;
Qwen3-4B production run chỉ thực hiện khi snapshot và dependency SGLang tương
ứng đã có trong môi trường offline.

**Architecture:** Adapter mỏng gọi `externals/SpecForge` offline capture; HF là
backend mặc định/fallback. Capture CLI chọn backend, ghi metadata, và parity
validator chặn publish nếu sai.

---

## Shared Scaffold

### Existing infra (giữ nguyên)

- `src/Finetuning/capture_features.py`: adaptive batching, atomic publish,
  distributed merge.
- `src/Finetuning/adaptive_inference.py`: planner và OOM backoff.
- `src/Finetuning/features.py`: feature schema/manifest.
- `externals/SpecForge/specforge/offline_capture/`: implementation SGLang chính
  thức được vendored.
- `scripts/modal_finetuning_smoke.py`: Modal CUDA smoke hiện có.

### Needs setup

- Adapter backend và parity utility trong `src/Finetuning/`.
- CLI/config knobs cho capture backend và SGLang.
- Tests CPU và Modal smoke cập nhật.
- Tài liệu vận hành tiếng Việt.

## Subtask 1: Capture backend contract

**Role:** Chuẩn hóa interface chung cho HF và OfflineSGLangCapture.

**Implementation:** Tạo adapter có `capture_rows`/batched capture, metadata và
lỗi dependency rõ ràng; không import test/validation vào core.

**Unit Tests:** fake backend kiểm tra row splitting, layer order, dtype/shape và
lỗi khi backend không trả đủ hidden state.

### Steps

1. Viết test đỏ cho interface và row normalization.
2. Chạy test xác nhận fail vì adapter chưa tồn tại.
3. Implement adapter tối thiểu, reuse SpecForge import path.
4. Chạy test xanh và test hiện hữu của `capture_features`.

## Subtask 2: Integrate capture_features and configuration

**Role:** Cho phép chạy `hf`/`sglang` mà vẫn giữ manifest/output contract.

**Implementation:** Thêm `--capture-backend`, các SGLang flags, backend factory,
metadata stats, OOM backoff và guard không publish khi capture incomplete.

**Unit Tests:** CLI parsing, backend selection, manifest metadata và fallback
behavior; HF default phải giữ nguyên.

### Steps

1. Viết test đỏ cho CLI/config/backend selection.
2. Chạy test đỏ.
3. Nối adapter vào capture loop và cập nhật manifest/stats.
4. Chạy test xanh, compile và toàn bộ test Finetuning.

## Subtask 3: Numerical parity validator

**Role:** Xác nhận hidden SGLang tương thích với HF batch-1 trước full cache.

**Implementation:** Tạo utility/script parity chạy cùng tokenized fixture, so
sánh per-row/per-layer hidden và final hidden, ghi JSON report; ngưỡng được
cấu hình và lỗi gate có exit code khác 0.

**Unit Tests:** exact tensors pass; tensor sai layer/shape/giá trị fail; metric
ổn định với zero vectors và BF16 conversion.

### Steps

1. Viết test đỏ cho metrics/gate.
2. Chạy test đỏ.
3. Implement metrics, report và CLI.
4. Chạy test xanh.

## Subtask 4: Modal GPU integration [INTEGRATION]

**Role:** Xác nhận end-to-end trên GPU Modal với tiny Qwen3; nếu dependency
SGLang không có trong image thì fail rõ ràng, không giả vờ đã test SGLang.

**Implementation:** Cập nhật Modal image để mount source/SpecForge và pin
SGLang-compatible dependency; chạy HF cache, SGLang cache, parity, manifest
và kiểm tra output. Giữ biến `FAST_INFER_VENV` cho local launcher/preflight;
không cố copy host venv vào container.

**Integration Tests:** CUDA availability, capture backend selection, parity gate,
sample count, atomic manifest, feature shapes và retry behavior.

**Validation Pyramid:** L0 static/import + L1 GPU smoke; full Qwen3-4B chỉ khi
snapshot local được mount.

### Steps

1. Viết assertion đỏ cho Modal smoke cần cả HF/SGLang/parity artifacts.
2. Chạy test đỏ.
3. Cập nhật Modal script và command contract.
4. Chạy unit/preflight trước.
5. Chạy Modal GPU smoke, sửa lỗi dependency/API theo output thực tế.
6. Chạy lại Modal smoke và toàn bộ test.
