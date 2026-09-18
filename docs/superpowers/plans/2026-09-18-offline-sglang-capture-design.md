# Thiết kế OfflineSGLangCapture cho cache hidden state tiếng Việt

## Mục tiêu

Tạo target hidden-state cache cho pipeline DFlash bằng backend SGLang nội bộ
theo cơ chế đã kiểm chứng trong SpecForge, nhưng vẫn giữ backend Transformers
làm fallback và không cho phép chạy full cache nếu parity với forward HF chưa
đạt.

## Phạm vi

- Áp dụng cho `src/Finetuning/capture_features.py`, không thay đổi schema
  `FeatureManifest`/feature record hiện tại.
- Sinh teacher trajectory vẫn giữ pipeline hiện tại; SGLang HTTP là một vấn đề
  riêng và không được dùng để suy ra hidden state.
- Reuse các module `externals/SpecForge/specforge/offline_capture` thay vì
  tự viết lại ModelRunner/capture hook.
- Hỗ trợ CUDA single GPU, TP/DP qua `torchrun`, batch packed prefill, adaptive
  batch và OOM backoff.
- Mặc định vẫn là Transformers; SGLang phải được chọn rõ ràng bằng CLI/config.

## Contract hidden state

Backend SGLang phải trả hidden state theo cùng thứ tự token, layer và dtype với
backend HF. Với mỗi sample, output được tách theo sequence length thực tế trước
khi ghi atomic feature record. `aux_hidden_states` được ánh xạ vào
`hidden_states` của schema repo; `last_hidden_states` được dùng trong parity
nhưng không làm thay đổi schema hiện tại.

## Parity gate

Một validator sẽ chạy cùng model snapshot, tokenizer, input IDs, attention mask,
layer IDs và dtype qua:

1. HF batch size 1;
2. HF batch size N;
3. SGLang OfflineCapture batch size N.

Validator ghi shape, max absolute error, mean absolute error, relative L2 error,
cosine similarity và greedy next-token/logit parity. Các ngưỡng được cấu hình;
không đạt gate thì trả lỗi và không publish cache SGLang.

## An toàn vận hành

- `capture_backend=hf` là mặc định.
- `capture_backend=sglang` yêu cầu import được SGLang/SpecForge-compatible
  internals và có parity gate bật.
- SGLang version được ghi vào manifest/stats cùng capture backend, layer IDs,
  TP/DP và batch size.
- Lỗi/OOM trong một batch dùng backoff; cache chỉ publish sau khi toàn bộ rank
  hoàn tất.

## Validation scope

- TDD CPU: contract adapter, layer mapping, row splitting, config và parity
  metric bằng tensor giả lập.
- Shared venv preflight: compile/import/CLI help và xác nhận thiếu dependency
  được báo rõ, không silent fallback.
- GPU Modal: tiny Qwen3 CUDA smoke, HF-vs-SGLang parity, feature manifest và
  multi-batch cache output. Full Qwen3-4B parity cần model snapshot được mount
  trong Modal; không tải model gated trong test.
