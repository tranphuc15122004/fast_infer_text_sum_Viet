# Logging cho pipeline Finetuning

Trên B200, dùng `scripts/run_finetuning_b200.sh` để mọi stage có log trong
`--output-root`. Launcher hiển thị trạng thái ngắn trên terminal và lưu đúng
các dòng trạng thái đó vào `logs/run.log`. Output stdout/stderr đầy đủ của từng
subprocess được lưu riêng theo stage:

```text
<output-root>/
├── logs/
│   ├── run.log
│   ├── generate_train.log
│   ├── generate_eval.log
│   ├── validate_teacher_train.log
│   ├── validate_teacher_eval.log
│   ├── cache_train.log
│   ├── cache_eval.log
│   ├── target_servers_launcher.log
│   └── target_servers/
│       ├── sglang_<port>_<index>.log
│       └── vllm_<port>_<index>.log
└── .state/
    └── <stage>.json
```

`run.log` ghi thời gian UTC, stage, split, model, backend, số record, số worker,
tiến độ định kỳ, warning, kết quả và lỗi. Các dòng warning/error từ vLLM/SGLang
được lấy từ log server do launcher quản lý và hiển thị gọn trên terminal; nội
dung đầy đủ vẫn nằm trong file server tương ứng. Log stage lưu cả stdout và
stderr của worker, bao gồm các sự kiện tiến độ.

Generate, kiểm định teacher và cache hiển thị tqdm theo số record đã đọc, kèm
tốc độ và ETA khi terminal hỗ trợ cập nhật động. Với `torchrun`, chỉ rank 0
hiển thị một thanh tiến trình. Launcher đếm record JSONL trước mỗi stage để
tạo mẫu số chính xác; lần đếm này đọc file tuần tự nhưng không đưa dữ liệu vào
RAM.

Khi chạy trực tiếp `python -m Finetuning.generate_targets` hoặc
`python -m Finetuning.capture_features`, lệnh hiển thị tqdm và summary cuối trên
terminal. Để có `run.log`, log stage và log backend được gom trong cùng run
folder, chạy qua `scripts/run_finetuning_b200.sh`.

Nếu generation endpoint SGLang/vLLM do một job khác vận hành, launcher chỉ có
thể lưu log client của stage generate. Muốn lưu log server vào run folder và
đưa warning server lên terminal, bật `--generation-launch-servers` để launcher
quản lý các process đó.
