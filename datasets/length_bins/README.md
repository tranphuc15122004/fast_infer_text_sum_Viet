# Length bins tiếng Việt

Bộ dữ liệu được tạo từ `datasets/normalized/` và dùng chung cho các baseline tóm
tắt tiếng Việt. Mỗi record giữ schema chuẩn `document` / `reference` và có thêm
`input_tokens`, `length_bin`, `length_bin_name`.

`input_tokens` là số token của prompt đầy đủ sau khi áp dụng prompt tiếng Việt,
chat template và tokenizer Qwen3-4B, cùng quy ước với benchmark inference.

| File | Khoảng token |
|---|---:|
| `bin1_00k_01k.jsonl` | `[0, 1.000)` |
| `bin2_01k_02k.jsonl` | `[1.000, 2.000)` |
| `bin3_02k_04k.jsonl` | `[2.000, 4.000)` |
| `bin4_04k_06k.jsonl` | `[4.000, 6.000)` |
| `bin5_06k_12k.jsonl` | `[6.000, 12.000)` |

`manifest.json` ghi cấu hình, số record khả dụng/được chọn, phân bố theo nguồn và
SHA-256 để kiểm tra tính toàn vẹn.

## Dùng khi benchmark

```bash
DATA_INPUT=datasets/length_bins/bin3_02k_04k.jsonl bash scripts/run.sh <baseline>
```

## Tạo lại

Local dùng tokenizer đã có trong cache:

```bash
FAST_INFER_VENV=/home/tuantb/fast_infer_text_sum/.venv \
  /home/tuantb/fast_infer_text_sum/.venv/bin/python scripts/data/build_length_bins.py
```

Trên server offline, truyền đường dẫn tokenizer Qwen3-4B local:

```bash
python3 scripts/data/build_length_bins.py --tokenizer "$MODEL_TARGET"
```

Mặc định chọn tối đa 50 mẫu/bin với seed 42. Nếu output đã tồn tại, thêm
`--overwrite` để tạo lại. Checkpoint trong thư mục output cho phép tiếp tục lần
quét bị gián đoạn.
