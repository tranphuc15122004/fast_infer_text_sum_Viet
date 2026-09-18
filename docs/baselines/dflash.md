# Baseline `dflash`

Runner dùng implementation DFlash Transformers vendored trong
`externals/dflash`, qua `scripts/infer_dflash.py`. Draft checkpoint được cấu
hình bằng `MODEL_DFLASH_DRAFT` hoặc `LONG_BENCH_DFLASH_MODEL`.

Khi chạy trong matrix, DFlash nhận `--skip-reference`; dense timing được join
từ output Vanilla cùng dataset/sample, tránh chạy lại block-size-1.

```bash
bash scripts/run.sh dflash
```
