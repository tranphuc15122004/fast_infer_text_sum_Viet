# Baseline `eagle3`

Runner dùng code EAGLE-3 Qwen3 vendored trong `externals/EAGLE`, qua entry
`scripts/infer_eagle3.py`. Draft checkpoint được cấu hình bằng
`MODEL_EAGLE_DRAFT` hoặc `LONG_BENCH_EAGLE_MODEL`.

Runner bỏ naive pass nội bộ của EAGLE-3 khi đã có output `vanilla_fa`/`vanilla_hf`;
timing dense được join theo `sample_id` sau inference.

```bash
bash scripts/run.sh eagle3
```
