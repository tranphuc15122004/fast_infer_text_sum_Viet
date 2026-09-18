# Baseline `dspark`

Runner dùng algorithm DSpark chính thức của SGLang/SpecForge, với source
vendored tại `externals/SpecForge`; repo này không reimplement draft/correction
head. Entry là `scripts/infer_dspark.py`.

Draft checkpoint dùng `MODEL_DSPARK_DRAFT` hoặc `LONG_BENCH_DSPARK_MODEL`.
Các raw field `spec_*`, draft/verification latency, server E2E và response
metadata được giữ lại để tính metric sau này.

```bash
bash scripts/run.sh dspark
```
