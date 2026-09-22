# Baseline `domino`

Runner dùng algorithm DFLASH/Domino chính thức của SGLang, được phát hành
qua `externals/Domino`/runtime SGLang; repo này chỉ chuẩn hóa JSONL, prompt,
server lifecycle và metadata. Implementation là `src/Benchmark/infer_domino.py`;
launcher là `scripts/run_domino.sh`.

Draft checkpoint dùng `MODEL_DOMINO_DRAFT` hoặc `LONG_BENCH_DOMINO_MODEL`.
Batch tự động trên B200 180 GiB bắt đầu ở 8, có thể override bằng
`LONG_BENCH_AUTO_BATCH_SIZE` hoặc `LONG_BENCH_BATCH_SIZE`; `LONG_BENCH_TP_SIZE`
điều khiển tensor parallel.

```bash
bash scripts/run.sh domino
```
