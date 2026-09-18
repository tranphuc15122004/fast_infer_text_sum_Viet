# Baseline `vanilla_hf`

Target Qwen3-4B chạy bằng implementation Transformers eager attention trong
`scripts/common/vanilla_inference.py`. Đây là dense reference batch-1 của
`vanilla_hf`; runner ghi prefill, TTFT, decode, E2E, TPOT, throughput, peak
VRAM và ROUGE.

Chạy riêng:

```bash
bash scripts/run.sh vanilla_hf
```
