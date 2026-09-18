# Baseline `vanilla_fa`

Target Qwen3-4B chạy cùng implementation Transformers nhưng yêu cầu
`flash_attention_2` và wheel `flash-attn` tương thích CUDA/GPU. Output batch-1
được dùng làm dense reference ưu tiên cho các baseline speculative; không có
một lần vanilla batch-1 phụ nào được chạy thêm.

```bash
bash scripts/run.sh vanilla_fa
```
