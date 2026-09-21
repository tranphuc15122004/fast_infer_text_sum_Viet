# Quy trình đánh giá VietBench trên B200

## Phạm vi và đối chiếu với runner canonical

Runner tham chiếu `/home/tuantb/fast_infer_text_sum/scripts/run_longbench_200.py`
điều phối theo ma trận `(baseline, dataset)`, chọn sample deterministic, chạy
preflight, stream log, ghi manifest và gọi collector. Runner Việt giữ nguyên
luồng đó nhưng thay profile bằng:

| Thành phần | LongBench tham chiếu | VietBench hiện tại |
|---|---|---|
| Dataset | `gov_report`, `qmsum`, `multi_news`, `lcc`, `repobench-p` | `vietnews`, `wikilingua`, `vims`, `vlsp` |
| Profile full | 100 mẫu/dataset | 100 mẫu/dataset từ `datasets/eval_100/*_100.jsonl` |
| Baseline | matrix canonical của project tham chiếu | `vanilla_hf`, `vanilla_fa`, `eagle3`, `dflash`, `domino`, `dspark` |
| Reference dense | Vanilla FA, fallback Vanilla HF | giống canonical |
| Recovery | retry shard/sample, giữ failure null timing | giữ retry sample, DP shard và port SGLang riêng |
| Strict | coverage + metric contract | coverage + metric contract theo 6 baseline Việt |

Không copy `config/master.path` của project tham chiếu: repo này phải trỏ tới
`fast_infer_master_Viet.env`.

## Kiểm tra trước khi chạy

Trên máy local chỉ chạy CPU preflight/smoke logic:

```bash
export FAST_INFER_VENV=/home/tuantb/fast_infer_text_sum/.venv
DEVICE=cpu CUDA_VISIBLE_DEVICES='' \
  python3 scripts/run_longbench_200.py \
  --mode smoke \
  --baselines 'vanilla_hf vanilla_fa eagle3 dflash domino dspark' \
  --datasets 'vietnews wikilingua vims vlsp' \
  --data-dir datasets/eval_100 \
  --output-dir /tmp/longbench_viet_preflight \
  --preflight-only --allow-unsupported --no-collect \
  --no-retry-failed-samples
```

Lệnh này không load model. `unsupported_cpu`, `missing_dependency`,
`missing_checkpoint` và `unsupported_dataset` chỉ là trạng thái môi trường,
không phải timing.

Trên B200, wrapper tự resolve master config và runtime:

```bash
bash scripts/run_longbench_200.sh \
  --mode smoke \
  --baselines 'vanilla_hf vanilla_fa eagle3 dflash domino dspark' \
  --datasets 'vietnews wikilingua vims vlsp'
```

## Profile benchmark

- `smoke`: 1 mẫu/dataset, tối đa 8 output token; dùng để kiểm tra wiring.
- `representative`: 20 mẫu/dataset, giữ phân tầng 5 bin; dùng trước full.
- `full`: 100 mẫu/dataset; chỉ chạy khi preflight B200 đã sẵn sàng.

Mọi subset được chọn deterministic bằng seed 42 mặc định. Full data-parallel
trên B200 dùng:

```bash
bash scripts/run.sh longbench_200 \
  --mode full --data-parallel --gpu-ids 0,1,2,3,4,5,6,7 \
  --sample-retries 2 --retry-backoff-seconds 5
```

`--baselines` và `--datasets` là một chuỗi; phải đặt trong quote. Runner lưu
raw output theo `outputs/longbench_viet_100/<run-id>/` và không ghi retry files
vào collector canonical.

## Điều kiện để số liệu được dùng

Mỗi success record phải có `measurement_scope`, input/output tokens, batch/device,
E2E và các phase timing cần thiết. `eagle3`, `dflash`, `domino`, `dspark` phải
giữ acceptance/phase telemetry khi upstream cung cấp. Semantic metrics được
ghi trên record; collector vẫn tính lại từ text/reference để báo cáo.

Runner chọn `vanilla_fa` làm dense reference trước, fallback `vanilla_hf`.
Reference bị malformed hoặc có `degenerate_repetition` sẽ bị bỏ qua. External
speedup chỉ hợp lệ khi join được bằng `sample_id` và số output token bằng nhau;
không diễn giải speedup của cặp lệch budget.

Nếu cell còn thiếu sample sau retry, runner ghi một failure row với timing
`null`, giữ `retry_history`, `unresolved_sample_ids` và không để sample đó biến
mất khỏi artifact. Nếu child exit 0 nhưng metric contract không complete,
strict vẫn đánh dấu cell `metric_incomplete`.

## Audit và tổng hợp

Cuối run, collector tạo:

- `metrics_summary.json`: dữ liệu đầy đủ;
- `metrics_summary.csv`: bảng tổng hợp;
- `metrics_summary.md`: báo cáo đọc được;
- `logs/<baseline>_<dataset>.metrics.json`: audit coverage/scope/quality.

Audit lại không chạy inference:

```bash
python3 scripts/audit_benchmark_metrics.py \
  --run-dir outputs/longbench_viet_100/<run-id> \
  --expected-output-tokens 2048
```

Đọc `run_manifest.json` trước khi dùng số liệu. Chỉ những cell có
`status=success`, `safe_eval_complete=true` và `metric_contract.status=complete`
mới được xem là kết quả benchmark strict; các status khác là thông tin chẩn
đoán môi trường hoặc failure recovery.
