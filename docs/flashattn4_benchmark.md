# Benchmark native Transformers với FlashAttention-4

Runner so sánh năm method `vanilla_hf`, `eagle3`, `dflash`, `domino`, `dspark`
trực tiếp trên GPU B200 của server. Cả target và draft đều phải dispatch qua
FlashAttention-4 (FA4); runtime gate dừng run nếu phát hiện attention fallback.
Inference chạy bằng Transformers native, batch size 1, greedy decoding, không
import hay gọi vLLM. Runner dùng Python 3.12 và FA4 `4.0.0b32` trên server;
nó không tạo virtualenv, cài package, hoặc tải checkpoint từ internet.

Target và draft checkpoint được lấy từ master config mà `config/master.path`
trỏ tới. Bốn JSONL dưới `datasets/eval_100/` là dữ liệu chung, mỗi file có 100
mẫu. Mọi method dùng cùng tokenizer, prompt tiếng Việt, seed và ngân sách token.
Mặc định representative chọn 20 mẫu/dataset phủ dải độ dài; lệnh bên dưới
override còn hai mẫu/dataset. Full chạy đủ 100 mẫu/dataset. Input dài hơn giới
hạn sẽ được truncate bằng helper dùng chung và ghi lại số token nguồn cùng cờ
truncation.

## Phiên bản attention và cơ chế native

Ghim `flash-attn-4[cu13]==4.0.0b32` trong requirements và image Modal để đồng bộ
với stack đã chạy trên B200: Torch 2.13/CUDA 13, CUTLASS DSL 4.7.1. Đây là đổi
pin của repo từ b19 sang b32; log B200 hiện tại đã có b32 nên không cần cài lại.
Preflight kiểm tra đúng phiên bản này. Không tự nâng Torch/Transformers trong
shared environment của server.

[Tài liệu FlashAttention chính thức](https://github.com/Dao-AILab/flash-attention)
chỉ định FA4 CuTe cho Hopper/Blackwell, gồm B200, và extra `cu13` cho CUDA 13;
FA3 nhắm Hopper. Chọn
[b32 trên PyPI](https://pypi.org/project/flash-attn-4/4.0.0b32/)
vì stack này đã có bằng chứng chạy trên server; không hạ sang FA2/FA3 chỉ để
cố lấy speedup cao hơn. Nếu server dùng phiên bản khác, cần mirror b32 cùng
các dependency cu13 vào wheelhouse trước khi cài offline. Không chạy installer
online trên server.

Runner gọi trực tiếp các entrypoint trong `externals/`; không sao chép lại
vòng draft/verify. Các adapter FA4 và compatibility Transformers vẫn cần thiết,
nên đây là bản port backend, không phải bản chạy nguyên xi môi trường tác giả.

| Method | Entry native và cấu hình |
|---|---|
| `vanilla_hf` | Target `model.generate()`, greedy, BF16, batch 1, FA4 |
| `eagle3` | `EaModel.eagenerate()` qua wrapper chuẩn hoá output/EOS; mặc định profile AR của project: total token 17, depth 16, top-k 1 |
| `dflash` | `dflash.model.dflash_generate()`, block size 16, greedy, giữ acceptance stats |
| `domino` | `DFlashDraftModel.spec_generate()`, causal correction bật; dùng `DraftCorrectionGraphRunner` gốc của Domino |
| `dspark` | `Qwen3DSparkEvaluator.generate_one_sample()`, threshold 0.0 đúng mặc định evaluator native |

EAGLE 17/16/1 là profile AR có trong launcher của project, không phải mặc định
constructor upstream (60/7/10). CLI cho phép đổi `--eagle-total-token`,
`--eagle-depth`, `--eagle-top-k`. Phiên bản trước cố định cây 18/4/2 để xử lý
vấn đề pruning; chạy smoke lại sau khi đổi profile là bắt buộc trước khi lấy
số liệu. Kích thước cây được kiểm tra trước khi tải model.

Domino dùng đúng kích thước graph như benchmark HF native: tính correction
steps từ block size, `shift_label` và `pure_draft_prefix_len` trong checkpoint.
Graph có bảng projection và bộ đệm bổ sung, nên phải đo lại VRAM; các peak cũ
21–23 GiB không đảm bảo phiên bản mới nằm trong 30 GB. Có thể chọn
`--no-domino-cuda-graph` để đo cấu hình không graph, được ghi rõ trong artifact.
Runner không tự chuyển sang cấu hình này khi graph thất bại.

Benchmark native DFlash dùng `dflash_generate(block_size=1)` làm reference và
báo speedup từ decode TPOT. Reference của bảng hiện tại vẫn là Vanilla HF
`model.generate()`. Vì khác reference và phạm vi đo, speedup 2–3x trước đây
không so trực tiếp với ESR/E2E hiện tại.

## Chạy trên server B200

Chạy từ thư mục checkout repo Việt trên server. Mặc định launcher đọc
`config/master.path`; để chỉ rõ master config `_Viet`, truyền path làm đối số
đầu tiên hoặc đặt `FAST_INFER_MASTER_CONFIG`:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum_Viet-main
export FAST_INFER_MASTER_CONFIG=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master_Viet.env
```

Preflight kiểm tra Python 3.12, FA4 import qua compatibility shim của repo,
B200 (SM100+), kernel tree-mask của EAGLE và tính toàn vẹn của `eval_100/`; nó
không nạp model checkpoint:

```bash
FI_GPU_IDS=0 bash scripts/run_fa4_benchmark.sh --preflight-only
```

Sau khi preflight pass, smoke nhanh một mẫu của riêng `vietnews` qua đủ năm
baseline:

```bash
FI_GPU_IDS=0 bash scripts/run_fa4_benchmark.sh \
  --mode smoke --datasets vietnews --samples-per-dataset 1 \
  --max-new-tokens 64 --warmup-tokens 64
```

Representative dưới đây chạy hai mẫu/dataset trên cả bốn dataset (tám prompt,
đủ năm baseline):

```bash
FI_GPU_IDS=0 bash scripts/run_fa4_benchmark.sh \
  --mode representative --datasets all --samples-per-dataset 2 \
  --max-new-tokens 512 --warmup-tokens 512 --repetitions 3
```

Full dùng đủ 100 mẫu/dataset:

```bash
FI_GPU_IDS=0 bash scripts/run_fa4_benchmark.sh \
  --mode full --datasets all --max-new-tokens 512 --repetitions 1
```

`FI_GPU_IDS=0` dùng GPU vật lý số 0. Có thể bỏ nếu master config đã chọn đúng
GPU hoặc scheduler đã gán `CUDA_VISIBLE_DEVICES`. Mỗi method được nạp và chạy
tuần tự trên cùng một GPU, vì vậy có thể chạy khi chỉ GPU 0 đang rảnh; runner
không chia một run sang nhiều card.

Các option chính: `--datasets`, `--methods`, `--samples-per-dataset`,
`--max-input-tokens`, `--max-new-tokens`, `--warmup-tokens`, `--repetitions`,
`--seed`, `--sample-retries`, `--checkpoint-interval`, `--run-id`, `--resume`,
`--direct-target-audit`, `--verifier-audit`, `--preflight-only`, `--output-dir`.
Mặc định output nằm dưới `outputs/fa4_native_benchmark/<run-id>/`; có thể nối
log ra file bằng `2>&1 | tee <log-file>`. Khi resume phải truyền cùng run ID
và cấu hình như lần chạy trước.

Warmup mặc định 512 token, được chặn bởi `max_new_tokens` và EOS. Mỗi prompt
được warmup trước khi đo để giảm ảnh hưởng biên dịch theo shape. Cấu hình native
và phiên bản package, SHA256 entrypoint native/adapter nằm trong signature resume;
dùng run ID mới cho bản port
này, không resume vào kết quả của runner cũ.

Nếu vLLM hoặc distribution liên quan nằm trong shared Python environment vì
các job khác, điều đó không làm benchmark này thành vLLM: runtime guard ghi tên
các distribution đang cài nhưng yêu cầu không module vLLM nào được import trong
process benchmark. Không gỡ hoặc cài lại package toàn cục.

## Metric và artifact

Mỗi sample/repeat lưu output IDs/text, model, input/output tokens, peak memory,
E2E, prefill/TTFT, decode, TPOT, throughput, QPS, draft/verification time,
acceptance counters, ROUGE-1/2/L, ROUGE-Lsum, BLEU-1..4, length ratio,
repetition/quality guard, exact greedy match và token LCS với Vanilla. Summary
báo mean/median/p90/std, DSR, ESR và metric theo từng dataset. DSR/ESR ghép cùng
sample và repeat; ESR tính trên số token output nhỏ hơn của cặp. Speedup báo theo
số đo thực tế, có thể nhỏ hơn 1.

Mặc định `--phase-timing-mode separate`: E2E vẫn đồng bộ CUDA ở đầu/cuối call,
nhưng bỏ các barrier chỉ dùng để profiling bên trong module native. Phép đo
không thay global `torch.cuda`, các tensor `.item()`, kernel FA4 hay vòng
accept/reject. Sau lượt đo, DFlash/EAGLE chạy thêm một generation có các timer
native gốc để lấy draft/verify time. Chỉ ghép các pha nếu token IDs, acceptance
lengths, accepted/proposed counts và số vòng verify khớp giữa hai lượt. Nếu
khác, các pha để `null` và metadata báo mismatch. Lượt profiling không được cộng
vào E2E, peak memory hay dispatch counters của lượt đo, nhưng làm thời gian
chạy toàn bộ benchmark dài hơn. Phase time của Domino/DSpark không được native
implementation cung cấp, để `null`.

`--phase-timing-mode inline` giữ các barrier native trong lượt đo, phục vụ
đối chiếu cách đo cũ. `--phase-timing-mode off` bỏ profiling generation riêng,
giữ acceptance/content/latency metrics nhưng để phase time `null`. Native
DFlash benchmark cũng dùng `return_stats=True`; không quy toàn bộ chênh lệch
tốc độ cũ cho overhead stats khi chưa có A/B cùng cấu hình.

Trong chế độ separate/off, `prefill_ms` dùng CUDA-event duration của target
forward đầu tiên cho mọi method. Đây là proxy GPU, không phải toàn bộ TTFT
(chưa tính tất cả công việc để lấy token đầu). Field `ttft_ms` giữ proxy này để
tương thích schema; nguồn đo ghi trong `extra_metrics`. Inline giữ native TTFT
khi có. `tpot_ms` là
`(e2e_ms - prefill_ms) / (output_tokens - 1)`. Queue wait, server startup, batch
wait và server-reported E2E để `null` vì đây không phải server request API.

Exact greedy và speedup > 1 mặc định là thông tin chẩn đoán. Bật
`--strict-greedy-parity` hoặc `--require-speedup` nếu muốn dùng chúng làm gate.
Runtime FA4/no-fallback, đầy đủ execution, output validity và schema vẫn phải
đạt. Status `success` không chứng minh nội dung tương đương: cần so ROUGE/BLEU
theo từng dataset và kiểm tra output. Target verify cũng không thay thế việc
kiểm chứng adapter mask/cache. Không sửa công thức hay ép speedup > 1.

Các sửa đổi mới đã kiểm tra logic trên CPU; chưa có smoke GPU B200 cho profile
mới. Preflight và smoke trên server phải đạt trước representative/full.

Artifact trong mỗi run directory gồm `results.jsonl` (sample records và summary
cuối), `run_report.json`, `report_vi.md`, `metrics_summary.csv`, `warmup.jsonl`,
`events.jsonl`, `samples.jsonl`, `excluded_samples.jsonl`, `progress.json`,
`state.json` và `results.partial.jsonl`.

## Chẩn đoán khi speedup full khác representative

Với `--samples-per-dataset 2`, bộ chọn lấy mẫu ngắn nhất và dài nhất sau
truncation. Đây là kiểm tra hai đầu dải độ dài, không phải ước lượng trung bình
toàn bộ dataset. Tuy nhiên, full chứa lại chính các sample đó: cần đối chiếu
latency của cùng sample/repeat giữa hai run trước khi kết luận khác biệt chỉ
đến từ cách lấy mẫu. FA4 dispatch không fallback chứng minh đường attention;
nó chưa xác nhận thuật toán hoặc adapter đã tối ưu.

Chạy công cụ CPU trên server, dùng artifact đã có; không nạp model hoặc gọi GPU:

```bash
python3 scripts/analyze_fa4_benchmark.py \
  --run-dir outputs/fa4_native_benchmark/fa4-b200-full-20261006 \
  --compare-dir outputs/fa4_native_benchmark/20261006T141035Z-67788a
```

Nếu thư mục representative trên server có tên khác, thay `--compare-dir`.
Có thể bỏ option đó để chỉ phân tích một run. Công cụ ghi
`diagnostics/diagnostics.json`, `diagnostics/report_vi.md`,
`diagnostics/inference_components.csv` và
`diagnostics/component_summary.csv` trong thư mục run hiện tại.
`inference_components.csv` có một dòng cho mỗi method/sample/repeat, gồm E2E,
prefill, decode đã ghi, draft/verify profile, acceptance, token count, parity,
dispatch/fallback FA4 và chênh lệch với Vanilla của cùng sample. Nếu dùng
`--compare-dir`, công cụ còn ghi `diagnostics/common_samples.csv`. Artifact
benchmark gốc được giữ nguyên.

`decode_ms` của native FA4 runner được suy ra từ `E2E - prefill`; nó không phải
timer kernel decode độc lập. Draft/verify phase ở chế độ `separate` được đo trong
một lượt profiling bổ sung, chỉ gắn với lượt chính nếu output và acceptance
khớp. Vì vậy không cộng draft/verify profile vào E2E. Domino và DSpark không có
timer pha native trong runner này; CSV sẽ giữ chúng là rỗng thay vì giả định 0 ms.

Ví dụ cho full run đã hoàn tất trên B200:

```bash
python3 scripts/analyze_fa4_benchmark.py \
  --run-dir outputs/fa4_native_benchmark/fa4-b200-full-20261006 \
  --output-dir /tmp/fa4-b200-full-20261006-diagnostics
```

Lệnh chỉ đọc JSONL/metadata, chạy bằng Python chuẩn, không nạp model hay cần GPU.

Báo cáo ghép đúng dataset/sample/repeat, gộp retry trùng và loại record lỗi hoặc
cặp thiếu timing/input không khớp. Nó báo E2E speedup trực tiếp, số mẫu method
nhanh hơn Vanilla, acceptance từ tổng counters, avg accept length, độ phủ
draft/verification timing và các nhóm độ dài input. Phase timing không có được
để `null`; không cộng timer khác phạm vi vào E2E. Bảng giữa hai run báo thay đổi
config/version/checksum, độ dài output và tỷ số thời gian của cùng sample.
Exact output IDs là chẩn đoán; kết luận chất lượng nội dung dùng metric trên
reference và margin đã chọn trước.

Nếu cần kiểm tra độ ổn định timing, bắt đầu bằng Vanilla–DFlash trên hai sample
VietNews đã có. Tăng warmup để chạy toàn bộ generation trước khi đo, lặp ba
lần và dùng run ID mới:

```bash
FI_GPU_IDS=0 bash scripts/run_fa4_benchmark.sh \
  --mode representative --datasets vietnews --samples-per-dataset 2 \
  --methods vanilla_hf,dflash \
  --max-input-tokens 8192 --max-new-tokens 512 \
  --warmup-tokens 512 --repetitions 3 \
  --run-id fa4-b200-dflash-diagnostic-20261007
```

Warmup dừng ở EOS nếu output ngắn hơn 512 token. Thử nghiệm này thay đổi warmup
so với run cũ, nên phải ghi rõ khi diễn giải. Dùng GPU không có job khác tranh
tài nguyên và giữ stack/model/prompt/greedy seed như run cũ. Cờ parity/speedup
của runner có thể vẫn làm exit code là 1 sau khi đã ghi đủ artifact; đọc
execution/runtime/quality/latency riêng để biết điều kiện nào chưa đạt. Khi
timing ổn định nhưng vẫn chậm, profile phần draft, target verification và
CPU/CUDA synchronization hoặc đối chiếu implementation native tác giả trên
cùng token IDs, checkpoint và FA4 trước khi sửa inference.

## Chạy qua Modal

Modal là đường chạy riêng, không sử dụng GPU server. Nếu cần gọi lại Modal, dùng
launcher tương thích này:

```bash
bash scripts/run_fa4_modal_benchmark.sh \
  --mode smoke --datasets vietnews --samples-per-dataset 1 --max-new-tokens 32
```
