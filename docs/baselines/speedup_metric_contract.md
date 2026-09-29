# Hợp đồng đo speedup cho ma trận baseline

## Mục tiêu và hai phép so sánh

Mỗi ô `(baseline, dataset)` dùng cùng `sample_id`, target model, prompt, giới hạn
input/output, sampling, phần cứng và cấu hình tải. Không dùng độ giống nhau của
output để quyết định có tính thời gian hay không. Output và chất lượng được báo
cáo riêng.

1. **Speedup chung giữa các baseline** dùng đúng **một** reference cố định cho
   cả run: `vanilla_fa` theo mặc định. Có thể chọn `vanilla_hf` *trước khi bắt đầu
   run*, nhưng không được fallback từng dataset hoặc từng sample. Reference này
   chạy một lần trên mỗi dataset, rồi join bằng `sample_id` với cả sáu baseline.
   Đây là số dùng để so sánh/rank baseline.
2. **Speedup nội bộ của thuật toán** dùng target-only trong chính runtime của
   từng baseline. DFlash: `block_size=1`; EAGLE3: `naivegenerate`; Domino và
   DSpark: SGLang `target_only`. Các tỷ số này trả lời “speculation giúp được
   bao nhiêu trong runtime đó”; không so sánh trực tiếp tỷ số nội bộ của hai
   runtime khác nhau. Vanilla HF/FA không cần reference nội bộ.

Không ghi cả hai phép so sánh vào cùng key `dense_*`/`speedup` rồi suy ra ý nghĩa
qua `speedup_scope`. Lưu riêng `common_reference_*` và `native_reference_*`, cùng
`reference_id`, `timing_scope`, `phase_definition` và `valid_pair_count`.

| Baseline | Reference chung | Reference nội bộ |
|---|---|---|
| `vanilla_fa` | chính nó: 1,0× khi là reference của run | không áp dụng |
| `vanilla_hf` | `vanilla_fa` | không áp dụng |
| `eagle3` | `vanilla_fa` | EAGLE `naivegenerate` |
| `dflash` | `vanilla_fa` | DFlash `block_size=1` |
| `domino` | `vanilla_fa` | SGLang `target_only` |
| `dspark` | `vanilla_fa` | SGLang `target_only` |

Nếu cả run chọn `vanilla_hf` làm reference, đổi cột reference chung đồng loạt;
`vanilla_hf` nhận 1,0× và `vanilla_fa` được so với `vanilla_hf`. Không gán 1,0×
cho một baseline khi reference không có dữ liệu thành công.

## Ranh giới thời gian đã triển khai

**End to end chung** (`request_wall_ms`) là monotonic wall time phía client từ
trước format/tokenize prompt đến khi output text sẵn sàng. Nó không gồm load
model, start server, warmup, ghi JSONL hay ROUGE. Với SGLang gồm loopback HTTP
request và response parsing; các adapter trong process đồng bộ CUDA quanh vùng
GPU tính giờ. Paper profile khóa batch/concurrency/TP ở 1 và tắt reuse prefix
cache giữa các request.

**Decode chung** ghi `decode_active_ms`, `decode_token_count` và một phase
identifier chung. Các adapter không tái sử dụng `e2e_ms`, `completion_latency`
hoặc `prefill_ms` bằng cách đổi tên. Vanilla và DFlash dùng phase timer đã
instrument; EAGLE dùng patch có version và SHA nguồn; SGLang paper profile khóa
SGLang 0.5.20, bật server metrics và tính thời gian decode từ
`(completion_tokens - 1) / decode_throughput`. Định nghĩa timestamp/thông lượng
của bản SGLang được pin ở [source `req_time_stats.py` v0.5.20](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/observability/req_time_stats.py).
Chỉ ghép DSR khi cả hai record cùng khai báo phase đã xác minh; output một token
không có decode-rate pair.

`native_elapsed_ms` vẫn tách khỏi `request_wall_ms`: DFlash/EAGLE dùng clock
generation native trong cặp cùng runtime; Domino/DSpark dùng SGLang target-only
client request cùng server configuration. Common end-to-end dùng outer wall
cho cả sáu baseline.

## Điều kiện ghép cặp và công thức

Một cặp timing hợp lệ khi cả hai run thành công, cùng `sample_id`, dataset,
target model, prompt thực tế sau cắt input (hash/token count), generation config,
điều kiện tải và timing tương ứng hữu hạn, dương. Với decode, thêm điều kiện
`phase_definition` trùng nhau. Không yêu cầu cùng text hoặc cùng output length.
Không dùng sample thiếu timing để tạo tỷ số; lưu lý do loại và số cặp hợp lệ.

Với tập cặp hợp lệ `I` của **từng metric**:

```text
end_to_end_speedup = Σ[i∈I] reference_request_wall_ms[i]
                   / Σ[i∈I] method_request_wall_ms[i]

decode_time_ratio  = Σ[i∈I] reference_decode_active_ms[i]
                   / Σ[i∈I] method_decode_active_ms[i]

decode_token_rate_ratio =
  (Σ method_decode_tokens / Σ method_decode_active_ms)
  / (Σ reference_decode_tokens / Σ reference_decode_active_ms)
```

`decode_time_ratio` là tỷ số thời gian quan sát được; khi output dài khác nhau,
nó cũng phản ánh lượng việc khác nhau. `decode_token_rate_ratio` chuẩn hóa theo
số token sinh trong phase, nhưng vẫn không bảo đảm hai output tương đương về
chất lượng. Chỉ gọi nó là tỷ số tốc độ token, không gọi là speedup cùng công
việc. Tính tỷ số từ tổng thời gian trên đúng cùng tập cặp, không lấy trung bình
của các tỷ số từng sample. Với `output_tokens <= 1`, không đưa sample vào tỷ số
token/giây decode; end to end vẫn có thể hợp lệ.

Mỗi ô báo `n_expected`, `n_success`, `n_common_e2e_pairs`,
`n_common_decode_pairs`, `n_native_e2e_pairs`, `n_native_decode_pairs`; báo
`null` khi không có cặp. Bảng xếp hạng sáu baseline dùng **giao các sample thành
công của cả sáu** để tránh so sánh các subset dễ/khó khác nhau; báo thêm kết
quả theo cặp với reference để không mất dữ liệu khi một baseline lỗi.

## Output và chất lượng đi kèm

Cho cả reference chung và nội bộ, báo tỷ lệ trùng token/text, tỷ lệ độ dài
output, số token trung bình, ROUGE theo gold, tỷ lệ output suy biến, số sample
thiếu reference. Không loại timing vì output lệch. Cặp có output lệch vẫn là
*observed latency* hợp lệ, nhưng không diễn giải tỷ số đó thành lợi ích trên
cùng một câu trả lời. Không retry chỉ để ép output giống nhau; cách này giữ chi
phí GPU trong một pass reference chung và các pass target-only đang có.

## Trạng thái triển khai

Đã có estimator v2 và report offline trong
`src/Benchmark/common/paired_reference.py`; collector tạo `audit_v2.json`,
`paper_speedup.csv` và `paper_speedup.md`. Mỗi metric có tập ID, số cặp, lý do
loại, CI paired bootstrap và quality fields riêng. Report có pairwise rows,
six-way shared-set theo từng metric, và geometric mean bốn dataset với trọng số
bằng nhau.

Runner `--paper-speedup` pin một common reference cho cả run, giới hạn đúng sáu
baseline/bốn dataset, yêu cầu batch/concurrency/TP 1, tắt retry và data parallel,
chạy chung SGLang target-only sidecar cho Domino/DSpark, và từ chối full run nếu
smoke audit không qua. Full profile chạy ba anchor ngắn/vừa/dài ở đầu và cuối;
độ drift tuyệt đối tối đa phải không quá 10%.

Timing mismatch/degenerate output không xóa cặp thời gian. Chúng được ghi như
quality signals. Các cặp chỉ bị loại theo status, sample/config/prompt/resource
identity hoặc timing/phase thiếu/không hợp lệ. Đây là hợp đồng của report v2;
các cột legacy trong `metrics_summary.*` không dùng để xếp hạng paper.

Code và CPU fixture đã triển khai; chưa có số benchmark B200 trong repo. Chỉ
sau khi chạy smoke và full trên B200, audit đạt `paper_ready`, và coverage/quality
được kiểm tra thì mới có thể dùng các giá trị thực nghiệm trong bài báo. CI này
đo biến thiên theo mẫu trong một phiên, không đo độ ổn định giữa nhiều phiên.

## Domino và DSpark: reference nội bộ dùng chung

`run_longbench_200.py` tạo một `references/sglang_target_only/<dataset>.jsonl`
trước cell speculative đầu tiên. Domino và DSpark đọc đúng sidecar đó theo
`sample_id`; adapter xác minh contract v2, prompt/config/hardware/runtime
fingerprints, sample coverage, token count sau truncate, batch/concurrency,
cache policy và TP. Sai khác làm sidecar bị từ chối với reason cụ thể; không
chạy lại reference lặng lẽ.

Sidecar và method đều đo `request_wall_ms` phía client trong SGLang; native
reference dùng cùng SGLang 0.5.20 và target config, còn draft model/algorithm
không nằm trong fingerprint target-only. Paper profile tắt radix/prefix reuse,
để không dùng cache prefix chéo sample. SGLang target-only cũng cấp strict
decode timing qua metric server đã pin. Domino và DSpark vẫn có common pair
riêng với Vanilla, nên hai phép so sánh không bị trộn.

## Giao thức một lượt B200 để xuất số liệu cho bài báo

Giao thức này áp dụng cho bốn dataset tiếng Việt `vietnews`, `wikilingua`,
`vims`, `vlsp`, mỗi dataset 100 sample, và đúng sáu baseline trong bảng trên.
Một lượt production nghĩa là: mỗi sample chạy một lần với mỗi method và mỗi
reference bắt buộc; các bước chuẩn bị/kiểm toán/tổng hợp chạy trên CPU hoặc từ
JSONL đã lưu. Không dùng fixed-K sau EOS để ép output bằng nhau: benchmark đo
chính tác vụ tóm tắt với quy tắc dừng tự nhiên, còn độ dài/chất lượng là metric
đi kèm.

Ngân sách generation tối thiểu cho 400 sample là **9 lượt/sample = 3.600 lượt**:
Vanilla HF (1), Vanilla FA (1), DFlash method + block-1 (2), EAGLE method +
naive (2), Domino (1), DSpark (1), SGLang target-only dùng chung (1); cộng
warmup ngắn và ba calibration repeats/phase. Nếu Domino/DSpark không cùng
fingerprint, target-only cần thêm một lượt/sample. Không có cách hợp lệ để
báo cả common lẫn native speedup cho sáu baseline với ít generation hơn mà
không bỏ một loại reference. Có thể gom bốn dataset vào một server session
khi config hệt nhau để giảm số lần load/startup, nhưng vẫn tách JSONL và
bootstrap theo dataset.

### 1. Khóa manifest trước khi dùng GPU

Manifest v2 phải khóa SHA-256 của bốn input JSONL và danh sách/sample order;
model target và revision/checkpoint thực tế; tokenizer/chat template; draft
checkpoint từng method; dtype, attention backend, GPU model/ID/số GPU/TP;
phiên bản torch, transformers, SGLang và code commit; seed, greedy
`temperature=0`, EOS/stop, input cap, `max_new_tokens`, warmup, batch size 1,
concurrency 1, cache policy và clock boundary. Chọn **một** common reference
(`vanilla_fa` mặc định) trước run. Nếu FA không khả dụng, chọn `vanilla_hf`
trước khi đo bất kỳ sample nào và ghi lựa chọn vào manifest; không fallback
sau đó. Ghi thứ tự thực thi baseline; có thể đảo thứ tự method giữa dataset
để giảm bias do chạy method nào cũng đầu/cuối.

Cùng target checkpoint và tokenizer là điều kiện bắt buộc. Manifest cũng
phải ghi **speculative algorithm thực chạy**: adapter hiện mặc định map
`domino` sang SGLang `DFLASH`, còn `dspark` sang `DSPARK`. Chỉ gọi kết quả
Domino khi checkpoint Domino thực sự chạy đúng backend DFLASH đã định nghĩa
trước; nếu không, sửa adapter/config trước production và không gắn nhãn sai.
Nếu method cần số GPU/TP khác reference, tỷ số common chỉ là so sánh độ trễ với tài nguyên khác
nhau, phải ghi rõ tài nguyên và không gọi đó là speedup trên cùng phần cứng.
Với paper table chính, khóa cùng GPU budget/TP hoặc tách nhóm tài nguyên.
Trước run, CPU preflight tạo token ID đã chat-format và truncate cho **mọi**
method, so hash token IDs trên từng `sample_id`. Kiểm tra stop IDs và max-token
budget, không chỉ so prompt gốc. Không bắt đầu production khi có mismatch chưa
được giải quyết.

### Adapter map đã instrument

| Adapter | Outer `request_wall_ms` | Strict decode / native reference |
|---|---|---|
| Vanilla HF/FA | Bao prompt preparation, tokenization, generation, text decode | Local phase clock; là common control, không có native reference |
| DFlash | Bao tokenize, generation, decode | Version/SHA pinned phase timer; block-size 1 luôn chạy ở paper profile |
| EAGLE3 | Bao prompt path, paired generation và decode | Version/SHA pinned EAGLE timer; `naivegenerate` là native reference |
| Domino/DSpark | Client timer bao tokenize, HTTP request và parse output | SGLang 0.5.20 strict decode; một target-only sidecar dùng chung khi fingerprint khớp |

Runner/adapter ghi raw timing, sample ID, prompt hash, generation identity,
resource identity và phase definition. Khi field strict decode thiếu hoặc
không xác minh được, metric đó là `null` và smoke gate dừng trước full profile.
### 2. Đo trong một lượt, giữ mọi observation

Một cell chạy model/server warmup ngắn nhưng cùng request path rồi đo đúng
100 sample theo thứ tự khóa trước. GPU synchronization trước/sau timer trong
process; SGLang dùng client monotonic wall clock cho request và server metric
cho phase. Timer chung bắt đầu trước prompt preparation và kết thúc sau text
sẵn sàng; một adapter không được dùng thời gian chỉ quanh `generate()` để
điền `request_wall_ms`. Mỗi record lưu raw timing **chưa làm tròn**, status,
run/cell ID, sample ID, actual prompt token hash, actual input/output tokens,
EOS reason, text và token IDs nếu có, reference quality, config fingerprint,
clock source, timing boundary và thứ tự sample. Ghi theo kiểu append/flush để
một cell đứt giữa chừng vẫn giữ được mẫu đã hoàn tất; summary chỉ tạo sau khi
kiểm toán số record.

Lưu các phase native kèm `phase_definition` và số token thuộc phase. Record
không đủ timestamp/phase evidence không được nâng thành strict decode; estimator
để metric tương ứng `null` thay vì suy từ một field có tên gần giống.

Giữ reference nội bộ: DFlash `block_size=1`, EAGLE `naivegenerate`, và một
SGLang target-only artifact dùng chung cho Domino/DSpark. Paper runner giữ
DFlash block-1 ngay cả khi có Vanilla reference, tái sử dụng đúng sidecar theo
sample/fingerprint và chạy tuần tự trên cùng GPU.

Để phát hiện drift, runner đo ba anchor đại diện ngắn/vừa/dài trên cả sáu
baseline trước và sau ma trận full. Các anchor lưu tách khỏi JSONL dataset và
không đi vào estimator. Một drift >10% làm hỏng qualification gate; phép đo
này không thay thế independent-run variance, vì vậy cần ghi rõ rằng số liệu
phản ánh một phiên production.

### 3. Tính offline bằng một aggregator duy nhất

Không tin speedup summary tự ghi bởi từng adapter. Adapter chỉ ghi observation
thô và metadata; estimator trong `common/paired_reference.py` join common/native
reference, kiểm toán, tính metric và tái tạo report từ JSONL mà không gọi GPU.
Không để `dense_*`, `baseline_*`, `external_*` cùng đi vào một estimator.

Với mỗi metric `m`, tạo tập cặp hợp lệ riêng `I_m` từ các sample thành công,
cùng dataset/sample/prompt-token hash, target/checkpoint, generation config,
timing boundary của metric. Cả hai timing phải hữu hạn và dương.
Ghi footprint GPU cho mọi cặp: nếu khác GPU budget, vẫn có thể tính tỷ số
latency hệ thống với nhãn `unequal_resource`, nhưng loại khỏi shared-set
ranking công bằng và không gọi là speedup trên cùng phần cứng.
Output giống nhau **không** là điều kiện. Missing reference, unsupported
phase, prompt mismatch, nonpositive time, failure và cấu hình khác nhau có
reason code riêng. Metric nào thiếu cặp thì `null`, không lấy decode làm e2e
hoặc đổi reference.

```text
ESR_common(d,b) = sum_{i in I_e2e} request_wall_ms[common_ref,d,i]
                / sum_{i in I_e2e} request_wall_ms[b,d,i]
ESR_native(d,b) = sum_{i in I_native_e2e} native_elapsed_ms[native_ref,d,i]
                / sum_{i in I_native_e2e} native_elapsed_ms[b,d,i]
DSR_time(d,b)  = sum_{i in I_decode} decode_active_ms[ref,d,i]
                / sum_{i in I_decode} decode_active_ms[b,d,i]
DSR_rate(d,b)  = (sum decode_token_count[b] / sum decode_active_ms[b])
                / (sum decode_token_count[ref] / sum decode_active_ms[ref])
```

`DSR_time` là tỷ số thời gian quan sát được; `DSR_rate` là tỷ số token/giây
trong phase. Cả hai cần suffix `common` hoặc `native`. `ESR_common` là metric
xếp hạng chính giữa các method, `ESR_native` cho biết lợi ích thuật toán trong
runtime đó, với `native_timing_scope` giống nhau trong cặp. DFlash và
EAGLE có thể dùng generation-only wall time đã đồng bộ GPU; SGLang dùng
client request wall time. Tỷ số native không được rank xuyên runtime.
Vanilla reference nhận 1,0× identity; Vanilla còn lại được tính
bằng cùng aggregator. Không làm trung bình speedup từng sample. Tính tốc độ
absolute bằng tổng token / tổng thời gian trên đúng tập sample được công bố.

Làm hai bảng: **pairwise** mỗi baseline với common reference trên tất cả cặp
hợp lệ; **shared-set** chỉ dùng giao sample có observation hợp lệ của cả sáu
baseline cho cùng metric. Bảng xếp hạng dùng shared-set, ghi rõ `n` cho từng
metric; không lấy mean của các tỷ số theo dataset làm chỉ số chung. Nếu cần
một số gộp bốn dataset, dùng geometric mean của bốn `ESR_common` theo
shared-set, mỗi dataset một trọng số, đồng thời luôn in bốn tỷ số gốc. Tính
95% interval bằng paired bootstrap sample IDs trong từng dataset, seed và số
lần bootstrap cố định trong manifest; stratified bootstrap qua dataset cho số
gộp. Bootstrap này đo biến thiên **theo sample**, không đo biến thiên giữa
những lần chạy GPU độc lập.

### 4. Bảng paper và điều kiện được dùng để kết luận

Mỗi `(dataset, baseline)` công bố tối thiểu: target/draft, `n_expected`,
`n_success`, `n_common_e2e_pairs`, `n_common_decode_pairs`,
`n_native_e2e_pairs`, `n_native_decode_pairs`, ESR common/native và 95% paired
bootstrap interval, DSR time/rate nếu phase hợp lệ, median/p90 end to end
latency, output tokens trung bình, output-token ratio, exact/token match với
native target, ROUGE-1/2/L và tỷ lệ output suy biến. Tách biệt sample failures
với sample có output lệch. Chỉ gọi speculative method **lossless** khi đã xác
minh điều kiện đúng tương ứng; timing ratio vẫn được báo khi output lệch nhưng
được gọi là observed task latency, kèm quality.

Trước run, khóa ngưỡng coverage và drift trong manifest (ví dụ ít nhất 95/100
successful mỗi cell và 90/100 ở shared-set, anchor drift không quá 10%). Nếu
không đạt, vẫn xuất raw result và coverage nhưng không xếp hạng cell đó bằng
claim chắc chắn. Không thay ngưỡng sau khi thấy số. Không dùng confidence
interval theo sample để tuyên bố đã đo được độ ổn định giữa các phiên GPU.

### 5. Thứ tự triển khai và bước còn lại

Các bước 1–5 đã có trong source: schema/fingerprint v2; outer timing và strict
decode instrumentation cho các adapter; DFlash/EAGLE vendor patch có version
và SHA; common reference cùng native sidecars; runner smoke/full gates; paired
estimator, audit, bootstrap và bảng paper; CPU fixture tests. Bước còn lại là
B200 preflight, paper smoke, xem `audit_v2.json`, rồi full matrix chỉ khi smoke
đạt. Sửa lỗi từ raw JSONL/collector có thể làm offline, không cần chạy lại
model.

## Đối chiếu benchmark của các external repo

Các repo upstream không dùng một mẫu số hay một estimator chung, vì vậy không
copy nguyên tỷ số của họ vào cùng cột paper. Mượn **đường chạy reference**,
nhưng đo và aggregate theo contract thống nhất ở trên.

| Repo | Reference và cách tính upstream | Điều áp dụng trong ma trận này |
|---|---|---|
| `externals/dflash/dflash/benchmark.py` | Chạy `dflash_generate` với `block_size=1` và `block_size=K` trên cùng prompt; báo decoding speedup bằng tỷ số **mean time-per-output-token** | Giữ block-1 native pair; aggregator dùng tổng token/tổng phase time, tách ESR common/native và output quality |
| `externals/Domino/code/benchmark.py` | HF path cũng chạy `spec_generate(block_size=1)` và `block_size=K`, báo tỷ số mean per-sample token time | Đây là HF benchmark riêng của Domino; adapter production đang dùng SGLang nên native reference phù hợp là cùng SGLang `target_only` |
| `externals/Domino/code/benchmark_sglang.py` | Khởi động target-only và speculative server riêng, warmup/flush cache, báo tỷ số **aggregate output-token throughput** ở từng concurrency | Cùng ý tưởng target-only; thêm paired per-sample latency, hash prompt và quality cho workload Việt, batch 1 |
| `externals/EAGLE/eagle/evaluation/` | Sinh file EAGLE và baseline riêng; `speed.py` lấy tỷ số mean per-question token/speed của hai file | Dùng `naivegenerate` native pair ngay trong adapter, join sample ID; không dùng mean per-sample speed |
| `externals/DeepSpec/eval.py` | DSpark evaluator báo acceptance length/verify rate theo dataset; không có reference timing/speedup trong bảng này | Lấy DSpark speed từ SGLang method pass và target-only SGLang pass, không suy tốc độ từ acceptance |
| `externals/SpecForge/docs/sections/benchmarks/benchmark.md` | Config `1,0,0,0` là SGLang target-only; docs đề nghị so throughput của server target-only/spec cùng setting | Hỗ trợ lựa chọn reference SGLang; vẫn cần boundary, paired sample và quality do benchmark paper của mình yêu cầu |

Triển khai hiện cung cấp **phép so sánh task-level chung** bằng Vanilla và
**phép so sánh thuật toán nội bộ** theo runtime. Paper gate chỉ cho trạng thái
`paper_ready` khi smoke/full, phase, prompt count, coverage, shared-set và
anchor checks đạt. Với natural EOS, latency đo output thực tế có thể dài khác
nhau. Báo output length, quality và rate ratio; không suy ra các method đã tạo
cùng một chuỗi token hay làm cùng lượng decode work.


Dựng lại report sau này từ output đã lưu mà không nạp model/GPU:

```bash
python3 src/Benchmark/collect_metrics.py \
  --outputs-dir outputs/longbench_viet_100/<run_id> \
  --data-dir datasets/eval_100 --paper-speedup
```
