Đúng. Quy trình trên B200 là:

1. Tạo/chỉnh sửa master env ở đường dẫn mong muốn, gồm các model/checkpoint:

```bash
MASTER=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master_Viet.env
export FAST_INFER_MASTER_CONFIG="$MASTER"
```

2. Chạy setup + preflight:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects

python3 scripts/setup_server_env.py \
  --all \
  --master-config "$MASTER"
```

Lệnh này sẽ:

- kiểm tra Python 3.12 và dependency;
- kiểm tra `datasets/eval_100`;
- cập nhật `config/master.path` trỏ tới master env;
- kiểm tra runtime offline/CUDA.

Nếu thiếu package và đã có wheelhouse offline:

```bash
python3 scripts/setup_server_env.py \
  --all \
  --master-config "$MASTER" \
  --install-dependencies \
  --offline \
  --wheelhouse /path/to/offline_wheelhouse
```

3. Chạy smoke benchmark:

```bash
bash scripts/run_longbench_200.sh \
  --mode smoke \
  --baselines "vanilla_hf vanilla_fa eagle3 dflash domino dspark" \
  --datasets "vietnews wikilingua vims vlsp"
```

4. Sau khi smoke pass, chạy representative rồi full:

```bash
bash scripts/run.sh longbench_200 \
  --mode full \
  --data-parallel \
  --gpu-ids 0,1,2,3,4,5,6,7
```

Nếu master đã ở đúng đường dẫn canonical và `config/master.path` đã trỏ đúng, có thể bỏ `--master-config "$MASTER"`. Điểm quan trọng là launcher và `setup_server_env.py` phải cùng dùng `fast_infer_master_Viet.env`, không dùng master của project tham chiếu.a