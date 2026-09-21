# fast_infer_text_sum_Viet

## Kiểm tra và setup môi trường server

Script không tải model/dataset. `--check` chỉ kiểm tra Python 3.12, master
config, bộ `datasets/eval_100` và import dependency trong chế độ offline.

```bash
# Server B200: setup + preflight
python3 scripts/setup_server_env.py --all

# Chỉ kiểm tra, không sửa pointer/master/dataset
python3 scripts/setup_server_env.py --check

# Cài manifest từ wheelhouse offline rồi kiểm tra lại
python3 scripts/setup_server_env.py --all \
  --install-dependencies --offline \
  --wheelhouse /workspace/storage-shared/nlp/dungdx4/phuc_projects/offline_wheelhouse
```

Máy local T4 chỉ dùng profile dependency tối thiểu và venv mô phỏng B200:

```bash
/home/tuantb/fast_infer_text_sum/.venv/bin/python \
  scripts/check_shared_env.py --profile minimal
```
