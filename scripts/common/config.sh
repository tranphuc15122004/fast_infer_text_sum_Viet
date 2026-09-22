#!/usr/bin/env bash

# Shared configuration loader.
#
# The repository contains only a pointer (config/master.path). The actual
# master config is kept outside the repository so code updates do not require
# copying the server's model/runtime configuration again.

if [[ -z "${ROOT:-}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

fast_infer_default() {
  local destination="$1"
  local value="$2"

  if [[ ! -v "$destination" ]]; then
    printf -v "$destination" '%s' "$value"
  fi
  export "$destination"
}

fast_infer_default_from() {
  local destination="$1"
  shift
  local source_name
  local value

  if [[ -v "$destination" ]]; then
    export "$destination"
    return 0
  fi

  for source_name in "$@"; do
    if [[ -v "$source_name" ]]; then
      value="${!source_name}"
      if [[ -n "$value" ]]; then
        printf -v "$destination" '%s' "$value"
        export "$destination"
        return 0
      fi
    fi
  done

  return 0
}

fast_infer__resolve_path() {
  local candidate="$1"

  if [[ "$candidate" == /* ]]; then
    printf '%s\n' "$candidate"
  else
    printf '%s/%s\n' "$ROOT" "$candidate"
  fi
}

fast_infer_master_path() {
  local pointer_file master_file pointer_value

  if [[ -n "${FAST_INFER_MASTER_CONFIG:-}" ]]; then
    master_file="$(fast_infer__resolve_path "$FAST_INFER_MASTER_CONFIG")"
  else
    pointer_file="${FAST_INFER_MASTER_POINTER:-$ROOT/config/master.path}"
    pointer_file="$(fast_infer__resolve_path "$pointer_file")"

    if [[ ! -f "$pointer_file" ]]; then
      echo "fast-infer: master pointer not found: $pointer_file" >&2
      return 1
    fi

    pointer_value="$(awk '
      { sub(/[[:space:]]*#.*/, ""); gsub(/^[[:space:]]+|[[:space:]]+$/, "") }
      length { print; exit }
    ' "$pointer_file")"

    if [[ -z "$pointer_value" ]]; then
      echo "fast-infer: master pointer is empty: $pointer_file" >&2
      return 1
    fi

    master_file="$(fast_infer__resolve_path "$pointer_value")"
  fi

  if [[ ! -f "$master_file" ]]; then
    echo "fast-infer: master config not found: $master_file" >&2
    return 1
  fi

  printf '%s\n' "$master_file"
}

fast_infer__export_compatibility_aliases() {
  # Runtime/cache names.
  fast_infer_default_from FAST_INFER_PYTHON FI_PYTHON
  fast_infer_default_from CUDA_VISIBLE_DEVICES FI_GPU_IDS
  fast_infer_default_from B200_TARGET_GPU FI_TARGET_GPU
  fast_infer_default_from B200_DEVICE FI_DEVICE
  fast_infer_default_from HF_HOME FI_HF_HOME
  fast_infer_default_from TRANSFORMERS_CACHE FI_TRANSFORMERS_CACHE
  fast_infer_default_from TRITON_CACHE_DIR FI_TRITON_CACHE
  fast_infer_default_from FLASHINFER_WORKSPACE_BASE FI_FLASHINFER_CACHE
  fast_infer_default_from TORCH_EXTENSIONS_DIR FI_TORCH_EXTENSIONS_CACHE

  # Shared model/data names used by preflight and older helpers.
  fast_infer_default_from B200_TARGET_MODEL MODEL_TARGET
  fast_infer_default_from B200_DFLASH_MODEL MODEL_DFLASH_DRAFT
  fast_infer_default_from B200_DOMINO_MODEL MODEL_DOMINO_DRAFT
  fast_infer_default_from B200_DSPARK_MODEL MODEL_DSPARK_DRAFT
  fast_infer_default_from B200_SPEC_MODEL MODEL_SPEC_DRAFT
  fast_infer_default_from B200_EAGLE_MODEL MODEL_EAGLE_DRAFT
  fast_infer_default_from B200_LONGSPEC_DRAFT_MODEL MODEL_LONGSPEC_DRAFT
  fast_infer_default_from B200_VICUNA_MODEL MODEL_LONGSPEC_TARGET
  fast_infer_default_from B200_COMPRESSOR_MODEL MODEL_COMPRESSOR
  fast_infer_default_from B200_EMBEDDING_MODEL MODEL_EMBEDDING
  fast_infer_default_from B200_MAGICDEC_MODEL_PTH CHECKPOINT_MAGICDEC
  fast_infer_default_from B200_MAGICDEC_MODEL_NAME MODEL_MAGICDEC_NAME
  fast_infer_default_from B200_DATA_FILE DATA_INPUT

  # B200 runner compatibility names. The canonical master names are shorter
  # and belong to the B200 namespace.
  fast_infer_default_from B200_SMOKE_MAX_SAMPLES B200_MAX_SAMPLES
  fast_infer_default_from B200_SMOKE_MAX_NEW_TOKENS B200_MAX_NEW_TOKENS
  fast_infer_default_from B200_TIMEOUT B200_TIMEOUT_SECONDS

}

fast_infer_load_master() {
  if [[ "${FAST_INFER_MASTER_LOADED:-0}" == "1" ]]; then
    return 0
  fi

  local master_file
  master_file="$(fast_infer_master_path)" || return 1

  # Export every assignment from the shell-env master file. Caller-provided
  # variables are restored below as defaults are applied, so CLI/environment
  # overrides remain authoritative.
  set -a
  # shellcheck disable=SC1090
  source "$master_file"
  set +a

  fast_infer_default FI_OFFLINE 1
  if [[ "${FI_OFFLINE:-1}" == "1" ]]; then
    fast_infer_default HF_HUB_OFFLINE 1
    fast_infer_default TRANSFORMERS_OFFLINE 1
    fast_infer_default HF_DATASETS_OFFLINE 1
  fi

  fast_infer__export_compatibility_aliases
  export FAST_INFER_MASTER_CONFIG="$master_file"
  export FAST_INFER_MASTER_LOADED=1
}

fast_infer__load_common() {
  fast_infer_default_from MAX_SAMPLES RUN_SAMPLES
  fast_infer_default_from MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from MAX_INPUT_TOKENS RUN_MAX_INPUT_TOKENS
  fast_infer_default_from TEMPERATURE RUN_TEMPERATURE
  fast_infer_default_from SKIP_NAIVE RUN_SKIP_NAIVE
  fast_infer_default_from DOCUMENT_FIELD RUN_DOCUMENT_FIELD
  fast_infer_default_from ID_FIELD RUN_ID_FIELD
  fast_infer_default_from REFERENCE_FIELD RUN_REFERENCE_FIELD

  if [[ ! -v SMOKE ]]; then
    if [[ "${RUN_MODE:-}" == "smoke" ]]; then
      SMOKE=1
    else
      SMOKE=0
    fi
    export SMOKE
  fi

  if [[ ! -v FULL ]]; then
    if [[ "${RUN_MODE:-}" == "full" ]]; then
      FULL=1
    else
      FULL=0
    fi
    export FULL
  fi
}

fast_infer__load_longbench() {
  # LongBench deliberately has its own directory namespace.  Do not inherit
  # DATA_INPUT/OUTPUT_ROOT here: those variables may still point to the legacy
  # representative_100 experiment and are files/single-run outputs rather
  # than the canonical LongBench directory.
  fast_infer_default_from LONG_BENCH_DATA_FILE DATA_INPUT
  fast_infer_default_from LONG_BENCH_OUTPUT_FILE OUTPUT_FILE
  fast_infer_default_from LONG_BENCH_OUTPUT_DIR
  fast_infer_default_from LONG_BENCH_MODEL MODEL_TARGET
  fast_infer_default_from LONG_BENCH_DEVICE FI_DEVICE
  fast_infer_default_from LONG_BENCH_GPU_IDS FI_GPU_IDS
  fast_infer_default_from LONG_BENCH_DTYPE DTYPE
  fast_infer_default_from LONG_BENCH_BASELINES
  fast_infer_default_from LONG_BENCH_DATASETS
  fast_infer_default_from LONG_BENCH_MODE RUN_MODE
  fast_infer_default_from LONG_BENCH_SMOKE_SAMPLES
  fast_infer_default_from LONG_BENCH_REPRESENTATIVE_SAMPLES
  fast_infer_default_from LONG_BENCH_FULL_SAMPLES
  fast_infer_default_from LONG_BENCH_REPRESENTATIVE_DATASETS
  fast_infer_default_from LONG_BENCH_MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from LONG_BENCH_SMOKE_MAX_NEW_TOKENS
  fast_infer_default_from LONG_BENCH_TEMPERATURE RUN_TEMPERATURE
  fast_infer_default_from LONG_BENCH_WARMUP_RUNS
  fast_infer_default_from LONG_BENCH_SEED SEMANTIC_RANDOM_SEED
  fast_infer_default_from LONG_BENCH_MAX_INPUT_TOKENS RUN_MAX_INPUT_TOKENS
  fast_infer_default_from LONG_BENCH_LOCAL_FILES_ONLY
  fast_infer_default_from LONG_BENCH_TIMEOUT_SECONDS B200_TIMEOUT_SECONDS
  fast_infer_default_from LONG_BENCH_STRICT

  fast_infer_default_from LONG_BENCH_EAGLE_MODEL MODEL_EAGLE_DRAFT
  fast_infer_default_from LONG_BENCH_DFLASH_MODEL MODEL_DFLASH_DRAFT
  fast_infer_default_from LONG_BENCH_DOMINO_MODEL MODEL_DOMINO_DRAFT
  fast_infer_default_from LONG_BENCH_DSPARK_MODEL MODEL_DSPARK_DRAFT
  fast_infer_default_from LONG_BENCH_LONGSPEC_TARGET_MODEL MODEL_LONGSPEC_TARGET MODEL_TARGET
  fast_infer_default_from LONG_BENCH_LONGSPEC_DRAFT_MODEL MODEL_LONGSPEC_DRAFT
  fast_infer_default_from LONG_BENCH_SPECEXTEND_DRAFT_MODEL MODEL_EAGLE_DRAFT MODEL_SPEC_DRAFT
  fast_infer_default_from LONG_BENCH_SSSD_DATASTORE_PATH SSSD_DATASTORE_PATH
  fast_infer_default_from LONG_BENCH_MAGICDEC_MODEL_PTH CHECKPOINT_MAGICDEC
  fast_infer_default_from LONG_BENCH_MAGICDEC_MODEL_NAME MODEL_TARGET MODEL_MAGICDEC_NAME
  fast_infer_default_from LONG_BENCH_FAFO_KV_METHOD FAFO_KV_METHOD
  fast_infer_default_from LONG_BENCH_LONGSPEC_MODEL_NAME LONGSPEC_MODEL_NAME
  fast_infer_default_from LONG_BENCH_SPECEXTEND_MODEL_NAME SPECEXTEND_MODEL_NAME
  fast_infer_default_from LONG_BENCH_EAGLE_TOTAL_TOKEN EAGLE_TOTAL_TOKENS
  fast_infer_default_from LONG_BENCH_EAGLE_DEPTH EAGLE_DEPTH
  fast_infer_default_from LONG_BENCH_EAGLE_TOP_K EAGLE_TOP_K

  fast_infer_default LONG_BENCH_DATA_DIR "datasets/eval_100"
  fast_infer_default LONG_BENCH_OUTPUT_DIR "outputs/longbench_viet_100"
  fast_infer_default LONG_BENCH_MODEL "${MODEL_TARGET:-}"
  fast_infer_default LONG_BENCH_DEVICE "${FI_DEVICE:-cuda}"
  # Empty means "inherit CUDA_VISIBLE_DEVICES".  Do not default to physical
  # GPU 0: cluster schedulers can expose a different allocated index, and the
  # runner must preserve that mapping for both the parent and child process.
  fast_infer_default LONG_BENCH_GPU_IDS "${FI_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
  fast_infer_default LONG_BENCH_DTYPE "bfloat16"
  fast_infer_default LONG_BENCH_BASELINES "vanilla_hf vanilla_fa eagle3 dflash domino dspark"
  fast_infer_default LONG_BENCH_REFERENCE_BASELINE "vanilla_fa"
  fast_infer_default LONG_BENCH_DATASETS "vietnews wikilingua vims vlsp"
  fast_infer_default LONG_BENCH_MODE "smoke"
  fast_infer_default LONG_BENCH_SMOKE_SAMPLES "1"
  fast_infer_default LONG_BENCH_REPRESENTATIVE_SAMPLES "20"
  fast_infer_default LONG_BENCH_FULL_SAMPLES "100"
  fast_infer_default LONG_BENCH_REPRESENTATIVE_DATASETS "vietnews wikilingua vims vlsp"
  fast_infer_default LONG_BENCH_MAX_NEW_TOKENS "2048"
  fast_infer_default LONG_BENCH_SMOKE_MAX_NEW_TOKENS "8"
  fast_infer_default LONG_BENCH_REPRESENTATIVE_TIMEOUT_SECONDS "3600"
  fast_infer_default LONG_BENCH_FULL_TIMEOUT_SECONDS "21600"
  fast_infer_default LONG_BENCH_TEMPERATURE "0"
  fast_infer_default LONG_BENCH_WARMUP_RUNS "3"
  fast_infer_default LONG_BENCH_SEED "42"
  fast_infer_default LONG_BENCH_MAX_INPUT_TOKENS "0"
  fast_infer_default LONG_BENCH_BATCH_SIZE "auto"
  fast_infer_default LONG_BENCH_MAX_RUNNING_REQUESTS "auto"
  fast_infer_default LONG_BENCH_TP_SIZE "1"
  fast_infer_default LONG_BENCH_MEM_FRACTION_STATIC "0.9"
  fast_infer_default LONG_BENCH_SGLANG_ATTENTION_BACKEND "flashinfer"
  fast_infer_default LONG_BENCH_SGLANG_PORT "30000"
  fast_infer_default LONG_BENCH_LOCAL_FILES_ONLY "1"
  fast_infer_default LONG_BENCH_TIMEOUT_SECONDS "900"
  fast_infer_default LONG_BENCH_STRICT "1"
  # Upstream EAGLE3 Llama-3.1 defaults; override explicitly for a hardware
  # ablation, but do not silently shrink the tree and call it the baseline.
  fast_infer_default LONG_BENCH_EAGLE_TOTAL_TOKEN "60"
  fast_infer_default LONG_BENCH_EAGLE_DEPTH "5"
  fast_infer_default LONG_BENCH_EAGLE_TOP_K "10"
  fast_infer_default LONG_BENCH_LONGSPEC_MODEL_NAME "llama8b"
  fast_infer_default LONG_BENCH_SPECEXTEND_MODEL_NAME "llama3_1_8b"
  fast_infer_default LONG_BENCH_MAGICDEC_MODEL_NAME "${MODEL_TARGET:-}"
}

fast_infer__load_dflash() {
  fast_infer_default_from TARGET_MODEL MODEL_TARGET
  fast_infer_default_from DRAFT_MODEL MODEL_DFLASH_DRAFT
  fast_infer_default_from DATA_FILE DFLASH_DATA_FILE DATA_INPUT
  fast_infer_default_from MAX_SAMPLES DFLASH_MAX_SAMPLES RUN_SAMPLES
  fast_infer_default_from MAX_NEW_TOKENS DFLASH_MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from TEMPERATURE DFLASH_TEMPERATURE RUN_TEMPERATURE
  fast_infer_default_from BLOCK_SIZE DFLASH_BLOCK_SIZE
  fast_infer_default_from PROMPT DFLASH_PROMPT
  fast_infer_default_from OUTPUT_FILE DFLASH_OUTPUT_FILE
  fast_infer_default_from BACKEND DFLASH_BACKEND
  fast_infer_default_from DATASET DFLASH_DATASET
  fast_infer_default_from SMOKE_MAX_SAMPLES DFLASH_SMOKE_SAMPLES
  fast_infer_default_from SMOKE_MAX_NEW_TOKENS DFLASH_SMOKE_NEW_TOKENS
}

fast_infer__load_sglang_spec() {
  fast_infer_default_from MODEL MODEL_TARGET
  fast_infer_default_from DRAFT_MODEL MODEL_DOMINO_DRAFT MODEL_DSPARK_DRAFT
  fast_infer_default_from DATA_FILE DATA_INPUT
  fast_infer_default_from MAX_SAMPLES RUN_SAMPLES
  fast_infer_default_from MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from TEMPERATURE RUN_TEMPERATURE
  fast_infer_default_from BATCH_SIZE LONG_BENCH_BATCH_SIZE
  fast_infer_default_from TP_SIZE LONG_BENCH_TP_SIZE
  fast_infer_default_from MEM_FRACTION_STATIC LONG_BENCH_MEM_FRACTION_STATIC
  fast_infer_default_from SGLANG_PORT LONG_BENCH_SGLANG_PORT
  fast_infer_default_from ATTENTION_BACKEND LONG_BENCH_SGLANG_ATTENTION_BACKEND
  fast_infer_default_from OUTPUT_FILE OUTPUT_ROOT
}

fast_infer__load_sssd() {
  fast_infer_default_from MODEL SSSD_MODEL MODEL_TARGET
  fast_infer_default_from DATA_FILE SSSD_DATA_FILE DATA_INPUT
  fast_infer_default_from DATASTORE_PATH SSSD_DATASTORE_PATH
  fast_infer_default_from MAX_SAMPLES SSSD_MAX_SAMPLES RUN_SAMPLES
  fast_infer_default_from MAX_NEW_TOKENS SSSD_MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from NUM_DRAFT_TOKENS SSSD_NUM_DRAFT_TOKENS
  fast_infer_default_from NUM_STEPS SSSD_NUM_STEPS
  fast_infer_default_from TOPK SSSD_TOPK
  fast_infer_default_from ADAPTIVE SSSD_ADAPTIVE
  fast_infer_default_from OUTPUT_FILE SSSD_OUTPUT_FILE
}

fast_infer__load_fafo() {
  fast_infer_default_from MODEL FAFO_MODEL MODEL_TARGET
  fast_infer_default_from DATA_FILE FAFO_DATA_FILE DATA_INPUT
  fast_infer_default_from MAX_SAMPLES FAFO_MAX_SAMPLES RUN_SAMPLES
  fast_infer_default_from MAX_NEW_TOKENS FAFO_MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from KV_METHOD FAFO_KV_METHOD
  fast_infer_default_from USE_FLASH FAFO_USE_FLASH
  fast_infer_default_from OUTPUT_FILE FAFO_OUTPUT_FILE
}

fast_infer__load_eagle3() {
  fast_infer_default_from BASE_MODEL MODEL_TARGET
  fast_infer_default_from EAGLE_MODEL MODEL_EAGLE_DRAFT
  fast_infer_default_from DATA_FILE EAGLE_DATA_FILE
  fast_infer_default_from BENCH_NAME EAGLE_BENCHMARK
  fast_infer_default_from QUESTION_BEGIN EAGLE_QUESTION_BEGIN
  fast_infer_default_from QUESTION_END EAGLE_QUESTION_END
  fast_infer_default_from MAX_NEW_TOKENS EAGLE_MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from NUM_CHOICES EAGLE_NUM_CHOICES
  fast_infer_default_from TEMPERATURE EAGLE_TEMPERATURE RUN_TEMPERATURE
  fast_infer_default_from TOTAL_TOKEN EAGLE_TOTAL_TOKENS
  fast_infer_default_from DEPTH EAGLE_DEPTH
  fast_infer_default_from TOP_K EAGLE_TOP_K
  fast_infer_default_from SKIP_NAIVE EAGLE_SKIP_NAIVE RUN_SKIP_NAIVE
  fast_infer_default_from OUTPUT_FILE EAGLE_OUTPUT_FILE
}

fast_infer__load_fastkv() {
  fast_infer_default_from MODEL MODEL_TARGET
  fast_infer_default_from DATA_FILE FASTKV_DATA_FILE DATA_INPUT
  fast_infer_default_from METHOD FASTKV_METHOD
  fast_infer_default_from ATTN_IMPL FASTKV_ATTN_IMPL
  fast_infer_default_from MAX_NEW_TOKENS FASTKV_MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from WINDOW_SIZE FASTKV_WINDOW_SIZE
  fast_infer_default_from MAX_CAPACITY_PROMPTS FASTKV_MAX_CAPACITY_PROMPTS
  fast_infer_default_from RETAIN_RATE FASTKV_RETAIN_RATE
  fast_infer_default_from EVICTION_MODE FASTKV_EVICTION_MODE
  fast_infer_default_from NUM_RUNS FASTKV_NUM_RUNS
  fast_infer_default_from FASTKV_DATA_ROOT DATA_ROOT
  fast_infer_default_from OUTPUT_FILE FASTKV_OUTPUT_FILE
}

fast_infer__load_flexprefill() {
  fast_infer_default_from MODEL MODEL_TARGET
  fast_infer_default_from PATTERN FLEXPREFILL_PATTERN
  fast_infer_default_from MAX_SAMPLES FLEXPREFILL_MAX_SAMPLES RUN_SAMPLES
  fast_infer_default_from MAX_NEW_TOKENS FLEXPREFILL_MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from DATA_FILE FLEXPREFILL_DATA_FILE DATA_INPUT
  fast_infer_default_from OUTPUT_FILE FLEXPREFILL_OUTPUT_FILE
  fast_infer_default_from SKIP_NAIVE FLEXPREFILL_SKIP_NAIVE RUN_SKIP_NAIVE
}

fast_infer__load_gemfilter() {
  fast_infer_default_from MODEL MODEL_TARGET
  fast_infer_default_from DATA_FILE GEMFILTER_DATA_FILE DATA_INPUT
  fast_infer_default_from TOPK GEMFILTER_TOP_K
  fast_infer_default_from SELECT_LAYER_IDX GEMFILTER_SELECT_LAYER
  fast_infer_default_from PROMPT GEMFILTER_PROMPT
  fast_infer_default_from MAX_GEN_LEN GEMFILTER_MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from NUM_RUNS GEMFILTER_NUM_RUNS
  fast_infer_default_from OUTPUT_FILE GEMFILTER_OUTPUT_FILE
}

fast_infer__load_higoe() {
  fast_infer_default_from RETRIEVER_MODEL HIGOE_RETRIEVER_MODEL MODEL_EMBEDDING
  fast_infer_default_from NUM_DOCS HIGOE_NUM_DOCS
  fast_infer_default_from OUTPUT_FILE HIGOE_OUTPUT_FILE
}

fast_infer__load_llmlingua() {
  fast_infer_default_from COMPRESSOR_MODEL MODEL_COMPRESSOR
  fast_infer_default_from TARGET_MODEL MODEL_TARGET
  fast_infer_default_from DOC_FILE LLMLINGUA_DATA_FILE DATA_INPUT
  fast_infer_default_from COMPRESSION_RATE LLMLINGUA_COMPRESSION_RATE
  fast_infer_default_from MAX_SAMPLES LLMLINGUA_MAX_SAMPLES RUN_SAMPLES
  fast_infer_default_from MAX_NEW_TOKENS LLMLINGUA_MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from DEVICE LLMLINGUA_DEVICE FI_DEVICE
  fast_infer_default_from OUTPUT_FILE LLMLINGUA_OUTPUT_FILE
}

fast_infer__load_longspec() {
  fast_infer_default_from MODEL_NAME LONGSPEC_MODEL_NAME MODEL_LONGSPEC_TARGET MODEL_TARGET
  fast_infer_default_from TARGET_MODEL MODEL_LONGSPEC_TARGET MODEL_TARGET
  fast_infer_default_from DRAFT_MODEL MODEL_LONGSPEC_DRAFT
  fast_infer_default_from METHOD LONGSPEC_METHOD
  fast_infer_default_from TASK LONGSPEC_TASK
  fast_infer_default_from MAX_GEN_LEN LONGSPEC_MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from TREE_SHAPE LONGSPEC_TREE_SHAPE
  fast_infer_default_from DATA_PATH_PREFIX LONGSPEC_DATA_PREFIX
  fast_infer_default_from DATA_FILE LONGSPEC_DATA_FILE
  fast_infer_default_from MAX_SAMPLES LONGSPEC_MAX_SAMPLES RUN_SAMPLES
  fast_infer_default_from OUTPUT_FILE LONGSPEC_OUTPUT_FILE
}

fast_infer__load_magicdec() {
  fast_infer_default_from MAGICDEC_CACHE_ROOT MAGICDEC_CACHE
  fast_infer_default_from MAGICDEC_DATA_ROOT DATA_ROOT
  fast_infer_default_from MODEL_PTH CHECKPOINT_MAGICDEC
  fast_infer_default_from MODEL_NAME MODEL_TARGET
  fast_infer_default_from BATCH_SIZE MAGICDEC_BATCH_SIZE
  fast_infer_default_from PREFIX_LEN MAGICDEC_PREFIX_LEN
  fast_infer_default_from MAX_LEN MAGICDEC_MAX_INPUT_TOKENS
  fast_infer_default_from NUM_RUNS MAGICDEC_NUM_RUNS
  fast_infer_default_from WINDOW_SIZE MAGICDEC_WINDOW_SIZE
  fast_infer_default_from SELF_SPEC MAGICDEC_SELF_SPEC
  fast_infer_default_from GAMMA MAGICDEC_GAMMA
  fast_infer_default_from DRAFT_BUDGET MAGICDEC_DRAFT_BUDGET
  fast_infer_default_from PREPARE_CHECKPOINT MAGICDEC_PREPARE_CHECKPOINT
  fast_infer_default_from OUTPUT_FILE MAGICDEC_OUTPUT_FILE
  fast_infer_default_from REPO_ID MAGICDEC_REPO_ID
  fast_infer_default_from MODEL_KEY MAGICDEC_MODEL_KEY
}

fast_infer__load_minference() {
  fast_infer_default_from MODEL MODEL_TARGET
  fast_infer_default_from DATA_FILE MINFERENCE_DATA_FILE DATA_INPUT
  fast_infer_default_from ATTN_TYPE MINFERENCE_ATTN_TYPE
  fast_infer_default_from MAX_NEW_TOKENS MINFERENCE_MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from MAX_MODEL_LEN MINFERENCE_MAX_INPUT_TOKENS
  fast_infer_default_from DEVICE MINFERENCE_DEVICE FI_DEVICE
  fast_infer_default_from ATTN_IMPLEMENTATION MINFERENCE_ATTN_IMPLEMENTATION
  fast_infer_default_from OUTPUT_FILE MINFERENCE_OUTPUT_FILE
}

fast_infer__load_qwen3_long_profile() {
  fast_infer_default_from MODEL MODEL_TARGET
  fast_infer_default_from INPUT_FILE QWEN3_PROFILE_DATA_FILE DATA_INPUT
  fast_infer_default_from WORD_MARKS QWEN3_PROFILE_WORD_MARKS
  fast_infer_default_from MAX_NEW_TOKENS QWEN3_PROFILE_MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from REPEATS QWEN3_PROFILE_REPEATS
  fast_infer_default_from WARMUP_RUNS QWEN3_PROFILE_WARMUP_RUNS
  fast_infer_default_from DEVICE QWEN3_PROFILE_DEVICE FI_DEVICE
  fast_infer_default_from ATTN_IMPLEMENTATION QWEN3_PROFILE_ATTN_IMPLEMENTATION
  fast_infer_default_from LOCAL_FILES_ONLY QWEN3_PROFILE_LOCAL_FILES_ONLY
  fast_infer_default_from OUTPUT_DIR QWEN3_PROFILE_OUTPUT_DIR OUTPUT_ROOT
}

fast_infer__load_rocketkv() {
  fast_infer_default_from TOKEN_BUDGET ROCKETKV_TOKEN_BUDGET
  fast_infer_default_from SEQ_LEN ROCKETKV_SEQ_LEN
  fast_infer_default_from MAX_NEW_TOKENS ROCKETKV_MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from HEAD_DIM ROCKETKV_HEAD_DIM
  fast_infer_default_from NUM_RUNS ROCKETKV_NUM_RUNS
  fast_infer_default_from OUTPUT_FILE ROCKETKV_OUTPUT_FILE
}

fast_infer__load_semantic_selection() {
  fast_infer_default_from MODEL MODEL_TARGET
  fast_infer_default_from INPUT_FILE SEMANTIC_DATA_FILE DATA_INPUT
  fast_infer_default_from SELECTORS SEMANTIC_SELECTORS
  fast_infer_default_from TOKEN_BUDGETS SEMANTIC_TOKEN_BUDGETS
  fast_infer_default_from SMOKE_TOKEN_BUDGETS SEMANTIC_SMOKE_TOKEN_BUDGETS
  fast_infer_default_from MAX_SAMPLES SEMANTIC_MAX_SAMPLES RUN_SAMPLES
  fast_infer_default_from MAX_NEW_TOKENS SEMANTIC_MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from DEVICE SEMANTIC_DEVICE FI_DEVICE
  fast_infer_default_from DTYPE SEMANTIC_DTYPE
  fast_infer_default_from ATTN_IMPLEMENTATION SEMANTIC_ATTN_IMPLEMENTATION
  fast_infer_default_from WARMUP_ROUNDS SEMANTIC_WARMUP_ROUNDS
  fast_infer_default_from EMBEDDING_MODEL MODEL_EMBEDDING
  fast_infer_default_from EMBEDDING_DEVICE SEMANTIC_EMBEDDING_DEVICE
  fast_infer_default_from RANDOM_SEED SEMANTIC_RANDOM_SEED
  fast_infer_default_from MMR_LAMBDA SEMANTIC_MMR_LAMBDA
  fast_infer_default_from OUTPUT_FILE SEMANTIC_OUTPUT_FILE
}

fast_infer__load_specextend() {
  fast_infer_default_from SCRIPT SPECEXTEND_SCRIPT
  fast_infer_default_from MODEL_NAME MODEL_TARGET
  fast_infer_default_from BASE_MODEL MODEL_TARGET
  fast_infer_default_from DRAFT_MODEL MODEL_SPEC_DRAFT
  fast_infer_default_from INPUT_FILE SPECEXTEND_DATA_FILE DATA_INPUT
  fast_infer_default_from MAX_SAMPLES SPECEXTEND_MAX_SAMPLES RUN_SAMPLES
  fast_infer_default_from MAX_GEN_LEN SPECEXTEND_MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from MAX_INPUT_TOKENS SPECEXTEND_MAX_INPUT_TOKENS RUN_MAX_INPUT_TOKENS
  fast_infer_default_from USE_SPECEXTEND SPECEXTEND_ENABLED
  fast_infer_default_from WARMUP_RUNS SPECEXTEND_WARMUP_RUNS
  fast_infer_default_from OUTPUT_FILE SPECEXTEND_OUTPUT_FILE
  fast_infer_default_from REPO_ID SPECEXTEND_REPO_ID
}

fast_infer__load_specprefill() {
  fast_infer_default_from TARGET_MODEL MODEL_TARGET
  fast_infer_default_from SPEC_MODEL MODEL_SPEC_DRAFT
  fast_infer_default_from DATA_FILE SPECPREFILL_DATA_FILE DATA_INPUT
  fast_infer_default_from SPEC_CONFIG SPECPREFILL_CONFIG
  fast_infer_default_from MAX_TOKENS SPECPREFILL_MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from GPU_MEMORY_UTILIZATION SPECPREFILL_GPU_MEMORY_UTILIZATION
  fast_infer_default_from OUTPUT_FILE SPECPREFILL_OUTPUT_FILE
}

fast_infer__load_syncspec() {
  fast_infer_default_from TARGET_MODEL SYNCSPEC_TARGET_MODEL MODEL_TARGET
  fast_infer_default_from DRAFTER_CHECKPOINT SYNCSPEC_DRAFTER_CHECKPOINT
  fast_infer_default_from SELECTOR_CHECKPOINT SYNCSPEC_SELECTOR_CHECKPOINT
  fast_infer_default_from SURVIVAL_CHECKPOINT SYNCSPEC_SURVIVAL_CHECKPOINT
  fast_infer_default_from DATA_FILE SYNCSPEC_DATA_FILE DATA_INPUT
  fast_infer_default_from OUTPUT_FILE SYNCSPEC_OUTPUT_FILE
  fast_infer_default_from DEVICE SYNCSPEC_DEVICE FI_DEVICE
  fast_infer_default_from DTYPE SYNCSPEC_DTYPE DTYPE
  fast_infer_default_from PROFILE SYNCSPEC_PROFILE
  fast_infer_default_from GATE_TABLE SYNCSPEC_GATE_TABLE
  fast_infer_default_from BACKEND SYNCSPEC_BACKEND
  fast_infer_default_from MAX_SAMPLES SYNCSPEC_MAX_SAMPLES RUN_SAMPLES
  fast_infer_default_from BATCH_SIZE SYNCSPEC_BATCH_SIZE
  fast_infer_default_from KD SYNCSPEC_KD
  fast_infer_default_from KV SYNCSPEC_KV
  fast_infer_default_from BUDGET_PROFILES SYNCSPEC_BUDGET_PROFILES
  fast_infer_default_from MAX_NEW_TOKENS SYNCSPEC_MAX_NEW_TOKENS RUN_MAX_NEW_TOKENS
  fast_infer_default_from MAX_INPUT_TOKENS SYNCSPEC_MAX_INPUT_TOKENS RUN_MAX_INPUT_TOKENS
  fast_infer_default_from STOCHASTIC SYNCSPEC_STOCHASTIC
  fast_infer_default_from LOCAL_FILES_ONLY SYNCSPEC_LOCAL_FILES_ONLY
}

fast_infer_load_config() {
  local baseline="${1:-}"

  fast_infer_load_master || return 1

  case "$baseline" in
    longbench) fast_infer__load_longbench ;;
    dflash) fast_infer__load_dflash ;;
    domino|dspark) fast_infer__load_sglang_spec ;;
    fafo) fast_infer__load_fafo ;;
    eagle3) fast_infer__load_eagle3 ;;
    fastkv) fast_infer__load_fastkv ;;
    flexprefill) fast_infer__load_flexprefill ;;
    gemfilter) fast_infer__load_gemfilter ;;
    higoe) fast_infer__load_higoe ;;
    llmlingua) fast_infer__load_llmlingua ;;
    longspec) fast_infer__load_longspec ;;
    magicdec|magicdec_prepare) fast_infer__load_magicdec ;;
    minference) fast_infer__load_minference ;;
    qwen3_long_profile) fast_infer__load_qwen3_long_profile ;;
    rocketkv) fast_infer__load_rocketkv ;;
    semantic_selection) fast_infer__load_semantic_selection ;;
    sssd) fast_infer__load_sssd ;;
    specextend) fast_infer__load_specextend ;;
    specprefill) fast_infer__load_specprefill ;;
    syncspec) fast_infer__load_syncspec ;;
    *)
      echo "fast-infer: unsupported baseline in master config loader: $baseline" >&2
      return 1
      ;;
  esac

  # Baseline-specific values are applied first; common RUN_* values fill only
  # parameters that the selected baseline did not set.
  fast_infer__load_common

  if [[ -z "${OUTPUT_FILE:-}" && -n "${OUTPUT_ROOT:-}" ]]; then
    local output_root="$OUTPUT_ROOT"
    if [[ "$output_root" != /* ]]; then
      output_root="$ROOT/$output_root"
    fi
    OUTPUT_FILE="$output_root/${baseline}.jsonl"
    export OUTPUT_FILE
  fi
}
