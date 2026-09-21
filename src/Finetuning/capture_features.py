"""Offline target hidden-state capture using a local Transformers snapshot."""

from __future__ import annotations

import argparse
import json
from itertools import chain
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterable, Mapping

import torch

from .adaptive_inference import (
    AdaptiveInferenceSettings,
    BackoffResult,
    PreparedExample,
    adaptive_settings_from_args,
    add_adaptive_cli_args,
    length_bucket_batches,
    run_with_oom_backoff,
    select_cuda_batch_size,
    validate_adaptive_inference_settings,
)
from .data import DEFAULT_SUMMARY_PROMPT_TEMPLATE
from .distributed import (
    cleanup_distributed,
    initialize_distributed,
    ranked_path,
)
from .features import (
    FEATURE_MANIFEST_FILENAME,
    FeatureManifest,
    _canonicalize_input_ids,
    _canonicalize_loss_mask,
    _dtype_name,
    _reject_symlink_components,
    validate_feature_record,
)
from .offline_sglang_capture import (
    OfflineSGLangCaptureAdapter,
    ParityThresholds,
    compare_hidden_rows,
    destroy_specforge_distributed,
    initialize_specforge_distributed,
    sglang_capture_available,
)


def _as_dtype(value: torch.dtype | str) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    name = str(value).removeprefix("torch.")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported capture dtype: {value!r}")
    return dtype


def _as_1d_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value)
    if value.ndim == 2 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim != 1:
        raise ValueError(f"prepared example {name!r} must be one-dimensional")
    return value.detach().cpu()


def _atomic_torch_save(payload: Mapping[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_feature_output_dir(destination: Path) -> None:
    """Reject replacement of a directory containing unrelated user files."""

    if destination.is_symlink() or not destination.is_dir():
        raise ValueError(f"feature output path is not a directory: {destination}")
    allowed_suffixes = (".pt", ".pth", ".ckpt", ".ckpt.gz")
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                "refusing to use feature output containing a symlink: " f"{path}"
            )
        if path.is_file() and path.name != FEATURE_MANIFEST_FILENAME and not path.name.endswith(allowed_suffixes):
            raise ValueError(
                "refusing to replace feature output containing unrelated file: "
                f"{path}"
            )


def _publish_feature_generation(
    staging: Path,
    destination: Path,
    manifest: FeatureManifest,
) -> None:
    """Publish a complete immutable generation behind an atomic manifest."""

    generations = destination / ".generations"
    generations.mkdir(parents=True, exist_ok=True)
    generation_name = staging.name.lstrip(".")
    generation_dir = generations / f"generation-{generation_name}"
    os.replace(staging, generation_dir)
    manifest.generation_dir = generation_dir.relative_to(destination).as_posix()
    # Readers either retain the old manifest/generation or observe this new
    # manifest after its complete generation has already been renamed in.
    _atomic_json_save(manifest.to_dict(), destination / FEATURE_MANIFEST_FILENAME)


def merge_feature_shards(
    shard_dirs: Iterable[str | Path],
    destination: str | Path,
) -> FeatureManifest:
    """Merge rank-local feature generations into one atomic feature store."""

    final_dir = Path(destination)
    if final_dir.exists():
        _validate_feature_output_dir(final_dir)
    manifests: list[FeatureManifest] = []
    record_paths: list[Path] = []
    for raw_shard in shard_dirs:
        shard = Path(raw_shard)
        manifest_path = shard / FEATURE_MANIFEST_FILENAME
        if not manifest_path.is_file():
            continue
        manifest = FeatureManifest.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        manifests.append(manifest)
        record_root = shard
        if manifest.generation_dir is not None:
            record_root = shard / manifest.generation_dir
        record_paths.extend(
            path
            for path in record_root.rglob("*")
            if path.is_file() and path.name.endswith((".pt", ".pth", ".ckpt", ".ckpt.gz"))
        )
    if not manifests or not record_paths:
        raise ValueError("distributed feature merge found no captured records")
    baseline = manifests[0].to_dict()
    baseline["generation_dir"] = None
    for manifest in manifests[1:]:
        candidate = manifest.to_dict()
        candidate["generation_dir"] = None
        if candidate != baseline:
            raise ValueError("distributed feature shards have incompatible manifests")
    record_names = [path.name for path in record_paths]
    if len(set(record_names)) != len(record_names):
        raise ValueError("distributed feature shards contain duplicate record names")

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.merge-", dir=final_dir.parent))
    try:
        for source in record_paths:
            shutil.copy2(source, staging / source.name)
        merged_manifest = FeatureManifest.from_dict(baseline)
        _publish_feature_generation(staging, final_dir, merged_manifest)
        return merged_manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _target_config(model: Any) -> Any:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("local target model has no config")
    text_config = getattr(config, "text_config", None)
    return text_config if text_config is not None else config


def _extract_hidden_states(outputs: Any) -> Any:
    if isinstance(outputs, Mapping):
        return outputs.get("hidden_states")
    return getattr(outputs, "hidden_states", None)


def _load_local_target(
    target_model_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> Any:
    if not target_model_path.is_dir():
        raise FileNotFoundError(
            f"local target model snapshot not found: {target_model_path}"
        )
    # Import at the capture boundary only.  Data loading and feature reading
    # remain usable without Transformers, and this call cannot resolve remote
    # model identifiers because local_files_only is unconditional.
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(target_model_path),
        local_files_only=True,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model = model.to(device)
    model.eval()
    return model


def _resolve_layer_ids(target_layer_ids: Iterable[int], num_layers: int) -> list[int]:
    layer_ids = list(target_layer_ids)
    if not layer_ids:
        raise ValueError("target_layer_ids must not be empty")
    if any(
        isinstance(layer_id, bool)
        or not isinstance(layer_id, int)
        or layer_id < 0
        or layer_id >= num_layers
        for layer_id in layer_ids
    ):
        raise ValueError(
            f"target_layer_ids must be within [0, {num_layers}), got {layer_ids!r}"
        )
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError("target_layer_ids must not contain duplicates")
    return layer_ids


def _capture_hidden_feature(
    model: Any,
    input_ids: torch.Tensor,
    layer_ids: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    model_input = input_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=model_input,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = _extract_hidden_states(outputs)
    if hidden_states is None:
        raise ValueError("target model did not return output_hidden_states")
    selected: list[torch.Tensor] = []
    for layer_id in layer_ids:
        index = layer_id + 1  # SpecForge layer output includes embedding at index 0.
        if index >= len(hidden_states):
            raise ValueError(
                f"target model hidden_states missing selected layer {layer_id}"
            )
        value = hidden_states[index]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"target layer {layer_id} output is not a tensor")
        if value.ndim != 3 or value.shape[0] != 1:
            raise ValueError(
                f"target layer {layer_id} hidden state must have shape [1, seq, hidden], "
                f"got {tuple(value.shape)}"
            )
        selected.append(value[0])
    feature = torch.cat(selected, dim=-1).detach().to(device="cpu", dtype=dtype)
    return feature


def _capture_hidden_features(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_ids: list[int],
    lengths: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Capture a right-padded batch and trim each row back to its true length."""

    model_input = input_ids.to(device=device, dtype=torch.long)
    model_mask = attention_mask.to(device=device, dtype=torch.long)
    with torch.inference_mode():
        try:
            outputs = model(
                input_ids=model_input,
                attention_mask=model_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        except TypeError as exc:
            if "attention_mask" not in str(exc):
                raise
            outputs = model(
                input_ids=model_input,
                output_hidden_states=True,
                use_cache=False,
            )
    hidden_states = _extract_hidden_states(outputs)
    if hidden_states is None:
        raise ValueError("target model did not return output_hidden_states")
    selected: list[torch.Tensor] = []
    for layer_id in layer_ids:
        index = layer_id + 1
        if index >= len(hidden_states):
            raise ValueError(f"target model hidden_states missing selected layer {layer_id}")
        value = hidden_states[index]
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise ValueError(
                f"target layer {layer_id} hidden state must have shape [batch, seq, hidden]"
            )
        selected.append(value)
    records: list[torch.Tensor] = []
    for row, length in enumerate(lengths):
        row_features = torch.cat(
            [value[row, :length] for value in selected],
            dim=-1,
        ).detach().to(device="cpu", dtype=dtype)
        records.append(row_features)
    return records


def _pad_capture_batch(
    examples: list[PreparedExample],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if not examples:
        raise ValueError("cannot pad an empty feature batch")
    records = [example.payload for example in examples]
    lengths = [int(record["input_ids"].shape[0]) for record in records]
    width = max(lengths)
    input_ids = torch.zeros((len(records), width), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(records), width), dtype=torch.long, device=device)
    for row, record in enumerate(records):
        length = lengths[row]
        input_ids[row, :length] = record["input_ids"].to(device=device, dtype=torch.long)
        attention_mask[row, :length] = 1
    return input_ids, attention_mask, lengths


def capture_dataset(
    target_model_path: str | Path,
    prepared_examples: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    target_layer_ids: Iterable[int] | None,
    max_length: int,
    device: str | torch.device,
    dtype: torch.dtype | str,
    *,
    num_draft_layers: int | None = None,
    trust_remote_code: bool = False,
    tokenizer_id: str | None = None,
    prompt_contract: Mapping[str, Any] | None = None,
    rank: int = 0,
    world_size: int = 1,
    allow_empty: bool = False,
    adaptive_settings: AdaptiveInferenceSettings | None = None,
    stats: dict[str, Any] | None = None,
    capture_backend: str = "hf",
    capture_method: str = "dflash",
    sglang_attention_backend: str = "flashinfer",
    sglang_mem_fraction_static: float = 0.40,
    sglang_max_running_requests: int = 8,
    sglang_max_total_tokens: int = 0,
    sglang_context_length: int | None = None,
    sglang_disable_radix_cache: bool = False,
    parity_samples: int = 2,
    parity_thresholds: ParityThresholds | None = None,
) -> FeatureManifest | None:
    """Capture prepared token examples into an atomic local feature store."""

    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    requested_dtype = _as_dtype(dtype)
    capture_backend = str(capture_backend).lower()
    if capture_backend not in {"hf", "sglang"}:
        raise ValueError("capture_backend must be 'hf' or 'sglang'")
    if capture_method not in {"eagle3", "dflash", "dspark"}:
        raise ValueError("capture_method must be 'eagle3', 'dflash', or 'dspark'")
    if parity_samples < 0:
        raise ValueError("parity_samples must be non-negative")
    if capture_backend == "sglang" and torch.device(device).type != "cuda":
        raise ValueError("OfflineSGLangCapture requires a CUDA device")
    if sglang_max_running_requests <= 0:
        raise ValueError("sglang_max_running_requests must be positive")
    if sglang_max_total_tokens < 0:
        raise ValueError("sglang_max_total_tokens must be non-negative")
    settings = adaptive_settings or AdaptiveInferenceSettings(
        enabled=False,
        min_batch_size=1,
        max_batch_size=1,
        bucket_window=1,
    )
    validate_adaptive_inference_settings(settings)
    if not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if not isinstance(rank, int) or not 0 <= rank < world_size:
        raise ValueError("rank must be within [0, world_size)")
    model_path = Path(target_model_path)
    device_obj = torch.device(device)
    destination = Path(output_dir)
    _reject_symlink_components(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _validate_feature_output_dir(destination)
    else:
        destination.mkdir(parents=True)
    reference_model = None
    if capture_backend == "hf":
        model = _load_local_target(
            model_path,
            device_obj,
            requested_dtype,
            trust_remote_code=trust_remote_code,
        )
        config = _target_config(model)
    else:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
    num_layers = int(getattr(config, "num_hidden_layers", 0))
    hidden_size = int(getattr(config, "hidden_size", 0))
    if num_layers < 1 or hidden_size < 1:
        raise ValueError("local target model config lacks num_hidden_layers/hidden_size")
    if target_layer_ids is None:
        if num_draft_layers is None or num_draft_layers < 1:
            raise ValueError(
                "capture_dataset requires target_layer_ids or a positive "
                "num_draft_layers"
            )
        from .model import build_target_layer_ids

        target_layer_ids = build_target_layer_ids(num_layers, num_draft_layers)
    layer_ids = _resolve_layer_ids(target_layer_ids, num_layers)

    if capture_backend == "sglang":
        if sglang_max_total_tokens == 0:
            sglang_max_total_tokens = sglang_max_running_requests * max_length
        model = OfflineSGLangCaptureAdapter.from_pretrained(
            model_path,
            layer_ids=layer_ids,
            capture_method=capture_method,
            dtype=requested_dtype,
            attention_backend=sglang_attention_backend,
            mem_fraction_static=sglang_mem_fraction_static,
            max_running_requests=sglang_max_running_requests,
            max_total_tokens=sglang_max_total_tokens,
            context_length=sglang_context_length,
            disable_radix_cache=sglang_disable_radix_cache,
            trust_remote_code=trust_remote_code,
        )
        if parity_samples > 0:
            reference_model = _load_local_target(
                model_path,
                device_obj,
                requested_dtype,
                trust_remote_code=trust_remote_code,
            )

    examples = (
        (source_index, example)
        for source_index, example in enumerate(prepared_examples)
        if source_index % world_size == rank
    )
    try:
        first_source_index, first_example = next(examples)
    except StopIteration as exc:
        if allow_empty:
            return None
        raise ValueError("capture_dataset requires at least one prepared example") from exc

    def prepare_example(example: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        first_ids = _as_1d_tensor(example["input_ids"], "input_ids")
        first_mask = _as_1d_tensor(example["loss_mask"], "loss_mask")
        if first_ids.shape[0] != first_mask.shape[0]:
            raise ValueError(
                "prepared example has mismatched sequence lengths: "
                f"input_ids={first_ids.shape[0]}, loss_mask={first_mask.shape[0]}"
            )
        return (
            _canonicalize_input_ids(first_ids)[:max_length],
            _canonicalize_loss_mask(first_mask)[:max_length],
        )

    first_ids, first_mask = prepare_example(first_example)
    capture_metadata: dict[str, Any] | None = None
    if capture_backend == "sglang":
        try:
            from importlib import metadata as importlib_metadata

            sglang_version = importlib_metadata.version("sglang")
        except importlib_metadata.PackageNotFoundError:
            sglang_version = None
        capture_metadata = {
            "sglang_version": sglang_version,
            "attention_backend": sglang_attention_backend,
            "mem_fraction_static": float(sglang_mem_fraction_static),
            "max_running_requests": int(sglang_max_running_requests),
            "max_total_tokens": int(sglang_max_total_tokens),
            "context_length": sglang_context_length,
            "radix_cache_disabled": bool(sglang_disable_radix_cache),
            "parity_samples": int(parity_samples),
        }
    manifest = FeatureManifest(
        model_id=str(model_path),
        revision=getattr(config, "_commit_hash", None),
        tokenizer_id=tokenizer_id,
        prompt_contract=dict(prompt_contract) if prompt_contract is not None else None,
        layer_ids=layer_ids,
        hidden_size=hidden_size,
        max_length=max_length,
        input_ids_dtype=_dtype_name(first_ids.dtype),
        loss_mask_dtype=_dtype_name(first_mask.dtype),
        hidden_states_dtype=_dtype_name(requested_dtype),
        capture_backend=capture_backend,
        capture_method=capture_method if capture_backend == "sglang" else None,
        capture_metadata=capture_metadata,
    )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.new-", dir=destination.parent)
    )
    captured = 0
    processed_tokens = 0
    oom_retries = 0
    started = time.perf_counter()
    current_batch_size = settings.min_batch_size
    selection = None
    observed_peak_reserved = 0
    parity_report: dict[str, Any] | None = None
    try:
        window: list[PreparedExample] = []

        def flush_window(items: list[PreparedExample]) -> None:
            nonlocal captured, processed_tokens, oom_retries
            nonlocal current_batch_size, selection, observed_peak_reserved, parity_report
            nonlocal reference_model
            if not items:
                return
            if (
                capture_backend == "sglang"
                and parity_report is None
                and parity_samples > 0
                and reference_model is not None
            ):
                parity_items = items[: min(parity_samples, len(items))]
                input_ids, attention_mask, lengths = _pad_capture_batch(
                    parity_items,
                    device=device_obj,
                )
                reference_rows = _capture_hidden_features(
                    reference_model,
                    input_ids,
                    attention_mask,
                    layer_ids,
                    lengths,
                    device_obj,
                    requested_dtype,
                )
                candidate_rows = model.capture_rows(
                    [item.payload["input_ids"] for item in parity_items],
                    expected_width=len(layer_ids) * hidden_size,
                ).aux_rows
                parity = compare_hidden_rows(
                    reference_rows,
                    candidate_rows,
                    thresholds=parity_thresholds,
                )
                parity_report = parity.to_dict()
                if not parity.passed:
                    raise RuntimeError(
                        "OfflineSGLangCapture parity gate failed: "
                        f"{json.dumps(parity_report, sort_keys=True)}"
                    )
                del reference_model
                reference_model = None
            if selection is None:
                probe_items = sorted(items, key=lambda item: (-item.length, item.index))

                def probe(batch_size: int) -> None:
                    selected = probe_items[:batch_size]
                    for _ in range(settings.probe_batches):
                        if capture_backend == "hf":
                            input_ids, attention_mask, lengths = _pad_capture_batch(
                                selected, device=device_obj
                            )
                            _capture_hidden_features(
                                model,
                                input_ids,
                                attention_mask,
                                layer_ids,
                                lengths,
                                device_obj,
                                requested_dtype,
                            )
                        else:
                            model.capture_rows(
                                [item.payload["input_ids"] for item in selected],
                                expected_width=len(layer_ids) * hidden_size,
                            )
                    return None

                selection = select_cuda_batch_size(
                    probe,
                    settings=settings,
                    device=device_obj,
                    maximum=min(settings.max_batch_size, len(probe_items)),
                )
                current_batch_size = selection.batch_size
                if device_obj.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device_obj)
            for batch in length_bucket_batches(
                items,
                batch_size=current_batch_size,
                max_tokens=settings.max_tokens_per_batch,
                window=max(settings.bucket_window, current_batch_size),
            ):
                offset = 0
                while offset < len(batch):
                    subset = batch[offset : offset + current_batch_size]

                    def work(size: int) -> list[torch.Tensor]:
                        selected = subset[:size]
                        if capture_backend == "hf":
                            input_ids, attention_mask, lengths = _pad_capture_batch(
                                selected,
                                device=device_obj,
                            )
                            return _capture_hidden_features(
                                model,
                                input_ids,
                                attention_mask,
                                layer_ids,
                                lengths,
                                device_obj,
                                requested_dtype,
                            )
                        return list(
                            model.capture_rows(
                                [item.payload["input_ids"] for item in selected],
                                expected_width=len(layer_ids) * hidden_size,
                            ).aux_rows
                        )

                    if settings.oom_backoff:
                        result = run_with_oom_backoff(
                            work,
                            initial_batch_size=len(subset),
                            minimum_batch_size=min(settings.min_batch_size, len(subset)),
                        )
                    else:
                        result = BackoffResult(work(len(subset)), len(subset), 0)
                    for example, hidden_states in zip(
                        subset[: result.batch_size], result.value, strict=True
                    ):
                        input_ids = example.payload["input_ids"]
                        loss_mask = example.payload["loss_mask"]
                        if hidden_states.shape != (input_ids.shape[0], manifest.feature_width):
                            raise ValueError(
                                "captured feature width mismatch: "
                                f"got {tuple(hidden_states.shape)}, expected "
                                f"({input_ids.shape[0]}, {manifest.feature_width})"
                            )
                        record = {
                            "input_ids": input_ids,
                            "loss_mask": loss_mask,
                            "hidden_states": hidden_states,
                        }
                        validate_feature_record(record, manifest)
                        _atomic_torch_save(
                            record,
                            staging / f"feature_{example.index:08d}.pt",
                        )
                        captured += 1
                        processed_tokens += example.length
                    oom_retries += result.oom_retries
                    if device_obj.type == "cuda" and torch.cuda.is_available():
                        observed_peak_reserved = max(
                            observed_peak_reserved,
                            int(torch.cuda.max_memory_reserved(device_obj)),
                        )
                    current_batch_size = min(current_batch_size, result.batch_size)
                    offset += result.batch_size

        stream = chain(
            ((first_source_index, first_example),),
            examples,
        )
        for source_index, example in stream:
            input_ids, loss_mask = prepare_example(example)
            window.append(
                PreparedExample(
                    source_index,
                    int(input_ids.shape[0]),
                    {"input_ids": input_ids, "loss_mask": loss_mask},
                )
            )
            if len(window) >= max(settings.bucket_window, settings.min_batch_size):
                flush_window(window)
                window = []
        flush_window(window)

        # Publish only a complete generation.  A loader either sees the old
        # manifest/generation or the new one, never a partial mixture.
        _publish_feature_generation(staging, destination, manifest)
        if stats is not None:
            elapsed = max(time.perf_counter() - started, 1e-9)
            stats.update(
                {
                    "captured": captured,
                    "tokens": processed_tokens,
                    "elapsed_ms": int(elapsed * 1000),
                    "samples_per_sec_x1000": int(captured / elapsed * 1000),
                    "tokens_per_sec_x1000": int(processed_tokens / elapsed * 1000),
                    "oom_retries": oom_retries,
                    "adaptive_batch_size": int(current_batch_size),
                    "adaptive_target_memory_bytes": int(
                        selection.target_memory_bytes if selection is not None else 0
                    ),
                    "adaptive_peak_reserved_bytes": int(
                        max(
                            observed_peak_reserved,
                            selection.peak_reserved_bytes if selection is not None else 0,
                        )
                    ),
                    "adaptive_probe_count": int(selection.probes if selection is not None else 0),
                    "adaptive_hit_maximum": int(
                        selection.hit_maximum if selection is not None else False
                    ),
                    "capture_backend": capture_backend,
                    "capture_method": capture_method,
                    "parity": parity_report,
                }
            )
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _layer_ids_argument(value: str) -> list[int]:
    try:
        layer_ids = [int(part) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--target-layer-ids must be comma-separated integers"
        ) from exc
    if not layer_ids:
        raise argparse.ArgumentTypeError("--target-layer-ids must not be empty")
    return layer_ids


def resolve_sglang_capacity(
    *,
    max_length: int,
    adaptive_max_batch_size: int,
    requested_max_running_requests: int,
    requested_max_total_tokens: int,
) -> tuple[int, int]:
    """Resolve SGLang limits from the adaptive batch policy.

    ``0`` means auto: use the same request count selected by the adaptive
    capture policy and reserve enough token capacity for that batch.  This
    avoids the old fixed ``8`` request ceiling leaving B200 VRAM idle.
    """

    if max_length <= 0 or adaptive_max_batch_size <= 0:
        raise ValueError("max_length and adaptive_max_batch_size must be positive")
    running = requested_max_running_requests or adaptive_max_batch_size
    total_tokens = requested_max_total_tokens or running * max_length
    if running <= 0 or total_tokens <= 0:
        raise ValueError("resolved SGLang request/token limits must be positive")
    return int(running), int(total_tokens)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture frozen Qwen hidden states for offline DFlash training"
    )
    parser.add_argument("--input", required=True, help="Teacher-trajectory JSONL")
    parser.add_argument("--output", required=True, help="Feature-store directory")
    parser.add_argument("--target-model-path", required=True)
    layer_selection = parser.add_mutually_exclusive_group(required=True)
    layer_selection.add_argument("--target-layer-ids", type=_layer_ids_argument)
    layer_selection.add_argument("--num-draft-layers", type=int)
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--max-source-tokens", required=True, type=int)
    parser.add_argument("--max-summary-tokens", required=True, type=int)
    parser.add_argument("--chat-template", default="qwen3")
    parser.add_argument("--prompt-template", default=DEFAULT_SUMMARY_PROMPT_TEMPLATE)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--capture-backend",
        choices=("hf", "sglang"),
        default="hf",
        help="hidden-state capture backend; SGLang uses SpecForge OfflineCapture",
    )
    parser.add_argument(
        "--capture-method",
        choices=("eagle3", "dflash", "dspark"),
        default="dflash",
        help="algorithm-owned SGLang hidden-state capture hook",
    )
    parser.add_argument("--sglang-tp-size", type=int, default=1)
    parser.add_argument(
        "--sglang-attention-backend",
        default="flashinfer",
        help="SGLang attention backend for OfflineSGLangCapture",
    )
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.88)
    parser.add_argument(
        "--sglang-max-running-requests",
        type=int,
        default=0,
        help="0 = follow adaptive max batch size",
    )
    parser.add_argument("--sglang-max-total-tokens", type=int, default=0)
    parser.add_argument("--sglang-context-length", type=int, default=None)
    parser.add_argument("--sglang-disable-radix-cache", action="store_true")
    parser.add_argument(
        "--parity-samples",
        type=int,
        default=2,
        help="number of initial samples used for mandatory HF-vs-SGLang parity",
    )
    parser.add_argument("--parity-max-abs-error", type=float, default=5e-2)
    parser.add_argument("--parity-mean-abs-error", type=float, default=1e-2)
    parser.add_argument("--parity-relative-l2-error", type=float, default=5e-2)
    parser.add_argument("--parity-min-cosine-similarity", type=float, default=0.999)
    add_adaptive_cli_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Capture one immutable feature generation from a local JSONL trajectory."""

    args = _parser().parse_args(argv)
    if args.max_source_tokens < 0 or args.max_summary_tokens < 1:
        raise ValueError("source budget must be non-negative and summary budget positive")
    if args.num_draft_layers is not None and args.num_draft_layers < 1:
        raise ValueError("--num-draft-layers must be positive")
    if args.max_source_tokens + args.max_summary_tokens > args.max_length:
        raise ValueError("source and summary token budgets exceed --max-length")
    context = None
    specforge_initialized = False
    try:
        if args.capture_backend == "sglang":
            if args.sglang_tp_size != 1:
                raise ValueError(
                    "capture_features currently uses data parallel workers; "
                    "set --sglang-tp-size 1. TP capture requires a dedicated "
                    "DP-aware launcher so TP ranks do not process different samples."
                )
            available, reason = sglang_capture_available()
            if not available:
                raise RuntimeError(
                    "OfflineSGLangCapture backend is unavailable before distributed "
                    f"initialization: {reason}"
                )
            initialize_specforge_distributed(tp_size=args.sglang_tp_size)
            specforge_initialized = True
        context = initialize_distributed(args.device)
        from transformers import AutoTokenizer

        device = (
            torch.device("cuda", context.local_rank)
            if context.is_distributed and torch.cuda.is_available()
            else torch.device(args.device)
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=True,
        )
        from .prepare_data import iter_summary_examples

        adaptive_settings = adaptive_settings_from_args(args, device)
        sglang_max_running_requests, sglang_max_total_tokens = resolve_sglang_capacity(
            max_length=args.max_length,
            adaptive_max_batch_size=adaptive_settings.max_batch_size,
            requested_max_running_requests=args.sglang_max_running_requests,
            requested_max_total_tokens=args.sglang_max_total_tokens,
        )
        capture_stats: dict[str, Any] = {}
        parity_thresholds = ParityThresholds(
            max_abs_error=args.parity_max_abs_error,
            mean_abs_error=args.parity_mean_abs_error,
            relative_l2_error=args.parity_relative_l2_error,
            min_cosine_similarity=args.parity_min_cosine_similarity,
        )

        prompt_contract = {
            "chat_template": args.chat_template,
            "max_source_tokens": args.max_source_tokens,
            "max_summary_tokens": args.max_summary_tokens,
            "prompt_template": args.prompt_template,
        }
        manifest = capture_dataset(
            target_model_path=args.target_model_path,
            prepared_examples=iter_summary_examples(
                args.input,
                tokenizer,
                max_length=args.max_length,
                chat_template=args.chat_template,
                max_samples=args.max_samples,
                max_source_tokens=args.max_source_tokens,
                max_summary_tokens=args.max_summary_tokens,
                prompt_template=args.prompt_template,
            ),
            output_dir=ranked_path(args.output, context.rank, context.world_size),
            target_layer_ids=args.target_layer_ids,
            num_draft_layers=args.num_draft_layers,
            trust_remote_code=args.trust_remote_code,
            max_length=args.max_length,
            device=device,
            dtype=args.torch_dtype,
            tokenizer_id=str(Path(args.target_model_path)),
            prompt_contract=prompt_contract,
            rank=context.rank,
            world_size=context.world_size,
            allow_empty=context.is_distributed,
            adaptive_settings=adaptive_settings,
            stats=capture_stats,
            capture_backend=args.capture_backend,
            capture_method=args.capture_method,
            sglang_attention_backend=args.sglang_attention_backend,
            sglang_mem_fraction_static=args.sglang_mem_fraction_static,
            sglang_max_running_requests=sglang_max_running_requests,
            sglang_max_total_tokens=sglang_max_total_tokens,
            sglang_context_length=args.sglang_context_length,
            sglang_disable_radix_cache=args.sglang_disable_radix_cache,
            parity_samples=args.parity_samples,
            parity_thresholds=parity_thresholds,
        )
        totals = context.all_reduce_sum(
            torch.tensor(
                [
                    capture_stats.get("captured", 0),
                    capture_stats.get("tokens", 0),
                    capture_stats.get("oom_retries", 0),
                ],
                dtype=torch.float64,
                device=device,
            )
        ).cpu()
        context.barrier()
        if context.is_main_process:
            if context.is_distributed:
                manifest = merge_feature_shards(
                    [
                        ranked_path(args.output, rank, context.world_size)
                        for rank in range(context.world_size)
                    ],
                    args.output,
                )
            if manifest is None:
                raise ValueError("feature capture produced no records")
            capture_stats["captured"] = int(totals[0])
            capture_stats["tokens"] = int(totals[1])
            capture_stats["oom_retries"] = int(totals[2])
            output = manifest.to_dict()
            output["stats"] = capture_stats
            print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    finally:
        cleanup_distributed(context)
        if specforge_initialized:
            destroy_specforge_distributed()


if __name__ == "__main__":
    main()


__all__ = [
    "capture_dataset",
    "main",
    "merge_feature_shards",
    "resolve_sglang_capacity",
]
