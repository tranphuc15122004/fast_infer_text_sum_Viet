"""Validated CPU-readable offline feature records for DFlash training."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset


FEATURE_MANIFEST_FILENAME = "manifest.json"
FEATURE_SCHEMA_VERSION = "dflash_offline_features_v1"
FEATURE_KEYS = ("input_ids", "loss_mask", "hidden_states")


def _dtype_name(dtype: torch.dtype | str) -> str:
    if isinstance(dtype, torch.dtype):
        return str(dtype)
    text = str(dtype)
    return text if text.startswith("torch.") else f"torch.{text}"


def _resolve_dtype(name: str) -> torch.dtype:
    normalized = _dtype_name(name).removeprefix("torch.")
    value = getattr(torch, normalized, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported tensor dtype in feature manifest: {name!r}")
    return value


_INTEGER_DTYPES = {
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _reject_symlink_components(path: Path) -> None:
    """Reject symlinks in every lexical component of a feature path."""

    absolute = path.absolute()
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"feature path cannot contain symlink component: {cursor}")


@dataclass
class FeatureManifest:
    """Provenance and tensor-shape contract for one feature directory."""

    model_id: str
    layer_ids: list[int]
    hidden_size: int
    max_length: int
    revision: str | None = None
    tokenizer_id: str | None = None
    prompt_contract: dict[str, Any] | None = None
    feature_width: int | None = None
    input_ids_dtype: str = "torch.int64"
    loss_mask_dtype: str = "torch.float32"
    hidden_states_dtype: str = "torch.float32"
    capture_backend: str = "hf"
    capture_method: str | None = None
    capture_metadata: dict[str, Any] | None = None
    schema_version: str = FEATURE_SCHEMA_VERSION
    generation_dir: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("feature manifest model_id must be a non-empty string")
        if not isinstance(self.layer_ids, (list, tuple)):
            raise ValueError("feature manifest layer_ids must be a list of integers")
        if any(
            isinstance(layer_id, bool) or not isinstance(layer_id, int)
            for layer_id in self.layer_ids
        ):
            raise ValueError("feature manifest layer_ids must be integers")
        self.layer_ids = list(self.layer_ids)
        if not self.layer_ids or any(layer_id < 0 for layer_id in self.layer_ids):
            raise ValueError("feature manifest layer_ids must be non-empty and non-negative")
        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("feature manifest layer_ids must not contain duplicates")
        if self.hidden_size < 1 or self.max_length < 1:
            raise ValueError("feature manifest hidden_size and max_length must be positive")
        expected_width = len(self.layer_ids) * int(self.hidden_size)
        if self.feature_width is None:
            self.feature_width = expected_width
        if int(self.feature_width) != expected_width:
            raise ValueError(
                "feature width does not match layer_ids * hidden_size: "
                f"{self.feature_width} != {expected_width}"
            )
        self.feature_width = int(self.feature_width)
        self.input_ids_dtype = _dtype_name(self.input_ids_dtype)
        self.loss_mask_dtype = _dtype_name(self.loss_mask_dtype)
        self.hidden_states_dtype = _dtype_name(self.hidden_states_dtype)
        if self.capture_backend not in {"hf", "sglang"}:
            raise ValueError("feature manifest capture_backend must be 'hf' or 'sglang'")
        if self.capture_method is not None and self.capture_method not in {
            "eagle3",
            "dflash",
            "dspark",
        }:
            raise ValueError("feature manifest capture_method is unsupported")
        if self.capture_metadata is not None:
            if not isinstance(self.capture_metadata, Mapping):
                raise ValueError("feature manifest capture_metadata must be an object")
            self.capture_metadata = dict(self.capture_metadata)
        for dtype_name in (
            self.input_ids_dtype,
            self.loss_mask_dtype,
            self.hidden_states_dtype,
        ):
            _resolve_dtype(dtype_name)
        if _resolve_dtype(self.input_ids_dtype) not in _INTEGER_DTYPES:
            raise ValueError(
                "feature manifest input_ids_dtype must be an integer dtype, "
                f"got {self.input_ids_dtype}"
            )
        if self.schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported feature manifest schema {self.schema_version!r}"
            )
        if self.prompt_contract is not None:
            if not isinstance(self.prompt_contract, Mapping):
                raise ValueError("feature manifest prompt_contract must be an object")
            self.prompt_contract = dict(self.prompt_contract)

    @property
    def dtype(self) -> str:
        """Compatibility alias for the stored hidden-state dtype."""

        return self.hidden_states_dtype

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "revision": self.revision,
            "tokenizer_id": self.tokenizer_id,
            "prompt_contract": self.prompt_contract,
            "layer_ids": list(self.layer_ids),
            "hidden_size": int(self.hidden_size),
            "feature_width": int(self.feature_width),
            "max_length": int(self.max_length),
            "input_ids_dtype": self.input_ids_dtype,
            "loss_mask_dtype": self.loss_mask_dtype,
            "hidden_states_dtype": self.hidden_states_dtype,
            "capture_backend": self.capture_backend,
            "capture_method": self.capture_method,
            "capture_metadata": self.capture_metadata,
            "generation_dir": self.generation_dir,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureManifest":
        if not isinstance(payload, Mapping):
            raise ValueError("feature manifest must be a JSON object")
        required = ("model_id", "layer_ids", "hidden_size", "max_length")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"feature manifest missing fields {missing}")
        return cls(
            model_id=str(payload["model_id"]),
            revision=payload.get("revision"),
            tokenizer_id=payload.get("tokenizer_id"),
            prompt_contract=(
                dict(payload["prompt_contract"])
                if payload.get("prompt_contract") is not None
                else None
            ),
            layer_ids=list(payload["layer_ids"]),
            hidden_size=int(payload["hidden_size"]),
            feature_width=(
                int(payload["feature_width"])
                if payload.get("feature_width") is not None
                else None
            ),
            max_length=int(payload["max_length"]),
            input_ids_dtype=str(payload.get("input_ids_dtype", "torch.int64")),
            loss_mask_dtype=str(payload.get("loss_mask_dtype", "torch.float32")),
            hidden_states_dtype=str(
                payload.get("hidden_states_dtype", payload.get("dtype", "torch.float32"))
            ),
            capture_backend=str(payload.get("capture_backend", "hf")),
            capture_method=(
                str(payload["capture_method"])
                if payload.get("capture_method") is not None
                else None
            ),
            capture_metadata=(
                dict(payload["capture_metadata"])
                if payload.get("capture_metadata") is not None
                else None
            ),
            schema_version=str(payload.get("schema_version", FEATURE_SCHEMA_VERSION)),
            generation_dir=(
                str(payload["generation_dir"])
                if payload.get("generation_dir") is not None
                else None
            ),
        )


def _one_dimensional(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"feature {name!r} must be a tensor")
    if value.ndim == 2 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim != 1:
        raise ValueError(f"feature {name!r} must have shape [sequence_length]")
    return value


def _hidden_2d(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError("feature 'hidden_states' must be a tensor")
    if value.ndim == 3 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim != 2:
        raise ValueError(
            "feature 'hidden_states' must have shape "
            "[sequence_length, feature_width]"
        )
    return value


def validate_feature_record(
    record: Mapping[str, Any],
    manifest: FeatureManifest,
) -> None:
    """Validate one CPU feature record against its manifest."""

    if not isinstance(record, Mapping):
        raise ValueError("feature record must be a mapping")
    missing = [key for key in FEATURE_KEYS if key not in record]
    if missing:
        raise ValueError(f"feature record missing required keys {missing}")
    input_ids = _one_dimensional(record["input_ids"], "input_ids")
    loss_mask = _one_dimensional(record["loss_mask"], "loss_mask")
    hidden_states = _hidden_2d(record["hidden_states"])
    lengths = (input_ids.shape[0], loss_mask.shape[0], hidden_states.shape[0])
    if len(set(int(length) for length in lengths)) != 1:
        raise ValueError(
            "feature record has mismatched sequence lengths: "
            f"input_ids={lengths[0]}, loss_mask={lengths[1]}, "
            f"hidden_states={lengths[2]}"
        )
    if int(hidden_states.shape[1]) != manifest.feature_width:
        raise ValueError(
            "feature width mismatch: "
            f"{hidden_states.shape[1]} != {manifest.feature_width}"
        )
    if input_ids.dtype not in _INTEGER_DTYPES:
        raise ValueError(
            "feature input_ids must use an integer dtype, "
            f"got {input_ids.dtype}"
        )
    expected_dtypes = {
        "input_ids": _resolve_dtype(manifest.input_ids_dtype),
        "loss_mask": _resolve_dtype(manifest.loss_mask_dtype),
        "hidden_states": _resolve_dtype(manifest.hidden_states_dtype),
    }
    actual_dtypes = {
        "input_ids": input_ids.dtype,
        "loss_mask": loss_mask.dtype,
        "hidden_states": hidden_states.dtype,
    }
    for name in FEATURE_KEYS:
        if actual_dtypes[name] != expected_dtypes[name]:
            raise ValueError(
                f"feature {name!r} dtype mismatch: {actual_dtypes[name]} != "
                f"{expected_dtypes[name]}"
            )
    if not torch.isfinite(loss_mask.float()).all():
        raise ValueError("feature loss_mask contains non-finite values")
    if not torch.all((loss_mask == 0) | (loss_mask == 1)):
        raise ValueError("feature loss_mask values must be binary (0 or 1)")
    if not torch.isfinite(hidden_states.float()).all():
        raise ValueError("feature hidden_states contains non-finite values")
    # SpecForge truncates aligned tensors before checking the minimum
    # supervised span for the shifted objective.
    mask = loss_mask[: manifest.max_length].detach().cpu().tolist()
    if not any(bool(current) and bool(following) for current, following in zip(mask, mask[1:])):
        raise ValueError(
            "offline feature requires two consecutive supervised tokens"
        )


def _load_record(path: Path) -> dict[str, torch.Tensor]:
    opener = gzip.open if path.name.endswith(".gz") else open
    try:
        with opener(path, "rb") as handle:
            payload = torch.load(handle, map_location="cpu", weights_only=True)
    except TypeError:  # compatibility with older local torch builds
        with opener(path, "rb") as handle:
            payload = torch.load(handle, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"feature file must contain a tensor mapping: {path}")
    return payload


class OfflineFeatureDataset(Dataset[dict[str, torch.Tensor]]):
    """Read one validated CPU tensor mapping per feature file."""

    def __init__(
        self,
        root: str | Path,
        manifest: FeatureManifest | Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        _reject_symlink_components(self.root)
        if self.root.is_symlink() or not self.root.is_dir():
            raise FileNotFoundError(f"offline feature directory not found: {self.root}")
        manifest_path = self.root / FEATURE_MANIFEST_FILENAME
        if manifest is None:
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"offline feature manifest must be written before loading: {manifest_path}"
                )
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid feature manifest: {manifest_path}") from exc
            manifest = FeatureManifest.from_dict(payload)
        elif isinstance(manifest, Mapping):
            manifest = FeatureManifest.from_dict(manifest)
        if not isinstance(manifest, FeatureManifest):
            raise TypeError("manifest must be a FeatureManifest or mapping")
        self.manifest = manifest
        self.record_root = self.root
        if manifest.generation_dir is not None:
            raw_record_root = self.root / manifest.generation_dir
            cursor = self.root
            for part in Path(manifest.generation_dir).parts:
                cursor = cursor / part
                if cursor.is_symlink():
                    raise ValueError(
                        "feature manifest generation_dir contains a symlink: "
                        f"{manifest.generation_dir!r}"
                    )
            root_resolved = self.root.resolve()
            record_root = raw_record_root.resolve()
            if root_resolved not in record_root.parents or not record_root.is_dir():
                raise ValueError(
                    "feature manifest generation_dir must name a directory inside root: "
                    f"{manifest.generation_dir!r}"
                )
            self.record_root = record_root
        suffixes = (".pt", ".pth", ".ckpt", ".ckpt.gz")
        self._paths = sorted(
            (
                path
                for path in self.record_root.rglob("*")
                if (
                    (not path.is_symlink())
                    and path.is_file()
                    and path.name.endswith(suffixes)
                )
            ),
            key=lambda path: path.relative_to(self.root).as_posix(),
        )
        for path in self.record_root.rglob("*"):
            if path.is_symlink():
                raise ValueError(
                    "offline feature directory cannot contain symlinks: " f"{path}"
                )
        if not self._paths:
            raise ValueError(f"offline feature directory has no tensor records: {self.root}")
        # Do not deserialize a whole corpus here: a Qwen hidden-state store can
        # be hundreds of GiB.  Capture validates each record before publishing
        # its immutable generation, while __getitem__ validates again at the
        # consumption boundary (including worker processes).

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        payload = _load_record(self._paths[index])
        validate_feature_record(payload, self.manifest)
        input_ids = _one_dimensional(payload["input_ids"], "input_ids")
        loss_mask = _one_dimensional(payload["loss_mask"], "loss_mask")
        hidden_states = _hidden_2d(payload["hidden_states"])
        max_length = self.manifest.max_length
        return {
            "input_ids": input_ids[:max_length].detach().cpu(),
            "loss_mask": loss_mask[:max_length].detach().cpu(),
            "hidden_states": hidden_states[:max_length].detach().cpu(),
        }


def _canonicalize_input_ids(value: torch.Tensor) -> torch.Tensor:
    """Convert integral ids to the long dtype required by embeddings."""

    if value.dtype in _INTEGER_DTYPES:
        return value.to(dtype=torch.long)
    if value.dtype.is_floating_point:
        if not torch.isfinite(value).all() or not torch.equal(value, value.round()):
            raise ValueError("input_ids must contain integer values")
        return value.to(dtype=torch.long)
    raise ValueError(f"input_ids must use an integer dtype, got {value.dtype}")


def _canonicalize_loss_mask(value: torch.Tensor) -> torch.Tensor:
    """Normalize binary masks to the float dtype consumed by the objective."""

    if value.dtype.is_floating_point and not torch.isfinite(value).all():
        raise ValueError("loss_mask contains non-finite values")
    if not torch.all((value == 0) | (value == 1)):
        raise ValueError("loss_mask values must be binary (0 or 1)")
    return value.to(dtype=torch.float32)


def collate_features(
    features: list[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Right-pad a batch, zero-filling all three sequence-aligned tensors."""

    if not features:
        raise ValueError("cannot collate an empty feature batch")
    normalized: list[dict[str, torch.Tensor]] = []
    for index, feature in enumerate(features):
        missing = [key for key in FEATURE_KEYS if key not in feature]
        if missing:
            raise ValueError(f"feature {index} missing required keys {missing}")
        input_ids = _canonicalize_input_ids(
            _one_dimensional(feature["input_ids"], "input_ids")
        )
        loss_mask = _canonicalize_loss_mask(
            _one_dimensional(feature["loss_mask"], "loss_mask")
        )
        hidden_states = _hidden_2d(feature["hidden_states"])
        lengths = (input_ids.shape[0], loss_mask.shape[0], hidden_states.shape[0])
        if len(set(int(length) for length in lengths)) != 1:
            raise ValueError(f"feature {index} has mismatched sequence lengths")
        normalized.append(
            {
                "input_ids": input_ids,
                "loss_mask": loss_mask,
                "hidden_states": hidden_states,
            }
        )
    max_length = max(int(item["input_ids"].shape[0]) for item in normalized)
    for key in FEATURE_KEYS:
        dtypes = {item[key].dtype for item in normalized}
        if len(dtypes) != 1:
            raise ValueError(f"cannot collate {key!r} tensors with different dtypes")
    widths = {int(item["hidden_states"].shape[1]) for item in normalized}
    if len(widths) != 1:
        raise ValueError(f"cannot collate hidden_states with different feature width: {widths}")

    batch: dict[str, torch.Tensor] = {}
    for key in ("input_ids", "loss_mask"):
        result = torch.zeros(
            (len(normalized), max_length),
            dtype=normalized[0][key].dtype,
            device=normalized[0][key].device,
        )
        for row, item in enumerate(normalized):
            length = item[key].shape[0]
            result[row, :length] = item[key]
        batch[key] = result
    width = next(iter(widths))
    hidden_result = torch.zeros(
        (len(normalized), max_length, width),
        dtype=normalized[0]["hidden_states"].dtype,
        device=normalized[0]["hidden_states"].device,
    )
    for row, item in enumerate(normalized):
        length = item["hidden_states"].shape[0]
        hidden_result[row, :length] = item["hidden_states"]
    batch["hidden_states"] = hidden_result
    return batch


__all__ = [
    "FEATURE_KEYS",
    "FEATURE_MANIFEST_FILENAME",
    "FEATURE_SCHEMA_VERSION",
    "FeatureManifest",
    "OfflineFeatureDataset",
    "collate_features",
    "validate_feature_record",
]
