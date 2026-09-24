"""Adapt local Vietnamese summarization JSONL for DFlash fine-tuning.

Supported sources use either ``input/output`` or ``id/text/summary`` fields.
The output follows the canonical ``id/document/summary`` contract and receives
a deterministic train/eval split without separating duplicate documents.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterator
from typing import Any

from .data import SummaryRecord, iter_summary_jsonl, render_summary_example


LOGGER = logging.getLogger(__name__)
_JSONL_SUFFIXES = (".jsonl", ".ndjson", ".json", ".txt")
_MOJIBAKE_PATTERN = re.compile(
    r"(?:Ã[\u0080-\u00bf]|[áÁ][º»]|Ä[\u0080-\u009f]|Æ[\u0080-\u00bf])"
)


def resolve_source_path(path_file: str | Path) -> Path:
    """Read the single dataset path recorded by the server handoff.

    Relative paths are resolved relative to ``path_file`` rather than the
    current working directory.  This keeps the handoff portable while leaving
    absolute server paths untouched.
    """

    source_path_file = Path(path_file).expanduser()
    if not source_path_file.is_file():
        raise FileNotFoundError(f"dataset path file not found: {source_path_file}")
    lines = [
        line.strip()
        for line in source_path_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise ValueError(
            f"dataset path file must contain exactly one non-empty path: {source_path_file}"
        )
    source = Path(lines[0]).expanduser()
    if not source.is_absolute():
        source = source_path_file.parent / source
    return source.resolve(strict=False)


def discover_source_files(path: str | Path) -> list[Path]:
    """Resolve a JSONL file or a directory containing JSONL shards.

    Directory contents are sorted by relative path so IDs, split assignment,
    and manifest output remain deterministic across runs.
    """

    source = Path(path).expanduser()
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(
            f"fine-tuning source file or directory not found: {source}"
        )

    files = [
        candidate
        for candidate in source.rglob("*")
        if candidate.is_file()
        and not candidate.name.startswith(".")
        and (
            candidate.name.lower().endswith(_JSONL_SUFFIXES)
            or candidate.name.lower().endswith((".jsonl.gz", ".ndjson.gz"))
        )
    ]
    files.sort(key=lambda candidate: candidate.relative_to(source).as_posix())
    if not files:
        raise FileNotFoundError(
            f"no JSONL/NDJSON source files found under directory: {source}"
        )
    return files


def _open_jsonl(path: Path):
    if path.name.lower().endswith((".jsonl.gz", ".ndjson.gz")):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _source_relative_path(source: Path, source_root: Path) -> str:
    if source_root.is_dir():
        return source.relative_to(source_root).as_posix()
    return source.name


def _canonical_text(value: str) -> tuple[str, bool]:
    stripped = value.strip()
    normalized = unicodedata.normalize("NFC", stripped)
    return normalized, normalized != stripped


def _iter_source_file(
    source: Path,
    *,
    source_root: Path,
    multi_file: bool,
    max_samples: int | None,
    already_yielded: int,
    seen_ids: set[str],
) -> Iterator[SummaryRecord]:
    source_name = _source_relative_path(source, source_root)
    emitted = already_yielded
    with _open_jsonl(source) as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_samples is not None and emitted >= max_samples:
                return
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid fine-tuning JSONL at {source_name}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"fine-tuning JSONL at {source_name}:{line_number} must be a JSON object"
                )

            if "input" in payload or "output" in payload:
                schema = "input_output"
                document_key, summary_key = "input", "output"
                required = (document_key, summary_key)
            elif "text" in payload:
                schema = "id_text_summary"
                document_key, summary_key = "text", "summary"
                required = (document_key, summary_key)
            elif "document" in payload or "summary" in payload:
                schema = "id_document_summary"
                document_key, summary_key = "document", "summary"
                required = (document_key, summary_key)
            else:
                raise ValueError(
                    f"fine-tuning JSONL at {source_name}:{line_number} must use "
                    "{input, output}, {id, text, summary}, or {id, document, summary}"
                )

            missing = [key for key in required if key not in payload]
            if missing:
                raise ValueError(
                    f"fine-tuning JSONL at {source_name}:{line_number} missing fields {missing}"
                )
            raw_document = payload[document_key]
            raw_summary = payload[summary_key]
            if not isinstance(raw_document, str) or not isinstance(raw_summary, str):
                raise ValueError(
                    f"fine-tuning JSONL at {source_name}:{line_number} fields "
                    f"{document_key}/{summary_key} must be strings"
                )
            document, document_nfc_changed = _canonical_text(raw_document)
            summary, summary_nfc_changed = _canonical_text(raw_summary)
            if not document or not summary:
                empty = document_key if not document else summary_key
                raise ValueError(
                    f"fine-tuning JSONL at {source_name}:{line_number} has empty {empty}"
                )

            raw_id = payload.get("id")
            if schema == "input_output" or raw_id is None:
                base_id = f"finetune-{line_number:08d}"
                if multi_file:
                    file_tag = hashlib.sha256(
                        source_name.encode("utf-8")
                    ).hexdigest()[:8]
                    base_id = f"finetune-{file_tag}-{line_number:08d}"
            elif isinstance(raw_id, (str, int, float)) and not isinstance(
                raw_id, bool
            ):
                base_id = str(raw_id).strip()
                if not base_id:
                    raise ValueError(
                        f"fine-tuning JSONL at {source_name}:{line_number} has empty id"
                    )
            else:
                raise ValueError(
                    f"fine-tuning JSONL at {source_name}:{line_number} id must be "
                    "a string or number"
                )

            record_id = base_id
            if record_id in seen_ids:
                record_id = f"{base_id}::{source_name}:{line_number}"
                suffix = 2
                while record_id in seen_ids:
                    record_id = f"{base_id}::{source_name}:{line_number}:{suffix}"
                    suffix += 1
            seen_ids.add(record_id)

            metadata: dict[str, Any] = {
                "source_line": line_number,
                "source_schema": schema,
            }
            if schema != "input_output" or multi_file:
                metadata["source_file"] = source_name
            if raw_id is not None:
                metadata["source_id"] = str(raw_id)
            if document_nfc_changed or summary_nfc_changed:
                metadata["unicode_nfc_normalized"] = True
            if _MOJIBAKE_PATTERN.search(document) or _MOJIBAKE_PATTERN.search(summary):
                # Keep source characters intact. This flag makes suspicious
                # samples visible in the manifest for human review.
                metadata["suspected_mojibake"] = True

            record = SummaryRecord(
                id=record_id,
                document=document,
                summary=summary,
                metadata=metadata,
            )
            emitted += 1
            yield record


def iter_finetune_jsonl(
    path: str | Path,
    max_samples: int | None = None,
) -> Iterator[SummaryRecord]:
    """Yield canonical records from supported JSONL schemas and directory shards."""

    source_root = Path(path).expanduser()
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    source_files = discover_source_files(source_root)
    seen_ids: set[str] = set()
    yielded = 0
    for source in source_files:
        for record in _iter_source_file(
            source,
            source_root=source_root,
            multi_file=len(source_files) > 1,
            max_samples=max_samples,
            already_yielded=yielded,
            seen_ids=seen_ids,
        ):
            yielded += 1
            yield record


def iter_input_output_jsonl(
    path: str | Path,
    max_samples: int | None = None,
) -> Iterator[SummaryRecord]:
    """Yield canonical records from the server ``input/output`` JSONL format.

    The function is streaming and fail-fast: malformed rows are reported with
    their physical line number instead of being silently dropped.  The
    document and summary are stripped only at their outer boundaries; all
    internal Vietnamese whitespace and Unicode content are preserved.
    """

    for record in iter_finetune_jsonl(path, max_samples=max_samples):
        if record.metadata.get("source_schema") != "input_output":
            raise ValueError("source JSONL is not in input/output format")
        yield record


def _eval_assignment(
    document: str,
    *,
    eval_ratio: float,
    split_seed: str,
) -> bool:
    """Assign all identical documents to one deterministic split."""

    key = f"{split_seed}\0{document}".encode("utf-8")
    score = int.from_bytes(hashlib.sha256(key).digest(), byteorder="big")
    return score < int(eval_ratio * (1 << 256))


def _jsonl_line(record: SummaryRecord) -> str:
    payload = {
        "id": record.id,
        "document": record.document,
        "summary": record.summary,
        **dict(record.metadata),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"


def _move_one_document_group(
    source_path: Path,
    destination_path: Path,
) -> tuple[int, int]:
    """Move one complete document group between split files."""

    lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines:
        return 0, 0
    documents = [json.loads(line)["document"] for line in lines]
    selected_document = documents[0]
    if all(document == selected_document for document in documents):
        raise ValueError(
            "cannot create non-leaking train/eval split: all records share one document"
        )
    moved = [
        line for line, document in zip(lines, documents, strict=True)
        if document == selected_document
    ]
    remaining = [
        line for line, document in zip(lines, documents, strict=True)
        if document != selected_document
    ]
    destination_path.write_text("".join(moved), encoding="utf-8")
    source_path.write_text("".join(remaining), encoding="utf-8")
    return len(remaining), len(moved)


def _write_non_empty_eval_fallback(
    train_path: Path,
    eval_path: Path,
    *,
    total: int,
    eval_ratio: float,
) -> tuple[int, int]:
    """Keep tiny smoke datasets usable by guaranteeing both split files.

    For a real 50k corpus this branch is practically unreachable.  It matters
    for a small sample whose hashes happen to land entirely in one split, as
    the DFlash launcher requires a non-empty validation input.
    """

    with train_path.open(encoding="utf-8") as handle:
        train_count = sum(1 for line in handle if line.strip())
    with eval_path.open(encoding="utf-8") as handle:
        eval_count = sum(1 for line in handle if line.strip())
    if total < 2 or eval_ratio <= 0 or (train_count and eval_count):
        return train_count, eval_count

    if eval_count == 0 and train_count > 1:
        remaining, moved = _move_one_document_group(train_path, eval_path)
        return remaining, moved

    if train_count == 0 and eval_count > 1:
        moved, remaining = _move_one_document_group(eval_path, train_path)
        return moved, remaining

    return train_count, eval_count


def prepare_finetune_dataset(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    eval_ratio: float = 0.02,
    split_seed: str = "42",
    max_samples: int | None = None,
    progress_interval: int = 1000,
    force: bool = False,
) -> dict[str, int | float]:
    """Normalize a JSONL file or directory of shards for DFlash.

    Outputs are written through a temporary staging directory and published
    only after the source has been read successfully.  Existing prepared files
    are protected unless ``force=True`` is explicitly requested.
    """

    source = Path(source_path).expanduser()
    destination = Path(output_dir).expanduser()
    if not source.is_file() and not source.is_dir():
        raise FileNotFoundError(
            f"fine-tuning source file or directory not found: {source}"
        )
    if not 0.0 <= eval_ratio < 1.0:
        raise ValueError("eval_ratio must satisfy 0 <= eval_ratio < 1")
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")

    destination.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "train": destination / "train.jsonl",
        "eval": destination / "eval.jsonl",
        "manifest": destination / "manifest.json",
    }
    if not force and any(path.exists() for path in final_paths.values()):
        raise FileExistsError(
            f"prepared dataset already exists under {destination}; use force=True to replace it"
        )

    stage = Path(tempfile.mkdtemp(prefix=".prepare-finetune-", dir=destination))
    train_stage = stage / "train.jsonl"
    eval_stage = stage / "eval.jsonl"
    try:
        total = 0
        train_count = 0
        eval_count = 0
        source_schema_counts: Counter[str] = Counter()
        source_file_counts: Counter[str] = Counter()
        nfc_normalized_records = 0
        suspected_mojibake_records = 0
        source_files = discover_source_files(source)
        with (
            train_stage.open("w", encoding="utf-8") as train_handle,
            eval_stage.open("w", encoding="utf-8") as eval_handle,
        ):
            for record in iter_finetune_jsonl(source, max_samples=max_samples):
                total += 1
                schema_name = str(record.metadata.get("source_schema", "unknown"))
                file_name = str(
                    record.metadata.get("source_file", source_files[0].name)
                )
                source_schema_counts[schema_name] += 1
                source_file_counts[file_name] += 1
                nfc_normalized_records += int(
                    bool(record.metadata.get("unicode_nfc_normalized"))
                )
                suspected_mojibake_records += int(
                    bool(record.metadata.get("suspected_mojibake"))
                )
                if _eval_assignment(
                    record.document,
                    eval_ratio=eval_ratio,
                    split_seed=split_seed,
                ):
                    eval_handle.write(_jsonl_line(record))
                    eval_count += 1
                else:
                    train_handle.write(_jsonl_line(record))
                    train_count += 1
                if total % progress_interval == 0:
                    LOGGER.info(
                        "prepared %d records (train=%d, eval=%d)",
                        total,
                        train_count,
                        eval_count,
                    )

        train_count, eval_count = _write_non_empty_eval_fallback(
            train_stage,
            eval_stage,
            total=total,
            eval_ratio=eval_ratio,
        )
        if total == 0:
            raise ValueError("fine-tuning source JSONL contains no usable records")
        if train_count == 0:
            raise ValueError("prepared dataset has no training records")

        stats: dict[str, int | float] = {
            "total": total,
            "train": train_count,
            "eval": eval_count,
            "eval_ratio": eval_ratio,
        }
        if suspected_mojibake_records:
            LOGGER.warning(
                "%d records contain likely mojibake markers; source text was preserved "
                "without automatic repair",
                suspected_mojibake_records,
            )
        manifest = {
            "schema": "dflash_summary_v1",
            "source_schema": (
                next(iter(source_schema_counts))
                if len(source_schema_counts) == 1
                else "mixed"
            ),
            "source_path": str(source.resolve(strict=False)),
            "source_files": [
                _source_relative_path(path, source) for path in source_files
            ],
            "source_schema_counts": dict(sorted(source_schema_counts.items())),
            "source_file_counts": dict(sorted(source_file_counts.items())),
            "unicode_nfc_normalized_records": nfc_normalized_records,
            "suspected_mojibake_records": suspected_mojibake_records,
            "split_seed": split_seed,
            "eval_ratio": eval_ratio,
            "max_samples": max_samples,
            **stats,
            "files": {"train": "train.jsonl", "eval": "eval.jsonl"},
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for name, stage_path in (
            ("train", train_stage),
            ("eval", eval_stage),
            ("manifest", stage / "manifest.json"),
        ):
            os.replace(stage_path, final_paths[name])
        return stats
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def iter_summary_examples(
    path: str | Path,
    tokenizer: Any,
    *,
    max_length: int,
    chat_template: str = "qwen3",
    max_samples: int | None = None,
    max_source_tokens: int | None = None,
    max_summary_tokens: int | None = None,
    prompt_template: str | None = None,
):
    """Render JSONL records lazily for capture without materializing the corpus.

    Malformed or too-short target trajectories are intentionally skipped.  The
    capture command rejects an entirely unusable input through
    ``capture_dataset``; this lets one bad document avoid invalidating a long
    offline batch while retaining deterministic input order for the rest.
    """

    for record in iter_summary_jsonl(path, max_samples=max_samples):
        try:
            kwargs = (
                {"prompt_template": prompt_template}
                if prompt_template is not None
                else {}
            )
            rendered = render_summary_example(
                record,
                tokenizer,
                max_length=max_length,
                chat_template=chat_template,
                max_source_tokens=max_source_tokens,
                max_summary_tokens=max_summary_tokens,
                **kwargs,
            )
        except ValueError:
            continue
        yield {**rendered, "id": record.id}


def prepare_summary_examples(
    path: str | Path,
    tokenizer: Any,
    *,
    max_length: int,
    chat_template: str = "qwen3",
    max_samples: int | None = None,
    max_source_tokens: int | None = None,
    max_summary_tokens: int | None = None,
    prompt_template: str | None = None,
) -> list[dict[str, Any]]:
    examples = list(
        iter_summary_examples(
            path,
            tokenizer,
            max_length=max_length,
            chat_template=chat_template,
            max_samples=max_samples,
            max_source_tokens=max_source_tokens,
            max_summary_tokens=max_summary_tokens,
            prompt_template=prompt_template,
        )
    )
    if not examples:
        raise ValueError("no trainable summary examples after rendering")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize Vietnamese summarization JSONL for DFlash fine-tuning"
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source",
        type=Path,
        help="JSONL file or directory of JSONL shards",
    )
    source_group.add_argument(
        "--source-path-file",
        type=Path,
        help="text file containing one server-side JSONL file or directory path",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--split-seed", default="42")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace train.jsonl, eval.jsonl and manifest.json in output-dir",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    source = (
        args.source.expanduser()
        if args.source is not None
        else resolve_source_path(args.source_path_file)
    )
    stats = prepare_finetune_dataset(
        source,
        args.output_dir,
        eval_ratio=args.eval_ratio,
        split_seed=args.split_seed,
        max_samples=args.max_samples,
        progress_interval=args.progress_interval,
        force=args.force,
    )
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


__all__ = [
    "discover_source_files",
    "iter_finetune_jsonl",
    "iter_input_output_jsonl",
    "iter_summary_examples",
    "prepare_finetune_dataset",
    "prepare_summary_examples",
    "resolve_source_path",
]


if __name__ == "__main__":
    main()
