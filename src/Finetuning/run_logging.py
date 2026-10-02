"""Progress helpers shared by the offline Finetuning preparation stages."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import TextIO

from tqdm import tqdm


PROGRESS_EVENT_PREFIX = "@@FINETUNE_PROGRESS "


def count_jsonl_records(path: str | Path, max_records: int | None = None) -> int:
    """Count non-empty JSONL rows without retaining them in memory."""

    if max_records is not None and max_records < 0:
        raise ValueError("max_records must be non-negative")
    if max_records == 0:
        return 0
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            if max_records is not None and count >= max_records:
                break
    return count


class ProgressReporter:
    """Render tqdm directly or emit line events for a parent launcher to render."""

    def __init__(
        self,
        description: str,
        total: int,
        *,
        unit: str = "records",
        enabled: bool = True,
        event_stream: TextIO | None = None,
        min_interval: float = 1.0,
    ) -> None:
        if total < 0:
            raise ValueError("progress total must be non-negative")
        self.description = description
        self.total = total
        self.unit = unit
        self.enabled = enabled
        self.event_mode = os.environ.get("FINETUNE_PROGRESS_MODE") == "events"
        self.event_stream = event_stream or sys.stderr
        self.min_interval = max(0.1, float(min_interval))
        self.n = 0
        self._last_emit = 0.0
        self._bar = None
        self._closed = False
        if self.enabled and not self.event_mode:
            self._bar = tqdm(
                total=total,
                desc=description,
                unit=unit,
                dynamic_ncols=True,
                mininterval=self.min_interval,
                file=sys.stderr,
            )
        elif self.enabled:
            self._emit(force=True)

    def _emit(self, *, force: bool = False, complete: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_emit < self.min_interval and self.n < self.total:
            return
        payload = {
            "description": self.description,
            "n": self.n,
            "total": self.total,
            "unit": self.unit,
            "complete": complete,
        }
        self.event_stream.write(
            PROGRESS_EVENT_PREFIX
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
        self.event_stream.flush()
        self._last_emit = now

    def update(self, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("progress update must be non-negative")
        if self._closed or amount == 0:
            return
        self.n = min(self.total, self.n + amount)
        if not self.enabled:
            return
        if self.event_mode:
            self._emit()
        elif self._bar is not None:
            self._bar.update(amount)

    def close(self, *, complete: bool = False) -> None:
        if self._closed:
            return
        if complete and self.n < self.total:
            self.update(self.total - self.n)
        if self.enabled:
            if self.event_mode:
                self._emit(force=True, complete=complete)
            elif self._bar is not None:
                self._bar.close()
        self._closed = True


__all__ = ["PROGRESS_EVENT_PREFIX", "ProgressReporter", "count_jsonl_records"]
