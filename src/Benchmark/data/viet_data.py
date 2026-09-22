#!/usr/bin/env python3
"""Loader chuẩn hoá 4 bộ dữ liệu tóm tắt tiếng Việt trong ``datasets/raw/``.

Mọi loader trả về record theo **một schema thống nhất** dùng cho eval baseline:

    {
      "id":             "<dataset>-<split>-<source_id>",
      "dataset":        "vietnews|wikilingua|vims|vlsp",
      "source_split":   "test" | "all",
      "source_index":   int,      # thứ tự trong pool nguồn (ổn định)
      "source_id":      str,      # định danh gốc (tên file / cluster / id)
      "task_type":      "summarization",
      "document":       str,      # văn bản nguồn -> loader render prompt từ đây
      "reference":      str,      # bản tóm tắt vàng -> rouge.py chấm từ đây
      "answers":        [str, ...],
      "num_source_docs": int,
      "metadata":       {...}
    }

``document`` và ``reference`` là đúng hai field mà ``data_loader.py`` và
``rouge.py`` của project tham chiếu tự nhận diện, nên record ở đây chạy được
trực tiếp qua ``DATA_INPUT`` mà không cần sửa loader.

Chỉ dùng thư viện chuẩn: server B200 không có internet và profile wheel có thể
không cài tokenizer/dataset library.

Quy tắc parse từng bộ (đã kiểm chứng trên dữ liệu thật):

VietNews (``*.txt.seg``)
    Tách theo dòng trống thành các block: ``block[0]`` = title,
    ``block[1]`` = summary, **toàn bộ ``block[2:]`` = document** (giữ nguyên
    xuống dòng, gồm cả byline và caption ảnh). Văn bản đã tách từ nên phải
    de-segment: ``_`` -> space rồi sửa khoảng trắng quanh dấu câu.
    Kiểm chứng: 22.643/22.644 file test (99,996%) tách được; 1 file lỗi bị bỏ.

WikiLingua (``*.json`` nhưng thực chất là **JSONL**)
    Mỗi dòng là ``{"src": [câu...], "tgt": [câu...]}``. Nối câu bằng space.
    Bản phân phối này là Việt -> Việt (không phải cặp cross-lingual en-vi).

ViMs (``Cluster_XXX``)
    Mỗi bài có header ``Title:/Source:/Link:/Published Date:/Author:/Tags:/
    Summary:/Content:``. Bỏ toàn bộ metadata, chỉ lấy phần sau ``Content:``,
    giữ ``Title`` làm marker. Tên file là **ID toàn cục** (``11.txt``,
    ``1939.txt``) nên phải sort theo số. Reference là bản gold của cluster,
    giữ cả hai bản gold trong ``answers``.

VLSP 2022 ABMUSU (``*.jsonl``)
    Mỗi record có ``single_documents`` (list ``{title, anchor_text, raw_text}``),
    ``summary`` và ``category``. Dùng ``raw_text`` làm nội dung, bỏ
    ``anchor_text`` (chỉ là đoạn dẫn ngắn).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterator

# --------------------------------------------------------------------------- #
# Vị trí dữ liệu
# --------------------------------------------------------------------------- #

#: Thư mục gốc chứa dữ liệu thô, tính từ gốc repository.
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[3] / "datasets" / "raw"

VIETNEWS_DIR = "vietnews-master/data"
WIKILINGUA_DIR = "wikilingua"
VIMS_DIR = "ViMs-Dataset-master/ViMs"
VLSP_DIR = "vlsp_raw"

#: Pool dùng để build bộ eval cho từng bộ dữ liệu.
#:
#: * ``vietnews``  -> split ``test`` (bản thân bộ này có train/val/test).
#: * ``wikilingua``-> split ``test``.
#: * ``vims``      -> cả 300 cluster (bộ không chia split).
#: * ``vlsp``      -> file ``vlsp_2022_abmusu.jsonl`` (300 record CÓ gold).
#:   KHÔNG dùng ``vlsp_abmusu_test_data.jsonl`` vì test chính thức không có
#:   ``summary`` -> không chấm được ROUGE.
POOLS: dict[str, dict[str, Any]] = {
    "vietnews": {"split": "test", "source_split": "test", "multi_doc": False},
    "wikilingua": {"split": "test", "source_split": "test", "multi_doc": False},
    "vims": {"split": "all", "source_split": "all", "multi_doc": True},
    "vlsp": {"split": "all", "source_split": "all", "multi_doc": True},
}

DATASET_ORDER = ("vietnews", "wikilingua", "vims", "vlsp")


# --------------------------------------------------------------------------- #
# Chuẩn hoá văn bản
# --------------------------------------------------------------------------- #

_BLANK_SPLIT = re.compile(r"\n\s*\n")
_MULTISPACE = re.compile(r"[ \t]{2,}")
#: Khoảng trắng thừa trước dấu câu, sinh ra do văn bản đã tách từ.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?%)\]}»”])")
#: Khoảng trắng thừa sau dấu mở ngoặc.
_SPACE_AFTER_OPEN = re.compile(r"([(\[{«“])\s+")
#: Space trước dấu ngoặc đóng/kết câu tiếng Việt.
_QUOTE_SPACE = re.compile(r"\s+([,.!?;:])(\s|$)")


def normalize_unicode(text: str) -> str:
    """Chuẩn hoá NFC để so sánh/khử trùng lặp ổn định (giữ nguyên dấu tiếng Việt)."""

    return unicodedata.normalize("NFC", text)


def collapse_whitespace(text: str) -> str:
    """Gộp khoảng trắng thừa nhưng **giữ nguyên** dấu xuống dòng."""

    text = _MULTISPACE.sub(" ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def deunderscore(text: str, *, punct_clean: bool = True) -> str:
    """Chuyển văn bản đã tách từ về văn bản tự nhiên.

    ``_`` nối các âm tiết của một từ ghép (``Khởi_tố``, ``thủ_tướng``) nên chỉ
    cần thay bằng space. Ngoài ra định dạng tách từ còn để lại space trước dấu
    câu (``mâu_thuẫn ,``) -- ``punct_clean`` dọn phần này để văn bản đọc tự
    nhiên khi đưa vào model. Có thể tắt bằng ``punct_clean=False`` nếu muốn giữ
    nguyên dấu vết của dữ liệu gốc.
    """

    text = text.replace("_", " ")
    if punct_clean:
        text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
        text = _SPACE_AFTER_OPEN.sub(r"\1", text)
        text = _QUOTE_SPACE.sub(r"\1\2", text)
    return collapse_whitespace(text)


def split_blocks(text: str) -> list[str]:
    """Tách văn bản thành các block ngăn bởi dòng trống."""

    return [normalize_unicode(b).strip() for b in _BLANK_SPLIT.split(text) if b.strip()]


def numeric_key(path: Path) -> tuple[int, str]:
    """Sort theo số trong tên file (``11.txt`` < ``1939.txt``), không theo chuỗi."""

    stem = path.name
    match = re.search(r"(\d+)", stem)
    return (int(match.group(1)) if match else 0, stem)


def word_count(text: str) -> int:
    """Đếm từ đơn giản bằng khoảng trắng; đủ ổn định để xếp bin độ dài."""

    return len(text.split())


# --------------------------------------------------------------------------- #
# Loader từng bộ
# --------------------------------------------------------------------------- #

def iter_vietnews(
    raw_dir: Path, split: str, *, punct_clean: bool = True
) -> Iterator[dict[str, Any]]:
    """Yield record từ ``vietnews-master/data/<split>_tokenized/*.txt.seg``."""

    folder = raw_dir / VIETNEWS_DIR / f"{split}_tokenized"
    if not folder.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục VietNews: {folder}")

    for index, path in enumerate(sorted(folder.glob("*.txt.seg"), key=numeric_key)):
        blocks = split_blocks(path.read_text(encoding="utf-8"))
        if len(blocks) < 3:
            # Không tách được title/summary/content -> bỏ, đếm ở manifest.
            yield {
                "_skip": True,
                "dataset": "vietnews",
                "source_id": path.stem.replace(".txt", ""),
                "reason": f"chỉ có {len(blocks)} block (<3)",
            }
            continue

        title = deunderscore(blocks[0], punct_clean=punct_clean)
        summary = deunderscore(blocks[1], punct_clean=punct_clean)
        # Toàn bộ phần còn lại là document, giữ nguyên xuống dòng.
        document = deunderscore("\n".join(blocks[2:]), punct_clean=punct_clean)

        if not document or not summary:
            yield {
                "_skip": True,
                "dataset": "vietnews",
                "source_id": path.stem.replace(".txt", ""),
                "reason": "document hoặc summary rỗng",
            }
            continue

        yield {
            "id": f"vietnews-{split}-{path.stem.replace('.txt', '')}",
            "dataset": "vietnews",
            "source_split": split,
            "source_index": index,
            "source_id": path.stem.replace(".txt", ""),
            "task_type": "summarization",
            "document": document,
            "reference": summary,
            "answers": [summary],
            "num_source_docs": 1,
            "metadata": {"title": title, "category": None, "lang": "vi"},
        }


def iter_wikilingua(
    raw_dir: Path, split: str
) -> Iterator[dict[str, Any]]:
    """Yield record từ ``wikilingua/<split>.json`` (thực chất là JSONL)."""

    path = raw_dir / WIKILINGUA_DIR / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy file WikiLingua: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            src = [str(s).strip() for s in row.get("src", []) if str(s).strip()]
            tgt = [str(s).strip() for s in row.get("tgt", []) if str(s).strip()]
            document = collapse_whitespace(" ".join(src))
            reference = collapse_whitespace(" ".join(tgt))

            if not document or not reference:
                yield {
                    "_skip": True,
                    "dataset": "wikilingua",
                    "source_id": str(index),
                    "reason": "document hoặc reference rỗng",
                }
                continue

            yield {
                "id": f"wikilingua-{split}-{index:06d}",
                "dataset": "wikilingua",
                "source_split": split,
                "source_index": index,
                "source_id": str(index),
                "task_type": "summarization",
                "document": document,
                "reference": reference,
                "answers": [reference],
                "num_source_docs": 1,
                "metadata": {
                    "title": None,
                    "category": None,
                    "lang": "vi",
                    "src_sentences": len(src),
                    "tgt_sentences": len(tgt),
                },
            }


def parse_vims_article(text: str) -> tuple[str, str]:
    """Tách ``Title`` và phần nội dung sau mốc ``Content:`` của một bài ViMs."""

    lines = text.split("\n")
    title = ""
    body_start: int | None = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Title:"):
            title = stripped[len("Title:"):].strip()
        elif stripped.startswith("Content:"):
            body_start = idx + 1
            break

    if body_start is None:
        # Không có mốc ``Content:``: bỏ các dòng header "Key:" ở đầu.
        body_start = 0
        for idx, line in enumerate(lines):
            if re.match(r"^(Title|Source|Link|Published Date|Author|Tags|Summary):", line.strip()):
                body_start = idx + 1
            else:
                break

    body = collapse_whitespace("\n".join(lines[body_start:]))
    return title, body


def iter_vims(raw_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield record ViMs: mỗi cluster là một mẫu tóm tắt đa văn bản."""

    root = raw_dir / VIMS_DIR
    orig_root = root / "original"
    sum_root = root / "summary"
    if not orig_root.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục ViMs: {orig_root}")

    clusters = sorted(orig_root.glob("Cluster_*"), key=numeric_key)
    for index, cluster in enumerate(clusters):
        article_dir = cluster / "original"
        articles = sorted(article_dir.glob("*.txt"), key=numeric_key) if article_dir.is_dir() else []

        parts: list[str] = []
        titles: list[str] = []
        for order, article in enumerate(articles, start=1):
            title, body = parse_vims_article(article.read_text(encoding="utf-8"))
            if not body:
                continue
            titles.append(title)
            header = f"### Tài liệu {order}: {title}" if title else f"### Tài liệu {order}"
            parts.append(f"{header}\n{body}")

        golds = []
        gold_dir = sum_root / cluster.name
        if gold_dir.is_dir():
            for gold in sorted(gold_dir.glob("*.gold.txt"), key=numeric_key):
                golds.append(collapse_whitespace(gold.read_text(encoding="utf-8")))
        golds = [g for g in golds if g]

        if not parts or not golds:
            yield {
                "_skip": True,
                "dataset": "vims",
                "source_id": cluster.name,
                "reason": f"{len(parts)} bài / {len(golds)} gold",
            }
            continue

        document = "\n\n".join(parts)
        yield {
            "id": f"vims-all-{cluster.name}",
            "dataset": "vims",
            "source_split": "all",
            "source_index": index,
            "source_id": cluster.name,
            "task_type": "summarization",
            "document": document,
            "reference": golds[0],
            "answers": golds,
            "num_source_docs": len(parts),
            "metadata": {
                "title": titles[0] if titles else None,
                "titles": titles,
                "category": None,
                "lang": "vi",
                "num_gold": len(golds),
            },
        }


def iter_vlsp(raw_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield record VLSP 2022 ABMUSU (tóm tắt đa văn bản)."""

    path = raw_dir / VLSP_DIR / "vlsp_2022_abmusu.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy file VLSP: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs = row.get("single_documents") or []
            summary = collapse_whitespace(str(row.get("summary") or ""))

            parts: list[str] = []
            titles: list[str] = []
            for order, doc in enumerate(docs, start=1):
                title = collapse_whitespace(str(doc.get("title") or ""))
                body = collapse_whitespace(str(doc.get("raw_text") or ""))
                if not body:
                    continue
                titles.append(title)
                header = f"### Tài liệu {order}: {title}" if title else f"### Tài liệu {order}"
                parts.append(f"{header}\n{body}")

            if not parts or not summary:
                yield {
                    "_skip": True,
                    "dataset": "vlsp",
                    "source_id": str(index),
                    "reason": f"{len(parts)} bài / summary {'có' if summary else 'rỗng'}",
                }
                continue

            category = row.get("category")
            yield {
                "id": f"vlsp-all-{index:04d}",
                "dataset": "vlsp",
                "source_split": "all",
                "source_index": index,
                "source_id": str(index),
                "task_type": "summarization",
                "document": "\n\n".join(parts),
                "reference": summary,
                "answers": [summary],
                "num_source_docs": len(parts),
                "metadata": {
                    "title": titles[0] if titles else None,
                    "titles": titles,
                    "category": collapse_whitespace(str(category)) if category else None,
                    "lang": "vi",
                },
            }


# --------------------------------------------------------------------------- #
# API chung
# --------------------------------------------------------------------------- #

def load_pool(
    dataset: str,
    raw_dir: Path | None = None,
    *,
    split: str | None = None,
    punct_clean: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load một pool và trả về ``(records, skipped)``.

    ``records`` chỉ chứa record hợp lệ; ``skipped`` ghi lại lý do từng record bị
    loại để manifest có thể báo cáo minh bạch.
    """

    if dataset not in POOLS:
        raise KeyError(f"Dataset không hỗ trợ: {dataset!r}. Có: {list(POOLS)}")
    raw_dir = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    split = split or POOLS[dataset]["split"]

    if dataset == "vietnews":
        stream = iter_vietnews(raw_dir, split, punct_clean=punct_clean)
    elif dataset == "wikilingua":
        stream = iter_wikilingua(raw_dir, split)
    elif dataset == "vims":
        stream = iter_vims(raw_dir)
    else:
        stream = iter_vlsp(raw_dir)

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in stream:
        if row.get("_skip"):
            skipped.append(row)
        else:
            records.append(row)
    return records, skipped


def load_all_pools(
    raw_dir: Path | None = None, *, punct_clean: bool = True
) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Load cả 4 pool theo ``DATASET_ORDER``."""

    return {
        name: load_pool(name, raw_dir, punct_clean=punct_clean)
        for name in DATASET_ORDER
    }


# --------------------------------------------------------------------------- #
# Thống kê & lấy mẫu
# --------------------------------------------------------------------------- #

def percentile(values: list[float], pct: float) -> float:
    """Percentile nội suy tuyến tính; ``pct`` trong [0, 100]."""

    if not values:
        raise ValueError("values rỗng")
    if not 0 <= pct <= 100:
        raise ValueError(f"pct ngoài [0,100]: {pct}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (pct / 100.0)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    frac = position - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


def length_stats(lengths: list[int]) -> dict[str, float]:
    """Thống kê mô tả cho một dãy độ dài."""

    if not lengths:
        return {}
    return {
        "count": len(lengths),
        "min": float(min(lengths)),
        "p1": percentile(lengths, 1),
        "p5": percentile(lengths, 5),
        "p25": percentile(lengths, 25),
        "median": percentile(lengths, 50),
        "mean": sum(lengths) / len(lengths),
        "p75": percentile(lengths, 75),
        "p95": percentile(lengths, 95),
        "p99": percentile(lengths, 99),
        "max": float(max(lengths)),
    }


def quantile_edges(values: list[int], bins: int) -> list[float]:
    """Biên bin theo quantile của ``values`` (trả về ``bins-1`` biên trong)."""

    if bins < 2:
        raise ValueError("bins phải >= 2")
    return [percentile(values, 100.0 * i / bins) for i in range(1, bins)]


def assign_bin(value: float, edges: list[float]) -> int:
    """Chỉ số bin của ``value`` theo ``edges`` đã sắp tăng."""

    index = 0
    for edge in edges:
        if value > edge:
            index += 1
        else:
            break
    return index


def evenly_spaced_indices(population_size: int, sample_size: int) -> list[int]:
    """Index cách đều trong ``[0, population_size)`` -- tái lập được, không random.

    Cùng công thức với ``data/extract_representative_samples.py`` của project
    tham chiếu để hai bên cho ra kết quả nhất quán.
    """

    if sample_size <= 0:
        raise ValueError("sample_size phải > 0")
    if population_size <= 0:
        return []
    if sample_size >= population_size:
        return list(range(population_size))
    if sample_size == 1:
        return [population_size // 2]
    return [
        round(i * (population_size - 1) / (sample_size - 1))
        for i in range(sample_size)
    ]


def _allocate(total: int, weights: list[int], floor: int) -> list[int]:
    """Chia ``total`` slot theo ``weights`` với sàn ``floor`` mỗi nhóm.

    Phần dư chia theo phương pháp largest-remainder để tổng luôn bằng ``total``
    (trừ khi ``floor * len(weights)`` đã vượt ``total``).
    """

    groups = len(weights)
    if groups == 0:
        return []
    population = sum(weights)
    if population <= 0:
        return [0] * groups

    floor = max(0, min(floor, total // groups))
    remaining = total - floor * groups
    if remaining < 0:
        floor = total // groups
        remaining = total - floor * groups

    exact = [remaining * w / population for w in weights]
    base = [int(x) for x in exact]
    allotment = [floor + b for b in base]

    leftover = total - sum(allotment)
    remainders = sorted(
        range(groups), key=lambda i: exact[i] - base[i], reverse=True
    )
    for i in remainders[: max(0, leftover)]:
        allotment[i] += 1

    return allotment


def filter_task_valid(
    records: list[dict[str, Any]],
    *,
    min_doc_words: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Loại record **không phải một instance tóm tắt hợp lệ**.

    Cắt OOD theo percentile chỉ xử lý được đuôi phân phối *tương đối*. Với bộ
    lệch mạnh như WikiLingua (p1 chỉ 24 từ, min 6 từ) thì vẫn lọt các mẫu suy
    biến: document 24 từ nhưng reference 73 từ, tức bản tóm tắt **dài hơn** văn
    bản nguồn -> ROUGE trên đó vô nghĩa và làm lệch điểm trung bình.

    Hai điều kiện, áp dụng trước khi lấy mẫu:

    1. ``document_words >= min_doc_words`` -- sàn độ dài tài liệu (0 = tắt).
    2. ``reference_words < document_words`` -- bản tóm tắt phải ngắn hơn nguồn.

    Trả về ``(kept, dropped)``; ``dropped`` ghi rõ lý do để manifest báo cáo.
    """

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in records:
        doc_words = int(row.get("document_words") or word_count(row["document"]))
        ref_words = int(row.get("reference_words") or word_count(row["reference"]))

        if min_doc_words and doc_words < min_doc_words:
            dropped.append(
                {
                    "id": row.get("id"),
                    "dataset": row.get("dataset"),
                    "reason": "document_duoi_san",
                    "document_words": doc_words,
                    "min_doc_words": min_doc_words,
                }
            )
            continue

        if ref_words >= doc_words:
            dropped.append(
                {
                    "id": row.get("id"),
                    "dataset": row.get("dataset"),
                    "reason": "reference_khong_ngan_hon_document",
                    "document_words": doc_words,
                    "reference_words": ref_words,
                }
            )
            continue

        kept.append(row)
    return kept, dropped


def stratified_sample(
    records: list[dict[str, Any]],
    samples: int,
    *,
    length_of: str = "document_words",
    bins: int = 5,
    clip_pct: tuple[float, float] = (1.0, 99.0),
    min_per_bin: int = 10,
) -> dict[str, Any]:
    """Chọn ``samples`` record đại diện, loại OOD, phân bổ theo phân phối độ dài.

    Quy trình:

    1. Tính độ dài document (``length_of``).
    2. **Loại OOD**: bỏ mẫu ngoài ``[clip_pct[0], clip_pct[1]]`` percentile.
    3. Chia ``bins`` bin theo quantile của phần đã lọc.
    4. Chia slot theo **tỷ lệ dân số từng bin** (bộ test phản ánh phân phối thật)
       với sàn ``min_per_bin``.
    5. Trong mỗi bin, lấy **điểm cách đều** theo độ dài => tái lập 100%.

    Trả về dict gồm ``selected`` (record đã gắn ``length_bin``), ``outliers`` và
    ``report`` (thống kê phục vụ manifest).
    """

    if not records:
        raise ValueError("records rỗng")
    if not 0 <= clip_pct[0] < clip_pct[1] <= 100:
        raise ValueError(f"clip_pct không hợp lệ: {clip_pct}")

    for row in records:
        row.setdefault(length_of, word_count(row["document"]))

    lengths = [int(row[length_of]) for row in records]
    low = percentile(lengths, clip_pct[0])
    high = percentile(lengths, clip_pct[1])

    inliers = [i for i, value in enumerate(lengths) if low <= value <= high]
    outliers = [i for i, value in enumerate(lengths) if not (low <= value <= high)]

    if not inliers:
        raise ValueError("Không còn mẫu sau khi loại OOD; nới --clip-pct")

    inlier_lengths = [lengths[i] for i in inliers]
    edges = quantile_edges(inlier_lengths, bins)

    buckets: list[list[int]] = [[] for _ in range(bins)]
    for i in inliers:
        buckets[assign_bin(lengths[i], edges)].append(i)
    for bucket in buckets:
        # Sắp theo (độ dài, index nguồn) để kết quả độc lập thứ tự đầu vào.
        bucket.sort(key=lambda i: (lengths[i], i))

    effective_bins = max(1, min(bins, samples))
    if effective_bins != bins:
        edges = quantile_edges(inlier_lengths, effective_bins)
        buckets = [[] for _ in range(effective_bins)]
        for i in inliers:
            buckets[assign_bin(lengths[i], edges)].append(i)
        for bucket in buckets:
            bucket.sort(key=lambda i: (lengths[i], i))

    weights = [len(b) for b in buckets]
    allotment = _allocate(samples, weights, min_per_bin)

    selected: list[dict[str, Any]] = []
    per_bin: list[dict[str, Any]] = []
    for bin_index, (bucket, want) in enumerate(zip(buckets, allotment)):
        want = min(want, len(bucket))
        picked = [bucket[i] for i in evenly_spaced_indices(len(bucket), want)]
        for i in picked:
            row = dict(records[i])
            row["length_bin"] = bin_index
            selected.append(row)
        per_bin.append(
            {
                "bin": bin_index,
                "range_words": [
                    round(edges[bin_index - 1], 1) if bin_index > 0 else round(low, 1),
                    round(edges[bin_index], 1) if bin_index < len(edges) else round(high, 1),
                ],
                "population": len(bucket),
                "allocated": want,
                "picked": len(picked),
            }
        )

    selected.sort(key=lambda r: (r["length_bin"], r.get(length_of, 0), r["id"]))

    report = {
        "requested_samples": samples,
        "selected_samples": len(selected),
        "clip_pct": list(clip_pct),
        "clip_bounds_words": [round(low, 1), round(high, 1)],
        "length_metric": length_of,
        "bins": effective_bins,
        "pool_size": len(records),
        "inlier_count": len(inliers),
        "outlier_count": len(outliers),
        "pool_length_stats": length_stats(lengths),
        "inlier_length_stats": length_stats(inlier_lengths),
        "per_bin": per_bin,
    }
    return {"selected": selected, "outliers": outliers, "report": report}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    """Ghi JSONL (ensure_ascii=False) và trả về sha256 của nội dung."""

    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True)
            handle.write(line + "\n")
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()
